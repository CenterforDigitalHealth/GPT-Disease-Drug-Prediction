import numpy as np
import torch
import re

def get_batch_composite(ix, data, p2i, select='center', index='patient', padding='regular',
                        block_size=48, device='cpu', lifestyle_augmentations=False,
                        no_event_token_rate=5, cut_batch=False, apply_token_shift=True,
                        shift_continuous=False, separate_shift_na_from_padding=False, shift_na_raw_token=4,
                        dose_continuous=None, separate_dose_na_from_padding=None, dose_na_raw_token=None):
    """
    Get a batch of composite data (DATA, DOSE/SHIFT, DURATION/TOTAL) from the dataset.
    
    Args:
        ix: list of indices to get data from
        data: structured numpy array with fields:
            - modern: (ID, AGE, DATA, DOSE, DURATION)
            - legacy: (ID, AGE, DATA, SHIFT, TOTAL)
        p2i: numpy array of the patient to index mapping
        select: 'left', 'right', 'random'
        index: 'patient', 'random'
        padding: 'regular', 'random', 'none'
        block_size: size of the block to get
        device: 'cpu' or 'cuda'
        lifestyle_augmentations: whether to perform augmentations
        no_event_token_rate: average rate of "no event" tokens in years
        cut_batch: whether to cut the batch to the smallest size possible
        apply_token_shift: whether to shift tokens by +1 to reserve 0 for padding
        shift_continuous: legacy name for whether DOSE/SHIFT is continuous
        separate_shift_na_from_padding: legacy name for DOSE/SHIFT N/A remapping
        shift_na_raw_token: legacy name for DOSE/SHIFT N/A token id
        dose_continuous: alias of shift_continuous (preferred name)
        separate_dose_na_from_padding: alias of separate_shift_na_from_padding
        dose_na_raw_token: alias of shift_na_raw_token

    Returns:
        x_data: input DATA tokens (B, T)
        x_shift: input SHIFT values (B, T)  
        x_total: input TOTAL values (B, T)
        ages: input ages (B, T)
        y_data: target DATA tokens (B, T)
        y_shift: target SHIFT values (B, T)
        y_total: target TOTAL values (B, T)
        y_ages: target ages (B, T)
    """
    # Backward-compatible argument aliases (preferred: dose_* names).
    if dose_continuous is not None:
        shift_continuous = bool(dose_continuous)
    if separate_dose_na_from_padding is not None:
        separate_shift_na_from_padding = bool(separate_dose_na_from_padding)
    if dose_na_raw_token is not None:
        shift_na_raw_token = int(dose_na_raw_token)

    # Backward-compatible field aliases (preferred: DOSE/DURATION).
    field_names = set(data.dtype.names)
    dose_field = 'DOSE' if 'DOSE' in field_names else 'SHIFT'
    dur_field = 'DURATION' if 'DURATION' in field_names else 'TOTAL'

    mask_time = -10000.

    x = torch.tensor(np.array([p2i[int(i)] for i in ix]))
    ix = torch.tensor(np.array(ix))

    gen = torch.Generator(device='cpu')
    gen.manual_seed(ix.sum().item())

    if index == 'patient':
        if select == 'left':
            traj_start_idx = x[:, 0]
        elif select == 'right':
            traj_start_idx = torch.clamp(x[:, 0] + x[:, 1] - block_size - 1, 0, data.shape[0])
        elif select == 'random':
            traj_start_idx = x[:, 0] + (torch.randint(2**63-1, (len(ix),), generator=gen) % torch.clamp(x[:, 1] - block_size, 1))
            traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0] - block_size - 1)
    traj_start_idx = traj_start_idx.numpy()

    batch_idx = np.arange(block_size + 1)[None, :] + traj_start_idx[:, None]

    # Create mask for valid patient data
    mask = torch.from_numpy(data['ID'][batch_idx].astype(np.int64))
    mask = mask == torch.tensor(data['ID'][p2i[ix.numpy()][:, 0]][:, None].astype(np.int64)).to(mask.dtype)

    # Extract all fields
    data_tokens = torch.from_numpy(data['DATA'][batch_idx].astype(np.int64))
    if shift_continuous:
        shift_values = torch.from_numpy(data[dose_field][batch_idx].astype(np.float32))
    else:
        shift_values = torch.from_numpy(data[dose_field][batch_idx].astype(np.int64))
    total_values = torch.from_numpy(data[dur_field][batch_idx].astype(np.int64))
    ages = torch.from_numpy(data['AGE'][batch_idx].astype(np.float32))

    # Mask invalid data
    data_tokens = data_tokens.masked_fill(~mask, -1)
    shift_values = shift_values.masked_fill(~mask, -1)
    total_values = total_values.masked_fill(~mask, -1)
    ages = ages.masked_fill(~mask, mask_time)

    # Optional: keep real SHIFT N/A (raw 0) distinct from no-event/padding (0).
    # Apply only to real rows (mask=True), before inserting synthetic no-event tokens.
    if separate_shift_na_from_padding and not shift_continuous:
        na_mask = (shift_values == 0) & mask
        shift_values = shift_values.masked_fill(na_mask, int(shift_na_raw_token))

    # Insert "no event" tokens
    if (padding.lower() == 'none' or
            padding is None or
            no_event_token_rate == 0 or
            no_event_token_rate is None):
        pad = torch.ones(len(ix), 0)
    elif padding == 'regular':
        pad = torch.arange(0, 36525, 365.25 * no_event_token_rate) * torch.ones(len(ix), 1) + 1
    elif padding == 'random':
        pad = torch.randint(1, 36525, (len(ix), int(100 / no_event_token_rate)), generator=gen)
    else:
        raise NotImplementedError
    
    m = ages.max(1, keepdim=True).values
    n_pad = pad.shape[1]

    # Stack "no event" tokens with real tokens
    # Fix: Padding should always be 0 (Ignore Index), not 1 (Decrease)
    no_event_token = 0
    data_tokens = torch.hstack([
        data_tokens,
        torch.full((len(ix), n_pad), no_event_token, dtype=torch.long),
    ])
    if shift_continuous:
        shift_pad = torch.full((len(ix), n_pad), float(no_event_token), dtype=torch.float32)
    else:
        shift_pad = torch.full((len(ix), n_pad), no_event_token, dtype=torch.long)
    shift_values = torch.hstack([shift_values, shift_pad])
    total_values = torch.hstack([total_values, torch.full((len(ix), n_pad), no_event_token, dtype=torch.long)])
    ages = torch.hstack([ages, pad])

    # Mask out tokens that are too far in the future
    data_tokens = data_tokens.masked_fill(ages > m, -1)
    shift_values = shift_values.masked_fill(ages > m, -1)
    total_values = total_values.masked_fill(ages > m, -1)
    ages = ages.masked_fill(ages > m, mask_time)

    # Sort by age
    s = torch.argsort(ages, 1)
    data_tokens = torch.gather(data_tokens, 1, s)
    shift_values = torch.gather(shift_values, 1, s)
    total_values = torch.gather(total_values, 1, s)
    ages = torch.gather(ages, 1, s)

    # Shift tokens by 1 (0 is reserved for padding)
    if apply_token_shift:
        data_tokens = data_tokens + 1
        total_values = total_values + 1
        if not shift_continuous:
            shift_values = shift_values + 1
    else:
        data_tokens = data_tokens.clamp_min(0)
        total_values = total_values.clamp_min(0)
        if not shift_continuous:
            shift_values = shift_values.clamp_min(0)

    # Cut padded tokens if possible
    if cut_batch:
        cut_margin = torch.min(torch.sum(data_tokens == 0, 1))
        data_tokens = data_tokens[:, cut_margin:]
        shift_values = shift_values[:, cut_margin:]
        total_values = total_values[:, cut_margin:]
        ages = ages[:, cut_margin:]

    # Cut to maintain block size
    if data_tokens.shape[1] > block_size + 1:
        cut_margin = data_tokens.shape[1] - block_size - 1
        data_tokens = data_tokens[:, cut_margin:]
        shift_values = shift_values[:, cut_margin:]
        total_values = total_values[:, cut_margin:]
        ages = ages[:, cut_margin:]

    # Split into input (x) and target (y)
    x_data = data_tokens[:, :-1]
    x_shift = shift_values[:, :-1]
    x_total = total_values[:, :-1]
    x_ages = ages[:, :-1]
    
    y_data = data_tokens[:, 1:]
    y_shift = shift_values[:, 1:]
    y_total = total_values[:, 1:]
    y_ages = ages[:, 1:]

    # Mask "no event" tokens
    x_data = x_data.masked_fill((x_data == 0) * (y_data == 1), 0)
    y_data = y_data.masked_fill(x_data == 0, 0)
    y_ages = y_ages.masked_fill(x_data == 0, mask_time)

    if device == 'cuda':
        x_data, x_shift, x_total, x_ages = [
            i.pin_memory().to(device, non_blocking=True) 
            for i in [x_data, x_shift, x_total, x_ages]
        ]
        y_data, y_shift, y_total, y_ages = [
            i.pin_memory().to(device, non_blocking=True) 
            for i in [y_data, y_shift, y_total, y_ages]
        ]
    else:
        x_data = x_data.to(device)
        x_shift = x_shift.to(device)
        x_total = x_total.to(device)
        x_ages = x_ages.to(device)
        y_data = y_data.to(device)
        y_shift = y_shift.to(device)
        y_total = y_total.to(device)
        y_ages = y_ages.to(device)

    return (x_data, x_shift, x_total, x_ages,
            y_data, y_shift, y_total, y_ages)


def get_p2i_composite(data):
    """
    Get the patient to index mapping for composite data.
    
    Args:
        data: structured numpy array with 'ID' field
    """
    px = data['ID'].astype('int')
    p2i = []
    j = 0
    q = px[0]
    for i, p in enumerate(px):
        if p != q:
            p2i.append([j, i - j])
            q = p
            j = i
        if i == len(px) - 1:
            p2i.append([j, i - j + 1])
    return np.array(p2i)


def shap_custom_tokenizer(s, return_offsets_mapping=True):
    """Custom tokenizers conform to a subset of the transformers API."""
    pos = 0
    offset_ranges = []
    input_ids = []
    for m in re.finditer(r"\W", s):
        start, end = m.span(0)
        offset_ranges.append((pos, start))
        input_ids.append(s[pos:start])
        pos = end
    if pos != len(s):
        offset_ranges.append((pos, len(s)))
        input_ids.append(s[pos:])
    out = {}
    out["input_ids"] = input_ids
    if return_offsets_mapping:
        out["offset_mapping"] = offset_ranges
    return out
