"""
common.py - Shared utilities for model & data loading
======================================================
Centralizes model loading, data loading, and dataset/dataloader creation
so that each notebook doesn't reinvent the wheel.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: ensure project root is importable
# ---------------------------------------------------------------------------
FIGURE_DIR = Path(__file__).resolve().parent.parent  # /gpt/figure
PROJECT_ROOT = FIGURE_DIR.parent                      # /gpt

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Composite binary dtype (shared across all data loading)
# ---------------------------------------------------------------------------
COMPOSITE_DTYPE = np.dtype([
    ('ID',    np.uint32),
    ('AGE',   np.uint32),
    ('DATA',  np.uint32),
    ('SHIFT', np.float32),
    ('TOTAL', np.uint32),
])


def load_model(ckpt_path, device='cpu', strip_prefix=True):
    """
    Load a CompositeDelphi model from a checkpoint.

    Parameters
    ----------
    ckpt_path : str or Path
        Path to the .pt checkpoint file.
    device : str
        Target device ('cpu', 'cuda', 'cuda:0', …).
    strip_prefix : bool
        If True, remove '_orig_mod.' prefix from state-dict keys
        (common when checkpoints are saved from torch.compile).

    Returns
    -------
    model : CompositeDelphi
        Model in eval mode on the requested device.
    checkpoint : dict
        Raw checkpoint dictionary (contains 'model_args', 'iter_num', etc.).
    """
    from model import CompositeDelphi, CompositeDelphiConfig

    print(f"[INFO] Loading model from {ckpt_path} → {device}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    state_dict = checkpoint['model']
    if strip_prefix:
        prefix = '_orig_mod.'
        state_dict = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }

    # ── Auto-detect model version from state_dict keys ──
    # model_v7: MDN shift_head (shift_head.proj.*)      — continuous regression
    # model_v2: BinaryChangeHead (shift_head.head.*)     — binary classification
    # model_v3: HierarchicalShiftHead (shift_head.change_head.* + direction_head.*)
    # model (v1): original Sequential (shift_head.0.weight)
    has_mdn_shift = any('shift_head.proj.' in k for k in state_dict)
    has_hierarchical = any('shift_head.change_head' in k for k in state_dict)
    has_binary_change = any('shift_head.head.' in k for k in state_dict)

    if has_mdn_shift and not has_binary_change:
        # v7: MDN continuous shift
        try:
            from model import CompositeDelphi, CompositeDelphiConfig
            print("[INFO] Detected MDN ShiftHead → using model_v7")
        except ImportError:
            try:
                from model import CompositeDelphi, CompositeDelphiConfig
                print("[WARN] model_v7 not found, trying model_v2")
            except ImportError:
                from model import CompositeDelphi, CompositeDelphiConfig
                print("[WARN] model_v7/v2 not found, falling back to model")
    elif has_binary_change:
        # v2: BinaryChangeHead
        try:
            from model import CompositeDelphi, CompositeDelphiConfig
            print("[INFO] Detected BinaryChangeHead → using model_v2")
        except ImportError:
            from model import CompositeDelphi, CompositeDelphiConfig
            print("[WARN] model_v2 not found, falling back to model")
    elif has_hierarchical:
        # v3: HierarchicalShiftHead
        try:
            from model import CompositeDelphi, CompositeDelphiConfig
            print("[INFO] Detected HierarchicalShiftHead → using model_v3")
        except ImportError:
            from model import CompositeDelphi, CompositeDelphiConfig
            print("[WARN] model_v3 not found, falling back to model")
    else:
        # v1: Original nn.Sequential shift_head
        from model import CompositeDelphi, CompositeDelphiConfig
        print("[INFO] Detected original Sequential ShiftHead → using model")

    # Filter model_args to only include fields defined in CompositeDelphiConfig
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(CompositeDelphiConfig)}
    model_args = {k: v for k, v in checkpoint['model_args'].items() if k in valid_fields}
    skipped = set(checkpoint['model_args']) - valid_fields
    if skipped:
        print(f"[WARN] Skipped unknown config fields: {skipped}")

    config = CompositeDelphiConfig(**model_args)
    model = CompositeDelphi(config)

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"[WARN] Missing keys (initialized randomly): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[WARN] Unexpected keys (ignored): {result.unexpected_keys}")
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[OK]  Model loaded ({n_params:.2f}M params)")
    return model, checkpoint


def load_composite_data(data_path):
    """
    Load a composite binary dataset.

    Parameters
    ----------
    data_path : str or Path
        Path to the .bin file.

    Returns
    -------
    data_raw : np.ndarray
        Structured array with fields (ID, AGE, DATA, SHIFT, TOTAL).
    data_2d : np.ndarray
        Same data reshaped to (N, 5) uint32 — needed by get_batch variants.
    p2i : np.ndarray
        Patient-to-index pointer array.
    """
    # Import from project-root utils.py (not our figutils package)
    from utils import get_p2i_composite

    print(f"[INFO] Loading data from {data_path}")
    data_raw = np.fromfile(str(data_path), dtype=COMPOSITE_DTYPE)
    data_2d = data_raw.view(np.uint32).reshape(-1, 5)

    p2i = get_p2i_composite(data_raw)
    print(f"[OK]  {len(data_raw):,} events, {len(p2i):,} patients")
    return data_raw, data_2d, p2i


def make_dataloader(data_raw, p2i, block_size=512, batch_size=32,
                    apply_token_shift=True, max_patients=-1,
                    shift_continuous=False, separate_shift_na_from_padding=False,
                    shift_na_raw_token=4, model=None):
    """
    Create a PyTorch DataLoader for CompositeDelphi evaluation.

    Parameters
    ----------
    data_raw : np.ndarray
    p2i : np.ndarray
    block_size : int
    batch_size : int
    apply_token_shift : bool
    max_patients : int
    shift_continuous : bool
    separate_shift_na_from_padding : bool
    shift_na_raw_token : int
    model : nn.Module, optional
        If provided, reads config from model to auto-set tokenization params.
    """
    from torch.utils.data import DataLoader, Dataset
    from utils import get_batch_composite

    # Auto-read config from model
    if model is not None:
        config = getattr(model, 'config', None)
        if config is not None:
            apply_token_shift = bool(getattr(config, 'apply_token_shift', apply_token_shift))
            shift_continuous = bool(getattr(config, 'shift_continuous', shift_continuous))
            separate_shift_na_from_padding = bool(getattr(config, 'separate_shift_na_from_padding', separate_shift_na_from_padding))
            shift_na_raw_token = int(getattr(config, 'shift_na_raw_token', shift_na_raw_token))

    if 0 < max_patients < len(p2i):
        p2i = p2i[:max_patients]
        print(f"[INFO] Limited to {max_patients} patients")

    _apply_ts = apply_token_shift
    _shift_cont = shift_continuous
    _sep_na = separate_shift_na_from_padding
    _na_tok = shift_na_raw_token

    class _Dataset(Dataset):
        def __init__(self):
            self.data = data_raw
            self.p2i = p2i
            self.block_size = block_size

        def __len__(self):
            return len(self.p2i)

        def __getitem__(self, idx):
            ix = torch.tensor([idx])
            return get_batch_composite(
                ix, self.data, self.p2i,
                block_size=self.block_size,
                device='cpu',
                select='left',
                padding='none',
                no_event_token_rate=0,
                cut_batch=True,
                apply_token_shift=_apply_ts,
                shift_continuous=_shift_cont,
                separate_shift_na_from_padding=_sep_na,
                shift_na_raw_token=_na_tok,
            )

    def _collate(batch):
        max_len = max(item[0].shape[1] for item in batch)

        def _pad(tensor, pad_val=0):
            if tensor.shape[1] < max_len:
                p = torch.full(
                    (tensor.shape[0], max_len - tensor.shape[1]),
                    pad_val, dtype=tensor.dtype,
                )
                return torch.cat([tensor, p], dim=1)
            return tensor

        # batch items: (x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages)
        pad_vals = [0, 0, 0, -10000, 0, 0, 0, -10000]
        return tuple(
            torch.cat([_pad(item[i], pad_vals[i]) for item in batch], dim=0)
            for i in range(8)
        )

    ds = _Dataset()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=0, collate_fn=_collate)
    print(f"[OK]  DataLoader: {len(dl)} batches (bs={batch_size})")
    return dl