"""
evaluate_shap.py - SHAP Aggregation for CompositeDelphi

Adapted from shap-agg-eval.py of the original Delphi implementation.

Uses the official SHAP library (shap.Explainer + shap.maskers.Text).
"""

import os
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import argparse
import shap
from torch.multiprocessing import Process, Queue
import traceback

from model import CompositeDelphiConfig, CompositeDelphi
from utils import get_batch_composite, get_p2i_composite

warnings.filterwarnings('ignore')


# =============================================================================
# SHAP Helper Functions (Probability-based computation)
# =============================================================================

def shap_custom_tokenizer(s, return_offsets_mapping=True):
    """Custom tokenizer for SHAP (from utils.py / shap-agg-eval.py)."""
    import re
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


def shap_model_creator_composite(model, disease_ids, person_tokens_ids, person_ages,
                                  person_shift, person_total, device):
    """
    SHAP model wrapper for CompositeDelphi.

    This function replicates the logic of shap_model_creator from shap-agg-eval.py:
      - Masked tokens (10000) are removed and the sequence is reconstructed.
      - Returns logits (not softmax probabilities) from the final time step.
    """
    def f(ps):
        xs_data = []
        xs_shift = []
        xs_total = []
        xs_ages = []

        for p in ps:
            if len(p) == 0:
                raise ValueError('No tokens found')
            p = list(map(int, p))

            new_tokens = []
            new_ages = []
            new_shift = []
            new_total = []

            for num, (masked, value, age, sh, tot) in enumerate(
                zip(p, person_tokens_ids, person_ages, person_shift, person_total)
            ):
                if num == 0:
                    # First token: handled identically to shap-agg-eval.py
                    new_ages.append(age)
                    new_shift.append(sh)
                    new_total.append(tot)
                    if masked == 10000:
                        # If masked: replace with alternate token (original implementation behavior)
                        new_tokens.append(2 if value == 3 else 3)
                    else:
                        new_tokens.append(value)
                else:
                    # Include only non-masked tokens or termination token (1)
                    if masked != 10000 or value == 1:
                        new_ages.append(age)
                        new_shift.append(sh)
                        new_total.append(tot)
                        new_tokens.append(value)

            xs_data.append(torch.tensor(new_tokens, device=device).unsqueeze(0))
            xs_shift.append(torch.tensor(new_shift, device=device).unsqueeze(0))
            xs_total.append(torch.tensor(new_total, device=device).unsqueeze(0))
            xs_ages.append(torch.tensor(new_ages, device=device, dtype=torch.float32).unsqueeze(0))

        # Pad all sequences in the batch to the maximum length
        max_length = max(x.shape[-1] for x in xs_data)

        xs_data = [torch.nn.functional.pad(x, (max_length - x.shape[-1], 0), value=0) for x in xs_data]
        xs_shift = [torch.nn.functional.pad(x, (max_length - x.shape[-1], 0), value=0) for x in xs_shift]
        xs_total = [torch.nn.functional.pad(x, (max_length - x.shape[-1], 0), value=0) for x in xs_total]
        xs_ages = [torch.nn.functional.pad(x, (max_length - x.shape[-1], 0), value=-10000) for x in xs_ages]

        x_data = torch.cat(xs_data)
        x_shift = torch.cat(xs_shift)
        x_total = torch.cat(xs_total)
        x_ages = torch.cat(xs_ages)

        with torch.no_grad():
            outputs, _, _ = model(x_data, x_shift, x_total, x_ages)
            # Return logits corresponding to disease_ids
            probs = outputs['data'][:, -1, disease_ids].detach().cpu().numpy()

        return probs

    return f


# =============================================================================
# GPU Worker Function
# =============================================================================
def gpu_worker(gpu_id, patient_indices, queue, ckpt_path, data_root, block_size,
               labels_path, debug=False):
    """
    Worker function executed on each GPU.

    Follows the original shap-agg-eval.py pipeline:
      1. Extract patient sequence via get_person
      2. Apply shap.maskers.Text + shap.Explainer
      3. Store (tokens, shap_values, time_passed, person_idx)
    """
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    torch.manual_seed(1337 + gpu_id)
    torch.cuda.manual_seed(1337 + gpu_id)

    print(f"[GPU {gpu_id}] Starting worker with {len(patient_indices)} patients")

    # Load validation dataset
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = dict(checkpoint['model_args'])
    eval_apply_token_shift = False  # must match get_batch_composite below
    if 'drug_token_min' not in model_args or 'drug_token_max' not in model_args:
        model_args['drug_token_min'] = 1279 if eval_apply_token_shift else 1278
        model_args['drug_token_max'] = 1289 if eval_apply_token_shift else 1288
        print(
            f"[GPU {gpu_id}] Checkpoint missing drug token range; using fallback "
            f"[{model_args['drug_token_min']}, {model_args['drug_token_max']}] "
            f"(apply_token_shift={eval_apply_token_shift})."
        )
    conf = CompositeDelphiConfig(**model_args)
    model = CompositeDelphi(conf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Load
    VAL_DATA_PATH = os.path.join(data_root, 'kr_val.bin')
    composite_dtype = np.dtype([
        ('ID', np.uint32),
        ('AGE', np.uint32),
        ('DATA', np.uint32),
        ('SHIFT', np.uint32),
        ('TOTAL', np.uint32)
    ])
    val = np.fromfile(VAL_DATA_PATH, dtype=composite_dtype)
    val_p2i = get_p2i_composite(val)

    # Load label mapping
    try:
        try:
            labels = pd.read_csv(labels_path, header=None, sep='\t')
            if labels.shape[1] == 1:
                labels = pd.read_csv(labels_path, header=None, sep=',')
        except Exception:
            labels = pd.read_csv(labels_path, header=None)
        id_to_token = labels[0].to_dict()
        token_to_id = {v: k for k, v in id_to_token.items()}
    except Exception as e:
        print(f"[GPU {gpu_id}] Warning: Could not load labels: {e}")
        id_to_token = {i: str(i) for i in range(conf.data_vocab_size)}
        token_to_id = {str(i): i for i in range(conf.data_vocab_size)}

    # Helper: extract patient sequence
    # Equivalent to get_person in shap-agg-eval.py
    def get_person(idx):
        batch = get_batch_composite(
            [idx], val, val_p2i,
            block_size=block_size,
            device=device,
            select='left',
            padding='random',
            no_event_token_rate=0,
            cut_batch=True,
            apply_token_shift=eval_apply_token_shift,
        )
        x_data, x_shift, x_total, x_ages = batch[0], batch[1], batch[2], batch[3]
        y_data, y_ages = batch[4], batch[7]

        # valid mask (x_data > 0 and y_data > -1)
        # shap-agg-eval.py: x, y = x[y > -1], y[y > -1]
        valid = y_data[0] > -1
        x_d = x_data[0][valid]
        x_s = x_shift[0][valid]
        x_t = x_total[0][valid]
        y_d = y_data[0][valid]
        x_a = x_ages[0][valid]
        y_a = y_ages[0][valid]

        # Construct person list as [(token_name, age), ...]
        person = []
        token_ids = x_d.cpu().numpy().tolist()
        ages = x_a.cpu().numpy().tolist()
        shifts = x_s.cpu().numpy().tolist()
        totals = x_t.cpu().numpy().tolist()

        for tid, age in zip(token_ids, ages):
            name = id_to_token.get(tid, str(tid))
            person.append((name, age))

        return person, token_ids, ages, shifts, totals, y_a

    # ----- SHAP computation loop (structure identical to shap-agg-eval.py) -----
    shaply_val = []
    error_count = 0

    for person_idx in tqdm(patient_indices, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            person_to_process, person_tokens_ids, person_ages, person_shift, person_total, target_ages = \
                get_person(person_idx)

            if len(person_tokens_ids) < 2:
                continue

            # time_passed:
            # In shap-agg-eval.py, (time_target - time) is stored.
            # Here we store the per-token time difference accordingly.
            time_target_val = target_ages[-1].cpu().item()
            time_passed = np.array(
                [time_target_val - a for a in person_ages], dtype=np.float32
            )

            # Same configuration as shap-agg-eval.py:
            # shap.maskers.Text + shap.Explainer
            masker = shap.maskers.Text(
                shap_custom_tokenizer,
                output_type='str',
                mask_token='10000',
                collapse_mask_token=False,
            )

            model_shap = shap_model_creator_composite(
                model,
                labels.index.values,   # disease_ids = full label index
                person_tokens_ids,
                person_ages,
                person_shift,
                person_total,
                device,
            )

            explainer = shap.Explainer(
                model_shap,
                masker,
                output_names=labels[0].values,
            )

            # Input string constructed using token_to_id mapping
            input_str = ' '.join([str(token_to_id.get(tok, tid))
                                  for tok, tid in zip(
                                      [p[0] for p in person_to_process],
                                      person_tokens_ids)])
            shap_values = explainer([input_str])

            # Store (token_name(age), ...) format in shap_values.data for visualization
            shap_values.data = np.array([
                [f"{p[0]}({p[1]/365:.1f}) " for p in person_to_process]
            ])

            shaply_val.append((
                np.array(person_tokens_ids, dtype=np.uint32),
                shap_values.values.astype(np.float16),
                time_passed,
                [person_idx] * len(person_tokens_ids),
            ))

            if debug and len(shaply_val) <= 3:
                print(f"\n[GPU {gpu_id}] Patient {person_idx}:")
                print(f"  Seq len: {len(person_tokens_ids)}")
                print(f"  SHAP shape: {shap_values.values.shape}")
                print(f"  SHAP range: [{shap_values.values.min():.6f}, {shap_values.values.max():.6f}]")

        except Exception as e:
            error_count += 1
            if debug and error_count <= 5:
                print(f"[GPU {gpu_id}] Error at patient {person_idx}: {repr(e)}")
                traceback.print_exc()

    print(f"\n[GPU {gpu_id}] Done: {len(shaply_val)} succeeded, {error_count} errors")
    queue.put((gpu_id, shaply_val))


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='SHAP aggregation for CompositeDelphi')
    parser.add_argument('--gpus', type=str, default='1,2,3,4',
                        help='GPU IDs to use (comma-separated)')
    parser.add_argument('--ckpt', type=str, default='./out-composite-large-shift/ckpt_0127.pt',
                        help='Checkpoint path')
    parser.add_argument('--data_root', type=str, default='/home/user02/GPT/data',
                        help='Data root directory')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to labels.csv (default: {data_root}/labels.csv)')
    parser.add_argument('--block_size', type=int, default=512)
    parser.add_argument('--output', type=str, default='shap_agg_composite.pickle')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--test_run', type=int, default=None,
                        help='Test run with N patients per GPU')
    args = parser.parse_args()

    if args.labels is None:
        args.labels = os.path.join(args.data_root, 'labels.csv')

    gpu_ids = [int(x) for x in args.gpus.split(',')]
    print(f"Using GPUs: {gpu_ids}")
    print(f"Method: shap.Explainer + shap.maskers.Text (same as shap-agg-eval.py)")
    print(f"Checkpoint: {args.ckpt}")

    # Determine total number of patients
    VAL_DATA_PATH = os.path.join(args.data_root, 'kr_val.bin')
    composite_dtype = np.dtype([
        ('ID', np.uint32), ('AGE', np.uint32), ('DATA', np.uint32),
        ('SHIFT', np.uint32), ('TOTAL', np.uint32)
    ])
    val = np.fromfile(VAL_DATA_PATH, dtype=composite_dtype)
    val_p2i = get_p2i_composite(val)
    total_patients = len(val_p2i)
    print(f"Total patients: {total_patients}")

    if args.test_run:
        print(f"TEST RUN: {args.test_run} patients per GPU")
        total_patients = min(total_patients, args.test_run * len(gpu_ids))

    # Split patients across GPUs
    patients_per_gpu = total_patients // len(gpu_ids)
    patient_splits = []
    for i, gid in enumerate(gpu_ids):
        start = i * patients_per_gpu
        end = (i + 1) * patients_per_gpu if i < len(gpu_ids) - 1 else total_patients
        if args.test_run:
            end = min(end, start + args.test_run)
        patient_splits.append(list(range(start, end)))
        print(f"  GPU {gid}: patients {start}-{end - 1} ({len(patient_splits[-1])} total)")

    # Launch multiprocessing workers
    queue = Queue()
    processes = []
    for i, gid in enumerate(gpu_ids):
        p = Process(target=gpu_worker, args=(
            gid, patient_splits[i], queue, args.ckpt, args.data_root,
            args.block_size, args.labels, args.debug,
        ))
        p.start()
        processes.append(p)

    # Collect results from workers
    all_results = {}
    for _ in range(len(gpu_ids)):
        gid, results = queue.get()
        all_results[gid] = results
        print(f"Main: Received {len(results)} samples from GPU {gid}")

    for p in processes:
        p.join()

    # Merge results (same aggregation format as shap-agg-eval.py)
    shaply_val = []
    for gid in sorted(all_results.keys()):
        shaply_val.extend(all_results[gid])

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {len(shaply_val)} patients processed")
    print(f"{'=' * 60}")

    if len(shaply_val) == 0:
        print("No SHAP values generated.")
        return

    # Aggregation strategy identical to shap-agg-eval.py
    all_tokens = np.concatenate([i[0] for i in shaply_val])
    all_values = np.concatenate([i[1] for i in shaply_val], axis=1)[0]
    # time_passed: each patient contributes a 1-D array → concatenate
    all_times_passed = np.concatenate([np.asarray(i[2]).ravel() for i in shaply_val])
    all_people = np.concatenate([np.asarray(i[3]).ravel() for i in shaply_val])

    # Save results (same key structure as shap-agg-eval.py)
    with open(args.output, 'wb') as f:
        pickle.dump({
            'tokens': all_tokens,
            'values': all_values,
            'times': all_times_passed,
            'model': args.ckpt,
            'people': all_people,
        }, f)

    print(f"Saved to {args.output}")
    print(f"  tokens: {all_tokens.shape}")
    print(f"  values: {all_values.shape}")
    print(f"  times:  {all_times_passed.shape}")
    print(f"  people: {all_people.shape}")


if __name__ == '__main__':
    main()
