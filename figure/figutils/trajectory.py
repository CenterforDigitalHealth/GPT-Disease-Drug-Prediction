"""
trajectory.py - Synthetic trajectory generation & rate-vs-age plots
=====================================================================
Adapted from delphi-sampling_trajectories.ipynb for CompositeDelphi.

Key figures:
  - Rate vs Age curves (Delphi Fig 2a style)
  - Predicted vs Observed incidence comparison
  - Lifestyle factor analysis
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_here))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils import get_batch_composite, get_p2i_composite


# =============================================================================
# Helper: aggregate rates over age bins
# =============================================================================

def _compute_incidence(tokens, ages, n_tokens, max_age=80):
    """
    Compute per-token incidence rate by yearly age bin.

    Parameters
    ----------
    tokens : np.ndarray  (N, T) int token IDs
    ages : np.ndarray    (N, T) float age in years
    n_tokens : int       total vocabulary size
    max_age : int

    Returns
    -------
    inc : np.ndarray (n_tokens, max_age)  incidence rate per year
    """
    ages_int = np.clip(np.nan_to_num(ages, nan=-1).astype(int), -1, max_age - 1)
    inc = np.zeros((n_tokens, max_age), dtype=np.float64)
    for t in range(max_age):
        mask = (ages_int == t)
        if mask.sum() == 0:
            continue
        counts = np.bincount(tokens[mask].flatten(), minlength=n_tokens)
        # Rate = count / person-years at risk
        n_at_risk = mask.any(axis=1).sum()
        if n_at_risk > 0:
            inc[:, t] = counts / max(n_at_risk, 1)
    return inc


# =============================================================================
# Generate synthetic trajectories
# =============================================================================

def generate_trajectories(model, val_data, val_p2i, device,
                          cutoff_age=60, n_patients=500,
                          max_new_tokens=100, block_size=512):
    """
    Generate synthetic trajectories from a cutoff age.

    1. Load real patient histories up to cutoff_age
    2. Use model.generate() to extend trajectories
    3. Return both real and synthetic token sequences

    Parameters
    ----------
    model : CompositeDelphi
    val_data : structured numpy array
    val_p2i : np.ndarray
    device : str
    cutoff_age : float  (years)
    n_patients : int
    max_new_tokens : int
    block_size : int

    Returns
    -------
    dict with keys:
        real_tokens, real_ages : observed data (N, T) numpy
        syn_tokens, syn_ages : generated continuations (N, T') numpy
        cutoff_age : float
    """
    config = getattr(model, 'config', None)
    apply_ts = bool(getattr(config, 'apply_token_shift', False))
    shift_cont = bool(getattr(config, 'shift_continuous', False))
    sep_na = bool(getattr(config, 'separate_shift_na_from_padding', False))
    na_tok = int(getattr(config, 'shift_na_raw_token', 4))

    print(f"[INFO] Generating trajectories: cutoff={cutoff_age}yr, n={n_patients}")

    # Get full patient data
    n_patients = min(n_patients, len(val_p2i))
    batch = get_batch_composite(
        range(n_patients), val_data, val_p2i,
        select='left', block_size=block_size,
        device=device, padding='random',
        apply_token_shift=apply_ts,
        shift_continuous=shift_cont,
        separate_shift_na_from_padding=sep_na,
        shift_na_raw_token=na_tok,
    )
    x_data, x_shift, x_total, x_ages = [b for b in batch[:4]]
    y_data, y_shift, y_total, y_ages = [b for b in batch[4:]]

    cutoff_days = cutoff_age * 365.25

    # Truncate to cutoff age
    d0 = x_data.clone()
    s0 = x_shift.clone()
    t0 = x_total.clone()
    a0 = x_ages.clone()

    d0[a0 > cutoff_days] = 0
    s0[a0 > cutoff_days] = 0
    t0[a0 > cutoff_days] = 0
    a0[a0 > cutoff_days] = -10000.

    # Filter: keep only patients who have data both before and after cutoff
    has_before = (a0 > 0).any(dim=1)
    has_after = (x_ages > cutoff_days).any(dim=1)
    valid = has_before & has_after
    if valid.sum() == 0:
        print("[WARN] No patients span the cutoff age")
        return None

    d0 = d0[valid]
    s0 = s0[valid]
    t0 = t0[valid]
    a0 = a0[valid]

    # Store real data for comparison
    real_tokens = x_data[valid].cpu().numpy()
    real_ages = x_ages[valid].cpu().numpy() / 365.25
    real_y_tokens = y_data[valid].cpu().numpy()
    real_y_ages = y_ages[valid].cpu().numpy() / 365.25

    print(f"  {valid.sum().item()} patients with data spanning cutoff")

    # Generate in batches
    gen_batch = 64
    all_syn_tokens = []
    all_syn_ages = []

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(d0), gen_batch), desc="Generating"):
            end = min(start + gen_batch, len(d0))
            out = model.generate(
                d0[start:end].to(device),
                s0[start:end].to(device),
                t0[start:end].to(device),
                a0[start:end].to(device),
                max_new_tokens=max_new_tokens,
                max_age=85 * 365.25,
            )
            # generate returns (data, shift, total, age, logits)
            gen_data = out[0].cpu().numpy()
            gen_age = out[3].cpu().numpy() / 365.25
            all_syn_tokens.append(gen_data)
            all_syn_ages.append(gen_age)

    # Pad and concatenate
    max_len = max(a.shape[1] for a in all_syn_tokens)
    syn_tokens = np.concatenate([
        np.pad(a, ((0, 0), (0, max_len - a.shape[1])), constant_values=0)
        for a in all_syn_tokens
    ])
    syn_ages = np.concatenate([
        np.pad(a, ((0, 0), (0, max_len - a.shape[1])), constant_values=-10000)
        for a in all_syn_ages
    ])

    print(f"[OK]  Generated {len(syn_tokens)} trajectories, max_len={max_len}")

    return {
        'real_tokens': real_y_tokens,
        'real_ages': real_y_ages,
        'syn_tokens': syn_tokens,
        'syn_ages': syn_ages,
        'cutoff_age': cutoff_age,
        'n_patients': len(syn_tokens),
    }


# =============================================================================
# Rate vs Age plot (Delphi Fig 2a)
# =============================================================================

def plot_rate_vs_age(model, val_data, val_p2i, device,
                     diseases, labels_path=None,
                     n_patients=2000, block_size=512,
                     figsize=None, save_path=None):
    """
    Plot predicted rate vs age for selected diseases (Delphi Fig 2a style).

    For each token position, compute softmax of data logits → per-disease rate.
    Plot rate (y, log scale) vs age (x) as scatter + observed training rate.

    Parameters
    ----------
    model : CompositeDelphi
    val_data, val_p2i : validation data
    device : str
    diseases : list of int or list of str
        Token IDs or disease names to plot
    labels_path : str
        Path to labels_chapter.csv for name lookup
    n_patients : int
    block_size : int
    """
    config = getattr(model, 'config', None)
    data_vocab = int(getattr(config, 'data_vocab_size', 1290))
    apply_ts = bool(getattr(config, 'apply_token_shift', False))
    shift_cont = bool(getattr(config, 'shift_continuous', False))
    sep_na = bool(getattr(config, 'separate_shift_na_from_padding', False))
    na_tok = int(getattr(config, 'shift_na_raw_token', 4))

    # Load labels
    token_names = {}
    token_colors = {}
    if labels_path and os.path.exists(labels_path):
        df = pd.read_csv(labels_path)
        for _, row in df.iterrows():
            tid = int(row['token_id'])
            if apply_ts:
                tid += 1
            token_names[tid] = row['name']
            token_colors[tid] = row.get('color', '#999999')

    # Resolve disease names to IDs
    name_to_id = {v: k for k, v in token_names.items()}
    disease_ids = []
    for d in diseases:
        if isinstance(d, str):
            if d in name_to_id:
                disease_ids.append(name_to_id[d])
            else:
                print(f"[WARN] Disease '{d}' not found in labels")
        else:
            disease_ids.append(int(d))

    if not disease_ids:
        print("No valid diseases to plot")
        return

    n_diseases = len(disease_ids)
    n_patients = min(n_patients, len(val_p2i))

    print(f"[INFO] Computing rates for {n_diseases} diseases, {n_patients} patients...")

    # Get batch
    batch = get_batch_composite(
        range(n_patients), val_data, val_p2i,
        select='left', block_size=block_size,
        device=device, padding='random',
        apply_token_shift=apply_ts,
        shift_continuous=shift_cont,
        separate_shift_na_from_padding=sep_na,
        shift_na_raw_token=na_tok,
    )
    x_data, x_shift, x_total, x_ages = batch[:4]
    y_data, y_shift, y_total, y_ages = batch[4:]

    # Run model in batches
    all_rates = []
    all_ages_out = []
    all_targets = []
    all_target_ages = []

    gen_batch = 64
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, n_patients, gen_batch), desc="Computing rates"):
            end = min(start + gen_batch, n_patients)
            logits, _, _ = model(
                x_data[start:end].to(device),
                x_shift[start:end].to(device),
                x_total[start:end].to(device),
                x_ages[start:end].to(device),
            )
            # Softmax → per-disease rate
            rates = torch.softmax(logits['data'], dim=-1)  # (B, T, V)
            all_rates.append(rates[:, :, disease_ids].cpu().numpy())
            all_ages_out.append(x_ages[start:end].cpu().numpy() / 365.25)
            all_targets.append(y_data[start:end].cpu().numpy())
            all_target_ages.append(y_ages[start:end].cpu().numpy() / 365.25)

    rates_np = np.concatenate(all_rates)      # (N, T, n_diseases)
    ages_np = np.concatenate(all_ages_out)     # (N, T)
    targets_np = np.concatenate(all_targets)   # (N, T)
    target_ages_np = np.concatenate(all_target_ages)

    # Compute observed incidence from target data
    obs_inc = _compute_incidence(targets_np, target_ages_np, data_vocab)

    # Plot
    cols = min(5, n_diseases)
    rows = (n_diseases + cols - 1) // cols
    if figsize is None:
        figsize = (4 * cols, 3.5 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for idx, tid in enumerate(disease_ids):
        ax = axes[idx // cols][idx % cols]
        name = token_names.get(tid, f'Token {tid}')

        # Scatter: predicted rate at each position
        valid = ages_np > 0
        age_flat = ages_np[valid].flatten()
        rate_flat = rates_np[:, :, idx][valid].flatten()

        # Filter positive rates for log scale
        pos = rate_flat > 0
        ax.scatter(age_flat[pos], rate_flat[pos],
                   c='lightblue', s=0.5, alpha=0.1, rasterized=True)

        # Highlight: predictions just before this disease occurs
        for i in range(min(len(targets_np), 500)):
            hit = targets_np[i] == tid
            if hit.any():
                hit_pos = np.where(hit)[0]
                for hp in hit_pos:
                    if hp > 0 and ages_np[i, hp-1] > 0:
                        ax.scatter(ages_np[i, hp-1], rates_np[i, hp-1, idx],
                                   c='darkblue', s=3, alpha=0.5, zorder=5)

        # Observed rate line
        obs_rate = obs_inc[tid, :]
        age_bins = np.arange(len(obs_rate))
        obs_pos = obs_rate > 0
        if obs_pos.any():
            ax.plot(age_bins[obs_pos], obs_rate[obs_pos],
                    color='purple', lw=1.5, alpha=0.8, label='Observed')

        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1)
        ax.set_xlim(0, 80)
        ax.set_title(name, fontsize=8, pad=3)
        ax.tick_params(labelsize=7)
        if idx % cols == 0:
            ax.set_ylabel('Rate (yr⁻¹)', fontsize=8)
        if idx // cols == rows - 1:
            ax.set_xlabel('Age (yr)', fontsize=8)
        ax.grid(True, alpha=0.2)

    # Hide unused
    for idx in range(n_diseases, rows * cols):
        axes[idx // cols][idx % cols].axis('off')

    fig.suptitle('Predicted Disease Rates vs Age', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK]  Saved → {save_path}")
    plt.show()
    plt.close(fig)
    return fig
