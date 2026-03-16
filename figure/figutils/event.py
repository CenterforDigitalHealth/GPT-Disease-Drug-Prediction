"""
event.py - Waiting-time prediction vs. observation
====================================================
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

# Ensure project root is importable
_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_here))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils import get_batch_composite

# =============================================================================
# Computation
# =============================================================================

def compute_waiting_times(model, val_data, val_p2i, device,
                          batch_indices=range(256), block_size=128):
    """
    Run inference and return (expected_t, t_observed) arrays.

    Parameters
    ----------
    model : nn.Module
        CompositeDelphi model.
    val_data : np.ndarray
        Structured array from load_composite_data (with fields ID, AGE, DATA, SHIFT, TOTAL).
    val_p2i : np.ndarray
        Patient-to-index pointer array.
    device : str
    batch_indices : range or list
    block_size : int

    Returns
    -------
    expected_t : np.ndarray
    t_observed : np.ndarray
    """
    print("[INFO] Running inference for waiting times...")
    model.eval()

    # get_batch_composite returns:
    # (x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages)
    # Read tokenization settings from model config
    config = getattr(model, 'config', None)
    apply_ts = bool(getattr(config, 'apply_token_shift', False))
    shift_cont = bool(getattr(config, 'shift_continuous', False))
    sep_na = bool(getattr(config, 'separate_shift_na_from_padding', False))
    na_tok = int(getattr(config, 'shift_na_raw_token', 4))

    batch = get_batch_composite(
        batch_indices, val_data, val_p2i,
        select='left', padding='random',
        block_size=block_size, device=device,
        apply_token_shift=apply_ts,
        shift_continuous=shift_cont,
        separate_shift_na_from_padding=sep_na,
        shift_na_raw_token=na_tok,
    )

    x_data, x_shift, x_total, x_ages = batch[0], batch[1], batch[2], batch[3]
    y_data, y_shift, y_total, y_ages = batch[4], batch[5], batch[6], batch[7]

    with torch.no_grad():
        outputs = model(
            x_data, x_shift, x_total, x_ages,
            targets_data=y_data,
            targets_shift=y_shift,
            targets_total=y_total,
            targets_age=y_ages,
        )
        logits = outputs[0]

        # logits dict keys: 'data', 'shift', 'total', 'time_scale', ...
        # 'time_scale' contains λ parameters for competing-exponentials model
        if isinstance(logits, dict) and 'time_scale' in logits:
            time_logits = logits['time_scale']
        elif isinstance(logits, dict):
            print("[WARN] 'time_scale' not found, falling back to 'data'")
            time_logits = logits['data']
        else:
            time_logits = logits

    p = time_logits.cpu().numpy()

    # Observed waiting time: difference between target age and input age (in days)
    t_observed = (y_ages - x_ages).cpu().numpy()

    # Expected time via competing-exponentials: E[T] = 1 / sum(lambda_i)
    expected_t = 1.0 / np.exp(logsumexp(p, axis=-1))

    # Flatten to 1D (all positions across all patients)
    expected_t = expected_t.flatten()
    t_observed = t_observed.flatten()

    # Filter: only valid positions (non-padding, positive observed time)
    valid = (t_observed > 0) & np.isfinite(expected_t) & (expected_t > 0)
    expected_t = expected_t[valid]
    t_observed = t_observed[valid]

    print(f"[OK]  {len(expected_t):,} valid token predictions")
    return expected_t, t_observed


def compute_binned_averages(expected_t, t_observed, delta_log_t=0.1,
                            log_min=1.75, log_max=4.0):
    """
    Compute binned averages of observed times on a log scale.

    Returns
    -------
    bin_centers : np.ndarray
    bin_means : list[float]
    """
    log_range = np.arange(log_min, log_max, delta_log_t)
    means = []
    for i in log_range:
        lo, hi = 10**i, 10**(i + delta_log_t)
        mask = (expected_t > lo) & (expected_t <= hi) & (t_observed > 0)
        means.append(t_observed[mask].mean() if mask.sum() > 0 else np.nan)

    bin_centers = 10 ** (log_range + delta_log_t / 2.0)
    return bin_centers, means


# =============================================================================
# Plotting
# =============================================================================

def draw_waiting_time_plot(expected_t, t_observed, bin_centers, bin_means,
                           figsize=(4, 4), save_path=None):
    """
    Scatter + average-line plot on log-log axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_aspect('equal')

    ax.scatter(expected_t, t_observed + 0.5, marker='.', c='steelblue',
               alpha=0.15, s=3, rasterized=True, label='Observed')
    ax.plot(bin_centers, bin_means, label='Average', color='tab:red', lw=2)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes,
            c='k', ls=(0, (5, 5)), lw=0.7)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, 2e3)
    ax.set_ylim(1, 2e3)
    ax.set_xlabel('Expected days to next token')
    ax.set_ylabel('Observed days to next token')
    ax.legend()

    ax.tick_params(length=1.15, width=0.3, labelsize=8,
                   grid_alpha=1, grid_linewidth=0.45, grid_linestyle=':')
    ax.tick_params(length=1.15, width=0.3, labelsize=8,
                   grid_alpha=0.0, grid_linewidth=0.35, which='minor')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK]  Saved -> {save_path}")

    return fig
