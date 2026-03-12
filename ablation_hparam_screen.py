#!/usr/bin/env python3
"""
Hyperparameter screening ablation (v6 pipeline).

- Training:   train_model_v6
- Evaluation: evaluate_auc_v6

Screens 5 hyperparameters:
  1) block_size
  2) n_embd
  3) n_layer
  4) n_head
  5) time_distribution

Per trial artifacts:
  - train/eval logs
  - checkpoint
  - flattened metrics in trials.csv

Global artifacts:
  - sampled_combinations.csv
  - trials.csv
  - best_trial.json
  - parallel_coords_val_loss.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Use a writable matplotlib cache dir in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Search space
# -----------------------------------------------------------------------------
BLOCK_SIZES = [128, 192, 256, 320, 384, 512]
N_EMBD_VALUES = [288, 336, 384, 432, 480]
N_LAYER_VALUES = [8, 10, 12, 14, 16]
N_HEAD_VALUES = [8, 12, 16]
TIME_DISTRIBUTIONS = ["exponential", "weibull"]
FIXED_N_KV_HEAD = 4


@dataclass(frozen=True)
class TrialSpec:
    trial_id: int
    block_size: int
    n_embd: int
    n_layer: int
    n_head: int
    n_kv_head: int
    time_distribution: str


def _run(cmd: Sequence[str], cwd: Path, env: Dict[str, str], log_path: Path) -> float:
    """Run a command and tee stdout/stderr into a log file. Returns elapsed seconds."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSee log: {log_path}"
        )
    return elapsed


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    keys = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _to_scalar(v):
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    return None


def _load_prefixed_metrics(eval_dir: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for fp in sorted(eval_dir.glob("*_composite_metrics.json")):
        prefix = fp.name.replace("_composite_metrics.json", "")
        with fp.open("r", encoding="utf-8") as f:
            out[prefix] = json.load(f)
    return out


def _flatten_metrics(metrics_by_prefix: Dict[str, Dict]) -> Dict[str, object]:
    row: Dict[str, object] = {}
    for prefix, metrics in metrics_by_prefix.items():
        for k, v in metrics.items():
            s = _to_scalar(v)
            if s is None:
                continue
            row[f"{prefix}.{k}"] = s
    return row


def _checkpoint_summary(ckpt_path: Path) -> Dict[str, object]:
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model_state = ckpt.get("model", {})
    model_args = ckpt.get("model_args", {})
    cfg = ckpt.get("config", {})

    num_params = int(sum(v.numel() for v in model_state.values()))

    out = {
        "checkpoint_iter": int(ckpt.get("iter_num", -1)),
        "best_val_loss": float(ckpt.get("best_val_loss", np.nan)),
        "num_params": num_params,
        "num_params_m": float(num_params) / 1e6,
        "shift_continuous_effective": bool(model_args.get("shift_continuous", False)),
        "shift_log_effective": bool(model_args.get("shift_log", False)),
        "shift_input_scale_effective": float(model_args.get("shift_input_scale", np.nan)),
        "shift_min_value_effective": float(model_args.get("shift_min_value", np.nan)),
        "shift_max_value_effective": float(model_args.get("shift_max_value", np.nan)),
        "batch_size": int(cfg.get("batch_size", -1)),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", -1)),
    }
    return out


def _resolve_checkpoint(train_out: Path) -> Path | None:
    """Resolve best/periodic checkpoint path across naming variants."""
    primary = [train_out / "ckpt.pt", train_out / "ckpt_composite.pt"]
    for p in primary:
        if p.exists():
            return p
    periodic = sorted(train_out.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if periodic:
        return periodic[0]
    return None


def _build_valid_combinations() -> List[Tuple[int, int, int, int, str]]:
    combos: List[Tuple[int, int, int, int, str]] = []
    for block_size in BLOCK_SIZES:
        for n_embd in N_EMBD_VALUES:
            for n_layer in N_LAYER_VALUES:
                for n_head in N_HEAD_VALUES:
                    if n_embd % n_head != 0:
                        continue
                    if n_head % FIXED_N_KV_HEAD != 0:
                        continue
                    for time_dist in TIME_DISTRIBUTIONS:
                        combos.append((block_size, n_embd, n_layer, n_head, time_dist))
    return combos


def _sample_trials(n_trials: int, seed: int) -> List[TrialSpec]:
    combos = _build_valid_combinations()
    if n_trials > len(combos):
        raise ValueError(
            f"Requested n_trials={n_trials}, but only {len(combos)} valid combinations exist."
        )

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(combos), size=n_trials, replace=False)
    sampled = [combos[i] for i in idx]

    specs: List[TrialSpec] = []
    for i, (block_size, n_embd, n_layer, n_head, time_dist) in enumerate(sampled, start=1):
        specs.append(
            TrialSpec(
                trial_id=i,
                block_size=block_size,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                n_kv_head=FIXED_N_KV_HEAD,
                time_distribution=time_dist,
            )
        )
    return specs


def _load_sampled_trials(path: Path) -> List[TrialSpec]:
    if not path.exists():
        raise FileNotFoundError(f"Sampled combinations file not found: {path}")
    specs: List[TrialSpec] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            specs.append(
                TrialSpec(
                    trial_id=int(float(row["trial_id"])),
                    block_size=int(float(row["block_size"])),
                    n_embd=int(float(row["n_embd"])),
                    n_layer=int(float(row["n_layer"])),
                    n_head=int(float(row["n_head"])),
                    n_kv_head=int(float(row["n_kv_head"])),
                    time_distribution=str(row["time_distribution"]),
                )
            )
    specs.sort(key=lambda s: s.trial_id)
    return specs


def _load_existing_trials(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out: Dict[str, object] = dict(row)
            # Normalize key fields used by resume and ranking logic.
            if "trial_id" in out and str(out["trial_id"]).strip() != "":
                out["trial_id"] = int(float(str(out["trial_id"])))
            if "is_best" in out:
                out["is_best"] = str(out["is_best"]).strip().lower() in {"1", "true", "yes"}
            rows.append(out)
    rows.sort(key=lambda r: int(r.get("trial_id", 0)))
    return rows


def _as_float(x) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    if not np.isfinite(v):
        return float("nan")
    return v


def _fmt_metric(x, digits: int = 4) -> str:
    v = _as_float(x)
    if np.isfinite(v):
        return f"{v:.{digits}f}"
    return "nan"


def _pick_best_trial(rows: List[Dict[str, object]]) -> int:
    """
    Return index in rows according to ranking:
      1) max val.auc_mean
      2) min mean(norm(val.shift_rmse_drug_cond), norm(val.total_rmse_drug_cond))
      3) max val.shift_r2_drug_cond
      4) max val.total_r2_drug_cond
      5) min best_val_loss
    """
    valid_idxs = [
        i
        for i, r in enumerate(rows)
        if str(r.get("status", "")) == "success" and np.isfinite(_as_float(r.get("val.auc_mean")))
    ]
    if not valid_idxs:
        return -1

    # Step 1: AUC max (with tolerance)
    auc_vals = np.array([_as_float(rows[i].get("val.auc_mean")) for i in valid_idxs], dtype=float)
    auc_max = float(np.nanmax(auc_vals))
    tie = [i for i in valid_idxs if abs(_as_float(rows[i].get("val.auc_mean")) - auc_max) <= 1e-6]

    if len(tie) == 1:
        return tie[0]

    # Step 2: normalized average RMSE score (lower better)
    shift_key = "val.shift_rmse_drug_cond"
    total_key = "val.total_rmse_drug_cond"
    shift_vals = np.array([_as_float(rows[i].get(shift_key)) for i in tie], dtype=float)
    total_vals = np.array([_as_float(rows[i].get(total_key)) for i in tie], dtype=float)

    def _normalize(vals: np.ndarray) -> np.ndarray:
        out = np.full_like(vals, fill_value=np.nan, dtype=float)
        finite = np.isfinite(vals)
        if not finite.any():
            return np.full_like(vals, fill_value=1.0, dtype=float)
        vmin = float(np.nanmin(vals[finite]))
        vmax = float(np.nanmax(vals[finite]))
        if vmax - vmin <= 1e-12:
            out[finite] = 0.0
        else:
            out[finite] = (vals[finite] - vmin) / (vmax - vmin)
        out[~finite] = 1.0
        return out

    shift_norm = _normalize(shift_vals)
    total_norm = _normalize(total_vals)
    score = 0.5 * shift_norm + 0.5 * total_norm
    best_score = float(np.nanmin(score))
    tie2 = [tie[j] for j, s in enumerate(score) if abs(float(s) - best_score) <= 1e-12]

    if len(tie2) == 1:
        return tie2[0]

    # Step 3: max shift R2
    shift_r2_key = "val.shift_r2_drug_cond"
    shift_r2_vals = np.array([_as_float(rows[i].get(shift_r2_key)) for i in tie2], dtype=float)
    if np.isfinite(shift_r2_vals).any():
        smax = float(np.nanmax(shift_r2_vals))
        tie3 = [tie2[j] for j, v in enumerate(shift_r2_vals) if np.isfinite(v) and abs(float(v) - smax) <= 1e-12]
    else:
        tie3 = tie2

    if len(tie3) == 1:
        return tie3[0]

    # Step 4: max total R2
    total_r2_key = "val.total_r2_drug_cond"
    total_r2_vals = np.array([_as_float(rows[i].get(total_r2_key)) for i in tie3], dtype=float)
    if np.isfinite(total_r2_vals).any():
        tmax = float(np.nanmax(total_r2_vals))
        tie4 = [tie3[j] for j, v in enumerate(total_r2_vals) if np.isfinite(v) and abs(float(v) - tmax) <= 1e-12]
    else:
        tie4 = tie3

    if len(tie4) == 1:
        return tie4[0]

    # Step 5: min best_val_loss
    bvl = np.array([_as_float(rows[i].get("best_val_loss")) for i in tie4], dtype=float)
    if np.isfinite(bvl).any():
        bmin = float(np.nanmin(bvl))
        tie5 = [tie4[j] for j, v in enumerate(bvl) if np.isfinite(v) and abs(float(v) - bmin) <= 1e-12]
        return tie5[0]

    return tie4[0]


def _plot_parallel_coords(rows: List[Dict[str, object]], out_png: Path) -> None:
    """Plot image-a-like parallel coordinates colored by best_val_loss."""
    valid_rows = [r for r in rows if str(r.get("status", "")) == "success"]
    if not valid_rows:
        raise RuntimeError("No successful trials to plot.")

    dims = [
        ("block_size", "Context size"),
        ("n_embd", "Embedding dim"),
        ("n_layer", "Layers"),
        ("n_head", "Heads"),
        ("num_params_m", "Parameters (M)"),
    ]

    data = {k: np.array([_as_float(r.get(k)) for r in valid_rows], dtype=float) for k, _ in dims}
    losses = np.array([_as_float(r.get("best_val_loss")) for r in valid_rows], dtype=float)

    finite_losses = losses[np.isfinite(losses)]
    if finite_losses.size == 0:
        raise RuntimeError("No finite best_val_loss among successful trials.")

    loss_min, loss_max = float(np.nanmin(finite_losses)), float(np.nanmax(finite_losses))
    if loss_max - loss_min <= 1e-12:
        loss_norm = np.zeros_like(losses)
    else:
        loss_norm = (losses - loss_min) / (loss_max - loss_min)

    # Normalize axis-wise for drawing
    norm_vals: Dict[str, np.ndarray] = {}
    axis_lims: Dict[str, Tuple[float, float]] = {}
    for key, _ in dims:
        v = data[key]
        vmin = float(np.nanmin(v))
        vmax = float(np.nanmax(v))
        if vmax - vmin <= 1e-12:
            norm = np.zeros_like(v)
        else:
            norm = (v - vmin) / (vmax - vmin)
        norm_vals[key] = norm
        axis_lims[key] = (vmin, vmax)

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(dims))
    cmap = plt.get_cmap("plasma")

    # optional style encode for time_distribution
    for i, row in enumerate(valid_rows):
        y = np.array([norm_vals[key][i] for key, _ in dims], dtype=float)
        dist = str(row.get("time_distribution", ""))
        linestyle = "--" if dist == "weibull" else "-"
        c = cmap(float(np.clip(loss_norm[i], 0.0, 1.0)))
        ax.plot(x, y, color=c, alpha=0.65, linewidth=1.0, linestyle=linestyle, zorder=2)

    # Highlight best row
    best_candidates = [r for r in valid_rows if bool(r.get("is_best", False))]
    if best_candidates:
        best = best_candidates[0]
        yb = np.array([
            (0.0 if axis_lims[k][1] - axis_lims[k][0] <= 1e-12 else (_as_float(best.get(k)) - axis_lims[k][0]) / (axis_lims[k][1] - axis_lims[k][0]))
            for k, _ in dims
        ])
        ax.plot(x, yb, color="#FDE725", linewidth=2.8, alpha=1.0, zorder=5)

    # Axis guides
    for j, (k, label) in enumerate(dims):
        ax.vlines(j, 0.0, 1.0, color="#DDDDDD", linewidth=1.0, zorder=1)
        vmin, vmax = axis_lims[k]
        ax.text(j, -0.08, f"{vmin:.3g}", ha="center", va="top", fontsize=8, color="#666666")
        ax.text(j, 1.03, f"{vmax:.3g}", ha="center", va="bottom", fontsize=8, color="#666666")

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.12, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in dims], fontsize=10)
    ax.set_yticks([])
    ax.set_title("Hyperparameter Screening (color = best validation loss)")

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=loss_min, vmax=loss_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("best_val_loss")

    # minimal legend for distribution line style
    ax.plot([], [], color="#555555", linestyle="-", label="exponential")
    ax.plot([], [], color="#555555", linestyle="--", label="weibull")
    ax.plot([], [], color="#FDE725", linewidth=2.8, label="best trial")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter screening ablation (v6 pipeline)")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--input_path", type=str, default="../data", help="Data root for evaluation")
    parser.add_argument("--output_root", type=str, default="ablation/hparam_screen", help="Root output dir")
    parser.add_argument("--run_name", type=str, default="screen_80_trials", help="Run subdirectory name")

    parser.add_argument("--n_trials", type=int, default=80, help="Number of sampled trials")
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for combo sampling")
    parser.add_argument("--trial_seed_base", type=int, default=1337, help="Base seed for train trials")
    parser.add_argument(
        "--resume_from_trial_id",
        type=int,
        default=1,
        help="Resume from this trial id (inclusive). Existing rows with smaller trial_id are kept.",
    )

    parser.add_argument("--max_iters", type=int, default=3000, help="Train max iterations per trial")
    parser.add_argument("--eval_interval", type=int, default=500, help="Train eval interval")
    parser.add_argument("--dataset_subset_size", type=int, default=10000, help="Eval subset size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Eval batch size")
    parser.add_argument(
        "--data_files",
        type=str,
        default="dose/kr_val.bin,dose/kr_test.bin,dose/JMDC_exval2.bin",
        help="Comma-separated eval files for evaluate_auc_v6",
    )

    parser.add_argument(
        "--extra_train_args",
        type=str,
        default="",
        help='Extra args forwarded to train_model_v6.py (e.g. "--batch_size=64")',
    )
    parser.add_argument(
        "--extra_eval_args",
        type=str,
        default="",
        help="Extra args forwarded to evaluate_auc_v6.py",
    )

    parser.add_argument("--skip_train", action="store_true", help="Skip training")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation")
    # Forward-only W&B options (same naming as train_model_v6).
    parser.add_argument("--wandb_log", action="store_true", help="Forward wandb_log to train_model_v6")
    parser.add_argument("--wandb_project", type=str, default="composite-delphi", help="Forwarded to train_model_v6")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="hparam_screen",
        help="Base run name forwarded to train_model_v6 (trial suffix added)",
    )
    args = parser.parse_args()

    if args.n_trials <= 0:
        raise ValueError("n_trials must be > 0")
    if args.resume_from_trial_id <= 0:
        raise ValueError("resume_from_trial_id must be >= 1")
    if args.max_iters <= 0:
        raise ValueError("max_iters must be > 0")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be > 0")

    repo_root = Path(__file__).resolve().parent
    run_root = (repo_root / args.output_root / args.run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    # Save run config
    with (run_root / "ablation_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Sample trial specs
    valid_combos = _build_valid_combinations()
    sampled_csv = run_root / "sampled_combinations.csv"
    if sampled_csv.exists():
        specs = _load_sampled_trials(sampled_csv)
        if len(specs) != args.n_trials:
            raise ValueError(
                f"Existing sampled_combinations.csv has {len(specs)} trials, "
                f"but --n_trials={args.n_trials}. "
                "Use matching --n_trials or a different --run_name."
            )
    else:
        specs = _sample_trials(args.n_trials, args.sample_seed)
        sampled_rows = []
        for s in specs:
            sampled_rows.append(
                {
                    "trial_id": s.trial_id,
                    "block_size": s.block_size,
                    "n_embd": s.n_embd,
                    "n_layer": s.n_layer,
                    "n_head": s.n_head,
                    "n_kv_head": s.n_kv_head,
                    "time_distribution": s.time_distribution,
                }
            )
        _write_csv(sampled_rows, sampled_csv)
    print("=" * 72, flush=True)
    print("Hyperparameter screening started", flush=True)
    print(f"Run root: {run_root}", flush=True)
    print(f"Valid combinations: {len(valid_combos)}", flush=True)
    print(f"Sampled trials: {len(specs)} (sample_seed={args.sample_seed})", flush=True)
    print(f"Resume from trial id: {args.resume_from_trial_id}", flush=True)
    print(
        "Forced options: shift_continuous=True, shift_log=True",
        flush=True,
    )
    if args.resume_from_trial_id > len(specs):
        print(
            f"[warning] resume_from_trial_id={args.resume_from_trial_id} is larger than sampled trial count ({len(specs)}).",
            flush=True,
        )
    print("=" * 72, flush=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    # Avoid OpenMP shared-memory initialization failures in restricted environments.
    env.setdefault("KMP_USE_SHM", "0")
    env.setdefault("OMP_NUM_THREADS", "1")

    extra_train = shlex.split(args.extra_train_args) if args.extra_train_args.strip() else []
    extra_eval = shlex.split(args.extra_eval_args) if args.extra_eval_args.strip() else []

    trials_csv = run_root / "trials.csv"
    existing_rows = _load_existing_trials(trials_csv)
    trial_rows: List[Dict[str, object]] = [
        r for r in existing_rows if int(r.get("trial_id", 0)) < args.resume_from_trial_id
    ]
    if existing_rows:
        print(
            f"Existing trials.csv detected: {len(existing_rows)} rows "
            f"(keeping {len(trial_rows)} rows with trial_id < {args.resume_from_trial_id})",
            flush=True,
        )

    total_trials = len(specs)
    for trial_idx, spec in enumerate(specs, start=1):
        if spec.trial_id < args.resume_from_trial_id:
            continue
        trial_root = run_root / f"trial_{spec.trial_id:03d}"
        train_out = trial_root / "train_out"
        eval_out = trial_root / "eval_out"
        logs_dir = trial_root / "logs"
        train_log = logs_dir / "train.log"
        eval_log = logs_dir / "eval.log"
        ckpt_path = train_out / "ckpt.pt"

        print(
            (
                f"\n[trial {trial_idx}/{total_trials}] START "
                f"(id={spec.trial_id:03d}) "
                f"block_size={spec.block_size}, n_embd={spec.n_embd}, "
                f"n_layer={spec.n_layer}, n_head={spec.n_head}, "
                f"time_distribution={spec.time_distribution}"
            ),
            flush=True,
        )

        row: Dict[str, object] = {
            "trial_id": spec.trial_id,
            "status": "running",
            "error_message": "",
            "is_best": False,
            "sample_seed": args.sample_seed,
            "train_seed": args.trial_seed_base + spec.trial_id,
            "max_iters": args.max_iters,
            "eval_interval": args.eval_interval,
            "block_size": spec.block_size,
            "n_embd": spec.n_embd,
            "n_layer": spec.n_layer,
            "n_head": spec.n_head,
            "n_kv_head": spec.n_kv_head,
            "time_distribution": spec.time_distribution,
            "shift_continuous_requested": True,
            "shift_log_requested": True,
            "checkpoint_path": str(ckpt_path),
            "train_log": str(train_log),
            "eval_log": str(eval_log),
        }

        t_total0 = time.time()
        train_seconds = 0.0
        eval_seconds = 0.0

        try:
            if not args.skip_train:
                print(
                    f"[trial {trial_idx}/{total_trials}] Training... (log: {train_log})",
                    flush=True,
                )
                train_cmd = [
                    sys.executable,
                    "-m",
                    "train_model_v6",
                    "--init_from=scratch",
                    "--out_dir_use_timestamp=False",
                    f"--out_dir={train_out}",
                    f"--max_iters={args.max_iters}",
                    f"--eval_interval={args.eval_interval}",
                    "--always_save_checkpoint=True",
                    f"--seed={args.trial_seed_base + spec.trial_id}",
                    f"--block_size={spec.block_size}",
                    f"--n_embd={spec.n_embd}",
                    f"--n_layer={spec.n_layer}",
                    f"--n_head={spec.n_head}",
                    f"--n_kv_head={spec.n_kv_head}",
                    f"--time_distribution={spec.time_distribution}",
                    "--shift_continuous=True",
                    "--shift_log=True",
                ] + extra_train
                if args.wandb_log:
                    train_cmd += [
                        "--wandb_log=True",
                        f"--wandb_project={args.wandb_project}",
                        f"--wandb_run_name={args.wandb_run_name}_trial_{spec.trial_id:03d}",
                    ]
                train_seconds = _run(train_cmd, cwd=repo_root, env=env, log_path=train_log)
                print(
                    f"[trial {trial_idx}/{total_trials}] Training done in {train_seconds:.1f}s",
                    flush=True,
                )

            resolved_ckpt = _resolve_checkpoint(train_out)
            if resolved_ckpt is None:
                raise FileNotFoundError(
                    f"Checkpoint not found under {train_out} "
                    "(checked: ckpt.pt, ckpt_composite.pt, ckpt_*.pt)"
                )
            ckpt_path = resolved_ckpt
            row["checkpoint_path"] = str(ckpt_path)

            ckpt_info = _checkpoint_summary(ckpt_path)
            row.update(ckpt_info)
            print(
                (
                    f"[trial {trial_idx}/{total_trials}] Checkpoint loaded: "
                    f"iter={row.get('checkpoint_iter')}, "
                    f"best_val_loss={_fmt_metric(row.get('best_val_loss'))}, "
                    f"num_params_m={_fmt_metric(row.get('num_params_m'), 2)}"
                ),
                flush=True,
            )

            # hard guard: requested shift_log/continuous must be effective
            if not bool(row.get("shift_continuous_effective", False)):
                raise RuntimeError("shift_continuous_effective=False (expected True)")
            if not bool(row.get("shift_log_effective", False)):
                raise RuntimeError("shift_log_effective=False (expected True)")

            metrics_by_prefix: Dict[str, Dict] = {}
            if not args.skip_eval:
                print(
                    f"[trial {trial_idx}/{total_trials}] Evaluating... (log: {eval_log})",
                    flush=True,
                )
                eval_cmd = [
                    sys.executable,
                    "-m",
                    "evaluate_auc_v6",
                    f"--input_path={args.input_path}",
                    f"--model_ckpt_path={ckpt_path}",
                    f"--output_path={eval_out}",
                    "--model_type=composite",
                    f"--dataset_subset_size={args.dataset_subset_size}",
                    f"--eval_batch_size={args.eval_batch_size}",
                    f"--data_files={args.data_files}",
                ] + extra_eval
                eval_seconds = _run(eval_cmd, cwd=repo_root, env=env, log_path=eval_log)
                metrics_by_prefix = _load_prefixed_metrics(eval_out)
                print(
                    f"[trial {trial_idx}/{total_trials}] Evaluation done in {eval_seconds:.1f}s",
                    flush=True,
                )

            row.update(_flatten_metrics(metrics_by_prefix))
            row["status"] = "success"
            print(
                (
                    f"[trial {trial_idx}/{total_trials}] RESULT "
                    f"val.auc_mean={_fmt_metric(row.get('val.auc_mean'))}, "
                    f"val.shift_rmse_drug_cond={_fmt_metric(row.get('val.shift_rmse_drug_cond'))}, "
                    f"val.total_rmse_drug_cond={_fmt_metric(row.get('val.total_rmse_drug_cond'))}, "
                    f"best_val_loss={_fmt_metric(row.get('best_val_loss'))}"
                ),
                flush=True,
            )
        except Exception as e:
            row["status"] = "failed"
            row["error_message"] = str(e)
            print(
                (
                    f"[trial {trial_idx}/{total_trials}] FAILED: {e}\n"
                    f"  train_log={train_log}\n"
                    f"  eval_log={eval_log}"
                ),
                flush=True,
            )
        finally:
            row["train_seconds"] = float(train_seconds)
            row["eval_seconds"] = float(eval_seconds)
            row["total_seconds"] = float(time.time() - t_total0)
            trial_rows.append(row)
            _write_csv(trial_rows, trials_csv)
            ok = sum(1 for r in trial_rows if str(r.get("status", "")) == "success")
            fail = len(trial_rows) - ok
            print(
                f"[trial {trial_idx}/{total_trials}] DONE status={row['status']} "
                f"(success={ok}, failed={fail})",
                flush=True,
            )

    # Pick best trial
    best_idx = _pick_best_trial(trial_rows)
    best_row = None
    if best_idx >= 0:
        trial_rows[best_idx]["is_best"] = True
        best_row = trial_rows[best_idx]

    # Rewrite CSV with is_best updated
    _write_csv(trial_rows, trials_csv)

    # Save best trial json
    with (run_root / "best_trial.json").open("w", encoding="utf-8") as f:
        json.dump(best_row if best_row is not None else {}, f, indent=2)

    # Plot
    try:
        _plot_parallel_coords(trial_rows, run_root / "parallel_coords_val_loss.png")
    except Exception as e:
        print(f"[warning] failed to generate parallel coordinates plot: {e}")

    success_count = sum(1 for r in trial_rows if str(r.get("status", "")) == "success")
    failed_count = len(trial_rows) - success_count
    print("\n" + "=" * 72, flush=True)
    print("Hyperparameter screening completed.", flush=True)
    print(f"Trials summary: success={success_count}, failed={failed_count}", flush=True)
    if best_row is not None:
        print(
            (
                f"Best trial id={best_row.get('trial_id')} "
                f"(val.auc_mean={_fmt_metric(best_row.get('val.auc_mean'))}, "
                f"val.shift_rmse_drug_cond={_fmt_metric(best_row.get('val.shift_rmse_drug_cond'))}, "
                f"val.total_rmse_drug_cond={_fmt_metric(best_row.get('val.total_rmse_drug_cond'))}, "
                f"best_val_loss={_fmt_metric(best_row.get('best_val_loss'))})"
            ),
            flush=True,
        )
    else:
        print("Best trial: none (no successful trials)", flush=True)
    print(f"Run root: {run_root}")
    print(f"Sampled combinations: {run_root / 'sampled_combinations.csv'}")
    print(f"Trials CSV: {trials_csv}")
    print(f"Best trial JSON: {run_root / 'best_trial.json'}")
    print(f"Figure: {run_root / 'parallel_coords_val_loss.png'}")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
