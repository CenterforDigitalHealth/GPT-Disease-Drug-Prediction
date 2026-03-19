#!/usr/bin/env python3
"""
Hyperparameter screening ablation.

- Training:   train_model
- Evaluation: evaluate_auc

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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Use a writable matplotlib cache dir in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from ablation._utils import (
    as_float,
    checkpoint_summary,
    flatten_metrics,
    fmt_metric,
    get_repo_root,
    load_existing_trials,
    load_prefixed_metrics,
    pick_best_trial,
    resolve_checkpoint,
    run_subprocess,
    to_scalar,
    write_csv,
)


# -----------------------------------------------------------------------------
# Search space
# -----------------------------------------------------------------------------
BLOCK_SIZES = [128, 256, 512]
N_EMBD_VALUES = [288, 384, 480]
N_LAYER_VALUES = [8, 12, 16]
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

    data = {k: np.array([as_float(r.get(k)) for r in valid_rows], dtype=float) for k, _ in dims}
    losses = np.array([as_float(r.get("best_val_loss")) for r in valid_rows], dtype=float)

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
            (0.0 if axis_lims[k][1] - axis_lims[k][0] <= 1e-12 else (as_float(best.get(k)) - axis_lims[k][0]) / (axis_lims[k][1] - axis_lims[k][0]))
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
    parser = argparse.ArgumentParser(description="Hyperparameter screening ablation")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--input_path", type=str, default="../data", help="Data root for evaluation")
    parser.add_argument("--output_root", type=str, default="ablation/hparam_screen", help="Root output dir")
    parser.add_argument("--run_name", type=str, default="screen_80_trials", help="Run subdirectory name")

    parser.add_argument("--n_trials", type=int, default=162, help="Number of sampled trials")
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for combo sampling")
    parser.add_argument("--trial_seed_base", type=int, default=1337, help="Base seed for train trials")
    parser.add_argument(
        "--resume_from_trial_id",
        type=int,
        default=1,
        help="Resume from this trial id (inclusive). Existing rows with smaller trial_id are kept.",
    )
    parser.add_argument(
        "--retry_eval_from_existing_ckpt",
        action="store_true",
        help=(
            "Retry eval from previously trained checkpoints without retraining. "
            "When enabled, all trials with existing ckpt are re-evaluated (no skip)."
        ),
    )

    parser.add_argument("--max_iters", type=int, default=3000, help="Train max iterations per trial")
    parser.add_argument("--eval_interval", type=int, default=500, help="Train eval interval")
    parser.add_argument("--dataset_subset_size", type=int, default=10000, help="Eval subset size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Eval batch size")
    parser.add_argument(
        "--data_files",
        type=str,
        default="dose/kr_val.bin,dose/kr_test.bin,dose/JMDC_extval.bin,UKB_extval.bin",
        help="Comma-separated eval files for evaluate_auc",
    )

    parser.add_argument(
        "--extra_train_args",
        type=str,
        default="",
        help='Extra args forwarded to train_model.py (e.g. "--batch_size=64")',
    )
    parser.add_argument(
        "--extra_eval_args",
        type=str,
        default="",
        help="Extra args forwarded to evaluate_auc.py",
    )

    parser.add_argument("--skip_train", action="store_true", help="Skip training")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation")
    # Forward-only W&B options (same naming as train_model).
    parser.add_argument("--wandb_log", action="store_true", help="Forward wandb_log to train_model")
    parser.add_argument("--wandb_project", type=str, default="composite-delphi", help="Forwarded to train_model")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="hparam_screen",
        help="Base run name forwarded to train_model (trial suffix added)",
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

    repo_root = get_repo_root()
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
        write_csv(sampled_rows, sampled_csv)
    print("=" * 72, flush=True)
    print("Hyperparameter screening started", flush=True)
    print(f"Run root: {run_root}", flush=True)
    print(f"Valid combinations: {len(valid_combos)}", flush=True)
    print(f"Sampled trials: {len(specs)} (sample_seed={args.sample_seed})", flush=True)
    print(f"Resume from trial id: {args.resume_from_trial_id}", flush=True)
    if args.retry_eval_from_existing_ckpt:
        print(
            "Eval retry mode: enabled (reuse existing ckpt and rerun eval on all matching trials)",
            flush=True,
        )
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
    existing_rows = load_existing_trials(trials_csv)
    sampled_ids = {s.trial_id for s in specs}
    trial_rows_by_id: Dict[int, Dict[str, object]] = {
        int(r.get("trial_id", 0)): dict(r)
        for r in existing_rows
        if int(r.get("trial_id", 0)) in sampled_ids
    }
    if existing_rows:
        kept_before_resume = sum(
            1
            for tid in trial_rows_by_id.keys()
            if tid < args.resume_from_trial_id
        )
        print(
            f"Existing trials.csv detected: {len(existing_rows)} rows "
            f"(loaded {len(trial_rows_by_id)} sampled rows, "
            f"{kept_before_resume} rows with trial_id < {args.resume_from_trial_id})",
            flush=True,
        )

    total_trials = len(specs)
    for trial_idx, spec in enumerate(specs, start=1):
        trial_root = run_root / f"trial_{spec.trial_id:03d}"
        train_out = trial_root / "train_out"
        eval_out = trial_root / "eval_out"
        logs_dir = trial_root / "logs"
        train_log = logs_dir / "train.log"
        eval_log = logs_dir / "eval.log"
        ckpt_path = train_out / "ckpt.pt"
        existing_ckpt = resolve_checkpoint(train_out)
        existing_row = trial_rows_by_id.get(spec.trial_id)
        retry_eval_this_trial = (
            args.retry_eval_from_existing_ckpt
            and existing_ckpt is not None
            and spec.trial_id >= args.resume_from_trial_id
        )

        if spec.trial_id < args.resume_from_trial_id and not retry_eval_this_trial:
            continue

        if (
            args.retry_eval_from_existing_ckpt
            and args.skip_train
            and spec.trial_id >= args.resume_from_trial_id
            and existing_ckpt is None
        ):
            print(
                (
                    f"\n[trial {trial_idx}/{total_trials}] SKIP "
                    f"(id={spec.trial_id:03d}) no existing checkpoint "
                    "(retry mode + --skip_train)."
                ),
                flush=True,
            )
            continue

        start_tag = "RETRY-EVAL" if retry_eval_this_trial else "START"

        print(
            (
                f"\n[trial {trial_idx}/{total_trials}] {start_tag} "
                f"(id={spec.trial_id:03d}) "
                f"block_size={spec.block_size}, n_embd={spec.n_embd}, "
                f"n_layer={spec.n_layer}, n_head={spec.n_head}, "
                f"time_distribution={spec.time_distribution}"
            ),
            flush=True,
        )

        row: Dict[str, object] = dict(existing_row) if existing_row is not None else {}
        for k in list(row.keys()):
            # Clear stale flattened metrics before re-evaluation.
            if "." in k:
                row.pop(k, None)

        row.update({
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
        })

        t_total0 = time.time()
        train_seconds = 0.0
        eval_seconds = 0.0

        try:
            if retry_eval_this_trial:
                print(
                    (
                        f"[trial {trial_idx}/{total_trials}] Reusing existing checkpoint "
                        f"for eval retry: {existing_ckpt}"
                    ),
                    flush=True,
                )
            elif not args.skip_train:
                print(
                    f"[trial {trial_idx}/{total_trials}] Training... (log: {train_log})",
                    flush=True,
                )
                train_cmd = [
                    sys.executable,
                    "-m",
                    "train_model",
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
                train_seconds = run_subprocess(train_cmd, cwd=repo_root, env=env, log_path=train_log)
                print(
                    f"[trial {trial_idx}/{total_trials}] Training done in {train_seconds:.1f}s",
                    flush=True,
                )

            resolved_ckpt = existing_ckpt if retry_eval_this_trial else resolve_checkpoint(train_out)
            if resolved_ckpt is None:
                raise FileNotFoundError(
                    f"Checkpoint not found under {train_out} "
                    "(checked: ckpt.pt, ckpt_composite.pt, ckpt_*.pt)"
                )
            ckpt_path = resolved_ckpt
            row["checkpoint_path"] = str(ckpt_path)

            ckpt_info = checkpoint_summary(ckpt_path)
            row.update(ckpt_info)
            print(
                (
                    f"[trial {trial_idx}/{total_trials}] Checkpoint loaded: "
                    f"iter={row.get('checkpoint_iter')}, "
                    f"best_val_loss={fmt_metric(row.get('best_val_loss'))}, "
                    f"num_params_m={fmt_metric(row.get('num_params_m'), 2)}"
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
                    "evaluate_auc",
                    f"--input_path={args.input_path}",
                    f"--model_ckpt_path={ckpt_path}",
                    f"--output_path={eval_out}",
                    "--model_type=composite",
                    f"--dataset_subset_size={args.dataset_subset_size}",
                    f"--eval_batch_size={args.eval_batch_size}",
                    f"--data_files={args.data_files}",
                ] + extra_eval
                eval_seconds = run_subprocess(eval_cmd, cwd=repo_root, env=env, log_path=eval_log)
                metrics_by_prefix = load_prefixed_metrics(eval_out)
                print(
                    f"[trial {trial_idx}/{total_trials}] Evaluation done in {eval_seconds:.1f}s",
                    flush=True,
                )

            row.update(flatten_metrics(metrics_by_prefix))
            row["status"] = "success"
            print(
                (
                    f"[trial {trial_idx}/{total_trials}] RESULT "
                    f"val.auc_mean={fmt_metric(row.get('val.auc_mean'))}, "
                    f"val.dose_rmse={fmt_metric(row.get('val.dose_rmse'))}, "
                    f"val.dur_rmse={fmt_metric(row.get('val.dur_rmse'))}, "
                    f"best_val_loss={fmt_metric(row.get('best_val_loss'))}"
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
            trial_rows_by_id[spec.trial_id] = row
            trial_rows = [trial_rows_by_id[tid] for tid in sorted(trial_rows_by_id.keys())]
            write_csv(trial_rows, trials_csv)
            ok = sum(1 for r in trial_rows if str(r.get("status", "")) == "success")
            fail = len(trial_rows) - ok
            print(
                f"[trial {trial_idx}/{total_trials}] DONE status={row['status']} "
                f"(success={ok}, failed={fail})",
                flush=True,
            )

    trial_rows = [trial_rows_by_id[tid] for tid in sorted(trial_rows_by_id.keys())]

    # Pick best trial
    best_idx = pick_best_trial(trial_rows)
    best_row = None
    if best_idx >= 0:
        trial_rows[best_idx]["is_best"] = True
        best_row = trial_rows[best_idx]

    # Rewrite CSV with is_best updated
    write_csv(trial_rows, trials_csv)

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
                f"(val.auc_mean={fmt_metric(best_row.get('val.auc_mean'))}, "
                f"val.dose_rmse={fmt_metric(best_row.get('val.dose_rmse'))}, "
                f"val.dur_rmse={fmt_metric(best_row.get('val.dur_rmse'))}, "
                f"best_val_loss={fmt_metric(best_row.get('best_val_loss'))})"
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
