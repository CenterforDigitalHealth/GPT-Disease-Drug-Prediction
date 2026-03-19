#!/usr/bin/env python3
"""
Ablation runner for post-hoc calibration only.

- Evaluation only (no training): uses a pre-trained checkpoint
- Screens posthoc_calibration: none, affine

Per trial artifacts:
  - eval logs
  - flattened metrics in trials.csv

Global artifacts:
  - trials.csv
  - best_trial.json
  - ablation_config.json
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Dict, List

from ablation._utils import (
    get_repo_root,
    run_subprocess,
    write_csv,
    load_existing_trials,
    load_prefixed_metrics,
    flatten_metrics,
    parse_list,
    parse_gpu_ids,
    build_env,
    as_float,
    fmt_metric,
    pick_best_trial,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation: post-hoc calibration only")
    parser.add_argument("--gpu_ids", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--input_path", type=str, default="../data", help="Data root for evaluation")
    parser.add_argument("--output_root", type=str, default="ablation/posthoc_calibration", help="Root output dir")
    parser.add_argument("--run_name", type=str, default="default", help="Run subdirectory name")
    parser.add_argument("--model_ckpt_path", type=str, required=True, help="Path to pre-trained checkpoint")
    parser.add_argument(
        "--posthoc_calibrations",
        type=str,
        default="none,affine",
        help="Comma-separated calibration methods to screen",
    )
    parser.add_argument("--dataset_subset_size", type=int, default=10000, help="Eval subset size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Eval batch size")
    parser.add_argument(
        "--data_files",
        type=str,
        default="dose/kr_val.bin,dose/kr_test.bin,dose/JMDC_extval.bin,UKB_extval.bin",
        help="Comma-separated eval files for evaluate_auc",
    )
    parser.add_argument("--extra_eval_args", type=str, default="", help="Extra args forwarded to evaluate_auc.py")
    parser.add_argument(
        "--resume_from_trial_id",
        type=int,
        default=1,
        help="Resume from this trial id (inclusive).",
    )
    parser.add_argument("--wandb_log", action="store_true", help="Log eval metrics to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="composite-delphi", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default="ablation_posthoc_calibration", help="W&B run name")
    args = parser.parse_args()

    # ---- Validate ----
    calibrations = parse_list(args.posthoc_calibrations)
    bad = [x for x in calibrations if x not in {"none", "affine"}]
    if bad:
        raise ValueError(f"Unknown calibration methods: {bad}")
    if args.resume_from_trial_id <= 0:
        raise ValueError("resume_from_trial_id must be >= 1")

    ckpt = Path(args.model_ckpt_path).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    # ---- Paths ----
    repo_root = get_repo_root()
    run_root = (repo_root / args.output_root / args.run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    with (run_root / "ablation_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # ---- GPU / env ----
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    env = build_env(gpu_ids)
    extra_eval = shlex.split(args.extra_eval_args) if args.extra_eval_args.strip() else []

    print("=" * 72, flush=True)
    print("Post-hoc calibration ablation started", flush=True)
    print(f"Run root: {run_root}", flush=True)
    print(f"Calibrations to screen: {calibrations}", flush=True)
    print(f"Checkpoint: {ckpt}", flush=True)
    print(
        f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
        f"(evaluate_auc uses the first visible GPU)",
        flush=True,
    )
    print(f"Resume from trial id: {args.resume_from_trial_id}", flush=True)
    print("=" * 72, flush=True)

    # ---- W&B ----
    wandb_run = None
    if args.wandb_log:
        try:
            import wandb  # type: ignore
        except Exception as e:
            raise RuntimeError("wandb_log was set but wandb import failed. Install wandb first.") from e
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # ---- Load existing trials for resume ----
    trials_csv = run_root / "trials.csv"
    existing_rows = load_existing_trials(trials_csv)
    trial_rows_by_id: Dict[int, Dict[str, object]] = {
        int(r.get("trial_id", 0)): dict(r) for r in existing_rows
    }

    # ---- Trial loop ----
    total_trials = len(calibrations)

    try:
        for trial_idx, cal in enumerate(calibrations, start=1):
            trial_id = trial_idx
            trial_name = f"t{trial_id:02d}_calibration-{cal}"
            trial_root = run_root / trial_name
            eval_out = trial_root / "eval_out"
            logs_dir = trial_root / "logs"
            eval_log = logs_dir / "eval.log"
            existing_row = trial_rows_by_id.get(trial_id)

            if trial_id < args.resume_from_trial_id:
                continue

            print(
                f"\n[trial {trial_idx}/{total_trials}] START "
                f"(id={trial_id:02d}) posthoc_calibration={cal}",
                flush=True,
            )

            row: Dict[str, object] = dict(existing_row) if existing_row is not None else {}
            for k in list(row.keys()):
                if "." in k:
                    row.pop(k, None)

            row.update({
                "trial_id": trial_id,
                "trial_name": trial_name,
                "status": "running",
                "error_message": "",
                "is_best": False,
                "posthoc_calibration": cal,
                "visible_gpu_count": len(gpu_ids),
                "visible_gpu_ids": env["CUDA_VISIBLE_DEVICES"],
                "checkpoint_path": str(ckpt),
                "eval_log": str(eval_log),
            })

            t_total0 = time.time()
            eval_seconds = 0.0

            try:
                print(
                    f"[trial {trial_idx}/{total_trials}] Evaluating... (log: {eval_log})",
                    flush=True,
                )
                eval_cmd = [
                    sys.executable,
                    "-m",
                    "evaluate_auc",
                    f"--input_path={args.input_path}",
                    f"--model_ckpt_path={ckpt}",
                    f"--output_path={eval_out}",
                    f"--dataset_subset_size={args.dataset_subset_size}",
                    f"--eval_batch_size={args.eval_batch_size}",
                    f"--data_files={args.data_files}",
                    f"--posthoc_calibration={cal}",
                ] + extra_eval
                eval_seconds = run_subprocess(eval_cmd, repo_root, env, eval_log)
                metrics_by_prefix = load_prefixed_metrics(eval_out)
                print(
                    f"[trial {trial_idx}/{total_trials}] Evaluation done in {eval_seconds:.1f}s",
                    flush=True,
                )

                row.update(flatten_metrics(metrics_by_prefix))
                row["status"] = "success"
                print(
                    f"[trial {trial_idx}/{total_trials}] RESULT "
                    f"val.auc_mean={fmt_metric(row.get('val.auc_mean'))}, "
                    f"val.dose_rmse={fmt_metric(row.get('val.dose_rmse'))}, "
                    f"val.dur_rmse={fmt_metric(row.get('val.dur_rmse'))}",
                    flush=True,
                )

                if wandb_run is not None:
                    wandb_payload = {
                        k: v for k, v in row.items()
                        if isinstance(v, (bool, int, float, str))
                    }
                    wandb_run.log(wandb_payload)

            except Exception as e:
                row["status"] = "failed"
                row["error_message"] = str(e)
                print(
                    f"[trial {trial_idx}/{total_trials}] FAILED: {e}\n"
                    f"  eval_log={eval_log}",
                    flush=True,
                )
            finally:
                row["eval_seconds"] = float(eval_seconds)
                row["total_seconds"] = float(time.time() - t_total0)
                trial_rows_by_id[trial_id] = row
                all_rows = [trial_rows_by_id[tid] for tid in sorted(trial_rows_by_id)]
                write_csv(all_rows, trials_csv)
                ok = sum(1 for r in all_rows if str(r.get("status", "")) == "success")
                fail = len(all_rows) - ok
                print(
                    f"[trial {trial_idx}/{total_trials}] DONE status={row['status']} "
                    f"(success={ok}, failed={fail})",
                    flush=True,
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    # ---- Pick best trial ----
    trial_rows = [trial_rows_by_id[tid] for tid in sorted(trial_rows_by_id)]

    best_idx = pick_best_trial(trial_rows)
    best_row = None
    if best_idx >= 0:
        trial_rows[best_idx]["is_best"] = True
        best_row = trial_rows[best_idx]

    write_csv(trial_rows, trials_csv)

    with (run_root / "best_trial.json").open("w", encoding="utf-8") as f:
        json.dump(best_row if best_row is not None else {}, f, indent=2)

    # ---- Summary ----
    success_count = sum(1 for r in trial_rows if str(r.get("status", "")) == "success")
    failed_count = len(trial_rows) - success_count
    print("\n" + "=" * 72, flush=True)
    print("Post-hoc calibration ablation completed.", flush=True)
    print(f"Trials summary: success={success_count}, failed={failed_count}", flush=True)
    if best_row is not None:
        print(
            f"Best trial id={best_row.get('trial_id')} "
            f"(val.auc_mean={fmt_metric(best_row.get('val.auc_mean'))}, "
            f"val.dose_rmse={fmt_metric(best_row.get('val.dose_rmse'))}, "
            f"val.dur_rmse={fmt_metric(best_row.get('val.dur_rmse'))})",
            flush=True,
        )
    else:
        print("Best trial: none (no successful trials)", flush=True)
    print(f"Run root: {run_root}")
    print(f"Trials CSV: {trials_csv}")
    print(f"Best trial JSON: {run_root / 'best_trial.json'}")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
