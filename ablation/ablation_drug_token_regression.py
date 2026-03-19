#!/usr/bin/env python3
"""
Ablation runner for drug-token-only regression.

- Training:   train_model
- Evaluation: evaluate_auc

Screens drug_token_only_regression and drug_token_loss_weight combinations:
  1. Baseline:   drug_token_only_regression=False, drug_token_loss_weight=1.0
  2. Drug-only:  drug_token_only_regression=True,  drug_token_loss_weight=1.0
  3. Weighted 5x: drug_token_only_regression=False, drug_token_loss_weight=5.0
  4. Weighted 10x: drug_token_only_regression=False, drug_token_loss_weight=10.0

Per trial artifacts:
  - train/eval logs
  - checkpoint
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
from typing import Dict, List, Tuple

from ablation._utils import (
    get_repo_root,
    run_subprocess,
    write_csv,
    load_existing_trials,
    load_prefixed_metrics,
    flatten_metrics,
    resolve_checkpoint,
    checkpoint_summary,
    parse_gpu_ids,
    build_env,
    ensure_ddp_compatible,
    as_float,
    fmt_metric,
    pick_best_trial,
)


# (drug_token_only_regression, drug_token_loss_weight, short_name)
DEFAULT_CONDITIONS: List[Tuple[bool, float, str]] = [
    (False, 1.0, "baseline"),
    (True, 1.0, "drug_only"),
    (False, 5.0, "weighted_5x"),
    (False, 10.0, "weighted_10x"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation: drug-token-only regression")
    parser.add_argument("--gpu_ids", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--input_path", type=str, default="../data", help="Data root for evaluation")
    parser.add_argument("--output_root", type=str, default="ablation/drug_token_regression", help="Root output dir")
    parser.add_argument("--run_name", type=str, default="default", help="Run subdirectory name")
    parser.add_argument(
        "--fixed_label_scaling",
        type=str,
        default="none",
        choices=["none", "zscore", "robust", "minmax"],
        help="Fixed label_scaling value for all trials",
    )
    parser.add_argument(
        "--fixed_loss_norm",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Fixed loss_normalize_by_variance value for all trials",
    )
    parser.add_argument(
        "--fixed_posthoc_calibration",
        type=str,
        default="none",
        choices=["none", "affine"],
        help="Fixed posthoc_calibration value for all trials",
    )
    parser.add_argument("--max_iters", type=int, default=10000, help="Train max iterations per trial")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Train eval interval")
    parser.add_argument("--dataset_subset_size", type=int, default=10000, help="Eval subset size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Eval batch size")
    parser.add_argument(
        "--data_files",
        type=str,
        default="dose/kr_val.bin,dose/kr_test.bin,dose/JMDC_extval.bin,UKB_extval.bin",
        help="Comma-separated eval files for evaluate_auc",
    )
    parser.add_argument("--extra_train_args", type=str, default="", help="Extra args forwarded to train_model.py")
    parser.add_argument("--extra_eval_args", type=str, default="", help="Extra args forwarded to evaluate_auc.py")
    parser.add_argument(
        "--resume_from_trial_id",
        type=int,
        default=1,
        help="Resume from this trial id (inclusive).",
    )
    parser.add_argument("--skip_train", action="store_true", help="Skip training")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation")
    parser.add_argument(
        "--retry_eval_from_existing_ckpt",
        action="store_true",
        help="Retry eval from previously trained checkpoints without retraining.",
    )
    parser.add_argument("--wandb_log", action="store_true", help="Forward wandb_log to train_model")
    parser.add_argument("--wandb_project", type=str, default="composite-delphi", help="Forwarded to train_model")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ablation_drug_token_regression",
        help="Base run name forwarded to train_model (trial suffix added)",
    )
    args = parser.parse_args()

    # ---- Validate ----
    if args.resume_from_trial_id <= 0:
        raise ValueError("resume_from_trial_id must be >= 1")

    conditions = DEFAULT_CONDITIONS
    loss_norm = args.fixed_loss_norm == "true"

    # ---- Paths ----
    repo_root = get_repo_root()
    run_root = (repo_root / args.output_root / args.run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    with (run_root / "ablation_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # ---- GPU / env ----
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    env = build_env(gpu_ids)
    extra_train = shlex.split(args.extra_train_args) if args.extra_train_args.strip() else []
    extra_eval = shlex.split(args.extra_eval_args) if args.extra_eval_args.strip() else []
    ensure_ddp_compatible(extra_train, gpu_ids)

    print("=" * 72, flush=True)
    print("Drug-token regression ablation started", flush=True)
    print(f"Run root: {run_root}", flush=True)
    print(f"Conditions: {[(c[2], c[0], c[1]) for c in conditions]}", flush=True)
    print(f"Fixed label_scaling: {args.fixed_label_scaling}", flush=True)
    print(f"Fixed loss_normalize_by_variance: {loss_norm}", flush=True)
    print(f"Fixed posthoc_calibration: {args.fixed_posthoc_calibration}", flush=True)
    print(
        f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
        f"(auto-DDP {'enabled' if len(gpu_ids) > 1 else 'single GPU'})",
        flush=True,
    )
    print(f"Resume from trial id: {args.resume_from_trial_id}", flush=True)
    print("=" * 72, flush=True)

    # ---- Load existing trials for resume ----
    trials_csv = run_root / "trials.csv"
    existing_rows = load_existing_trials(trials_csv)
    trial_rows_by_id: Dict[int, Dict[str, object]] = {
        int(r.get("trial_id", 0)): dict(r) for r in existing_rows
    }

    # ---- Trial loop ----
    total_trials = len(conditions)

    for trial_idx, (drug_only, drug_weight, short_name) in enumerate(conditions, start=1):
        trial_id = trial_idx
        trial_name = f"t{trial_id:02d}_drug_token-{short_name}"
        trial_root = run_root / trial_name
        train_out = trial_root / "train_out"
        eval_out = trial_root / "eval_out"
        logs_dir = trial_root / "logs"
        train_log = logs_dir / "train.log"
        eval_log = logs_dir / "eval.log"
        existing_ckpt = resolve_checkpoint(train_out)
        existing_row = trial_rows_by_id.get(trial_id)
        retry_eval_this_trial = (
            args.retry_eval_from_existing_ckpt
            and existing_ckpt is not None
            and trial_id >= args.resume_from_trial_id
        )

        if trial_id < args.resume_from_trial_id and not retry_eval_this_trial:
            continue

        if (
            args.retry_eval_from_existing_ckpt
            and args.skip_train
            and trial_id >= args.resume_from_trial_id
            and existing_ckpt is None
        ):
            print(
                f"\n[trial {trial_idx}/{total_trials}] SKIP "
                f"(id={trial_id:02d}) no existing checkpoint "
                "(retry mode + --skip_train).",
                flush=True,
            )
            continue

        start_tag = "RETRY-EVAL" if retry_eval_this_trial else "START"
        print(
            f"\n[trial {trial_idx}/{total_trials}] {start_tag} "
            f"(id={trial_id:02d}) {short_name}: "
            f"drug_token_only_regression={drug_only}, drug_token_loss_weight={drug_weight}",
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
            "drug_token_only_regression": drug_only,
            "drug_token_loss_weight": drug_weight,
            "label_scaling": args.fixed_label_scaling,
            "loss_normalize_by_variance": loss_norm,
            "posthoc_calibration": args.fixed_posthoc_calibration,
            "visible_gpu_count": len(gpu_ids),
            "visible_gpu_ids": env["CUDA_VISIBLE_DEVICES"],
            "checkpoint_path": "",
            "train_log": str(train_log),
            "eval_log": str(eval_log),
        })

        t_total0 = time.time()
        train_seconds = 0.0
        eval_seconds = 0.0

        try:
            # ---- Train ----
            if retry_eval_this_trial:
                print(
                    f"[trial {trial_idx}/{total_trials}] Reusing existing checkpoint "
                    f"for eval retry: {existing_ckpt}",
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
                    f"--label_scaling={args.fixed_label_scaling}",
                    f"--loss_normalize_by_variance={loss_norm}",
                    f"--drug_token_only_regression={drug_only}",
                    f"--drug_token_loss_weight={drug_weight}",
                ] + extra_train
                if args.wandb_log:
                    train_cmd += [
                        "--wandb_log=True",
                        f"--wandb_project={args.wandb_project}",
                        f"--wandb_run_name={args.wandb_run_name}_{trial_name}",
                    ]
                train_seconds = run_subprocess(train_cmd, repo_root, env, train_log)
                print(
                    f"[trial {trial_idx}/{total_trials}] Training done in {train_seconds:.1f}s",
                    flush=True,
                )

            # ---- Resolve checkpoint ----
            resolved_ckpt = existing_ckpt if retry_eval_this_trial else resolve_checkpoint(train_out)
            if resolved_ckpt is None:
                raise FileNotFoundError(
                    f"Checkpoint not found under {train_out} "
                    "(checked: ckpt.pt, ckpt_composite.pt, ckpt_*.pt)"
                )
            row["checkpoint_path"] = str(resolved_ckpt)

            ckpt_info = checkpoint_summary(resolved_ckpt)
            row.update(ckpt_info)
            print(
                f"[trial {trial_idx}/{total_trials}] Checkpoint loaded: "
                f"iter={row.get('checkpoint_iter')}, "
                f"best_val_loss={fmt_metric(row.get('best_val_loss'))}, "
                f"num_params_m={fmt_metric(row.get('num_params_m'), 2)}",
                flush=True,
            )

            # ---- Evaluate ----
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
                    f"--model_ckpt_path={resolved_ckpt}",
                    f"--output_path={eval_out}",
                    f"--dataset_subset_size={args.dataset_subset_size}",
                    f"--eval_batch_size={args.eval_batch_size}",
                    f"--data_files={args.data_files}",
                    f"--posthoc_calibration={args.fixed_posthoc_calibration}",
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
                f"val.dur_rmse={fmt_metric(row.get('val.dur_rmse'))}, "
                f"best_val_loss={fmt_metric(row.get('best_val_loss'))}",
                flush=True,
            )
        except Exception as e:
            row["status"] = "failed"
            row["error_message"] = str(e)
            print(
                f"[trial {trial_idx}/{total_trials}] FAILED: {e}\n"
                f"  train_log={train_log}\n"
                f"  eval_log={eval_log}",
                flush=True,
            )
        finally:
            row["train_seconds"] = float(train_seconds)
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
    print("Drug-token regression ablation completed.", flush=True)
    print(f"Trials summary: success={success_count}, failed={failed_count}", flush=True)
    if best_row is not None:
        print(
            f"Best trial id={best_row.get('trial_id')} "
            f"(val.auc_mean={fmt_metric(best_row.get('val.auc_mean'))}, "
            f"val.dose_rmse={fmt_metric(best_row.get('val.dose_rmse'))}, "
            f"val.dur_rmse={fmt_metric(best_row.get('val.dur_rmse'))}, "
            f"best_val_loss={fmt_metric(best_row.get('best_val_loss'))})",
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
