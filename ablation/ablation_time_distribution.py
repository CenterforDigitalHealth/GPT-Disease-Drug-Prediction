#!/usr/bin/env python3
"""
Run time-distribution ablation: exponential vs weibull.

For each distribution:
1) Train from scratch with identical settings except `time_distribution`
2) Evaluate using evaluate_auc.py
3) Collect key metrics into a single summary JSON/CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


TIME_DISTS = ("exponential", "weibull")


def _run(cmd: List[str], cwd: Path, env: Dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSee log: {log_path}")


def _load_composite_metrics(eval_dir: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for fp in sorted(eval_dir.glob("*_composite_metrics.json")):
        prefix = fp.name.replace("_composite_metrics.json", "")
        with fp.open("r", encoding="utf-8") as f:
            out[prefix] = json.load(f)
    return out


def _flatten_metrics(dist: str, metrics_by_prefix: Dict[str, Dict]) -> Dict[str, object]:
    row: Dict[str, object] = {"time_distribution": dist}
    wanted = (
        "auc_mean",
        "auc_median",
        "shift_accuracy",
        "shift_f1_macro",
        "shift_f1_macro_drug_cond",
        "total_mae",
        "total_rmse",
        "total_r2",
        "total_mae_drug_cond",
        "total_rmse_drug_cond",
        "total_r2_drug_cond",
    )
    for prefix, m in metrics_by_prefix.items():
        for k in wanted:
            if k in m:
                row[f"{prefix}.{k}"] = m[k]
    return row


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    keys = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation: time distribution (exp vs weibull)")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--input_path", type=str, default="../data", help="Data root for evaluation")
    parser.add_argument(
        "--output_root",
        type=str,
        default="ablation/time_distribution",
        help="Root directory for ablation outputs",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="exp_vs_weibull",
        help="Subdirectory name under output_root",
    )
    parser.add_argument("--max_iters", type=int, default=20000, help="Training max iterations")
    parser.add_argument("--eval_interval", type=int, default=2000, help="Training eval interval")
    parser.add_argument("--dataset_subset_size", type=int, default=10000, help="Eval subset size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Eval inference batch size")
    parser.add_argument(
        "--data_files",
        type=str,
        default="kr_val.bin,kr_test.bin",
        help="Comma-separated eval files for evaluate_auc.py",
    )
    parser.add_argument(
        "--extra_train_args",
        type=str,
        default="",
        help="Extra args forwarded to train_model.py (e.g. \"--batch_size=64 --learning_rate=3e-4\")",
    )
    parser.add_argument(
        "--extra_eval_args",
        type=str,
        default="",
        help="Extra args forwarded to evaluate_auc.py",
    )
    parser.add_argument("--skip_train", action="store_true", help="Skip training and only run evaluation")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation (training only)")
    args = parser.parse_args()

    if args.max_iters <= 0:
        raise ValueError("max_iters must be > 0")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be > 0")

    repo_root = Path(__file__).resolve().parent
    run_root = (repo_root / args.output_root / args.run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    ablation_cfg_path = run_root / "ablation_config.json"
    with ablation_cfg_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    extra_train = shlex.split(args.extra_train_args) if args.extra_train_args.strip() else []
    extra_eval = shlex.split(args.extra_eval_args) if args.extra_eval_args.strip() else []

    rows: List[Dict[str, object]] = []

    for dist in TIME_DISTS:
        dist_root = run_root / dist
        train_out = dist_root / "train_out"
        eval_out = dist_root / "eval_out"
        logs_dir = dist_root / "logs"
        train_log = logs_dir / "train.log"
        eval_log = logs_dir / "eval.log"
        ckpt_path = train_out / "ckpt_composite.pt"

        if not args.skip_train:
            train_cmd = [
                sys.executable,
                "-m",
                "train_model",
                "--init_from=scratch",
                f"--time_distribution={dist}",
                f"--out_dir={train_out}",
                f"--max_iters={args.max_iters}",
                f"--eval_interval={args.eval_interval}",
            ] + extra_train
            _run(train_cmd, cwd=repo_root, env=env, log_path=train_log)

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        metrics_by_prefix: Dict[str, Dict] = {}
        if not args.skip_eval:
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
            _run(eval_cmd, cwd=repo_root, env=env, log_path=eval_log)
            metrics_by_prefix = _load_composite_metrics(eval_out)

        row = _flatten_metrics(dist, metrics_by_prefix)
        row["checkpoint_path"] = str(ckpt_path)
        row["train_log"] = str(train_log)
        row["eval_log"] = str(eval_log)
        rows.append(row)

    summary_json = run_root / "summary.json"
    summary_csv = run_root / "summary.csv"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    _write_csv(rows, summary_csv)

    print("\nAblation completed.")
    print(f"Run root: {run_root}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
