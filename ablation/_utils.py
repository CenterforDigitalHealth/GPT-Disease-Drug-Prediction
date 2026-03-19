#!/usr/bin/env python3
"""Shared utilities for ablation study scripts."""
from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_repo_root() -> Path:
    """Return the project root directory (parent of ablation/)."""
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------

def run_subprocess(cmd: Sequence[str], cwd: Path, env: Dict[str, str], log_path: Path) -> float:
    """Run a command, tee stdout/stderr into a log file. Returns elapsed seconds."""
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


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    """Write list of dicts to CSV with auto-discovered sorted fieldnames."""
    keys = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_existing_trials(path: Path) -> List[Dict[str, object]]:
    """Load previously saved trial rows from CSV with type normalization."""
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out: Dict[str, object] = dict(row)
            if "trial_id" in out and str(out["trial_id"]).strip() != "":
                out["trial_id"] = int(float(str(out["trial_id"])))
            if "is_best" in out:
                out["is_best"] = str(out["is_best"]).strip().lower() in {"1", "true", "yes"}
            rows.append(out)
    rows.sort(key=lambda r: int(r.get("trial_id", 0)))
    return rows


# ---------------------------------------------------------------------------
# Scalar / metrics utilities
# ---------------------------------------------------------------------------

def to_scalar(v):
    """Convert value to JSON-serializable scalar."""
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    return None


def load_prefixed_metrics(eval_dir: Path) -> Dict[str, Dict]:
    """Load *_composite_metrics.json files from eval output directory."""
    out: Dict[str, Dict] = {}
    for fp in sorted(eval_dir.glob("*_composite_metrics.json")):
        prefix = fp.name.replace("_composite_metrics.json", "")
        with fp.open("r", encoding="utf-8") as f:
            out[prefix] = json.load(f)
    return out


def flatten_metrics(metrics_by_prefix: Dict[str, Dict]) -> Dict[str, object]:
    """Flatten nested metric dicts with prefix. Includes ALL scalar keys."""
    row: Dict[str, object] = {}
    for prefix, metrics in metrics_by_prefix.items():
        for k, v in metrics.items():
            s = to_scalar(v)
            if s is None:
                continue
            row[f"{prefix}.{k}"] = s
    return row


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def resolve_checkpoint(train_out: Path) -> Optional[Path]:
    """Resolve best/periodic checkpoint path across naming variants."""
    primary = [train_out / "ckpt.pt", train_out / "ckpt_composite.pt"]
    for p in primary:
        if p.exists():
            return p
    periodic = sorted(
        train_out.glob("ckpt_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if periodic:
        return periodic[0]
    return None


def checkpoint_summary(ckpt_path: Path) -> Dict[str, object]:
    """Extract metadata from a checkpoint file."""
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model_state = ckpt.get("model", {})
    model_args = ckpt.get("model_args", {})
    cfg = ckpt.get("config", {})

    num_params = int(sum(v.numel() for v in model_state.values()))

    return {
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


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------

def parse_list(raw: str) -> List[str]:
    """Split and strip a comma-separated string."""
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_bool_list(raw: str) -> List[bool]:
    """Parse comma-separated boolean tokens (true/1/yes/on -> True)."""
    out: List[bool] = []
    for tok in [x.strip().lower() for x in raw.split(",") if x.strip()]:
        if tok in {"true", "1", "yes", "on"}:
            out.append(True)
        elif tok in {"false", "0", "no", "off"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid bool token: {tok}")
    return out


def parse_gpu_ids(raw: str) -> List[str]:
    """Validate and parse GPU ID string."""
    gpu_ids = [x.strip() for x in raw.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("gpu_ids must contain at least one GPU id")
    invalid = [gid for gid in gpu_ids if not gid.isdigit()]
    if invalid:
        raise ValueError(f"Invalid gpu_ids token(s): {invalid}")
    return gpu_ids


def build_env(gpu_ids: List[str]) -> Dict[str, str]:
    """Build environment dict with CUDA_VISIBLE_DEVICES set."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    env.setdefault("KMP_USE_SHM", "0")
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def ensure_ddp_compatible(extra_train: List[str], gpu_ids: List[str]) -> None:
    """Raise if extra_train_args contains --gpu_id with multiple GPUs."""
    if len(gpu_ids) <= 1:
        return
    has_gpu_id_override = any(
        arg == "--gpu_id" or arg.startswith("--gpu_id=") for arg in extra_train
    )
    if has_gpu_id_override:
        raise ValueError(
            "extra_train_args contains --gpu_id, which disables train_model auto-DDP. "
            "Remove --gpu_id when using multiple --gpu_ids."
        )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def as_float(x) -> float:
    """Safe float conversion with NaN for invalid values."""
    try:
        v = float(x)
    except Exception:
        return float("nan")
    if not np.isfinite(v):
        return float("nan")
    return v


def fmt_metric(x, digits: int = 4) -> str:
    """Format float to N decimal places or 'nan'."""
    v = as_float(x)
    if np.isfinite(v):
        return f"{v:.{digits}f}"
    return "nan"


# ---------------------------------------------------------------------------
# Best trial selection (5-tier ranking)
# ---------------------------------------------------------------------------

def pick_best_trial(rows: List[Dict[str, object]]) -> int:
    """
    Return index of the best trial in rows according to ranking:
      1) max val.auc_mean
      2) min mean(norm(val.dose_rmse), norm(val.dur_rmse))
      3) max val.dose_r2
      4) max val.dur_r2
      5) min best_val_loss
    Returns -1 if no valid trials.
    """
    valid_idxs = [
        i
        for i, r in enumerate(rows)
        if str(r.get("status", "")) == "success"
        and np.isfinite(as_float(r.get("val.auc_mean")))
    ]
    if not valid_idxs:
        return -1

    # Step 1: AUC max (with tolerance)
    auc_vals = np.array(
        [as_float(rows[i].get("val.auc_mean")) for i in valid_idxs], dtype=float
    )
    auc_max = float(np.nanmax(auc_vals))
    tie = [
        i
        for i in valid_idxs
        if abs(as_float(rows[i].get("val.auc_mean")) - auc_max) <= 1e-6
    ]
    if len(tie) == 1:
        return tie[0]

    # Step 2: normalized average RMSE score (lower better)
    def _normalize(vals: np.ndarray) -> np.ndarray:
        out = np.full_like(vals, fill_value=np.nan, dtype=float)
        finite = np.isfinite(vals)
        if not finite.any():
            return np.full_like(vals, fill_value=1.0, dtype=float)
        vmin, vmax = float(np.nanmin(vals[finite])), float(np.nanmax(vals[finite]))
        if vmax - vmin <= 1e-12:
            out[finite] = 0.0
        else:
            out[finite] = (vals[finite] - vmin) / (vmax - vmin)
        out[~finite] = 1.0
        return out

    shift_vals = np.array(
        [as_float(rows[i].get("val.dose_rmse")) for i in tie], dtype=float
    )
    total_vals = np.array(
        [as_float(rows[i].get("val.dur_rmse")) for i in tie], dtype=float
    )
    score = 0.5 * _normalize(shift_vals) + 0.5 * _normalize(total_vals)
    best_score = float(np.nanmin(score))
    tie2 = [
        tie[j] for j, s in enumerate(score) if abs(float(s) - best_score) <= 1e-12
    ]
    if len(tie2) == 1:
        return tie2[0]

    # Step 3: max shift R2
    sr2 = np.array(
        [as_float(rows[i].get("val.dose_r2")) for i in tie2], dtype=float
    )
    if np.isfinite(sr2).any():
        smax = float(np.nanmax(sr2))
        tie3 = [
            tie2[j]
            for j, v in enumerate(sr2)
            if np.isfinite(v) and abs(float(v) - smax) <= 1e-12
        ]
    else:
        tie3 = tie2
    if len(tie3) == 1:
        return tie3[0]

    # Step 4: max total R2
    tr2 = np.array(
        [as_float(rows[i].get("val.dur_r2")) for i in tie3], dtype=float
    )
    if np.isfinite(tr2).any():
        tmax = float(np.nanmax(tr2))
        tie4 = [
            tie3[j]
            for j, v in enumerate(tr2)
            if np.isfinite(v) and abs(float(v) - tmax) <= 1e-12
        ]
    else:
        tie4 = tie3
    if len(tie4) == 1:
        return tie4[0]

    # Step 5: min best_val_loss
    bvl = np.array(
        [as_float(rows[i].get("best_val_loss")) for i in tie4], dtype=float
    )
    if np.isfinite(bvl).any():
        bmin = float(np.nanmin(bvl))
        tie5 = [
            tie4[j]
            for j, v in enumerate(bvl)
            if np.isfinite(v) and abs(float(v) - bmin) <= 1e-12
        ]
        return tie5[0]

    return tie4[0]
