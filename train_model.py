"""
Composite Delphi Training Script
Multi-GPU: Auto-detects GPUs and uses DDP (DistributedDataParallel)
"""

import os
import sys
import time
import math
import pickle
from datetime import datetime
from contextlib import nullcontext

import numpy as np
import torch

# =============================================================================
# Model Size Presets
# =============================================================================
MODEL_SIZE_PRESETS = {
    "small": {
        "n_layer": 8,
        "n_head": 8,
        "n_kv_head": 4,
        "n_embd": 256,
        "dropout": 0.2,
        "batch_size": 128,
        "gradient_accumulation_steps": 1,
    },
    "medium": {
        "n_layer": 12,
        "n_head": 12,
        "n_kv_head": 4,
        "n_embd": 384,
        "dropout": 0.2,
        "batch_size": 96,
        "gradient_accumulation_steps": 1,
    },
    "large": {
        "n_layer": 16,
        "n_head": 16,
        "n_kv_head": 4,
        "n_embd": 512,
        "dropout": 0.2,
        "batch_size": 48,
        "gradient_accumulation_steps": 2,
    },
}
MODEL_SIZE_KEYS = tuple(next(iter(MODEL_SIZE_PRESETS.values())).keys())


def _get_cli_arg_value(key: str):
    prefix = f"--{key}="
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _apply_model_size_preset(size_name: str) -> str:
    size_name = str(size_name).lower()
    if size_name == "custom":
        return size_name
    if size_name not in MODEL_SIZE_PRESETS:
        valid = ", ".join(sorted(list(MODEL_SIZE_PRESETS.keys()) + ["custom"]))
        raise ValueError(f"Unknown model_size '{size_name}'. Valid options: {valid}")
    preset = MODEL_SIZE_PRESETS[size_name]
    for key, value in preset.items():
        globals()[key] = value
    return size_name


# =============================================================================
# Auto Multi-GPU Detection & DDP Launch
# =============================================================================
def _auto_ddp():
    """Auto-detect multiple GPUs and re-launch with torchrun for DDP training.
    
    If multiple GPUs are available and we're not already running under torchrun,
    re-launches the script via torchrun with --nproc_per_node=<num_gpus>.
    Skipped if user specifies --gpu_id (explicit single-GPU mode).
    """
    if 'RANK' in os.environ:
        return  # Already running under torchrun/DDP
    
    # If user explicitly specified a single GPU, skip DDP
    for arg in sys.argv[1:]:
        if arg.startswith('--gpu_id='):
            return
    
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        return
    
    import subprocess
    import random
    port = random.randint(29500, 29999)
    script = os.path.abspath(__file__)
    
    print(f"\n{'='*60}")
    print(f"  Auto-detected {n_gpus} GPUs → launching DDP training")
    for i in range(n_gpus):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Master port: {port}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable, '-m', 'torch.distributed.run',
        f'--nproc_per_node={n_gpus}',
        f'--master_port={port}',
        script,
    ] + sys.argv[1:]
    
    sys.exit(subprocess.run(cmd).returncode)

_auto_ddp()

from model import CompositeDelphi, CompositeDelphiConfig
from utils import get_p2i_composite, get_batch_composite

# =============================================================================
# Default Configuration
# =============================================================================

out_dir = 'out_v6'
out_dir_use_timestamp = True  # when out_dir=='out_v6' and scratch, save to MMDD_HHMM_out_v6
eval_interval = 100
log_interval = 100
eval_iters = 100
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'  # 'scratch' or 'resume'
seed = 42

# wandb logging
wandb_log = False
wandb_project = 'composite-delphi'
wandb_run_name = 'run' + str(time.time())

# data
gradient_accumulation_steps = 1
batch_size = 96
block_size = 512

# model selection (composite only)
model_type = 'composite'
model_size = 'small'  # 'small' | 'medium' | 'large' | 'custom'

# Model config
n_layer = 12
n_head = 12
n_kv_head = 4  # GQA (must divide n_head evenly: 12/4=3 heads per group)
n_embd = 384
dropout = 0.2
bias = False

# Allow one-line architecture preset from CLI, e.g. --model_size=small
# Manual architecture args (e.g. --n_layer=10) still override this later via configurator.
_cli_model_size = _get_cli_arg_value('model_size')
if _cli_model_size is not None:
    model_size = _cli_model_size
model_size = _apply_model_size_preset(model_size)
_pre_config_model_size = model_size
_arch_before_configurator = {k: globals()[k] for k in MODEL_SIZE_KEYS}

# Composite Delphi model config (5-column data)
data_vocab_size = 1290   # DATA: 약품/질병 코드 수 (Classification)
shift_vocab_size = 5     # Legacy discrete SHIFT embedding size (unused when shift_continuous=True)
total_vocab_size = 552   # TOTAL: Embedding vocab

# SHIFT continuous regression settings
shift_continuous = True
shift_log = False                # if True: train SHIFT head on log1p(target) in continuous mode
shift_input_scale = -1.0         # <=0: auto from train data
shift_min_value = 0.0
shift_max_value = -1.0           # <=0: auto from train data percentile
shift_auto_min_percentile = 0.5  # used when shift_max_value <= shift_min_value
shift_auto_max_percentile = 99.5 # set 100.0 to use observed max
shift_exclude_na_token = True
shift_mdn_nll_weight = 0.05
label_scaling = 'none'  # 'none' | 'zscore' | 'robust' | 'minmax'
loss_normalize_by_variance = False

# SHIFT legacy imbalance options (kept for compatibility; ignored in continuous mode)
shift_loss_type = 'dice_focal'      # 'dice_focal', 'focal', 'ce'
shift_dice_weight = 0.5
shift_ignore_index = -1
shift_focal_gamma = 2.0  # Reduced from 5.0 to standard value to prevent hallucinations
shift_class_weights = []  # Empty list = unweighted
shift_maintain_idx = 2
shift_change_weight_max = 10.0
shift_class_weight_cap = 8.0
change_vocab_size = 2

# TOTAL MDN settings
mdn_n_components = 8
total_min_value = 0.0
total_max_value = 550.0

# Loss weights for composite model
loss_weight_data = 1.0
loss_weight_shift = 20.0
loss_weight_change = 0.0
loss_weight_total = 5.0
loss_weight_time = 1.0

# architecture features
use_moe = True
num_experts = 8
experts_per_token = 2
sliding_window = 128

# Drug-conditioning
use_drug_conditioning = True
rope_theta = 10000.0

# TOTAL regression (legacy fallback only; MDN path does not use this)
total_log_transform = False

# adamw optimizer
learning_rate = 6e-4
max_iters = 10000        # Increased from 10000
# early stopping
# Stop when validation loss has not improved for this many iterations.
# Set <= 0 to disable early stopping.
early_stop_patience_iters = 1000
# max_iters = 2000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 9000   # Adjusted for 20000 max_iters
min_lr = 3e-5

# system
gpu_id = 0  # GPU device ID (e.g., 0, 1, 2, ...)
device = 'cpu'  # Will be set after config parsing
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
compile = False  # torch.compile (requires PyTorch 2.0+)

# delphi training
token_dropout = 0.0
t_min = 0.1  # Prevent log(0) numerical instability
mask_ties = True
ignore_tokens = [0]
data_fraction = 1.0
no_event_token_rate = 5
apply_token_shift = False
separate_shift_na_from_padding = True
shift_na_raw_token = 4

# Time-to-Event distribution: 'exponential' or 'weibull'
time_distribution = 'exponential'

TRAIN_DATA_PATH = '../data/dose/kr_train.bin'
VAL_DATA_PATH = '../data/dose/kr_val.bin'
# JMDC path for domain generalization (mixing)
JMDC_DATA_PATH = '../data/dose/JMDC_exval2.bin'

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list))]
exec(open('configurator.py').read())

# Support model_size inside config files when architecture keys are not explicitly set there.
# If CLI already provided --model_size, pre-config application is sufficient.
if _cli_model_size is None:
    _post_config_model_size = str(model_size).lower()
    if _post_config_model_size != _pre_config_model_size:
        _arch_after_configurator = {k: globals()[k] for k in MODEL_SIZE_KEYS}
        if _arch_after_configurator == _arch_before_configurator:
            model_size = _apply_model_size_preset(_post_config_model_size)

# If SHIFT N/A is separated from padding/no-event, shifted mode needs one extra class:
# 0=pad, 1=no-event, 2=dec, 3=maint, 4=inc, 5=na
if separate_shift_na_from_padding and apply_token_shift and shift_vocab_size < 6:
    shift_vocab_size = 6

# Continuous SHIFT regression should not treat a numeric value as a special NA class token.
if shift_continuous and separate_shift_na_from_padding:
    separate_shift_na_from_padding = False
    print("[fix] shift_continuous=True -> forcing separate_shift_na_from_padding=False")

config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

if model_type != 'composite':
    raise ValueError("Only composite model_type is supported.")

# =============================================================================
# DDP Setup & Device Configuration
# =============================================================================
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(ddp_local_rank)
    master_process = (ddp_rank == 0)
    seed_offset = ddp_rank
    if master_process:
        print(f"DDP training: {ddp_world_size} GPUs")
        for i in range(ddp_world_size):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    ddp_world_size = 1
    master_process = True
    seed_offset = 0
    if torch.cuda.is_available():
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
        if master_process:
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon)")
    else:
        device = 'cpu'
        print("Using CPU")

# Resolve final checkpoint directory
# - v6 default behavior: out_v6 -> MMDD_HHMM_out_v6 for scratch runs
# - compatibility: normalize legacy out_v6_cont -> out_v6
out_dir_norm = os.path.normpath(out_dir)
out_dir_parent = os.path.dirname(out_dir_norm)
out_dir_base = os.path.basename(out_dir_norm)
if out_dir_base == 'out_v6_cont':
    out_dir_base = 'out_v6'
    out_dir_norm = os.path.join(out_dir_parent, out_dir_base) if out_dir_parent else out_dir_base
    out_dir = out_dir_norm
    if master_process:
        print("[path] normalized out_dir from 'out_v6_cont' to 'out_v6'")

if (
    init_from == 'scratch'
    and out_dir_use_timestamp
    and out_dir_base == 'out_v6'
):
    run_timestamp = os.environ.get('TRAIN_RUN_TIMESTAMP')
    if run_timestamp is None:
        if ddp:
            # Make sure all ranks use exactly the same run timestamp.
            obj = [datetime.now().strftime('%m%d_%H%M') if master_process else None]
            dist.broadcast_object_list(obj, src=0)
            run_timestamp = obj[0]
        else:
            run_timestamp = datetime.now().strftime('%m%d_%H%M')
        os.environ['TRAIN_RUN_TIMESTAMP'] = run_timestamp
    out_dir = os.path.join(out_dir_parent, f"{run_timestamp}_{out_dir_base}") if out_dir_parent else f"{run_timestamp}_{out_dir_base}"

# Keep saved config aligned with the effective checkpoint directory
config['out_dir'] = out_dir

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size * ddp_world_size
if master_process:
    print(
        f"Model size: {model_size} | "
        f"layers={n_layer}, heads={n_head}, kv_heads={n_kv_head}, embd={n_embd}, "
        f"batch_size={batch_size}, grad_acc={gradient_accumulation_steps}"
    )
    print(f"Checkpoint directory: {out_dir}")
    print(f"Tokens per iteration: {tokens_per_iter:,} ({ddp_world_size} GPU(s))")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
ptdtype = {'float32': torch.float32, 'float64': torch.float64,
           'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.set_default_dtype(ptdtype)

# =============================================================================
# Data Loading
# =============================================================================

# data_dir = '../data'

def _compute_shift_class_weights(shift_values, shift_vocab_size, shift_ignore_index):
    counts = np.bincount(shift_values, minlength=shift_vocab_size).astype(np.float64)
    if shift_ignore_index is not None and 0 <= shift_ignore_index < shift_vocab_size:
        counts[shift_ignore_index] = 0.0
    nonzero = counts > 0
    weights = np.zeros(shift_vocab_size, dtype=np.float32)
    if nonzero.any():
        weights[nonzero] = counts[nonzero].sum() / (counts[nonzero] * nonzero.sum())
        cap = float(globals().get('shift_class_weight_cap', 8.0))
        weights[nonzero] = np.clip(weights[nonzero], 1.0, cap)
    return weights.tolist()


def _remap_shift_to_change_np(shift_values: np.ndarray, shifted: bool) -> np.ndarray:
    """
    Remap SHIFT labels to binary change labels:
    - changed(1): Dec/Inc
    - maintain(0): Maint
    """
    out = np.full(shift_values.shape, -1, dtype=np.int64)
    if shifted:
        # shifted labels: Dec=2, Maint=3, Inc=4
        is_dec = shift_values == 2
        is_maint = shift_values == 3
        is_inc = shift_values == 4
    else:
        # raw labels: Dec=1, Maint=2, Inc=3
        is_dec = shift_values == 1
        is_maint = shift_values == 2
        is_inc = shift_values == 3
    out[is_maint] = 0
    out[is_dec | is_inc] = 1
    return out


def _compute_label_scaling_stats(values: np.ndarray, strategy: str):
    """Compute train-set-only scaling stats for ablations."""
    strategy = str(strategy).lower()
    values = values.astype(np.float32)
    if values.size == 0:
        return {
            'center': 0.0,
            'scale': 1.0,
            'min': 0.0,
            'max': 1.0,
            'var': 1.0,
        }

    out = {
        'center': 0.0,
        'scale': 1.0,
        'min': float(values.min()),
        'max': float(values.max()),
        'var': max(float(np.var(values)), 1e-8),
    }
    if strategy == 'zscore':
        out['center'] = float(values.mean())
        out['scale'] = max(float(values.std()), 1e-8)
    elif strategy == 'robust':
        q1, q2, q3 = np.percentile(values, [25.0, 50.0, 75.0])
        out['center'] = float(q2)
        out['scale'] = max(float(q3 - q1), 1e-8)
    elif strategy == 'minmax':
        out['center'] = out['min']
        out['scale'] = max(float(out['max'] - out['min']), 1e-8)
    elif strategy == 'none':
        pass
    else:
        raise ValueError(f"Unknown label_scaling strategy: {strategy}")
    return out

# 6-column structured data: (ID, AGE, DATA, DOSE, TOTAL, UNIT)
# composite_dtype = np.dtype([
#     ('ID', '<u4'),
#     ('AGE', '<u4'),
#     ('DATA', '<u4'),
#     ('DOSE', '<f4'),
#     ('TOTAL', '<u4'),
#     ('UNIT', '<u4')
# ])
composite_dtype = np.dtype([
    ('ID', np.uint32),
    ('AGE', np.uint32),
    ('DATA', np.uint32),
    ('SHIFT', np.float32),
    ('TOTAL', np.uint32)
])

# train_data = np.memmap(TRAIN_DATA_PATH, dtype=composite_dtype, mode='r')
# val_data = np.memmap(VAL_DATA_PATH, dtype=composite_dtype, mode='r')
train_data = np.fromfile(TRAIN_DATA_PATH, dtype=composite_dtype)
val_data = np.fromfile(VAL_DATA_PATH, dtype=composite_dtype)

train_p2i = get_p2i_composite(train_data)
val_p2i = get_p2i_composite(val_data)

if master_process:
    print(f"Loaded composite data: train={len(train_data)}, val={len(val_data)}")
    print(f"Unique patients: train={len(train_p2i)}, val={len(val_p2i)}")
    print(f"SHIFT N/A separation: {separate_shift_na_from_padding} (shift_na_raw_token={shift_na_raw_token})")

# Drug token range (used by both class-weighting and patient sampling)
drug_token_min = 1279 if apply_token_shift else 1278
drug_token_max = 1289 if apply_token_shift else 1288

# SHIFT continuous stats (used to set MDN range and input scaling)
shift_stats = train_data['SHIFT'].astype(np.float32)
if apply_token_shift:
    shift_stats = shift_stats + 1.0
shift_valid_stats = shift_stats >= 0
if shift_continuous and separate_shift_na_from_padding and shift_exclude_na_token:
    shift_na_token = float(shift_na_raw_token + (1 if apply_token_shift else 0))
    shift_valid_stats &= shift_stats != shift_na_token
shift_stats = shift_stats[shift_valid_stats]
if shift_continuous and shift_stats.size > 0:
    p_hi_ref = float(np.percentile(shift_stats, 99.5))
    if shift_max_value > 0 and shift_max_value > max(10.0, p_hi_ref * 3.0):
        if master_process:
            print(
                f"[warning] shift_max_value={shift_max_value:.4f} is much larger than "
                f"SHIFT p99.5={p_hi_ref:.4f}. "
                "This can weaken SHIFT gradients (especially drug-conditioned head). "
                "Consider --shift_max_value=-1 (auto) or a tighter cap."
            )
    if shift_max_value <= shift_min_value:
        p_lo = float(np.percentile(shift_stats, shift_auto_min_percentile))
        if float(shift_auto_max_percentile) >= 100.0:
            p_hi = float(shift_stats.max())
        else:
            p_hi = float(np.percentile(shift_stats, shift_auto_max_percentile))
        if p_hi <= p_lo:
            p_lo = float(shift_stats.min())
            p_hi = float(shift_stats.max())
        shift_min_value = max(0.0, p_lo)
        shift_max_value = max(shift_min_value + 1.0, p_hi)
        if master_process:
            print(
                f"SHIFT auto-range: p{shift_auto_min_percentile}={p_lo:.4f}, "
                f"p{shift_auto_max_percentile}={p_hi:.4f}, data_max={float(shift_stats.max()):.4f}"
            )
    if shift_input_scale <= 0:
        shift_scale_stats = shift_stats
        if shift_log:
            shift_scale_stats = np.log1p(np.clip(shift_scale_stats, a_min=0.0, a_max=None))
        shift_input_scale = max(float(np.percentile(np.abs(shift_scale_stats), 95.0)), 1.0)
    if master_process:
        print(
            f"SHIFT continuous mode: min={shift_min_value:.4f}, max={shift_max_value:.4f}, "
            f"input_scale={shift_input_scale:.4f}, shift_log={shift_log}"
        )

total_stats = train_data['TOTAL'].astype(np.float32)
total_valid_stats = total_stats >= 0
total_stats = total_stats[total_valid_stats]

label_scaling = str(label_scaling).lower()
shift_scaling_stats = _compute_label_scaling_stats(shift_stats, label_scaling)
total_scaling_stats = _compute_label_scaling_stats(total_stats, label_scaling)

if master_process:
    print(
        f"Label scaling={label_scaling} | "
        f"SHIFT(center={shift_scaling_stats['center']:.4f}, scale={shift_scaling_stats['scale']:.4f}, var={shift_scaling_stats['var']:.4f}) | "
        f"TOTAL(center={total_scaling_stats['center']:.4f}, scale={total_scaling_stats['scale']:.4f}, var={total_scaling_stats['var']:.4f})"
    )
    if loss_normalize_by_variance:
        print(
            f"Loss variance normalization enabled: shift_var={shift_scaling_stats['var']:.6f}, "
            f"total_var={total_scaling_stats['var']:.6f}"
        )

# Dynamic Class Weighting (SHIFT, legacy discrete mode only)
if not shift_continuous and not shift_class_weights:
    drug_mask = (train_data['DATA'] >= drug_token_min) & (train_data['DATA'] <= drug_token_max)
    shift_values = train_data['SHIFT'][drug_mask].astype(np.int64)
    if apply_token_shift:
        shift_values = shift_values + 1
    shift_values = _remap_shift_to_change_np(shift_values, shifted=apply_token_shift)
    shift_values = shift_values[shift_values >= 0]
    shift_class_weights = _compute_shift_class_weights(
        shift_values,
        change_vocab_size,
        shift_ignore_index,
    )
    if master_process:
        print(f"Computed binary change class weights (drug-token subset): {shift_class_weights}")
elif shift_continuous:
    shift_class_weights = []
    if master_process:
        print("SHIFT continuous mode: skipping binary class-weight computation.")

# WeightedRandomSampler: Patient-level sampling
if master_process:
    if shift_continuous:
        print("Computing patient-level sampling weights for continuous SHIFT signal...")
    else:
        print("Computing patient-level sampling weights for binary-change balancing...")

patient_weights = np.zeros(len(train_p2i), dtype=np.float32)
for pid, (start_idx, length) in enumerate(train_p2i):
    patient_data = train_data[start_idx:start_idx + length]
    drug_mask = (patient_data['DATA'] >= drug_token_min) & (patient_data['DATA'] <= drug_token_max)
    patient_shifts = patient_data['SHIFT'][drug_mask].astype(np.float32)
    if apply_token_shift:
        patient_shifts = patient_shifts + 1.0
    if shift_continuous:
        if separate_shift_na_from_padding and shift_exclude_na_token:
            na_token = float(shift_na_raw_token + (1 if apply_token_shift else 0))
            patient_shifts = patient_shifts[patient_shifts != na_token]
        signal_count = (patient_shifts > 0).sum()
        patient_weights[pid] = 1.0 + signal_count * 0.05
    else:
        patient_changes = _remap_shift_to_change_np(patient_shifts.astype(np.int64), shifted=apply_token_shift)
        minority_count = (patient_changes == 1).sum()
        patient_weights[pid] = 1.0 + minority_count * 0.3

patient_weights = patient_weights / patient_weights.sum()
patient_weights_tensor = torch.from_numpy(patient_weights)

minority_patient_count = (patient_weights > 1.0 / len(train_p2i)).sum()
if master_process:
    if shift_continuous:
        print(f"  Patients with non-zero SHIFT events: {minority_patient_count:,} / {len(train_p2i):,}")
    else:
        print(f"  Patients with changed SHIFT events: {minority_patient_count:,} / {len(train_p2i):,}")
    print(f"  Max sampling weight: {patient_weights.max():.4f}, Min: {patient_weights.min():.6f}")

# Downsample to requested fraction
if data_fraction < 1.0:
    train_p2i = train_p2i[:int(data_fraction * len(train_p2i))]
    if master_process:
        print(f"Using {data_fraction*100:.1f}% of training data: {len(train_p2i)} patients")

iter_num = 0
best_val_loss = 1e9

# =============================================================================
# Model Initialization
# =============================================================================

# Composite Delphi with multi-head output
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_kv_head=n_kv_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    dropout=dropout,
    token_dropout=token_dropout,
    t_min=t_min,
    mask_ties=mask_ties,
    ignore_tokens=ignore_tokens,
    use_moe=use_moe,
    num_experts=num_experts,
    experts_per_token=experts_per_token,
    sliding_window=sliding_window,
    rope_theta=rope_theta,
    use_drug_conditioning=use_drug_conditioning,
    drug_token_min=drug_token_min,
    drug_token_max=drug_token_max,
    mdn_n_components=mdn_n_components,
    shift_min_value=shift_min_value,
    shift_max_value=shift_max_value,
    shift_continuous=shift_continuous,
    shift_log=shift_log,
    shift_input_scale=shift_input_scale,
    shift_exclude_na_token=shift_exclude_na_token,
    shift_mdn_nll_weight=shift_mdn_nll_weight,
    shift_label_scaling=label_scaling,
    shift_label_center=shift_scaling_stats['center'],
    shift_label_scale=shift_scaling_stats['scale'],
    shift_label_min=shift_scaling_stats['min'],
    shift_label_max=shift_scaling_stats['max'],
    total_min_value=total_min_value,
    total_max_value=total_max_value,
    total_log_transform=total_log_transform,
    total_label_scaling=label_scaling,
    total_label_center=total_scaling_stats['center'],
    total_label_scale=total_scaling_stats['scale'],
    total_label_min=total_scaling_stats['min'],
    total_label_max=total_scaling_stats['max'],
    loss_normalize_by_variance=loss_normalize_by_variance,
    shift_loss_variance=shift_scaling_stats['var'],
    total_loss_variance=total_scaling_stats['var'],
    # Composite-specific
    data_vocab_size=data_vocab_size,
    shift_vocab_size=shift_vocab_size,
    total_vocab_size=total_vocab_size,
    # SHIFT loss options
    shift_loss_type=shift_loss_type,
    shift_dice_weight=shift_dice_weight,
    shift_ignore_index=shift_ignore_index,
    shift_maintain_idx=shift_maintain_idx,
    shift_change_weight_max=shift_change_weight_max,
    shift_focal_gamma=shift_focal_gamma,
    shift_class_weights=shift_class_weights,
    apply_token_shift=apply_token_shift,
    separate_shift_na_from_padding=separate_shift_na_from_padding,
    shift_na_raw_token=shift_na_raw_token,
    loss_weight_data=loss_weight_data,
    loss_weight_shift=loss_weight_shift,
    loss_weight_change=loss_weight_change,
    loss_weight_total=loss_weight_total,
    loss_weight_time=loss_weight_time,
    # Time-to-Event distribution
    time_distribution=time_distribution,
)

if init_from == 'scratch':
    if master_process:
        print("Initializing a new Composite Delphi model from scratch")
    gptconf = CompositeDelphiConfig(**model_args)
    model = CompositeDelphi(gptconf)
elif init_from == 'resume':
    if master_process:
        print(f"Resuming Composite Delphi training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_kv_head', 'n_embd', 'block_size', 'bias',
              'data_vocab_size', 'shift_vocab_size', 'total_vocab_size',
              'shift_min_value', 'shift_max_value', 'shift_continuous',
              'shift_log', 'shift_input_scale', 'shift_exclude_na_token', 'shift_mdn_nll_weight',
              'shift_label_scaling', 'shift_label_center', 'shift_label_scale', 'shift_label_min', 'shift_label_max',
              'total_label_scaling', 'total_label_center', 'total_label_scale', 'total_label_min', 'total_label_max',
              'loss_normalize_by_variance', 'shift_loss_variance', 'total_loss_variance']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    if 'shift_log' not in checkpoint_model_args:
        model_args['shift_log'] = False
    if 'shift_continuous' not in checkpoint_model_args:
        model_args['shift_continuous'] = False
        model_args['shift_log'] = False
        model_args['shift_input_scale'] = 1.0
        model_args['shift_exclude_na_token'] = True
        model_args['shift_mdn_nll_weight'] = 0.0
    gptconf = CompositeDelphiConfig(**model_args)
    model = CompositeDelphi(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

# raw_model: always points to the unwrapped model (before compile/DDP)
# Use for state_dict, configure_optimizers, get_num_params, etc.
raw_model = model

if master_process:
    print(f"Model type: {model_type}")
    print(f"Model parameters: {raw_model.get_num_params()/1e6:.2f}M")

# =============================================================================
# Optimizer & Scaler
# =============================================================================

scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16' and device_type == 'cuda'))
optimizer = raw_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# Track the iteration where best validation loss was last improved.
best_val_improve_iter = iter_num

# Compile (before DDP wrapping)
if compile:
    if master_process:
        print("Compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# DDP wrapping (after compile)
# find_unused_parameters=True: needed because drug-conditioned heads and MoE experts
# may not participate in every forward pass (conditional activation)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

# =============================================================================
# Loss Estimation Functions
# =============================================================================

@torch.no_grad()
def estimate_loss():
    """Estimate loss for Composite Delphi"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, 6)  # loss, data, shift, change, total, time
        data = train_data if split == 'train' else val_data
        p2i = train_p2i if split == 'train' else val_p2i
        for k in range(eval_iters):
            ix = torch.randint(len(p2i), (batch_size,))
            batch = get_batch_composite(ix, data, p2i, block_size=block_size,
                                        device=device, select='left',
                                        no_event_token_rate=no_event_token_rate,
                                        cut_batch=True,
                                        apply_token_shift=apply_token_shift,
                                        shift_continuous=shift_continuous,
                                        separate_shift_na_from_padding=separate_shift_na_from_padding,
                                        shift_na_raw_token=shift_na_raw_token)
            x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages = batch
            
            with ctx:
                logits, loss, _ = model(
                    x_data, x_shift, x_total, x_ages,
                    y_data, y_shift, y_total, y_ages,
                    validation_loss_mode=True
                )
            losses[k] = torch.stack([
                loss['loss'],
                loss['loss_data'],
                loss['loss_shift'],
                loss['loss_change'],
                loss['loss_total'],
                loss['loss_time']
            ])
        out[split] = losses.mean(0)
    model.train()
    return out

# =============================================================================
# Learning Rate Scheduler
# =============================================================================

def get_lr(it):
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # After decay, return min_lr
    if it > lr_decay_iters:
        return min_lr
    # Cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# =============================================================================
# Logging Setup
# =============================================================================

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


def _save_checkpoint(checkpoint_obj, filename: str, tag: str):
    ckpt_path = os.path.join(out_dir, filename)
    torch.save(checkpoint_obj, ckpt_path)
    if master_process:
        print(f"[checkpoint:{tag}] saved: {ckpt_path}")


def _save_loss_plot(train_steps, train_losses, val_steps, val_losses):
    if not master_process:
        return None
    if len(train_losses) == 0 and len(val_losses) == 0:
        print("[plot] skipped: no loss history")
        return None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped: matplotlib unavailable ({e})")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    if len(train_losses) > 0:
        ax.plot(train_steps, train_losses, label='train/loss', color='#1f77b4')
    if len(val_losses) > 0:
        ax.plot(val_steps, val_losses, label='val/loss', color='#ff7f0e')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(out_dir, 'loss_plot.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved: {plot_path}")
    return plot_path

# =============================================================================
# Training Loop
# =============================================================================

if master_process:
    print(f"{'='*60}")
    print(f"  Device: {device} ({'DDP x' + str(ddp_world_size) if ddp else 'single'})")
    print(f"  Batch size: {batch_size} (x{ddp_world_size} GPUs = {batch_size * ddp_world_size} effective)")
    print(f"  Block size: {block_size}")
    print(f"  Max iterations: {max_iters}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Validation interval: every {eval_interval} iterations")
    print(f"{'='*60}\n")

# Initial batch (weighted sampling for SHIFT class balance)
ix = torch.multinomial(patient_weights_tensor, batch_size, replacement=True)
batch = get_batch_composite(ix, train_data, train_p2i, block_size=block_size, device=device,
                            padding='random', lifestyle_augmentations=True, select='left',
                            no_event_token_rate=no_event_token_rate,
                            apply_token_shift=apply_token_shift,
                            shift_continuous=shift_continuous,
                            separate_shift_na_from_padding=separate_shift_na_from_padding,
                            shift_na_raw_token=shift_na_raw_token)
x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages = batch

t0 = time.time()
local_iter_num = 0
val_loss = None
early_stop_triggered = False
train_loss_steps, train_loss_history = [], []
val_loss_steps, val_loss_history = [], []

while True:
    # Set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate and checkpoint
    # All processes evaluate independently; only master prints/saves
    if iter_num % eval_interval == 0 and iter_num > 0:
        losses = estimate_loss()

        # Composite model loss components
        if val_loss is None:
            val_loss_unpooled = losses['val']
        val_loss_unpooled = 0.1 * losses['val'] + 0.9 * val_loss_unpooled
        val_loss = val_loss_unpooled[0].item()  # Total loss

        if master_process:
            train_breakdown = losses['train']
            val_breakdown = losses['val']
            print(f"step {iter_num}: train loss {train_breakdown[0].item():.4f}, val loss {val_breakdown[0].item():.4f} (ema {val_loss:.4f})")
            print(
                "  breakdown (train/val) - "
                f"data: {train_breakdown[1].item():.4f}/{val_breakdown[1].item():.4f}, "
                f"shift: {train_breakdown[2].item():.4f}/{val_breakdown[2].item():.4f}, "
                f"total: {train_breakdown[4].item():.4f}/{val_breakdown[4].item():.4f}, "
                f"time: {train_breakdown[5].item():.4f}/{val_breakdown[5].item():.4f}"
            )
            val_loss_steps.append(iter_num)
            val_loss_history.append(val_loss)

            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'][0].item(),
                    "val/loss": val_loss,
                    "val/loss_data": val_loss_unpooled[1].item(),
                    "val/loss_shift": val_loss_unpooled[2].item(),
                    "val/loss_total": val_loss_unpooled[4].item(),
                    "val/loss_time": val_loss_unpooled[5].item(),
                })

        # Save ckpt.pt only when validation loss improves.
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_improve_iter = iter_num
            if master_process and iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': val_loss,
                    'config': config,
                    'model_type': model_type,
                }
                _save_checkpoint(checkpoint, 'ckpt.pt', 'best')
                print(f"[best] ckpt.pt updated at iter {iter_num} (val/loss={val_loss:.6f})")
        elif master_process:
            print(f"[best] not updated at iter {iter_num} (val/loss={val_loss:.6f}, best={best_val_loss:.6f})")

        # Early stopping: no validation improvement for N iterations.
        should_early_stop = False
        if early_stop_patience_iters > 0:
            no_improve_iters = iter_num - best_val_improve_iter
            should_early_stop = no_improve_iters >= early_stop_patience_iters

        # Keep stop decision consistent across all DDP workers.
        if ddp:
            stop_tensor = torch.tensor(1 if should_early_stop else 0, device=device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            should_early_stop = bool(stop_tensor.item())

        if should_early_stop:
            if master_process:
                no_improve_iters = iter_num - best_val_improve_iter
                print(
                    f"[early-stop] no val loss improvement for {no_improve_iters} iterations "
                    f"(patience={early_stop_patience_iters}); stopping at iter {iter_num}."
                )
            early_stop_triggered = True
            break

        # Save periodic checkpoint (master only)
        if master_process and iter_num % 10_000 == 0:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
                'model_type': model_type,
            }
            _save_checkpoint(checkpoint, f'ckpt_{iter_num}.pt', f'periodic@{iter_num}')

    if iter_num == 0 and eval_only:
        break

    # Training step
    for micro_step in range(gradient_accumulation_steps):
        # DDP: only sync gradients on the last micro-step (performance optimization)
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            logits, loss, att = model(
                x_data, x_shift, x_total, x_ages,
                y_data, y_shift, y_total, y_ages
            )

        # Prefetch next batch (weighted sampling for SHIFT class balance)
        ix = torch.multinomial(patient_weights_tensor, batch_size, replacement=True)
        batch = get_batch_composite(ix, train_data, train_p2i, block_size=block_size, device=device,
                                    padding='random', lifestyle_augmentations=True, select='left',
                                    no_event_token_rate=no_event_token_rate, cut_batch=True,
                                    apply_token_shift=apply_token_shift,
                                    shift_continuous=shift_continuous,
                                    separate_shift_na_from_padding=separate_shift_na_from_padding,
                                    shift_na_raw_token=shift_na_raw_token)
        x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages = batch
        total_loss = loss['loss']

        scaler.scale(total_loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if master_process and iter_num % log_interval == 0:
        lossf = total_loss.item()
        train_loss_steps.append(iter_num)
        train_loss_history.append(lossf)
        valf = f"{val_loss:.4f}" if val_loss is not None else "n/a"
        # Show lr trend
        if iter_num > 0 and iter_num % (log_interval * 10) == 0:
            prev_lr = get_lr(iter_num - log_interval) if decay_lr else learning_rate
            lr_change = "↑" if lr > prev_lr else "↓" if lr < prev_lr else "="
            print(f"iter {iter_num}: loss {lossf:.4f}, val {valf}, time {dt*1000:.2f}ms, lr {lr:.2e} {lr_change} (warmup: {iter_num < warmup_iters}, decay: {iter_num > warmup_iters})")
        else:
            print(f"iter {iter_num}: loss {lossf:.4f}, val {valf}, time {dt*1000:.2f}ms, lr {lr:.2e}")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": total_loss.item(),
                "train/loss_data": loss['loss_data'].item(),
                "train/loss_shift": loss['loss_shift'].item(),
                "train/loss_total": loss['loss_total'].item(),
                "train/loss_time": loss['loss_time'].item(),
                "lr": lr,
            })

    iter_num += 1
    local_iter_num += 1

    # Termination
    if iter_num > max_iters:
        break

if master_process:
    loss_plot_path = _save_loss_plot(
        train_loss_steps, train_loss_history,
        val_loss_steps, val_loss_history,
    )
    if wandb_log and loss_plot_path is not None:
        wandb.log({"train/loss_plot": wandb.Image(loss_plot_path)})

    print(f"\n{'='*60}")
    print(f"Training completed!")
    if early_stop_triggered:
        print(f"Early stopping: triggered (patience={early_stop_patience_iters} iters)")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total iterations: {iter_num}")
    print(f"{'='*60}")

# DDP cleanup
if ddp:
    dist.destroy_process_group()
