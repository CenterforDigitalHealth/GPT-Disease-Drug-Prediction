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

out_dir = 'out'
out_dir_use_timestamp = True  # when out_dir=='out' and scratch, save to out/YYYYMMDD_HHMMSS
eval_interval = 2000
log_interval = 100
eval_iters = 200
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
model_size = 'medium'  # 'small' | 'medium' | 'large' | 'custom'

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
shift_vocab_size = 5     # SHIFT: Classification (values 0-4)
total_vocab_size = 552   # TOTAL: Embedding vocab

# SHIFT imbalance handling
shift_loss_type = 'dice_focal'      # 'dice_focal', 'focal', 'ce'
shift_dice_weight = 0.5
shift_ignore_index = 0
shift_focal_gamma = 2.0  # Reduced from 5.0 to standard value to prevent hallucinations
shift_class_weights = []  # Empty list = unweighted
shift_maintain_idx = 2
shift_change_weight_max = 10.0
shift_class_weight_cap = 8.0

# TOTAL MDN settings
mdn_n_components = 8
total_min_value = 0.0
total_max_value = 550.0

# Loss weights for composite model
loss_weight_data = 1.0
loss_weight_shift = 20.0
loss_weight_change = 5.0
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
# max_iters = 2000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 19000   # Adjusted for 20000 max_iters
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

# Time-to-Event distribution: 'exponential' or 'weibull'
time_distribution = 'exponential'

TRAIN_DATA_PATH = '../data/kr_train.bin'
VAL_DATA_PATH = '../data/kr_val.bin'
# JMDC path for domain generalization (mixing)
JMDC_DATA_PATH = '../data/JMDC_extval.bin'

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
# - default behavior: out -> out/YYYYMMDD_HHMMSS for scratch runs
# - keep explicit out_dir values unchanged (important for ablation scripts)
if (
    init_from == 'scratch'
    and out_dir_use_timestamp
    and os.path.normpath(out_dir) == 'out'
):
    run_timestamp = os.environ.get('TRAIN_RUN_TIMESTAMP')
    if run_timestamp is None:
        if ddp:
            # Make sure all ranks use exactly the same timestamped path.
            obj = [datetime.now().strftime('%Y%m%d_%H%M%S') if master_process else None]
            dist.broadcast_object_list(obj, src=0)
            run_timestamp = obj[0]
        else:
            run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.environ['TRAIN_RUN_TIMESTAMP'] = run_timestamp
    out_dir = os.path.join(out_dir, run_timestamp)

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
    ('SHIFT', np.uint32),
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

# Drug token range (used by both class-weighting and patient sampling)
drug_token_min = 1279 if apply_token_shift else 1278
drug_token_max = 1289 if apply_token_shift else 1288

# Dynamic Class Weighting (SHIFT)
if not shift_class_weights:
    drug_mask = (train_data['DATA'] >= drug_token_min) & (train_data['DATA'] <= drug_token_max)
    shift_values = train_data['SHIFT'][drug_mask].astype(np.int64)
    if apply_token_shift:
        shift_values = shift_values + 1
    shift_class_weights = _compute_shift_class_weights(
        shift_values,
        shift_vocab_size,
        shift_ignore_index,
    )
    if master_process:
        print(f"Computed shift class weights (drug-token subset): {shift_class_weights}")

# WeightedRandomSampler: Patient-level balanced sampling
if master_process:
    print("Computing patient-level sampling weights for SHIFT balancing...")

minority_classes = [1, 3] if not apply_token_shift else [2, 4]
patient_weights = np.zeros(len(train_p2i), dtype=np.float32)
for pid, (start_idx, length) in enumerate(train_p2i):
    patient_data = train_data[start_idx:start_idx + length]
    drug_mask = (patient_data['DATA'] >= drug_token_min) & (patient_data['DATA'] <= drug_token_max)
    patient_shifts = patient_data['SHIFT'][drug_mask]
    minority_count = sum((patient_shifts == c).sum() for c in minority_classes)
    patient_weights[pid] = 1.0 + minority_count * 0.3

patient_weights = patient_weights / patient_weights.sum()
patient_weights_tensor = torch.from_numpy(patient_weights)

minority_patient_count = (patient_weights > 1.0 / len(train_p2i)).sum()
if master_process:
    print(f"  Patients with minority SHIFT events: {minority_patient_count:,} / {len(train_p2i):,}")
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
    total_min_value=total_min_value,
    total_max_value=total_max_value,
    total_log_transform=total_log_transform,
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
              'data_vocab_size', 'shift_vocab_size', 'total_vocab_size']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
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
                                        apply_token_shift=apply_token_shift)
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

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


def _save_checkpoint(checkpoint_obj, filename: str, tag: str):
    ckpt_path = os.path.join(out_dir, filename)
    torch.save(checkpoint_obj, ckpt_path)
    if master_process:
        print(f"[checkpoint:{tag}] saved: {ckpt_path}")

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
    print(f"{'='*60}\n")

# Initial batch (weighted sampling for SHIFT class balance)
ix = torch.multinomial(patient_weights_tensor, batch_size, replacement=True)
batch = get_batch_composite(ix, train_data, train_p2i, block_size=block_size, device=device,
                            padding='random', lifestyle_augmentations=True, select='left',
                            no_event_token_rate=no_event_token_rate,
                            apply_token_shift=apply_token_shift)
x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages = batch

t0 = time.time()
local_iter_num = 0
val_loss = None

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
                f"change: {train_breakdown[3].item():.4f}/{val_breakdown[3].item():.4f}, "
                f"total: {train_breakdown[4].item():.4f}/{val_breakdown[4].item():.4f}, "
                f"time: {train_breakdown[5].item():.4f}/{val_breakdown[5].item():.4f}"
            )

            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'][0].item(),
                    "val/loss": val_loss,
                    "val/loss_data": val_loss_unpooled[1].item(),
                    "val/loss_shift": val_loss_unpooled[2].item(),
                    "val/loss_change": val_loss_unpooled[3].item(),
                    "val/loss_total": val_loss_unpooled[4].item(),
                    "val/loss_time": val_loss_unpooled[5].item(),
                })

        # Save best checkpoint (master only)
        if master_process and (always_save_checkpoint or val_loss < best_val_loss):
            best_val_loss = val_loss
            if iter_num > 0:
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
                                    apply_token_shift=apply_token_shift)
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
        # Show lr trend
        if iter_num > 0 and iter_num % (log_interval * 10) == 0:
            prev_lr = get_lr(iter_num - log_interval) if decay_lr else learning_rate
            lr_change = "↑" if lr > prev_lr else "↓" if lr < prev_lr else "="
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e} {lr_change} (warmup: {iter_num < warmup_iters}, decay: {iter_num > warmup_iters})")
        else:
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": total_loss.item(),
                "train/loss_data": loss['loss_data'].item(),
                "train/loss_shift": loss['loss_shift'].item(),
                "train/loss_change": loss['loss_change'].item(),
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
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total iterations: {iter_num}")
    print(f"{'='*60}")

# DDP cleanup
if ddp:
    dist.destroy_process_group()
