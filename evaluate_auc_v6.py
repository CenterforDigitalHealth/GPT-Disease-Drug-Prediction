import scipy.stats
import scipy
import warnings
import torch
# Suppress sklearn warnings about classes not in y_true
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics._classification')
from model_v6 import CompositeDelphi, CompositeDelphiConfig
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from utils import get_batch_composite, get_p2i_composite
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, 
    top_k_accuracy_score, 
    mean_absolute_error, 
    mean_squared_error,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    r2_score,
    confusion_matrix
)

def auc(x1, x2):
    n1 = len(x1)
    n2 = len(x2)
    R1 = np.concatenate([x1, x2]).argsort().argsort()[:n1].sum() + n1
    U1 = n1 * n2 + 0.5 * n1 * (n1 + 1) - R1
    if n1 == 0 or n2 == 0:
        return np.nan
    return U1 / n1 / n2


def get_common_diseases(labels_df, filter_min_total=100):
    """
    Get common diseases from labels DataFrame.
    
    Args:
        labels_df: DataFrame with columns including 'index' and optionally 'count'
        filter_min_total: Minimum count to include a token
    
    Returns:
        List of shifted token IDs (labels.csv index + 1, due to +1 shift in get_batch_composite)
    """
    if 'count' in labels_df.columns:
        labels_df_filtered = labels_df[labels_df['count'] > filter_min_total]
    else:
        # If no count column, use all non-special tokens
        # Assuming tokens 0-20 are special tokens (padding, no event, sex, lifestyle, etc.)
        labels_df_filtered = labels_df[labels_df['index'] > 20]
    
    # labels.csv 'index' = raw data value
    # get_batch_composite applies +1 shift to all DATA tokens
    # Therefore, shifted token ID = raw data value + 1
    raw_indices = labels_df_filtered['index'].tolist()
    shifted_tokens = [idx + 1 for idx in raw_indices]
    return shifted_tokens


def get_shift_label_indices(apply_token_shift: bool):
    """Return SHIFT label ids for (decrease, maintain, increase)."""
    if apply_token_shift:
        return 2, 3, 4
    return 1, 2, 3


def hierarchical_logits_to_shift_labels(
    change_logits: torch.Tensor,
    direction_logits: torch.Tensor,
    apply_token_shift: bool,
) -> torch.Tensor:
    """
    Convert hierarchical SHIFT predictions to 3-class labels.
    - stage1: maintain(0) vs changed(1)
    - stage2: decrease(0) vs increase(1) for changed
    """
    dec_idx, maintain_idx, inc_idx = get_shift_label_indices(apply_token_shift)
    change_pred = torch.argmax(change_logits, dim=-1)
    direction_pred = torch.argmax(direction_logits, dim=-1)
    shift_pred = torch.full_like(change_pred, fill_value=maintain_idx)
    changed_mask = change_pred == 1
    shift_pred[changed_mask & (direction_pred == 0)] = dec_idx
    shift_pred[changed_mask & (direction_pred == 1)] = inc_idx
    return shift_pred


def optimized_bootstrapped_auc_gpu(case, control, n_bootstrap=1000):
    """
    Computes bootstrapped AUC estimates using PyTorch on CUDA.

    Parameters:
        case: 1D tensor of scores for positive cases
        control: 1D tensor of scores for controls
        n_bootstrap: Number of bootstrap replicates

    Returns:
        Tensor of shape (n_bootstrap,) containing AUC for each bootstrap replicate
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This function requires a GPU.")

    # Convert inputs to CUDA tensors
    if not torch.is_tensor(case):
        case = torch.tensor(case, device="cuda", dtype=torch.float32)
    else:
        case = case.to("cuda", dtype=torch.float32)

    if not torch.is_tensor(control):
        control = torch.tensor(control, device="cuda", dtype=torch.float32)
    else:
        control = control.to("cuda", dtype=torch.float32)

    n_case = case.size(0)
    n_control = control.size(0)
    total = n_case + n_control

    # Generate bootstrap samples
    boot_idx_case = torch.randint(0, n_case, (n_bootstrap, n_case), device="cuda")
    boot_idx_control = torch.randint(0, n_control, (n_bootstrap, n_control), device="cuda")

    boot_case = case[boot_idx_case]
    boot_control = control[boot_idx_control]

    combined = torch.cat([boot_case, boot_control], dim=1)

    # Mask to identify case entries
    mask = torch.zeros((n_bootstrap, total), dtype=torch.bool, device="cuda")
    mask[:, :n_case] = True

    # Compute ranks and AUC
    ranks = combined.argsort(dim=1).argsort(dim=1)
    case_ranks_sum = torch.sum(ranks.float() * mask.float(), dim=1)
    min_case_rank_sum = n_case * (n_case - 1) / 2.0
    U = case_ranks_sum - min_case_rank_sum
    aucs = U / (n_case * n_control)
    return aucs.cpu().tolist()


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float32)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float32)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float32)
    ty = np.empty([k, n], dtype=np.float32)
    tz = np.empty([k, m + n], dtype=np.float32)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    
    # Handle cases with insufficient samples for covariance calculation
    # Suppress warnings for small sample sizes
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        # Calculate covariance with error handling
        if m > 1 and v01.shape[1] > 1:
            sx = np.cov(v01)
            # Handle case where covariance might be singular
            if np.any(np.isnan(sx)) or np.any(np.isinf(sx)):
                sx = np.zeros_like(sx)
        else:
            sx = np.zeros((k, k), dtype=np.float32)
        
        if n > 1 and v10.shape[1] > 1:
            sy = np.cov(v10)
            # Handle case where covariance might be singular
            if np.any(np.isnan(sy)) or np.any(np.isinf(sy)):
                sy = np.zeros_like(sy)
        else:
            sy = np.zeros((k, k), dtype=np.float32)
    
    # Calculate delongcov with protection against division by zero
    if m > 0 and n > 0:
        delongcov = sx / m + sy / n
        # Replace any invalid values with 0
        delongcov = np.nan_to_num(delongcov, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        delongcov = np.zeros((k, k), dtype=np.float32)
    
    return aucs, delongcov


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def get_auc_delong_var(healthy_scores, diseased_scores):
    """
    Computes ROC AUC value and variance using DeLong's method

    Args:
        healthy_scores: Values for class 0 (healthy/controls)
        diseased_scores: Values for class 1 (diseased/cases)
    Returns:
        AUC value and variance
    """
    # Create ground truth labels (1 for diseased, 0 for healthy)
    ground_truth = np.array([1] * len(diseased_scores) + [0] * len(healthy_scores))
    predictions = np.concatenate([diseased_scores, healthy_scores])

    # Compute statistics needed for DeLong method
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]

    # Calculate AUC and covariance
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"

    # Convert delongcov to scalar if it's an array (for single classifier case)
    if isinstance(delongcov, np.ndarray):
        delongcov = delongcov.item() if delongcov.size == 1 else delongcov[0, 0]
    
    return aucs[0], delongcov


def get_calibration_auc(j, k, d, p, diseases_chunk, offset=365.25, age_groups=range(45, 80, 5), precomputed_idx=None, n_bootstrap=1, use_delong=False):
    """
    Compute calibration AUC for a specific disease token.
    
    Args:
        j: index of disease in the chunk
        k: disease token ID (actual token value)
        d: data tuple [input_tokens, input_ages, target_tokens, target_ages]
        p: predictions (logits) from model, shape (B, T, chunk_size) - only for diseases in chunk
        diseases_chunk: array of disease token IDs in this chunk
        offset: time offset in days
        age_groups: age groups to evaluate
        precomputed_idx: precomputed prediction indices
        n_bootstrap: number of bootstrap samples
        use_delong: whether to use DeLong method
    
    Returns:
        list of dictionaries with AUC results (includes N/A results for insufficient data)
    """
    age_step = age_groups[1] - age_groups[0]

    # Indexes of cases with disease k
    # d[2] contains target tokens (disease tokens)
    wk = np.where(d[2] == k)
    n_cases = len(wk[0])

    # For controls, we need to exclude cases with disease k
    # Controls are positions where the target token is not k
    # We also exclude patients who have disease k anywhere in their trajectory
    patient_has_disease = (d[2] == k).any(axis=1)  # (B,) - True if patient has disease k
    wc = np.where((d[2] != k) & (~patient_has_disease[:, None]))  # Controls: not k and patient doesn't have k
    n_controls = len(wc[0])

    # If insufficient data, return N/A results for all age groups
    if n_cases < 2 or n_controls == 0:
        out = []
        reason = "insufficient_cases" if n_cases < 2 else "no_controls"
        for aa in age_groups:
            out_item = {
                "token": k,
                "auc": np.nan,
                "age": aa,
                "n_healthy": n_controls if n_controls > 0 else 0,
                "n_diseased": n_cases,
                "status": reason,
            }
            if use_delong:
                out_item["auc_delong"] = np.nan
                out_item["auc_variance_delong"] = np.nan
            if n_bootstrap > 1:
                out_item["bootstrap_idx"] = 0  # Only one record per age group for N/A
            out.append(out_item)
        return out

    wall = (np.concatenate([wk[0], wc[0]]), np.concatenate([wk[1], wc[1]]))  # All cases and controls

    # We need to take into account the offset t and use the tokens for prediction that are at least t before the event
    if precomputed_idx is None:
        pred_idx = (d[1][wall[0]] <= d[3][wall].reshape(-1, 1) - offset).sum(1) - 1
    else:
        pred_idx = precomputed_idx[wall]  # It's actually much faster to precompute this

    z = d[1][(wall[0], pred_idx)]  # Times of the tokens for prediction
    z = z[pred_idx != -1]

    zk = d[3][wall]  # Target times
    zk = zk[pred_idx != -1]

    # Extract predictions for disease k
    # p shape: (B, T, chunk_size) - logits only for diseases in chunk
    # j is the index within the chunk, so we use p[..., j]
    x = p[..., j][(wall[0], pred_idx)]
    x = x[pred_idx != -1]

    wk = (wk[0][pred_idx[: len(wk[0])] != -1], wk[1][pred_idx[: len(wk[0])] != -1])
    p_idx = wall[0][pred_idx != -1]

    out = []

    for i, aa in enumerate(age_groups):
        a = np.logical_and(z / 365.25 >= aa, z / 365.25 < aa + age_step)
        # Optionally, add extra filtering on the time difference, for example:
        # a *= (zk - z < 365.25)
        selected_groups = p_idx[a]
        perm = np.random.permutation(len(selected_groups))
        _, indices = np.unique(selected_groups[perm], return_index=True)
        indices = perm[indices]
        selected = np.zeros(np.sum(a), dtype=bool)
        selected[indices] = True
        a[a] = selected

        control = x[len(wk[0]) :][a[len(wk[0]) :]]
        case = x[: len(wk[0])][a[: len(wk[0])]]

        if len(control) == 0 or len(case) == 0:
            continue

        if use_delong:
            auc_value_delong, auc_variance_delong = get_auc_delong_var(control, case)
            # Ensure auc_variance_delong is a scalar for parquet compatibility
            if isinstance(auc_variance_delong, np.ndarray):
                auc_variance_delong = auc_variance_delong.item() if auc_variance_delong.size == 1 else float(auc_variance_delong[0, 0])
            else:
                auc_variance_delong = float(auc_variance_delong)
            auc_delong_dict = {"auc_delong": float(auc_value_delong), "auc_variance_delong": auc_variance_delong}
        else:
            auc_delong_dict = {}

        if n_bootstrap > 1:
            aucs_bootstrapped = optimized_bootstrapped_auc_gpu(case, control, n_bootstrap)

        for bootstrap_idx in range(n_bootstrap):
            if n_bootstrap == 1:
                if use_delong:
                    y = auc_value_delong
                else:
                    y = auc(case, control)
            else:
                y = aucs_bootstrapped[bootstrap_idx]
            
            out_item = {
                "token": k,
                "auc": y,
                "age": aa,
                "n_healthy": len(control),
                "n_diseased": len(case),
                "status": "ok",
            }
            if n_bootstrap > 1:
                out_item["bootstrap_idx"] = bootstrap_idx
            out.append(out_item | auc_delong_dict)
    return out


def evaluate_composite_fields(model, d100k, batch_size=64, device="mps"):
    """
    Evaluate SHIFT, TOTAL predictions for CompositeDelphi v6 model.
    
    Args:
        model: CompositeDelphi model
        d100k: Data batch from get_batch_composite
        batch_size: Batch size for inference
        device: Device identifier
    
    Returns:
        dict with evaluation metrics for each field
    """
    model.eval()
    model.to(device)
    all_predictions = {'shift': [], 'total': []}
    all_targets = {'shift': [], 'total': []}
    # For TOTAL regression, also compute "positive-only" metrics (targets > 0), since zeros dominate.
    all_predictions_pos = {'total': []}
    all_targets_pos = {'total': []}

    # Drug-conditioned predictions (if model uses drug-conditioning)
    all_predictions_shift_drug_cond = []
    all_predictions_total_drug_cond = []
    all_targets_shift_drug_cond = []
    all_targets_total_drug_cond = []
    use_drug_conditioning = getattr(model.config, 'use_drug_conditioning', False)
    eval_apply_token_shift = bool(getattr(model.config, 'apply_token_shift', False))
    eval_shift_continuous = bool(getattr(model.config, 'shift_continuous', False))
    eval_shift_exclude_na_token = bool(getattr(model.config, 'shift_exclude_na_token', True))
    eval_separate_shift_na = bool(getattr(model.config, 'separate_shift_na_from_padding', False))
    # Continuous SHIFT regression should not treat value 0 as a special NA token.
    if eval_shift_continuous and eval_separate_shift_na:
        eval_separate_shift_na = False
    eval_shift_na_token = int(getattr(model.config, 'shift_na_raw_token', 4))
    if eval_apply_token_shift:
        eval_shift_na_token += 1
    shift_min_eval = float(getattr(model.config, 'shift_min_value', -1.0))
    shift_max_eval = float(getattr(model.config, 'shift_max_value', -1.0))
    if shift_max_eval <= shift_min_eval:
        if eval_shift_continuous:
            shift_min_eval = 0.0
            shift_max_eval = float(getattr(model.config, 'total_max_value', 550.0))
        elif eval_apply_token_shift:
            shift_min_eval, shift_max_eval = 2.0, 4.0
        else:
            shift_min_eval, shift_max_eval = 1.0, 3.0
    dec_idx, maintain_idx, inc_idx = get_shift_label_indices(eval_apply_token_shift)
    drug_token_min = int(getattr(model.config, 'drug_token_min', 1278))
    drug_token_max = int(getattr(model.config, 'drug_token_max', 1288))
    drug_token_note = f"Metrics computed only for drug tokens ({drug_token_min}-{drug_token_max})"
    
    x_data, x_shift, x_total, x_ages = d100k[0], d100k[1], d100k[2], d100k[3]
    y_data, y_shift, y_total, y_ages = d100k[4], d100k[5], d100k[6], d100k[7]
    
    num_batches = (x_data.shape[0] + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating composite fields"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, x_data.shape[0])
            
            batch_x_data = x_data[start_idx:end_idx].to(device)
            batch_x_shift = x_shift[start_idx:end_idx].to(device)
            batch_x_total = x_total[start_idx:end_idx].to(device)
            batch_x_ages = x_ages[start_idx:end_idx].to(device)
            
            # Get targets on device
            batch_y_data = y_data[start_idx:end_idx].to(device)
            batch_y_shift = y_shift[start_idx:end_idx].to(device)
            batch_y_total = y_total[start_idx:end_idx].to(device)
            
            # Pass targets for drug-conditioning evaluation (uses GT drug embedding)
            outputs = model(
                batch_x_data, batch_x_shift, batch_x_total, batch_x_ages,
                targets_data=batch_y_data, targets_shift=batch_y_shift, 
                targets_total=batch_y_total,
                targets_age=y_ages[start_idx:end_idx].to(device)
            )[0]  # Get logits dict
            
            # v6:
            # - SHIFT: regression output (B, T)
            # - TOTAL: regression output (B, T)
            shift_pred = outputs['shift']
            total_pred = outputs['total']

            # SHIFT log handling:
            # - New checkpoints (shift_log=True): model already outputs raw-space SHIFT.
            # - Legacy checkpoints (shift_log_transform=True, non-MDN): outputs may be log-space.
            shift_log = bool(getattr(model.config, 'shift_log', False))
            shift_log_transform_legacy = bool(getattr(model.config, 'shift_log_transform', False))
            has_mdn_shift = isinstance(outputs.get('shift_mdn', None), dict)
            if shift_log_transform_legacy and (not shift_log) and (not has_mdn_shift):
                shift_pred = torch.expm1(shift_pred)

            # Inverse log-transform if model was trained with total_log_transform
            total_log_transform = getattr(model.config, 'total_log_transform', False)
            has_mdn_total = isinstance(outputs.get('total_mdn', None), dict)
            if total_log_transform and not has_mdn_total:
                total_pred = torch.expm1(total_pred)  # inverse of log1p

            # Defensive compatibility handling for unexpected checkpoint outputs.
            if shift_pred.dim() == 3:
                if shift_pred.size(-1) == 1:
                    shift_pred = shift_pred.squeeze(-1)
                else:
                    shift_pred = torch.argmax(shift_pred, dim=-1).float()
            
            # Use per-field valid masks
            valid_data_mask = (batch_y_data != -1) & (batch_y_data > 0)
            if eval_shift_continuous:
                shift_mask = valid_data_mask & (batch_y_shift != -1) & (batch_y_shift >= 0)
                if eval_separate_shift_na and eval_shift_exclude_na_token:
                    shift_mask = shift_mask & (batch_y_shift != eval_shift_na_token)
            else:
                shift_mask = (
                    (batch_y_shift == dec_idx)
                    | (batch_y_shift == maintain_idx)
                    | (batch_y_shift == inc_idx)
                )
                shift_mask = shift_mask & valid_data_mask
            total_mask = (batch_y_total != -1) & (batch_y_total >= 0)
            
            # Drug token mask: only evaluate drug-conditioned predictions for configured drug range
            drug_token_mask = (batch_y_data >= drug_token_min) & (batch_y_data <= drug_token_max)

            # SHIFT (regression)
            if shift_mask.any():
                all_predictions['shift'].append(shift_pred[shift_mask].float().cpu().numpy())
                all_targets['shift'].append(batch_y_shift[shift_mask].float().cpu().numpy())
                
                # Drug-conditioned SHIFT (regression)
                if use_drug_conditioning and 'shift_drug_cond' in outputs:
                    shift_drug_pred = outputs['shift_drug_cond']
                    has_mdn_drug_shift = isinstance(outputs.get('shift_mdn_drug_cond', None), dict)
                    if shift_log_transform_legacy and (not shift_log) and (not has_mdn_drug_shift):
                        shift_drug_pred = torch.expm1(shift_drug_pred)
                    if shift_drug_pred.dim() == 3:
                        if shift_drug_pred.size(-1) == 1:
                            shift_drug_pred = shift_drug_pred.squeeze(-1)
                        else:
                            shift_drug_pred = torch.argmax(shift_drug_pred, dim=-1).float()
                    # Filter: only drug tokens AND valid shift tokens
                    drug_shift_mask = shift_mask & drug_token_mask
                    if drug_shift_mask.any():
                        all_predictions_shift_drug_cond.append(shift_drug_pred[drug_shift_mask].float().cpu().numpy())
                        all_targets_shift_drug_cond.append(batch_y_shift[drug_shift_mask].float().cpu().numpy())

            # TOTAL (regression): store clipped predictions to match domain (non-negative)
            if total_mask.any():
                tp = total_pred[total_mask]
                tt = batch_y_total[total_mask].float()
                all_predictions['total'].append(torch.clamp(tp, min=0.0).cpu().numpy())
                all_targets['total'].append(tt.cpu().numpy())
                
                # Drug-conditioned TOTAL (if available) - ONLY for drug tokens
                # Create drug_total_mask BEFORE filtering by total_mask
                if use_drug_conditioning and 'total_drug_cond' in outputs:
                    # Filter: only drug tokens AND valid total tokens
                    drug_total_mask = total_mask & drug_token_mask
                    if drug_total_mask.any():
                        # Apply drug_total_mask directly to outputs (before total_mask filtering)
                        tp_drug = outputs['total_drug_cond'][drug_total_mask]
                        has_mdn_drug_total = isinstance(outputs.get('total_mdn_drug_cond', None), dict)
                        if total_log_transform and not has_mdn_drug_total:
                            tp_drug = torch.expm1(tp_drug)
                        tp_drug = torch.clamp(tp_drug, min=0.0)
                        all_predictions_total_drug_cond.append(tp_drug.cpu().numpy())
                        all_targets_total_drug_cond.append(batch_y_total[drug_total_mask].float().cpu().numpy())

                total_pos = total_mask & (batch_y_total > 0)
                if total_pos.any():
                    tp_pos = total_pred[total_pos]
                    tt_pos = batch_y_total[total_pos].float()
                    all_predictions_pos['total'].append(torch.clamp(tp_pos, min=0.0).cpu().numpy())
                    all_targets_pos['total'].append(tt_pos.cpu().numpy())

    # Concatenate all batches (skip empty)
    for field in ['shift', 'total']:
        if len(all_predictions[field]) > 0:
            all_predictions[field] = np.concatenate(all_predictions[field])
            all_targets[field] = np.concatenate(all_targets[field])
        else:
            all_predictions[field] = np.array([])
            all_targets[field] = np.array([])

    for field in ['total']:
        if len(all_predictions_pos[field]) > 0:
            all_predictions_pos[field] = np.concatenate(all_predictions_pos[field])
            all_targets_pos[field] = np.concatenate(all_targets_pos[field])
        else:
            all_predictions_pos[field] = np.array([])
            all_targets_pos[field] = np.array([])
    
    if len(all_predictions_shift_drug_cond) > 0:
        all_predictions_shift_drug_cond = np.concatenate(all_predictions_shift_drug_cond)
        all_targets_shift_drug_cond = np.concatenate(all_targets_shift_drug_cond)
    else:
        all_predictions_shift_drug_cond = np.array([])
        all_targets_shift_drug_cond = np.array([])
    
    if len(all_predictions_total_drug_cond) > 0:
        all_predictions_total_drug_cond = np.concatenate(all_predictions_total_drug_cond)
        all_targets_total_drug_cond = np.concatenate(all_targets_total_drug_cond)
    else:
        all_predictions_total_drug_cond = np.array([])
        all_targets_total_drug_cond = np.array([])
    
    # Calculate metrics
    results = {}
    
    # ============================================================
    # SHIFT: REGRESSION METRICS
    # ============================================================
    if len(all_targets['shift']) > 0:
        shift_pred = all_predictions['shift'].astype(np.float32)
        shift_target = all_targets['shift'].astype(np.float32)

        results['shift_mae'] = mean_absolute_error(shift_target, shift_pred)
        results['shift_rmse'] = float(np.sqrt(mean_squared_error(shift_target, shift_pred)))
        results['shift_median_ae'] = float(np.median(np.abs(shift_target - shift_pred)))
        try:
            results['shift_r2'] = r2_score(shift_target, shift_pred)
        except Exception:
            results['shift_r2'] = np.nan
        results['shift_mean_target'] = float(np.mean(shift_target))
        results['shift_mean_pred'] = float(np.mean(shift_pred))
        results['shift_std_target'] = float(np.std(shift_target))
        results['shift_std_pred'] = float(np.std(shift_pred))
        results['shift_support'] = int(len(shift_target))

        # Optional interpretability metric for near-discrete targets.
        shift_target_round = np.rint(shift_target)
        frac_integer_target = float(np.mean(np.isclose(shift_target, shift_target_round, atol=1e-6)))
        if frac_integer_target > 0.95:
            shift_pred_rounded = np.rint(np.clip(shift_pred, shift_min_eval, shift_max_eval)).astype(np.int64)
            shift_target_int = shift_target.astype(np.int64)
            results['shift_rounded_accuracy'] = accuracy_score(shift_target_int, shift_pred_rounded)

        # Drug-conditioned SHIFT metrics (ONLY for drug tokens)
        if len(all_predictions_shift_drug_cond) > 0:
            shift_pred_drug = all_predictions_shift_drug_cond.astype(np.float32)
            shift_target_drug = all_targets_shift_drug_cond.astype(np.float32)
            results['shift_mae_drug_cond'] = mean_absolute_error(shift_target_drug, shift_pred_drug)
            results['shift_rmse_drug_cond'] = float(np.sqrt(mean_squared_error(shift_target_drug, shift_pred_drug)))
            results['shift_median_ae_drug_cond'] = float(np.median(np.abs(shift_target_drug - shift_pred_drug)))
            try:
                results['shift_r2_drug_cond'] = r2_score(shift_target_drug, shift_pred_drug)
            except Exception:
                results['shift_r2_drug_cond'] = np.nan
            results['shift_mean_target_drug_cond'] = float(np.mean(shift_target_drug))
            results['shift_mean_pred_drug_cond'] = float(np.mean(shift_pred_drug))
            results['shift_support_drug_cond'] = int(len(shift_target_drug))
            shift_drug_target_round = np.rint(shift_target_drug)
            frac_integer_target_drug = float(np.mean(np.isclose(shift_target_drug, shift_drug_target_round, atol=1e-6)))
            if frac_integer_target_drug > 0.95:
                shift_drug_pred_rounded = np.rint(np.clip(shift_pred_drug, shift_min_eval, shift_max_eval)).astype(np.int64)
                shift_drug_target_int = shift_target_drug.astype(np.int64)
                results['shift_rounded_accuracy_drug_cond'] = accuracy_score(
                    shift_drug_target_int,
                    shift_drug_pred_rounded,
                )
            results['shift_drug_cond_note'] = drug_token_note
    
    # ============================================================
    # TOTAL: REGRESSION METRICS
    # ============================================================
    for field in ['total']:
        if len(all_targets[field]) == 0:
            continue
            
        pred = all_predictions[field]  # continuous regression output
        target = all_targets[field]    # continuous target
        
        # Regression metrics
        mae = mean_absolute_error(target, pred)
        rmse = np.sqrt(mean_squared_error(target, pred))
        median_ae = np.median(np.abs(target - pred))
        
        # R² score
        try:
            r2 = r2_score(target, pred)
        except:
            r2 = np.nan
        
        results[f'{field}_mae'] = mae
        results[f'{field}_rmse'] = rmse
        results[f'{field}_median_ae'] = median_ae
        results[f'{field}_r2'] = r2
        
        # Additional stats
        results[f'{field}_mean_target'] = np.mean(target)
        results[f'{field}_mean_pred'] = np.mean(pred)
        results[f'{field}_std_target'] = np.std(target)
        results[f'{field}_std_pred'] = np.std(pred)

        # Positive-only regression metrics (targets > 0)
        if len(all_targets_pos[field]) > 0:
            pred_pos = all_predictions_pos[field]
            target_pos = all_targets_pos[field]
            results[f'{field}_mae_pos'] = mean_absolute_error(target_pos, pred_pos)
            results[f'{field}_rmse_pos'] = float(np.sqrt(mean_squared_error(target_pos, pred_pos)))
            results[f'{field}_median_ae_pos'] = float(np.median(np.abs(target_pos - pred_pos)))
            try:
                results[f'{field}_r2_pos'] = r2_score(target_pos, pred_pos)
            except:
                results[f'{field}_r2_pos'] = np.nan
            results[f'{field}_support_pos'] = int(len(target_pos))

    # Drug-conditioned TOTAL metrics (ONLY for drug tokens)
    if len(all_predictions_total_drug_cond) > 0:
        pred_drug = all_predictions_total_drug_cond
        tgt_drug = all_targets_total_drug_cond
        results['total_mae_drug_cond'] = mean_absolute_error(tgt_drug, pred_drug)
        results['total_rmse_drug_cond'] = float(np.sqrt(mean_squared_error(tgt_drug, pred_drug)))
        results['total_median_ae_drug_cond'] = float(np.median(np.abs(tgt_drug - pred_drug)))
        try:
            results['total_r2_drug_cond'] = r2_score(tgt_drug, pred_drug)
        except:
            results['total_r2_drug_cond'] = np.nan
        results['total_mean_target_drug_cond'] = float(np.mean(tgt_drug))
        results['total_mean_pred_drug_cond'] = float(np.mean(pred_drug))
        results['total_support_drug_cond'] = int(len(tgt_drug))
        results['total_drug_cond_note'] = drug_token_note
    
    return results


# New internal function that performs the AUC evaluation pipeline.
def evaluate_auc_pipeline(
    model,
    d100k,
    output_path,
    labels_df,
    model_type='composite',
    evaluate_composite=True,
    diseases_of_interest=None,
    filter_min_total=100,
    disease_chunk_size=200,
    age_groups=np.arange(40, 80, 5),
    offset=0.1,
    batch_size=64,
    device="cpu",
    seed=1337,
    n_bootstrap=1,
    meta_info={},
    train_valid_tokens=None,  # Set of tokens present in train data (for filtering)
):
    """
    Runs the AUC evaluation pipeline.

    Args:
        model (torch.nn.Module): The loaded model set to eval().
        d100k: Data batch from get_batch_composite (CompositeDelphi).
        labels_df (pd.DataFrame): DataFrame with label info (token names, etc.).
        output_path (str | None): Directory where CSV files will be written. If None, files will not be saved.
        model_type (str): must be 'composite'
        diseases_of_interest (np.ndarray or list, optional): If provided, these disease indices are used.
        filter_min_total (int): Minimum total token count to include a token.
        disease_chunk_size (int): Maximum chunk size for processing diseases.
        age_groups (np.ndarray): Age groups to use in calibration.
        offset (float): Offset used in get_calibration_auc.
        batch_size (int): Batch size for model forwarding.
        device (str): Device identifier.
        seed (int): Random seed for reproducibility.
        n_bootstrap (int): Number of bootstrap samples. (1 for no bootstrap)
        meta_info (dict): Additional metadata to add to output DataFrames.
    Returns:
        tuple: (df_auc_unpooled, df_auc, df_both) DataFrames.
    """

    assert n_bootstrap > 0, "n_bootstrap must be greater than 0"

    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Get model vocab size to filter out invalid indices
    config_vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else model.config.data_vocab_size if hasattr(model.config, 'data_vocab_size') else 1290
    
    # Adjust vocab_size if labels indicate more tokens (e.g., Death token at 1289)
    # Note: labels_df indices are 0-based and represent raw data values.
    # If max index is 1288 (Death raw), after +1 shift max token is 1289.
    # We need vocab_size > 1289 (i.e. >= 1290) to include it.
    # Use model's vocab_size (trust the model config)
    vocab_size = config_vocab_size
    
    # Optional: warn if labels suggest a larger vocab_size
    if 'index' in labels_df.columns:
        max_label_index = labels_df['index'].max()
        # If labels contain indices >= vocab_size, they will be filtered out
        if max_label_index >= vocab_size - 1:
            print(f"Warning: labels contain index {max_label_index}, but model vocab_size is {vocab_size}.")
            print(f"         Tokens with index >= {vocab_size} will be excluded from evaluation.")
    
    # Get common diseases
    if diseases_of_interest is None:
        diseases_of_interest = get_common_diseases(labels_df, filter_min_total)
    
    # Filter out invalid indices (must be < vocab_size)
    # Note: token indices are 0-based, so valid range is [0, vocab_size)
    diseases_of_interest = [d for d in diseases_of_interest if 0 <= d < vocab_size]
    
    if model_type != 'composite':
        raise ValueError("Only composite model_type is supported.")

    # CRITICAL: Filter to only include tokens that actually exist in the evaluation data
    # This prevents evaluating tokens like SGLT-2 or Other that may not exist in val/test data
    target_data_np = d100k[4].cpu().detach().numpy()  # y_data (target DATA tokens)
    # d100k = (x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages)
    
    actual_tokens_in_data = set(np.unique(target_data_np).tolist())
    # Remove invalid tokens like -1 (padding)
    actual_tokens_in_data = {t for t in actual_tokens_in_data if t >= 0}
    
    # Filter diseases to only those present in the evaluation data
    diseases_before_filter = len(diseases_of_interest)
    diseases_of_interest = [d for d in diseases_of_interest if d in actual_tokens_in_data]
    diseases_filtered_eval = diseases_before_filter - len(diseases_of_interest)
    
    if diseases_filtered_eval > 0:
        print(f"Filtered out {diseases_filtered_eval} diseases not present in evaluation data")
    
    # CRITICAL: Filter to only include tokens present in train data
    # This ensures we only evaluate tokens the model was trained on
    # (e.g., excludes SGLT-2, Other if not in train)
    if train_valid_tokens is not None:
        diseases_before_train_filter = len(diseases_of_interest)
        diseases_of_interest = [d for d in diseases_of_interest if d in train_valid_tokens]
        diseases_filtered_train = diseases_before_train_filter - len(diseases_of_interest)
        
        if diseases_filtered_train > 0:
            print(f"Filtered out {diseases_filtered_train} diseases not present in train data")
    
    if len(diseases_of_interest) == 0:
        raise ValueError(f"No valid diseases found. All indices must be in range [0, {vocab_size}), present in evaluation data, and present in train data")
    
    print(f"Evaluating {len(diseases_of_interest)} diseases (vocab_size={vocab_size}, actual unique tokens in eval data={len(actual_tokens_in_data)})")

    # Split diseases into chunks for processing
    num_chunks = (len(diseases_of_interest) + disease_chunk_size - 1) // disease_chunk_size
    diseases_chunks = np.array_split(diseases_of_interest, num_chunks)

    # Precompute prediction indices for calibration
    data_tokens = d100k[0].cpu().detach().numpy()  # x_data
    ages = d100k[3].cpu().detach().numpy()  # x_ages
    target_data = d100k[4].cpu().detach().numpy()  # y_data
    target_ages = d100k[7].cpu().detach().numpy()  # y_ages
    d = [data_tokens, ages, target_data, target_ages]
    
    # Precompute prediction indices: find positions where input age <= target age - offset
    pred_idx_precompute = (d[1][:, :, np.newaxis] <= d[3][:, np.newaxis, :] - offset).sum(1) - 1

    all_aucs = []
    tqdm_options = {"desc": "Processing disease chunks", "total": len(diseases_chunks)}
    for disease_chunk_idx, diseases_chunk in tqdm(enumerate(diseases_chunks), **tqdm_options):
        # Filter out invalid indices for this chunk
        diseases_chunk = np.array(diseases_chunk)
        valid_mask = (diseases_chunk >= 0) & (diseases_chunk < vocab_size)
        diseases_chunk = diseases_chunk[valid_mask].tolist()
        
        if len(diseases_chunk) == 0:
            print(f"Skipping chunk {disease_chunk_idx}: no valid diseases")
            continue
        
        p100k = []
        model.to(device)
        model.eval()
        with torch.no_grad():
            # Process the evaluation data in batches
            x_data, x_shift, x_total, x_ages = d100k[0], d100k[1], d100k[2], d100k[3]
            num_batches = (x_data.shape[0] + batch_size - 1) // batch_size
            for batch_idx in tqdm(range(num_batches), desc=f"Model inference, chunk {disease_chunk_idx}"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, x_data.shape[0])

                batch_x_data = x_data[start_idx:end_idx].to(device)
                batch_x_shift = x_shift[start_idx:end_idx].to(device)
                batch_x_total = x_total[start_idx:end_idx].to(device)
                batch_x_ages = x_ages[start_idx:end_idx].to(device)

                outputs = model(
                    batch_x_data, batch_x_shift, batch_x_total, batch_x_ages
                )[0]  # Get logits dict

                data_logits = outputs['data'].cpu().detach().numpy()
                p100k.append(data_logits[:, :, diseases_chunk].astype("float16"))
        
        if len(p100k) == 0:
            print(f"Skipping chunk {disease_chunk_idx}: no predictions generated")
            continue
        
        p100k = np.vstack(p100k)

        # Loop over each disease (token) in the current chunk, sexes separately
        # Note: For now, we process all data together. Sex filtering can be added if needed.
        for j, k in tqdm(
            list(enumerate(diseases_chunk)), desc=f"Processing diseases in chunk {disease_chunk_idx}"
        ):
            # Get calibration AUC for the current disease token.
            out = get_calibration_auc(
                j,
                k,
                d,
                p100k,
                diseases_chunk,
                age_groups=age_groups,
                offset=offset,
                precomputed_idx=pred_idx_precompute,
                n_bootstrap=n_bootstrap,
                use_delong=True,
            )
            if out is None:
                continue
            for out_item in out:
                all_aucs.append(out_item)

    df_auc_unpooled = pd.DataFrame(all_aucs)

    for key, value in meta_info.items():
        df_auc_unpooled[key] = value

    # Merge with labels if available
    if 'index' in labels_df.columns:
        labels_df_subset = labels_df[['index']].copy()
        if 'name' in labels_df.columns:
            labels_df_subset['name'] = labels_df['name']
        # labels.csv 'index' = raw data value, but df_auc_unpooled 'token' = shifted token ID
        # Therefore, create shifted_token column for merge
        labels_df_subset['shifted_token'] = labels_df_subset['index'] + 1
        df_auc_unpooled_merged = df_auc_unpooled.merge(
            labels_df_subset, left_on="token", right_on="shifted_token", how="inner"
        )
    else:
        df_auc_unpooled_merged = df_auc_unpooled.copy()

    def aggregate_age_brackets_delong(group):
        # For normal distributions, when averaging n of them:
        # The variance of the sum is the sum of variances
        # The variance of the average is the sum of variances divided by n^2
        n = len(group)
        
        # Handle cases where all AUC values are NaN (insufficient data)
        valid_aucs = group['auc_delong'].dropna()
        if len(valid_aucs) == 0:
            mean = np.nan
            var = np.nan
            status = group['status'].iloc[0] if 'status' in group.columns else 'unknown'
        else:
            mean = valid_aucs.mean()
            # Since we're taking the average, divide combined variance by n^2
            valid_vars = group.loc[valid_aucs.index, 'auc_variance_delong']
            var = valid_vars.sum() / (len(valid_vars)**2) if len(valid_vars) > 0 else np.nan
            status = 'ok'
        
        # Ensure var is a scalar (not array) for parquet compatibility
        if isinstance(var, np.ndarray):
            var = var.item() if var.size == 1 else float(var[0, 0]) if var.ndim > 0 else float(var)
        elif not np.isnan(var):
            var = float(var)
        
        return pd.Series({
            'auc': mean,
            'auc_variance_delong': var,
            'n_samples': n, 
            'n_diseased': group['n_diseased'].sum(),
            'n_healthy': group['n_healthy'].sum(),
            'status': status,
        })

    print('Using DeLong method to calculate AUC confidence intervals..')
    
    # Use include_groups=False to suppress FutureWarning in pandas
    df_auc = df_auc_unpooled.groupby(["token"]).apply(aggregate_age_brackets_delong, include_groups=False).reset_index()
    
    if 'index' in labels_df.columns:
        # labels.csv 'index' = raw data value, but df_auc 'token' = shifted token ID
        # Create temporary shifted_token column for merge
        labels_df_for_merge = labels_df.copy()
        labels_df_for_merge['shifted_token'] = labels_df_for_merge['index'] + 1
        df_auc_merged = df_auc.merge(labels_df_for_merge, left_on="token", right_on="shifted_token", how="inner")
    else:
        df_auc_merged = df_auc.copy()
    
    # Evaluate composite fields (SHIFT, TOTAL) if composite model and enabled
    composite_metrics = None
    if evaluate_composite:
        print("\nEvaluating composite fields (SHIFT, TOTAL)...")
        composite_metrics = evaluate_composite_fields(
            model, d100k, batch_size=batch_size, device=device
        )
        
        # Print results
        print("\nComposite Field Evaluation Results:")
        print("=" * 60)
        
        # SHIFT: regression
        if 'shift_mae' in composite_metrics:
            print("SHIFT:")
            print(f"  MAE: {composite_metrics['shift_mae']:.4f}")
            if 'shift_rmse' in composite_metrics:
                print(f"  RMSE: {composite_metrics['shift_rmse']:.4f}")
            if 'shift_median_ae' in composite_metrics:
                print(f"  Median AE: {composite_metrics['shift_median_ae']:.4f}")
            if 'shift_r2' in composite_metrics and not np.isnan(composite_metrics['shift_r2']):
                print(f"  R²: {composite_metrics['shift_r2']:.4f}")
            if 'shift_rounded_accuracy' in composite_metrics:
                print(f"  Rounded Accuracy: {composite_metrics['shift_rounded_accuracy']:.4f}")
            if 'shift_support' in composite_metrics:
                print(f"  Support: {composite_metrics['shift_support']}")

            # Drug-conditioned SHIFT regression metrics (ONLY for configured drug token range)
            if 'shift_mae_drug_cond' in composite_metrics:
                shift_drug_note = composite_metrics.get('shift_drug_cond_note', 'Metrics computed only for configured drug token range')
                print(f"\n  Drug-Conditioned ({shift_drug_note}):")
                print(f"    MAE: {composite_metrics['shift_mae_drug_cond']:.4f}")
                if 'shift_rmse_drug_cond' in composite_metrics:
                    print(f"    RMSE: {composite_metrics['shift_rmse_drug_cond']:.4f}")
                if 'shift_median_ae_drug_cond' in composite_metrics:
                    print(f"    Median AE: {composite_metrics['shift_median_ae_drug_cond']:.4f}")
                if 'shift_r2_drug_cond' in composite_metrics and not np.isnan(composite_metrics['shift_r2_drug_cond']):
                    print(f"    R²: {composite_metrics['shift_r2_drug_cond']:.4f}")
                if 'shift_rounded_accuracy_drug_cond' in composite_metrics:
                    print(f"    Rounded Accuracy: {composite_metrics['shift_rounded_accuracy_drug_cond']:.4f}")
                if 'shift_mean_target_drug_cond' in composite_metrics:
                    print(f"    Mean Target: {composite_metrics['shift_mean_target_drug_cond']:.4f}")
                if 'shift_mean_pred_drug_cond' in composite_metrics:
                    print(f"    Mean Prediction: {composite_metrics['shift_mean_pred_drug_cond']:.4f}")
                if 'shift_support_drug_cond' in composite_metrics:
                    print(f"    Support: {composite_metrics['shift_support_drug_cond']}")
        else:
            print("SHIFT:")
            print("  n/a (no valid SHIFT targets under current filtering/mask)")
        
        # TOTAL: regression
        field = 'total'
        if f'{field}_mae' in composite_metrics:
            print(f"{field.upper()}:")
            print(f"  MAE: {composite_metrics[f'{field}_mae']:.4f}")
            if f'{field}_rmse' in composite_metrics:
                print(f"  RMSE: {composite_metrics[f'{field}_rmse']:.4f}")
            if f'{field}_median_ae' in composite_metrics:
                print(f"  Median AE: {composite_metrics[f'{field}_median_ae']:.4f}")
            if f'{field}_r2' in composite_metrics and not np.isnan(composite_metrics[f'{field}_r2']):
                print(f"  R²: {composite_metrics[f'{field}_r2']:.4f}")
            if f'{field}_mae_pos' in composite_metrics:
                print(f"  MAE (target>0): {composite_metrics[f'{field}_mae_pos']:.4f} (n={composite_metrics.get(f'{field}_support_pos', 'NA')})")
            # TOTAL drug-conditioned extras (ONLY for configured drug token range)
            if 'total_mae_drug_cond' in composite_metrics:
                total_drug_note = composite_metrics.get('total_drug_cond_note', 'Metrics computed only for configured drug token range')
                print(f"\n  Drug-Conditioned ({total_drug_note}):")
                print(f"    MAE: {composite_metrics['total_mae_drug_cond']:.4f}")
                if 'total_rmse_drug_cond' in composite_metrics:
                    print(f"    RMSE: {composite_metrics['total_rmse_drug_cond']:.4f}")
                if 'total_median_ae_drug_cond' in composite_metrics:
                    print(f"    Median AE: {composite_metrics['total_median_ae_drug_cond']:.4f}")
                if 'total_r2_drug_cond' in composite_metrics and not np.isnan(composite_metrics['total_r2_drug_cond']):
                    print(f"    R²: {composite_metrics['total_r2_drug_cond']:.4f}")
                if 'total_mean_target_drug_cond' in composite_metrics:
                    print(f"    Mean Target: {composite_metrics['total_mean_target_drug_cond']:.4f}")
                if 'total_mean_pred_drug_cond' in composite_metrics:
                    print(f"    Mean Prediction: {composite_metrics['total_mean_pred_drug_cond']:.4f}")
                if 'total_support_drug_cond' in composite_metrics:
                    print(f"    Support: {composite_metrics['total_support_drug_cond']}")
        print("=" * 60)
        
        # Save composite metrics
        if output_path is not None:
            import json
            with open(f"{output_path}/composite_metrics.json", 'w') as f:
                # Convert numpy types to native Python types for JSON
                json_metrics = {}
                for k, v in composite_metrics.items():
                    if isinstance(v, dict):
                        json_metrics[k] = {str(k2): float(v2) if isinstance(v2, (np.integer, np.floating, int, float)) else v2 
                                          for k2, v2 in v.items()}
                    elif isinstance(v, (np.integer, np.floating, np.ndarray)):
                        if isinstance(v, np.ndarray):
                            json_metrics[k] = v.tolist()
                        else:
                            json_metrics[k] = float(v)
                    elif isinstance(v, (int, float, bool, str)):
                        json_metrics[k] = v
                    else:
                        # Try to convert to float, if fails, convert to string
                        try:
                            json_metrics[k] = float(v)
                        except (ValueError, TypeError):
                            json_metrics[k] = str(v)
                json.dump(json_metrics, f, indent=2)
            print(f"Composite metrics saved to {output_path}/composite_metrics.json")
    
    if output_path is not None:
        Path(output_path).mkdir(exist_ok=True, parents=True)
        df_auc_merged.to_parquet(f"{output_path}/df_both.parquet", index=False)
        df_auc_unpooled_merged.to_parquet(f"{output_path}/df_auc_unpooled.parquet", index=False)

    return df_auc_unpooled_merged, df_auc_merged, composite_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate AUC")
    parser.add_argument("--input_path", type=str, default="../data", help="Path to the dataset")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to the output. If omitted or set to 'auto', use the checkpoint directory.",
    )
    parser.add_argument("--model_ckpt_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--model_type", type=str, default='composite', choices=['composite'],
                        help="Model type (composite only)")
    parser.add_argument("--no_event_token_rate", type=int, default=5, help="No event token rate")
    parser.add_argument(
        "--health_token_replacement_prob", default=0.0, type=float, help="Health token replacement probability"
    )
    parser.add_argument("--dataset_subset_size", type=int, default=10000, help="Dataset subset size for evaluation (-1 for all)")
    parser.add_argument("--n_bootstrap", type=int, default=1, help="Number of bootstrap samples")
    # Optional filtering/chunking parameters:
    parser.add_argument("--filter_min_total", type=int, default=0, help="Minimum total count to filter tokens (0=include all)")
    parser.add_argument("--disease_chunk_size", type=int, default=200, help="Chunk size for processing diseases")
    parser.add_argument("--labels_path", type=str, default=None, help="Path to labels CSV file")
    parser.add_argument("--block_size", type=int, default=80, help="Block size for data loading")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size for model inference during evaluation")
    parser.add_argument("--data_files", type=str, default=None, 
                        help="Comma-separated list of data files to evaluate (e.g., 'kr_val.bin,kr_test.bin'). If None, evaluates all: kr_val.bin, kr_test.bin, JMDC_extval.bin, UKB_extval.bin")
    parser.add_argument("--train_data_file", type=str, default="dose/kr_train.bin",
                        help="Train data file to filter valid tokens. Only tokens present in train data will be evaluated.")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_type = args.model_type
    no_event_token_rate = args.no_event_token_rate
    dataset_subset_size = args.dataset_subset_size
    ckpt_path = args.model_ckpt_path

    # Resolve output path: default to checkpoint directory.
    if output_path is None or str(output_path).strip() == "" or str(output_path).lower() == "auto":
        output_path = str(Path(ckpt_path).resolve().parent)

    # Create output folder if it doesn't exist.
    if output_path is not None:
        Path(output_path).mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
    print(device)
    seed = 1337

    # Load model checkpoint and initialize model.
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = dict(checkpoint["model_args"])
    eval_apply_token_shift = bool(model_args.get('apply_token_shift', False))
    eval_separate_shift_na = bool(model_args.get('separate_shift_na_from_padding', False))
    eval_shift_na_raw_token = int(model_args.get('shift_na_raw_token', 4))
    eval_shift_continuous = bool(model_args.get('shift_continuous', False))
    eval_shift_log = bool(model_args.get('shift_log', False))
    eval_shift_log_transform_legacy = bool(model_args.get('shift_log_transform', False))
    eval_shift_min = float(model_args.get('shift_min_value', -1.0))
    eval_shift_max = float(model_args.get('shift_max_value', -1.0))
    eval_shift_input_scale = float(model_args.get('shift_input_scale', 1.0))
    print(f"apply_token_shift from checkpoint: {eval_apply_token_shift}")
    print(f"separate_shift_na_from_padding from checkpoint: {eval_separate_shift_na}")
    print(f"shift_continuous from checkpoint: {eval_shift_continuous}")
    print(f"shift_log from checkpoint: {eval_shift_log}")
    print(f"shift_log_transform (legacy) from checkpoint: {eval_shift_log_transform_legacy}")
    print(f"shift_range from checkpoint: [{eval_shift_min}, {eval_shift_max}]")
    print(f"shift_input_scale from checkpoint: {eval_shift_input_scale}")
    if eval_shift_continuous and eval_separate_shift_na:
        print("[fix] shift_continuous=True -> forcing separate_shift_na_from_padding=False for evaluation")
        eval_separate_shift_na = False
    if 'drug_token_min' not in model_args or 'drug_token_max' not in model_args:
        model_args['drug_token_min'] = 1279 if eval_apply_token_shift else 1278
        model_args['drug_token_max'] = 1289 if eval_apply_token_shift else 1288
        print(
            f"Checkpoint missing drug token range; using fallback "
            f"[{model_args['drug_token_min']}, {model_args['drug_token_max']}] "
            f"(apply_token_shift={eval_apply_token_shift})."
        )
    
    # Extract MoE and other architecture info for metadata
    use_moe = model_args.get('use_moe', False)
    num_experts = model_args.get('num_experts', 0)
    experts_per_token = model_args.get('experts_per_token', 0)
    
    conf = CompositeDelphiConfig(**model_args)
    model = CompositeDelphi(conf)
    
    state_dict = checkpoint["model"]
    # Strip DDP 'module.' or torch.compile '_orig_mod.' prefixes if present
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '').replace('_orig_mod.', '')
        cleaned[k] = v
    model.load_state_dict(cleaned)
    model.eval()
    model = model.to(device)
    
    # Print model architecture info
    print(f"\n{'='*60}")
    print(f"Model Architecture Info:")
    print(f"  Model type: {model_type}")
    print(f"  Use MoE: {use_moe}")
    if use_moe:
        print(f"  Number of experts: {num_experts}")
        print(f"  Experts per token: {experts_per_token}")
    print(f"  Parameters: {model.get_num_params()/1e6:.2f}M")
    print(f"{'='*60}\n")

    # Load labels (external) to be passed in.
    # IMPORTANT: Use header=None to avoid treating first line as header!
    # labels.csv format: each line is "name," or "name" (may have trailing comma)
    # Line number = index (0-based), so line 0 = index 0, line 1288 = index 1288 (Death)
    if args.labels_path:
        labels_df = pd.read_csv(args.labels_path, header=None, usecols=[0], names=['name'])
        labels_df['index'] = range(len(labels_df))
    else:
        # Try to load from default location
        labels_path = f"{input_path}/labels.csv"
        if Path(labels_path).exists():
            labels_df = pd.read_csv(labels_path, header=None, usecols=[0], names=['name'])
            labels_df['index'] = range(len(labels_df))
        else:
            # Create a minimal labels DataFrame
            print(f"Warning: labels file not found at {labels_path}. Creating minimal labels DataFrame.")
            labels_df = pd.DataFrame({'index': range(2000), 'name': [f'token_{i}' for i in range(2000)]})

    # Define data files to evaluate with their prefixes
    # Format: (filename, prefix)
    if args.data_files:
        # Parse user-specified files
        data_files_list = []
        for f in args.data_files.split(','):
            f = f.strip()
            if f:
                # Generate prefix from filename
                f_lower = f.lower()
                if 'ukb' in f_lower and 'extval' in f_lower:
                    prefix = 'extval_ukb'
                elif 'jmdc' in f_lower and 'extval' in f_lower:
                    prefix = 'extval_jmdc'
                elif 'extval' in f_lower:
                    prefix = 'extval'
                elif 'val' in f_lower:
                    prefix = 'val'
                elif 'test' in f_lower:
                    prefix = 'test'
                else:
                    prefix = Path(f).stem
                data_files_list.append((f, prefix))
    else:
        # Default: evaluate all files (internal val/test + external validations)
        data_files_list = [
            ("dose/kr_val.bin", "val"),
            ("dose/kr_test.bin", "test"),
            ("dose/JMDC_extval.bin", "extval_jmdc"),
            ("UKB_extval.bin", "extval_ukb"),
        ]
    
    # Prepare meta info for results (base)
    base_meta_info = {
        'model_type': model_type,
        'use_moe': use_moe,
    }
    if use_moe:
        base_meta_info['num_experts'] = num_experts
        base_meta_info['experts_per_token'] = experts_per_token
    
    # Add checkpoint info if available
    if 'iter_num' in checkpoint:
        base_meta_info['checkpoint_iter'] = checkpoint['iter_num']
    if 'best_val_loss' in checkpoint:
        base_meta_info['checkpoint_val_loss'] = checkpoint['best_val_loss']

    # Define dtype for composite data
    # IMPORTANT: Must match train_model.py exactly!
    # Format: (ID, AGE, DATA, SHIFT, TOTAL) - NO DOSE, NO UNIT
    composite_dtype = np.dtype([
        ('ID', np.uint32),
        ('AGE', np.uint32),
        ('DATA', np.uint32),
        ('SHIFT', np.float32),
        ('TOTAL', np.uint32)
    ])
    
    # ============================================================
    # DIAGNOSTIC: Check SHIFT values in raw data (before +1 shift)
    # Only run for first data file to avoid spam
    # ============================================================
    diagnostic_run = False

    # Load train data to get valid tokens (only tokens in train should be evaluated)
    # NOTE: Train filtering only applies to data files with same prefix (e.g., kr_train → kr_val, kr_test)
    #       External validation (e.g., JMDC_extval) should NOT be filtered by kr_train
    train_data_path = f"{input_path}/{args.train_data_file}"
    train_valid_tokens = None
    train_prefix = args.train_data_file.split('_')[0] if '_' in args.train_data_file else None
    
    if Path(train_data_path).exists():
        print(f"\nLoading train data to filter valid tokens: {train_data_path}")
        print(f"  Train prefix: '{train_prefix}' (filtering will only apply to data files with same prefix)")
        train_data_raw = np.fromfile(train_data_path, dtype=composite_dtype)
        train_raw_tokens = np.unique(train_data_raw['DATA'])
        if eval_apply_token_shift:
            train_valid_tokens = set((train_raw_tokens + 1).tolist())
            token_shift_note = "after +1 shift"
        else:
            train_valid_tokens = set(train_raw_tokens.tolist())
            token_shift_note = "raw token space (no shift)"

        print(f"  Train data contains {len(train_valid_tokens)} unique tokens ({token_shift_note})")
        
        # Show which drug tokens are in train (token ids depend on apply_token_shift)
        base_drug_token = 1279 if eval_apply_token_shift else 1278
        drug_token_names = {
            base_drug_token + 0: 'Metformin',
            base_drug_token + 1: 'Sulfonylurea',
            base_drug_token + 2: 'DPP-4',
            base_drug_token + 3: 'Insulin',
            base_drug_token + 4: 'Meglitinide',
            base_drug_token + 5: 'Thiazolidinedione',
            base_drug_token + 6: 'Alpha-glucosidase',
            base_drug_token + 7: 'GLP-1',
            base_drug_token + 8: 'SGLT-2',
            base_drug_token + 9: 'Other',
            base_drug_token + 10: 'Death',
        }
        print("  Drug tokens in train data:")
        for token, name in drug_token_names.items():
            status = "✓" if token in train_valid_tokens else "✗"
            print(f"    {status} {name} ({token})")
    else:
        print(f"\n[WARNING] Train data not found at {train_data_path}. Skipping train-based token filtering.")
        train_prefix = None

    # Process each data file
    all_results = {}
    for data_filename, prefix in data_files_list:
        data_filepath = f"{input_path}/{data_filename}"
        
        # Check if file exists
        if not Path(data_filepath).exists():
            print(f"\n{'='*60}")
            print(f"[WARNING] Skipping {data_filename}: file not found at {data_filepath}")
            print(f"{'='*60}\n")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {data_filename} (prefix: {prefix})")
        print(f"{'='*60}\n")
        
        # Load data
        data = np.fromfile(data_filepath, dtype=composite_dtype)
        data_p2i = get_p2i_composite(data)
        
        # Determine subset size
        current_subset_size = dataset_subset_size
        if current_subset_size == -1:
            current_subset_size = len(data_p2i)
        else:
            current_subset_size = min(current_subset_size, len(data_p2i))
        
        print(f"Using {current_subset_size} patients for evaluation (out of {len(data_p2i)} total)")
        
        # Sample random patients for evaluation
        np.random.seed(seed)
        patient_indices = np.random.choice(len(data_p2i), size=current_subset_size, replace=False)
        patient_indices = sorted(patient_indices)
        
        # Get a subset batch for evaluation
        d100k = get_batch_composite(
            patient_indices,
            data,
            data_p2i,
            select="left",
            block_size=args.block_size,
            device=device,
            padding="random",
            no_event_token_rate=no_event_token_rate,
            apply_token_shift=eval_apply_token_shift,
            shift_continuous=eval_shift_continuous,
            separate_shift_na_from_padding=(eval_separate_shift_na and not eval_shift_continuous),
            shift_na_raw_token=eval_shift_na_raw_token,
        )
        
        # Prepare meta info with data source
        meta_info = base_meta_info.copy()
        meta_info['data_source'] = data_filename
        meta_info['data_prefix'] = prefix
        
        # Call the internal evaluation function (don't save files yet - we'll save with prefix)
        result = evaluate_auc_pipeline(
            model,
            d100k,
            output_path=None,  # Don't save internally, we'll save with prefix
            labels_df=labels_df,
            model_type=model_type,
            # UKB external validation: only AUC is needed (skip SHIFT/TOTAL)
            evaluate_composite=(prefix != 'extval_ukb'),
            diseases_of_interest=None,
            filter_min_total=args.filter_min_total,
            disease_chunk_size=args.disease_chunk_size,
            batch_size=args.eval_batch_size,
            device=device,
            seed=seed,
            n_bootstrap=args.n_bootstrap,
            meta_info=meta_info,
            train_valid_tokens=train_valid_tokens,
        )
        
        df_auc_unpooled, df_auc_merged, composite_metrics = result

        if composite_metrics is None:
            composite_metrics = {}

        if df_auc_merged is not None and not df_auc_merged.empty and 'auc' in df_auc_merged.columns:
            auc_values = df_auc_merged['auc'].dropna()
            if not auc_values.empty:
                composite_metrics['auc_mean'] = float(auc_values.mean())
                composite_metrics['auc_median'] = float(auc_values.median())
                composite_metrics['auc_min'] = float(auc_values.min())
                composite_metrics['auc_max'] = float(auc_values.max())
                composite_metrics['auc_std'] = float(auc_values.std())
                composite_metrics['n_diseases_auc'] = int(len(auc_values))

                print(f"\n[AUC Statistics] (Next Disease Prediction)")
                print(f"  Mean:   {composite_metrics['auc_mean']:.4f}")
                print(f"  Median: {composite_metrics['auc_median']:.4f}")
                print(f"  Min/Max: {composite_metrics['auc_min']:.4f} / {composite_metrics['auc_max']:.4f}")
        
        # Save results with prefix
        if output_path is not None:
            # Save parquet files with prefix (check for None/empty DataFrames)
            if df_auc_merged is not None and not df_auc_merged.empty:
                df_auc_merged.to_parquet(f"{output_path}/{prefix}_df_both.parquet", index=False)
                df_auc_merged.to_csv(f"{output_path}/{prefix}_df_both.csv", index=False)
            
            if df_auc_unpooled is not None and not df_auc_unpooled.empty:
                df_auc_unpooled.to_parquet(f"{output_path}/{prefix}_df_auc_unpooled.parquet", index=False)
                df_auc_unpooled.to_csv(f"{output_path}/{prefix}_df_auc_unpooled.csv", index=False)
            
            # Save composite metrics with prefix
            if composite_metrics:
                import json
                with open(f"{output_path}/{prefix}_composite_metrics.json", 'w') as f:
                    json_metrics = {}
                    for k, v in composite_metrics.items():
                        if isinstance(v, dict):
                            json_metrics[k] = {str(k2): float(v2) if isinstance(v2, (np.integer, np.floating, int, float)) else v2 
                                              for k2, v2 in v.items()}
                        elif isinstance(v, (np.integer, np.floating, np.ndarray)):
                            if isinstance(v, np.ndarray):
                                json_metrics[k] = v.tolist()
                            else:
                                json_metrics[k] = float(v)
                        elif isinstance(v, (int, float, bool, str)):
                            json_metrics[k] = v
                        else:
                            try:
                                json_metrics[k] = float(v)
                            except (ValueError, TypeError):
                                json_metrics[k] = str(v)
                    json.dump(json_metrics, f, indent=2)
        
        # Store results
        all_results[prefix] = {
            'df_auc_unpooled': df_auc_unpooled,
            'df_auc_merged': df_auc_merged,
            'composite_metrics': composite_metrics,
            'data_filename': data_filename,
        }
        
        print(f"\n[{prefix.upper()}] Evaluation completed!")
        print(f"  Total diseases evaluated: {len(df_auc_merged)}")
        if composite_metrics:
            print(f"  Composite field metrics saved to {output_path}/{prefix}_composite_metrics.json")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ALL EVALUATIONS COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}")
    for prefix, result_data in all_results.items():
        print(f"\n[{prefix.upper()}] {result_data['data_filename']}:")
        print(f"  - {prefix}_df_both.parquet / .csv")
        print(f"  - {prefix}_df_auc_unpooled.parquet / .csv")
        if result_data['composite_metrics']:
            print(f"  - {prefix}_composite_metrics.json")
        print(f"  - Diseases evaluated: {len(result_data['df_auc_merged'])}")


if __name__ == "__main__":
    main()
