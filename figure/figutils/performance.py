"""
performance.py - Shift / Total prediction performance analysis
===============================================================
Contains CompositeModelAnalyzer: collects predictions from the model
and produces separate figures for shift classification and total regression.

Moved from composite_model_analysis.py into figutils/ so notebooks can
import via ``from figutils.performance import CompositeModelAnalyzer``.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error, mean_absolute_error, r2_score,
)
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class CompositeModelAnalyzer:
    """
    Analyzer for Composite Delphi model predictions.
    Handles both shift (classification) and total (regression) tasks.
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()

        self.predictions = {'shift': [], 'total': [], 'disease': [],
                            'shift_change': [], 'shift_direction': []}
        self.targets     = {'shift': [], 'total': [], 'disease': []}
        self.attention_weights = []

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def collect_predictions(self, dataloader, max_batches=None):
        """Run the model on *dataloader* and store predictions + targets."""
        print("Collecting predictions...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break

                if len(batch) != 8:
                    print(f"Warning: unexpected batch length {len(batch)}, skipping")
                    continue

                x_data, x_shift, x_total, x_ages = [b.to(self.device) for b in batch[:4]]
                y_data, y_shift, y_total, y_ages = [b.to(self.device) for b in batch[4:]]

                logits, loss, att = self.model(
                    x_data, x_shift, x_total, x_ages,
                    targets_data=y_data,
                    targets_shift=y_shift,
                    targets_total=y_total,
                    targets_age=y_ages,
                )

                # Use drug-conditioned heads when available
                shift_logits = logits.get('shift_drug_cond', logits['shift'])
                total_pred   = logits.get('total_drug_cond', logits['total'])

                self.predictions['disease'].append(logits['data'].cpu())
                self.predictions['shift'].append(shift_logits.cpu())
                self.predictions['total'].append(total_pred.cpu())

                # Store hierarchical shift logits if available (v2: HierarchicalShiftHead)
                if 'shift_change_logits' in logits:
                    change_src = logits.get('shift_change_logits_drug_cond',
                                            logits['shift_change_logits'])
                    self.predictions['shift_change'].append(change_src.cpu())
                if 'shift_direction_logits' in logits:
                    dir_src = logits.get('shift_direction_logits_drug_cond',
                                         logits['shift_direction_logits'])
                    self.predictions['shift_direction'].append(dir_src.cpu())

                self.targets['disease'].append(y_data.cpu())
                self.targets['shift'].append(y_shift.cpu())
                self.targets['total'].append(y_total.cpu())

                if att is not None:
                    self.attention_weights.append(att.cpu())

                if (batch_idx + 1) % 10 == 0:
                    print(f"  {batch_idx + 1} batches processed")

        self._concatenate()
        print("Prediction collection complete!")

    # ------------------------------------------------------------------
    def _concatenate(self):
        """Pad variable-length batches and concatenate."""
        for key in ('disease', 'shift', 'total', 'shift_change', 'shift_direction'):
            preds = self.predictions.get(key, [])
            if not preds:
                continue

            max_len = max(p.shape[1] for p in preds)

            def _pad_list(tensors, max_len):
                out = []
                for t in tensors:
                    if t.shape[1] < max_len:
                        pad_shape = list(t.shape)
                        pad_shape[1] = max_len - t.shape[1]
                        t = torch.cat([t, torch.zeros(*pad_shape, dtype=t.dtype)], dim=1)
                    out.append(t)
                return torch.cat(out, dim=0)

            self.predictions[key] = _pad_list(preds, max_len)
            tgts = self.targets.get(key, [])
            if tgts:
                self.targets[key] = _pad_list(tgts, max_len)

    # ==================================================================
    # Shift classification figure (standalone)
    # ==================================================================
    def visualize_shift_performance(self, save_path=None):
        """
        Auto-detects model type and renders appropriate shift analysis:
        - model_v7 (MDN continuous): shift predictions (B, T) float → regression plot
        - model_v2 (BinaryChangeHead): shift logits (B, T, 2) → Maintain vs Changed
        - model_v1/v3: shift logits (B, T, 5) → Decrease/Maintain/Increase
        """
        shift_preds = self.predictions['shift']
        shift_target = self.targets['shift']
        if shift_preds is None or len(shift_preds) == 0:
            print("No shift predictions available")
            return None

        config = getattr(self.model, 'config', None)
        apply_ts = bool(getattr(config, 'apply_token_shift', False))
        shift_continuous = bool(getattr(config, 'shift_continuous', False))

        if apply_ts:
            dec_idx, maintain_idx, inc_idx = 2, 3, 4
        else:
            dec_idx, maintain_idx, inc_idx = 1, 2, 3

        shift_true_raw = shift_target.numpy().flatten()

        # Detect: continuous regression (v7) vs classification
        is_continuous = shift_continuous or (shift_preds.dim() <= 2 and shift_preds.dtype in (torch.float32, torch.float64))
        # Extra check: if values are clearly classification logits (dim=3), not continuous
        if shift_preds.dim() == 3:
            is_continuous = False

        if is_continuous:
            return self._visualize_regression_shift(
                shift_preds, shift_true_raw,
                dec_idx, maintain_idx, inc_idx,
                save_path=save_path
            )

        n_classes = shift_preds.shape[-1] if shift_preds.dim() >= 3 else 2
        is_binary = (n_classes == 2)

        if is_binary:
            return self._visualize_binary_shift(
                shift_preds, shift_true_raw,
                dec_idx, maintain_idx, inc_idx,
                save_path=save_path
            )
        else:
            return self._visualize_multiclass_shift(
                shift_preds, shift_true_raw,
                dec_idx, maintain_idx, inc_idx,
                save_path=save_path
            )

    def _visualize_regression_shift(self, shift_preds, shift_true_raw,
                                     dec_idx, maintain_idx, inc_idx,
                                     save_path=None):
        """
        Regression-based shift analysis (model_v7: MDN continuous).
        6-panel: scatter, residuals, distribution, error dist, rounded accuracy, metrics.
        """
        shift_pred = shift_preds.numpy().flatten()
        shift_true = shift_true_raw.astype(np.float32)

        # Filter to valid drug events
        valid = np.isin(shift_true.astype(int), [dec_idx, maintain_idx, inc_idx])
        shift_pred = shift_pred[valid]
        shift_true = shift_true[valid]

        if len(shift_true) == 0:
            print("No valid drug events for shift regression analysis")
            return None

        n_total = len(shift_true)
        print(f"Shift regression analysis — {n_total:,} drug events")
        print(f"  True range: [{shift_true.min():.2f}, {shift_true.max():.2f}]")
        print(f"  Pred range: [{shift_pred.min():.2f}, {shift_pred.max():.2f}]")

        mae = np.mean(np.abs(shift_true - shift_pred))
        rmse = np.sqrt(np.mean((shift_true - shift_pred)**2))
        try:
            r2 = 1 - np.sum((shift_true - shift_pred)**2) / np.sum((shift_true - np.mean(shift_true))**2)
        except:
            r2 = np.nan
        residuals = shift_pred - shift_true

        # Rounded accuracy (since targets are near-integer)
        pred_rounded = np.clip(np.rint(shift_pred), dec_idx, inc_idx).astype(int)
        true_int = shift_true.astype(int)
        rounded_acc = np.mean(pred_rounded == true_int)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Drug Dose Shift Prediction (Continuous Regression)',
                     fontsize=16, fontweight='bold')

        # 1. Scatter
        axes[0, 0].scatter(shift_true, shift_pred, alpha=0.2, s=10, c='steelblue')
        lim = [min(shift_true.min(), shift_pred.min()) - 0.5,
               max(shift_true.max(), shift_pred.max()) + 0.5]
        axes[0, 0].plot(lim, lim, 'r--', lw=2, label='Perfect')
        axes[0, 0].set_xlabel('True'); axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Predicted vs True'); axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residuals
        axes[0, 1].scatter(shift_true, residuals, alpha=0.2, s=10, c='coral')
        axes[0, 1].axhline(0, color='r', ls='--', lw=2)
        axes[0, 1].set_xlabel('True'); axes[0, 1].set_ylabel('Residual')
        axes[0, 1].set_title('Residual Plot'); axes[0, 1].grid(True, alpha=0.3)

        # 3. Distribution comparison
        bins = np.linspace(min(dec_idx-0.5, shift_pred.min()), max(inc_idx+0.5, shift_pred.max()), 30)
        axes[0, 2].hist(shift_true, bins=bins, alpha=0.5, label='True', color='blue')
        axes[0, 2].hist(shift_pred, bins=bins, alpha=0.5, label='Predicted', color='orange')
        axes[0, 2].set_xlabel('Shift Value'); axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Distribution Comparison'); axes[0, 2].legend()

        # 4. Error distribution
        axes[1, 0].hist(residuals, bins=50, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(0, color='r', ls='--', lw=2)
        axes[1, 0].set_xlabel('Error'); axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Error Distribution')

        # 5. Rounded confusion matrix
        from sklearn.metrics import confusion_matrix as cm_func
        class_labels = [dec_idx, maintain_idx, inc_idx]
        class_names = ['Decrease', 'Maintain', 'Increase']
        cm = cm_func(true_int, pred_rounded, labels=class_labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[1, 1])
        axes[1, 1].set_title(f'Rounded Confusion (Acc: {rounded_acc:.2%})')
        axes[1, 1].set_ylabel('True'); axes[1, 1].set_xlabel('Predicted')

        # 6. Metrics text
        txt  = f"Regression Metrics:\n\n"
        txt += f"MAE:  {mae:.4f}\n"
        txt += f"RMSE: {rmse:.4f}\n"
        txt += f"R²:   {r2:.4f}\n\n"
        txt += f"Rounded Accuracy: {rounded_acc:.2%}\n"
        txt += f"Samples: {n_total:,}\n\n"
        txt += f"True — Mean: {shift_true.mean():.3f}  Std: {shift_true.std():.3f}\n"
        txt += f"Pred — Mean: {shift_pred.mean():.3f}  Std: {shift_pred.std():.3f}"
        axes[1, 2].text(0.1, 0.5, txt, fontsize=11, family='monospace',
                        verticalalignment='center')
        axes[1, 2].axis('off'); axes[1, 2].set_title('Metrics')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved → {save_path}")
        plt.show()
        plt.close(fig)

        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'rounded_accuracy': rounded_acc,
                'n_samples': n_total}

    def _visualize_binary_shift(self, shift_logits, shift_true_raw,
                                 dec_idx, maintain_idx, inc_idx,
                                 save_path=None):
        """
        Binary change classification: Maintain(0) vs Changed(1).
        Used by model_v2 (BinaryChangeHead).
        """
        shift_pred = torch.argmax(shift_logits, dim=-1).numpy().flatten()

        # Remap targets to binary: maintain→0, dec/inc→1
        is_dec = shift_true_raw == dec_idx
        is_maintain = shift_true_raw == maintain_idx
        is_inc = shift_true_raw == inc_idx
        valid = is_dec | is_maintain | is_inc

        if valid.sum() == 0:
            print("No valid drug events for shift analysis")
            return None

        binary_true = np.full_like(shift_true_raw, -1)
        binary_true[is_maintain] = 0
        binary_true[is_dec | is_inc] = 1

        binary_true = binary_true[valid]
        binary_pred = shift_pred[valid]

        n_total = len(binary_true)
        n_changed = (binary_true == 1).sum()
        n_maintain = (binary_true == 0).sum()
        print(f"Binary shift analysis — {n_total:,} drug events")
        print(f"  Maintain: {n_maintain:,}  Changed: {n_changed:,}")

        pred_unique, pred_counts = np.unique(binary_pred, return_counts=True)
        print(f"[DEBUG] Prediction distribution:")
        for v, c in zip(pred_unique, pred_counts):
            lbl = 'Maintain' if v == 0 else 'Changed'
            print(f"  {lbl:>10s} ({v}): {c:>8,} ({c/n_total*100:.1f}%)")

        class_labels = [0, 1]
        class_names = ['Maintain', 'Changed']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Drug Dose Change Prediction (Binary: Maintain vs Changed)',
                     fontsize=16, fontweight='bold')

        # 1. Confusion matrix
        cm = confusion_matrix(binary_true, binary_pred, labels=class_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True'); axes[0, 0].set_xlabel('Predicted')

        # 2. Normalized confusion matrix
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1])
        axes[0, 1].set_title('Normalized Confusion Matrix')
        axes[0, 1].set_ylabel('True'); axes[0, 1].set_xlabel('Predicted')

        # 3. Class distribution
        dist_counts = [n_maintain, n_changed]
        axes[0, 2].bar(class_names, dist_counts, color=['lightgreen', 'salmon'])
        axes[0, 2].set_title('True Class Distribution'); axes[0, 2].set_ylabel('Count')

        # 4. Per-class accuracy
        per_class_acc = cm_norm.diagonal()
        axes[1, 0].bar(class_names, per_class_acc, color=['lightgreen', 'salmon'])
        axes[1, 0].set_title('Per-Class Accuracy'); axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim(0, 1)
        for i, v in enumerate(per_class_acc):
            axes[1, 0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')

        # 5. Precision / Recall / F1
        prec, rec, f1, _ = precision_recall_fscore_support(
            binary_true, binary_pred, average=None, labels=class_labels)
        x = np.arange(2); w = 0.25
        axes[1, 1].bar(x - w, prec, w, label='Precision', color='steelblue')
        axes[1, 1].bar(x,     rec,  w, label='Recall',    color='darkorange')
        axes[1, 1].bar(x + w, f1,   w, label='F1-Score',  color='green')
        axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(class_names)
        axes[1, 1].set_title('Precision / Recall / F1'); axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)

        # 6. Text report
        overall_acc = accuracy_score(binary_true, binary_pred)
        report = classification_report(binary_true, binary_pred,
                                       target_names=class_names,
                                       labels=class_labels, output_dict=True)
        report['accuracy'] = overall_acc

        txt  = f"Overall Accuracy: {overall_acc:.2%}\n\n"
        txt += f"Balanced Accuracy: {(per_class_acc[0]+per_class_acc[1])/2:.2%}\n\n"
        txt += f"Maintain — P: {prec[0]:.3f}  R: {rec[0]:.3f}  F1: {f1[0]:.3f}\n"
        txt += f"Changed  — P: {prec[1]:.3f}  R: {rec[1]:.3f}  F1: {f1[1]:.3f}\n\n"
        txt += f"Macro Avg:\n"
        txt += f"  Precision: {report['macro avg']['precision']:.2%}\n"
        txt += f"  Recall:    {report['macro avg']['recall']:.2%}\n"
        txt += f"  F1-Score:  {report['macro avg']['f1-score']:.2%}"
        axes[1, 2].text(0.1, 0.5, txt, fontsize=11, family='monospace',
                        verticalalignment='center')
        axes[1, 2].axis('off'); axes[1, 2].set_title('Classification Report')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved → {save_path}")
        plt.show()
        return report

    def _visualize_multiclass_shift(self, shift_logits, shift_true_raw,
                                     dec_idx, maintain_idx, inc_idx,
                                     save_path=None):
        """
        Multi-class shift: Decrease / Maintain / Increase.
        Used by model_v1 (Sequential) or model_v3 (HierarchicalShiftHead).
        """
        shift_pred = torch.argmax(shift_logits, dim=-1).numpy().flatten()

        valid_classes = [dec_idx, maintain_idx, inc_idx]
        mask = np.isin(shift_true_raw, valid_classes)
        shift_pred = shift_pred[mask]
        shift_true = shift_true_raw[mask]

        if len(shift_true) == 0:
            print("No drug-related shift predictions after filtering")
            return None

        pred_unique, pred_counts = np.unique(shift_pred, return_counts=True)
        print(f"[DEBUG] Raw prediction distribution:")
        label_map = {dec_idx: 'Decrease', maintain_idx: 'Maintain', inc_idx: 'Increase'}
        for v, c in zip(pred_unique, pred_counts):
            print(f"  {label_map.get(v, f'?{v}'):>10s} (idx={v}): {c:>8,} ({c/len(shift_pred)*100:.1f}%)")

        class_labels = valid_classes
        class_names = ['Decrease', 'Maintain', 'Increase']
        class_colors = ['salmon', 'lightgreen', 'orange']

        print(f"Shift analysis — {len(shift_true):,} drug events")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Drug Dose Shift Prediction (Decrease / Maintain / Increase)',
                     fontsize=16, fontweight='bold')

        cm = confusion_matrix(shift_true, shift_pred, labels=class_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True'); axes[0, 0].set_xlabel('Predicted')

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1])
        axes[0, 1].set_title('Normalized Confusion Matrix')
        axes[0, 1].set_ylabel('True'); axes[0, 1].set_xlabel('Predicted')

        dist_counts = [np.sum(shift_true == c) for c in class_labels]
        axes[0, 2].bar(class_names, dist_counts, color='skyblue')
        axes[0, 2].set_title('True Class Distribution'); axes[0, 2].set_ylabel('Count')

        per_class_acc = cm_norm.diagonal()
        axes[1, 0].bar(class_names, per_class_acc, color=class_colors)
        axes[1, 0].set_title('Per-Class Accuracy'); axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim(0, 1)
        for i, v in enumerate(per_class_acc):
            axes[1, 0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')

        prec, rec, f1, _ = precision_recall_fscore_support(
            shift_true, shift_pred, average=None, labels=class_labels)
        x = np.arange(len(class_names)); w = 0.25
        axes[1, 1].bar(x - w, prec, w, label='Precision', color='steelblue')
        axes[1, 1].bar(x,     rec,  w, label='Recall',    color='darkorange')
        axes[1, 1].bar(x + w, f1,   w, label='F1-Score',  color='green')
        axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(class_names)
        axes[1, 1].set_title('Precision / Recall / F1'); axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)

        report = classification_report(shift_true, shift_pred,
                                       target_names=class_names,
                                       labels=class_labels, output_dict=True)
        overall_acc = accuracy_score(shift_true, shift_pred)
        report['accuracy'] = overall_acc

        txt  = f"Overall Accuracy: {overall_acc:.2%}\n\n"
        txt += f"Macro Avg:\n"
        txt += f"  Precision: {report['macro avg']['precision']:.2%}\n"
        txt += f"  Recall:    {report['macro avg']['recall']:.2%}\n"
        txt += f"  F1-Score:  {report['macro avg']['f1-score']:.2%}\n\n"
        txt += f"Weighted Avg:\n"
        txt += f"  Precision: {report['weighted avg']['precision']:.2%}\n"
        txt += f"  Recall:    {report['weighted avg']['recall']:.2%}\n"
        txt += f"  F1-Score:  {report['weighted avg']['f1-score']:.2%}"
        axes[1, 2].text(0.1, 0.5, txt, fontsize=11, family='monospace',
                        verticalalignment='center')
        axes[1, 2].axis('off'); axes[1, 2].set_title('Classification Report')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved → {save_path}")
        plt.show()
        return report

    # ==================================================================
    # Hierarchical shift analysis (v3: change detection + direction)
    # ==================================================================
    def visualize_hierarchical_shift(self, save_path=None):
        """
        6-panel figure for hierarchical shift prediction (model_v2).

        Stage 1: Maintain vs Changed (binary)
        Stage 2: Decrease vs Increase (binary, only for changed samples)

        Falls back gracefully if hierarchical logits are not available.

        Returns
        -------
        report : dict with stage1/stage2 metrics
        """
        shift_change = self.predictions.get('shift_change')
        shift_direction = self.predictions.get('shift_direction')
        shift_target = self.targets.get('shift')

        if shift_change is None or not isinstance(shift_change, torch.Tensor) or len(shift_change) == 0:
            print("No hierarchical shift logits available (is this a v2 model?)")
            return None

        # Determine class indices based on apply_token_shift
        apply_ts = getattr(getattr(self.model, 'config', None), 'apply_token_shift', True)
        if apply_ts:
            dec_idx, maintain_idx, inc_idx = 2, 3, 4
        else:
            dec_idx, maintain_idx, inc_idx = 1, 2, 3

        shift_true = shift_target.numpy().flatten()
        change_logits = shift_change.numpy().reshape(-1, shift_change.shape[-1])
        direction_logits = shift_direction.numpy().reshape(-1, shift_direction.shape[-1])

        # ── Filter to valid drug events only ──
        mask = np.isin(shift_true, [dec_idx, maintain_idx, inc_idx])
        shift_true = shift_true[mask]
        change_logits = change_logits[mask]
        direction_logits = direction_logits[mask]

        if len(shift_true) == 0:
            print("No valid drug events for hierarchical analysis")
            return None

        # ── Stage 1: Maintain (0) vs Changed (1) ──
        is_dec = shift_true == dec_idx
        is_maintain = shift_true == maintain_idx
        is_inc = shift_true == inc_idx

        change_true = (is_dec | is_inc).astype(int)   # 0=maintain, 1=changed
        change_pred = np.argmax(change_logits, axis=-1)  # 0=maintain, 1=changed

        # ── Stage 2: Decrease (0) vs Increase (1), changed only ──
        changed_mask = change_true == 1
        if changed_mask.sum() > 0:
            dir_true = is_inc[changed_mask].astype(int)   # 0=decrease, 1=increase
            dir_pred = np.argmax(direction_logits[changed_mask], axis=-1)
        else:
            dir_true = np.array([])
            dir_pred = np.array([])

        n_total = len(shift_true)
        n_changed = changed_mask.sum()
        n_maintain = n_total - n_changed

        print(f"Hierarchical shift analysis — {n_total:,} drug events")
        print(f"  Maintain: {n_maintain:,}  Changed: {n_changed:,} "
              f"(Dec: {is_dec.sum():,}, Inc: {is_inc.sum():,})")

        # ── Figure ──
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hierarchical Shift Prediction Analysis (Drug Events Only)',
                     fontsize=16, fontweight='bold')

        # 1. Stage 1 Confusion Matrix
        cm1 = confusion_matrix(change_true, change_pred, labels=[0, 1])
        sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Maintain', 'Changed'],
                    yticklabels=['Maintain', 'Changed'], ax=axes[0, 0])
        axes[0, 0].set_title('Stage 1: Change Detection')
        axes[0, 0].set_ylabel('True'); axes[0, 0].set_xlabel('Predicted')

        # 2. Stage 1 Normalized
        cm1_norm = cm1.astype(float) / cm1.sum(axis=1, keepdims=True)
        sns.heatmap(cm1_norm, annot=True, fmt='.2%', cmap='Greens',
                    xticklabels=['Maintain', 'Changed'],
                    yticklabels=['Maintain', 'Changed'], ax=axes[0, 1])
        axes[0, 1].set_title('Stage 1: Normalized')
        axes[0, 1].set_ylabel('True'); axes[0, 1].set_xlabel('Predicted')

        # 3. Stage 1 Metrics
        s1_acc = accuracy_score(change_true, change_pred)
        s1_prec, s1_rec, s1_f1, _ = precision_recall_fscore_support(
            change_true, change_pred, average=None, labels=[0, 1])

        x_pos = np.arange(2); w = 0.25
        axes[0, 2].bar(x_pos - w, s1_prec, w, label='Precision', color='steelblue')
        axes[0, 2].bar(x_pos,     s1_rec,  w, label='Recall',    color='darkorange')
        axes[0, 2].bar(x_pos + w, s1_f1,   w, label='F1',        color='green')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(['Maintain', 'Changed'])
        axes[0, 2].set_title(f'Stage 1 Metrics (Acc: {s1_acc:.2%})')
        axes[0, 2].set_ylim(0, 1); axes[0, 2].legend()

        # 4. Stage 2 Confusion Matrix (changed samples only)
        if len(dir_true) > 0:
            cm2 = confusion_matrix(dir_true, dir_pred, labels=[0, 1])
            sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges',
                        xticklabels=['Decrease', 'Increase'],
                        yticklabels=['Decrease', 'Increase'], ax=axes[1, 0])
            axes[1, 0].set_title('Stage 2: Direction (Changed Only)')
            axes[1, 0].set_ylabel('True'); axes[1, 0].set_xlabel('Predicted')

            # 5. Stage 2 Normalized
            cm2_norm = cm2.astype(float) / cm2.sum(axis=1, keepdims=True)
            sns.heatmap(cm2_norm, annot=True, fmt='.2%', cmap='Purples',
                        xticklabels=['Decrease', 'Increase'],
                        yticklabels=['Decrease', 'Increase'], ax=axes[1, 1])
            axes[1, 1].set_title('Stage 2: Normalized')
            axes[1, 1].set_ylabel('True'); axes[1, 1].set_xlabel('Predicted')

            s2_acc = accuracy_score(dir_true, dir_pred)
            s2_prec, s2_rec, s2_f1, _ = precision_recall_fscore_support(
                dir_true, dir_pred, average=None, labels=[0, 1])
        else:
            axes[1, 0].text(0.5, 0.5, 'No changed samples', ha='center', va='center')
            axes[1, 0].axis('off')
            axes[1, 1].text(0.5, 0.5, 'No changed samples', ha='center', va='center')
            axes[1, 1].axis('off')
            s2_acc = 0.0
            s2_prec = s2_rec = s2_f1 = np.array([0.0, 0.0])

        # 6. Combined summary
        txt  = "═══ Stage 1: Change Detection ═══\n"
        txt += f"  Accuracy: {s1_acc:.2%}\n"
        txt += f"  Maintain  — P: {s1_prec[0]:.3f}  R: {s1_rec[0]:.3f}  F1: {s1_f1[0]:.3f}\n"
        txt += f"  Changed   — P: {s1_prec[1]:.3f}  R: {s1_rec[1]:.3f}  F1: {s1_f1[1]:.3f}\n\n"
        txt += "═══ Stage 2: Direction ═══\n"
        if len(dir_true) > 0:
            txt += f"  Accuracy: {s2_acc:.2%}  ({n_changed:,} samples)\n"
            txt += f"  Decrease  — P: {s2_prec[0]:.3f}  R: {s2_rec[0]:.3f}  F1: {s2_f1[0]:.3f}\n"
            txt += f"  Increase  — P: {s2_prec[1]:.3f}  R: {s2_rec[1]:.3f}  F1: {s2_f1[1]:.3f}\n\n"
        else:
            txt += "  (no changed samples)\n\n"

        # Compute combined 3-class accuracy from composed logits
        shift_logits = self.predictions['shift']
        if shift_logits is not None and len(shift_logits) > 0:
            combined_pred = torch.argmax(shift_logits, dim=-1).numpy().flatten()[mask]
            combined_acc = accuracy_score(
                self.targets['shift'].numpy().flatten()[mask],
                combined_pred
            )
            txt += f"═══ Combined 3-Class ═══\n"
            txt += f"  Accuracy: {combined_acc:.2%}"
        else:
            combined_acc = 0.0

        axes[1, 2].text(0.05, 0.5, txt, fontsize=11, family='monospace',
                        verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        axes[1, 2].axis('off'); axes[1, 2].set_title('Summary Report')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved → {save_path}")
        plt.show()

        return {
            'stage1_accuracy': s1_acc,
            'stage1_maintain': dict(precision=s1_prec[0], recall=s1_rec[0], f1=s1_f1[0]),
            'stage1_changed':  dict(precision=s1_prec[1], recall=s1_rec[1], f1=s1_f1[1]),
            'stage2_accuracy': s2_acc,
            'stage2_decrease': dict(precision=s2_prec[0], recall=s2_rec[0], f1=s2_f1[0]),
            'stage2_increase': dict(precision=s2_prec[1], recall=s2_rec[1], f1=s2_f1[1]),
            'combined_3class_accuracy': combined_acc,
            'n_total': n_total, 'n_maintain': int(n_maintain), 'n_changed': int(n_changed),
        }

    # ==================================================================
    # Total regression figure (standalone)
    # ==================================================================
    def visualize_total_performance(self, save_path=None):
        """
        6-panel figure for total-dosage regression.

        Returns
        -------
        dict with keys: n_samples, mse, rmse, mae, r2, mape
        """
        total_pred_all = self.predictions['total']
        total_true_all = self.targets['total']
        if total_pred_all is None or len(total_pred_all) == 0:
            print("No total predictions available")
            return None

        total_pred = total_pred_all.numpy().flatten()
        total_true = total_true_all.numpy().flatten()
        shift_true = self.targets['shift'].numpy().flatten()

        # Keep only drug events with valid positive totals
        config = getattr(self.model, 'config', None)
        apply_ts = bool(getattr(config, 'apply_token_shift', False))
        if apply_ts:
            drug_shift_mask = np.isin(shift_true, [2, 3, 4])
        else:
            drug_shift_mask = np.isin(shift_true, [1, 2, 3])
        mask = drug_shift_mask & (total_true > 0) & (total_pred > 0)
        total_pred = total_pred[mask]
        total_true = total_true[mask]

        if len(total_true) == 0:
            print("No drug-related total predictions after filtering")
            return None

        print(f"Total analysis — {len(total_true):,} drug events")

        mse  = mean_squared_error(total_true, total_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(total_true, total_pred)
        r2   = r2_score(total_true, total_pred)
        mape = np.mean(np.abs((total_true - total_pred) / total_true)) * 100
        residuals = total_pred - total_true

        # ── figure ──
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Drug Total Dosage Prediction Analysis (Drug Events Only)',
                     fontsize=16, fontweight='bold')

        # 1. Scatter
        axes[0, 0].scatter(total_true, total_pred, alpha=0.3, s=20)
        axes[0, 0].plot([total_true.min(), total_true.max()],
                        [total_true.min(), total_true.max()],
                        'r--', lw=2, label='Perfect')
        axes[0, 0].set_xlabel('True'); axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Predicted vs True'); axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residuals
        axes[0, 1].scatter(total_true, residuals, alpha=0.3, s=20)
        axes[0, 1].axhline(0, color='r', ls='--', lw=2)
        axes[0, 1].set_xlabel('True'); axes[0, 1].set_ylabel('Residual')
        axes[0, 1].set_title('Residual Plot'); axes[0, 1].grid(True, alpha=0.3)

        # 3. Distribution comparison
        axes[0, 2].hist(total_true, bins=50, alpha=0.5, label='True', color='blue')
        axes[0, 2].hist(total_pred, bins=50, alpha=0.5, label='Predicted', color='orange')
        axes[0, 2].set_xlabel('Dosage'); axes[0, 2].set_ylabel('Freq')
        axes[0, 2].set_title('Distribution Comparison'); axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Error distribution
        axes[1, 0].hist(residuals, bins=50, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(0, color='r', ls='--', lw=2)
        axes[1, 0].set_xlabel('Error'); axes[1, 0].set_ylabel('Freq')
        axes[1, 0].set_title('Error Distribution'); axes[1, 0].grid(True, alpha=0.3)

        # 5. Absolute error vs true
        axes[1, 1].scatter(total_true, np.abs(residuals), alpha=0.3, s=20)
        axes[1, 1].set_xlabel('True'); axes[1, 1].set_ylabel('|Error|')
        axes[1, 1].set_title('Abs Error vs True'); axes[1, 1].grid(True, alpha=0.3)

        # 6. Metrics text
        txt  = "Regression Metrics:\n\n"
        txt += f"MSE:  {mse:.4f}\n"
        txt += f"RMSE: {rmse:.4f}\n"
        txt += f"MAE:  {mae:.4f}\n"
        txt += f"R²:   {r2:.4f}\n"
        txt += f"MAPE: {mape:.2f}%\n\n"
        txt += f"Samples: {len(total_true):,}\n"
        txt += f"Mean True: {total_true.mean():.2f}  Pred: {total_pred.mean():.2f}\n"
        txt += f"Std  True: {total_true.std():.2f}  Pred: {total_pred.std():.2f}"
        axes[1, 2].text(0.1, 0.5, txt, fontsize=12, family='monospace',
                        verticalalignment='center')
        axes[1, 2].axis('off'); axes[1, 2].set_title('Regression Metrics')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved → {save_path}")
        plt.show()

        return dict(n_samples=len(total_true), mse=mse, rmse=rmse,
                    mae=mae, r2=r2, mape=mape)

    # ==================================================================
    # Attention maps (convenience)
    # ==================================================================
    def visualize_attention_maps(self, sample_indices=(0, 1, 2),
                                 save_path=None):
        """Quick attention heatmap for a few samples."""
        if not self.attention_weights:
            print("No attention weights available")
            return

        n = min(len(sample_indices), len(self.attention_weights))
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]
        fig.suptitle('Attention Map Visualization', fontsize=16, fontweight='bold')

        for idx, si in enumerate(sample_indices[:n]):
            attn = self.attention_weights[si]
            if isinstance(attn, torch.Tensor):
                attn = attn.cpu().numpy()
            if len(attn.shape) == 4:
                attn = attn[0].mean(axis=0)
            elif len(attn.shape) == 3:
                attn = attn.mean(axis=0)
            im = axes[idx].imshow(attn, cmap='viridis', aspect='auto')
            axes[idx].set_title(f'Sample {si}')
            axes[idx].set_xlabel('Key'); axes[idx].set_ylabel('Query')
            plt.colorbar(im, ax=axes[idx])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved → {save_path}")
        plt.show()
