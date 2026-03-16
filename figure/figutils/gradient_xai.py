"""
gradient_xai.py - Gradient-based XAI for SHIFT & TOTAL regression
===================================================================
Post-hoc explanation: works on any checkpoint, no model/training changes.

Uses forward hooks on composite_emb → works across v1/v2/v7 without
reimplementing the forward pass.

Methods:
  - Integrated Gradients (IG): principled, completeness axiom
  - Gradient × Input: fast exploratory baseline
  - Attention Rollout: information flow through layers

Figures:
  1. explain()              → per-patient waterfall (top-K bar + timeline)
  2. population_attribution → chapter-level & token-level importance
  3. compare_methods        → side-by-side method comparison

Usage
-----
```python
from figutils.gradient_xai import GradientExplainer

gxai = GradientExplainer(model, device='cuda',
                         labels_path='../data/labels_chapter.csv')

# Single patient: waterfall
gxai.explain_shift(dataloader, patient_idx=42)
gxai.explain_total(dataloader, patient_idx=42)

# Compare IG vs GradInput vs Attention
gxai.compare_methods(dataloader, patient_idx=42, target='shift')

# Population-level: which tokens drive SHIFT/TOTAL?
gxai.population_attribution(dataloader, target='shift', n_patients=200)
```
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_here))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# =====================================================================
# Core attribution engine (hook-based, model-agnostic)
# =====================================================================

class _EmbeddingCapture:
    """Context manager that hooks composite_emb to capture / replace output."""

    def __init__(self, model):
        self.model = model
        self.handle = None
        self.captured = None  # stores the embedding tensor after forward

    # ---- capture mode: just record the embedding ----
    def capture(self):
        def _hook(module, inp, out):
            out = out.detach().clone()
            self.captured = out
            return out
        self.handle = self.model.composite_emb.register_forward_hook(_hook)
        return self

    # ---- inject mode: replace embedding with custom tensor ----
    def inject(self, emb_tensor):
        """emb_tensor must have requires_grad=True for backprop."""
        def _hook(module, inp, out):
            self.captured = emb_tensor
            return emb_tensor
        self.handle = self.model.composite_emb.register_forward_hook(_hook)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _get_scalar(logits, target, position):
    """Extract scalar prediction value at given position."""
    if target == 'shift':
        out = logits['shift']
        if out.dim() == 2:
            return out[0, position]
        elif out.dim() == 3 and out.size(-1) == 1:
            return out[0, position, 0]
        else:
            # classification: take logit of most-likely class
            return out[0, position].max()
    elif target == 'total':
        out = logits['total']
        if out.dim() == 2:
            return out[0, position]
        else:
            return out[0, position, 0] if out.dim() == 3 else out[0, position]
    else:
        raise ValueError(f"Unknown target: {target}")


def gradient_x_input(model, x_data, x_shift, x_total, x_ages,
                     target='shift', position=-1):
    """
    Gradient × Input attribution.

    Returns
    -------
    attribution : np.ndarray (T,)
    position : int
    pred_value : float
    """
    device = x_data.device

    # 1) Capture actual embedding
    cap = _EmbeddingCapture(model)
    cap.capture()
    with torch.no_grad():
        model(x_data, x_shift, x_total, x_ages)
    cap.remove()
    emb_actual = cap.captured.clone()  # (1, T, D)

    # 2) Inject with grad enabled
    emb_input = emb_actual.detach().clone().requires_grad_(True)
    inj = _EmbeddingCapture(model)
    inj.inject(emb_input)

    logits, _, _ = model(x_data, x_shift, x_total, x_ages)

    if position < 0:
        position = _find_last_valid(x_data)

    scalar = _get_scalar(logits, target, position)
    pred_value = scalar.item()

    model.zero_grad()
    scalar.backward()
    inj.remove()

    grad = emb_input.grad  # (1, T, D)
    # Per-token attribution = sum of element-wise grad * input
    attr = (grad * emb_input).sum(dim=-1)[0].detach().cpu().numpy()

    return attr, position, pred_value


def integrated_gradients(model, x_data, x_shift, x_total, x_ages,
                         target='shift', position=-1, n_steps=50):
    """
    Integrated Gradients attribution.

    Satisfies completeness: sum(attr) ≈ f(x) - f(baseline).

    Returns
    -------
    attribution : np.ndarray (T,)
    position : int
    pred_value : float
    """
    device = x_data.device

    # 1) Capture actual embedding
    cap = _EmbeddingCapture(model)
    cap.capture()
    with torch.no_grad():
        model(x_data, x_shift, x_total, x_ages)
    cap.remove()
    emb_actual = cap.captured.clone()  # (1, T, D)
    emb_baseline = torch.zeros_like(emb_actual)

    if position < 0:
        position = _find_last_valid(x_data)

    # 2) Integrate gradients along interpolation path
    integrated_grad = torch.zeros_like(emb_actual)
    pred_value = None

    for step in range(n_steps + 1):
        alpha = step / n_steps
        emb_interp = emb_baseline + alpha * (emb_actual - emb_baseline)
        emb_interp = emb_interp.detach().clone().requires_grad_(True)

        inj = _EmbeddingCapture(model)
        inj.inject(emb_interp)

        logits, _, _ = model(x_data, x_shift, x_total, x_ages)
        scalar = _get_scalar(logits, target, position)

        if step == n_steps:
            pred_value = scalar.item()

        model.zero_grad()
        scalar.backward()
        inj.remove()

        if emb_interp.grad is not None:
            integrated_grad += emb_interp.grad.detach()

    # IG = (input - baseline) × mean(gradients)
    ig = (emb_actual - emb_baseline) * integrated_grad / (n_steps + 1)
    attr = ig.sum(dim=-1)[0].cpu().numpy()

    return attr, position, pred_value


def attention_rollout(model, x_data, x_shift, x_total, x_ages, position=-1):
    """
    Attention Rollout: multiply attention matrices across layers
    to track information flow to a specific output position.

    No gradients needed — just the forward pass attention weights.

    Returns
    -------
    attribution : np.ndarray (T,)  how much each input token contributed
    position : int
    """
    # Forward without targets → attention collected
    with torch.no_grad():
        logits, _, att = model(x_data, x_shift, x_total, x_ages)

    if att is None:
        raise RuntimeError(
            "No attention weights collected. Make sure targets are NOT passed "
            "(model only collects attention when targets_data=None)."
        )

    if position < 0:
        position = _find_last_valid(x_data)

    pred_value = _get_scalar(logits, 'shift', position).item()

    # att shape: (n_layers, B, n_heads, T, T)
    n_layers = att.shape[0]
    T = att.shape[-1]

    # Average over heads, then rollout (multiply across layers)
    rollout = torch.eye(T, device=att.device).unsqueeze(0)  # (1, T, T)

    for layer in range(n_layers):
        # (B, n_heads, T, T) → (B, T, T) mean over heads
        attn = att[layer].mean(dim=1)  # (B, T, T)
        # Add identity (residual connection)
        attn = 0.5 * attn + 0.5 * torch.eye(T, device=attn.device).unsqueeze(0)
        # Re-normalize rows
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        rollout = torch.bmm(attn, rollout)

    # rollout[0, position, :] = how much each input token contributed to position
    attr = rollout[0, position, :].cpu().numpy()
    return attr, position, pred_value


# =====================================================================
# Helpers
# =====================================================================

def _find_last_valid(x_data):
    """Find last non-padding position."""
    tokens = x_data[0].cpu().numpy()
    valid = np.where(tokens > 0)[0]
    return int(valid[-1]) if len(valid) > 0 else x_data.shape[1] - 1


def _find_last_drug(x_data, config):
    """Find last drug token position."""
    drug_min = int(getattr(config, 'drug_token_min', 1278))
    drug_max = int(getattr(config, 'drug_token_max', 1288))
    tokens = x_data[0].cpu().numpy()
    drug_mask = (tokens >= drug_min) & (tokens <= drug_max)
    if drug_mask.any():
        return int(np.where(drug_mask)[0][-1])
    return _find_last_valid(x_data)


# =====================================================================
# High-level explainer class
# =====================================================================

class GradientExplainer:
    """
    Post-hoc XAI for CompositeDelphi SHIFT & TOTAL predictions.
    Works on any checkpoint without model or training modifications.
    """

    def __init__(self, model, device='cuda', labels_path=None):
        self.model = model
        self.device = device
        self.model.eval()

        config = getattr(model, 'config', None)
        self.config = config
        self.apply_ts = bool(getattr(config, 'apply_token_shift', False))
        self.data_vocab = int(getattr(config, 'data_vocab_size', 1290))

        # Token name / chapter / color lookup
        self.token_names = {}
        self.token_chapters = {}
        self.token_colors = {}
        if labels_path and os.path.exists(labels_path):
            df = pd.read_csv(labels_path)
            for _, row in df.iterrows():
                tid = int(row['token_id'])
                if self.apply_ts:
                    tid += 1
                self.token_names[tid] = row['name']
                ch = row.get('Short Chapter', '')
                self.token_chapters[tid] = ch
                self.token_colors[tid] = row.get('color', '#999999')

    # ------------------------------------------------------------------
    # Sample extraction
    # ------------------------------------------------------------------
    def _get_sample(self, dataloader, patient_idx=0):
        """Extract a single patient sample from dataloader."""
        offset = 0
        for batch in dataloader:
            bs = batch[0].shape[0]
            if patient_idx < offset + bs:
                i = patient_idx - offset
                return tuple(b[i:i+1].to(self.device) for b in batch)
            offset += bs
        raise IndexError(f"Patient {patient_idx} out of range (total {offset})")

    # ------------------------------------------------------------------
    # Compute attribution
    # ------------------------------------------------------------------
    def _compute(self, sample, target='shift', method='ig',
                 position='drug', n_steps=50):
        """
        Compute per-token attribution.

        Parameters
        ----------
        sample : tuple of 8 tensors
        target : 'shift' or 'total'
        method : 'ig', 'grad_input', or 'attention'
        position : 'drug' (last drug token), 'last' (last valid), or int
        n_steps : int (for IG)

        Returns
        -------
        attr : np.ndarray (T,)
        pos : int
        pred : float
        true : float
        """
        x_data, x_shift, x_total, x_ages = sample[:4]
        y_shift, y_total = sample[5], sample[6]

        # Resolve position
        if position == 'drug':
            pos = _find_last_drug(x_data, self.config)
        elif position == 'last':
            pos = _find_last_valid(x_data)
        elif isinstance(position, int):
            pos = position
        else:
            pos = _find_last_valid(x_data)

        # Compute
        if method == 'ig':
            attr, pos, pred = integrated_gradients(
                self.model, x_data, x_shift, x_total, x_ages,
                target=target, position=pos, n_steps=n_steps)
        elif method == 'attention':
            attr, pos, pred = attention_rollout(
                self.model, x_data, x_shift, x_total, x_ages, position=pos)
        else:  # grad_input
            attr, pos, pred = gradient_x_input(
                self.model, x_data, x_shift, x_total, x_ages,
                target=target, position=pos)

        # True value
        true = (y_shift if target == 'shift' else y_total)[0, pos].item()

        return attr, pos, pred, true

    # ------------------------------------------------------------------
    # Figure 1: Per-patient waterfall + timeline
    # ------------------------------------------------------------------
    def explain(self, dataloader, patient_idx=0, target='shift',
                method='ig', top_k=20, position='drug', save_path=None):
        """
        Explain a single SHIFT or TOTAL prediction.

        Left panel: top-K tokens by |attribution| (waterfall bar).
        Right panel: attribution vs age timeline.
        """
        sample = self._get_sample(dataloader, patient_idx)
        attr, pos, pred, true = self._compute(
            sample, target, method, position)

        x_data = sample[0]
        x_ages = sample[3]
        tokens = x_data[0].cpu().numpy()
        ages = x_ages[0].cpu().numpy() / 365.25

        # Build per-token info (non-padding only)
        valid_idx = np.where(tokens > 0)[0]
        labels, attrs, ages_list = [], [], []
        for i in valid_idx:
            tid = int(tokens[i])
            name = self.token_names.get(tid, f'T{tid}')
            if len(name) > 30:
                name = name[:27] + '...'
            labels.append(f"{name} ({ages[i]:.0f}yr)")
            attrs.append(attr[i])
            ages_list.append(ages[i])

        attrs = np.array(attrs)
        ages_arr = np.array(ages_list)

        # Top-K by |attribution|
        top_idx = np.argsort(-np.abs(attrs))[:top_k]

        fig, axes = plt.subplots(
            1, 2, figsize=(16, max(6, top_k * 0.35)),
            gridspec_kw={'width_ratios': [3, 1]})

        # ---- Left: waterfall bar ----
        ax = axes[0]
        lab = [labels[i] for i in top_idx]
        val = [attrs[i] for i in top_idx]
        col = ['#d73027' if v > 0 else '#4575b4' for v in val]
        y_pos = np.arange(len(lab))

        ax.barh(y_pos, val, color=col, alpha=0.8,
                edgecolor='gray', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(lab, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel('Attribution Score')
        ax.grid(True, alpha=0.2, axis='x')

        target_lbl = 'SHIFT' if target == 'shift' else 'TOTAL'
        method_lbl = {'ig': 'Integrated Gradients',
                      'grad_input': 'Gradient × Input',
                      'attention': 'Attention Rollout'}[method]
        ax.set_title(
            f'{target_lbl} Prediction — {method_lbl}\n'
            f'Predicted: {pred:.2f}  |  True: {true:.2f}  |  '
            f'Position: {pos}',
            fontweight='bold', fontsize=11)

        # ---- Right: attribution over time ----
        ax2 = axes[1]
        pos_mask = attrs > 0
        neg_mask = attrs <= 0
        if pos_mask.any():
            ax2.scatter(ages_arr[pos_mask], attrs[pos_mask],
                        c='#d73027', s=12, alpha=0.5, label='↑')
        if neg_mask.any():
            ax2.scatter(ages_arr[neg_mask], attrs[neg_mask],
                        c='#4575b4', s=12, alpha=0.5, label='↓')
        ax2.axhline(0, color='gray', ls='--', lw=0.8)
        ax2.set_xlabel('Age (yr)', fontsize=9)
        ax2.set_ylabel('Attribution', fontsize=9)
        ax2.set_title('Over Time', fontsize=10, fontweight='bold')
        ax2.legend(fontsize=7, loc='best')
        ax2.grid(True, alpha=0.2)
        ax2.tick_params(labelsize=8)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved → {save_path}")
        plt.show()
        plt.close(fig)

        return {'attribution': attrs, 'labels': labels, 'ages': ages_arr,
                'pred': pred, 'true': true, 'position': pos}

    # Convenience
    def explain_shift(self, dl, patient_idx=0, **kw):
        return self.explain(dl, patient_idx, target='shift', **kw)

    def explain_total(self, dl, patient_idx=0, **kw):
        return self.explain(dl, patient_idx, target='total', **kw)

    # ------------------------------------------------------------------
    # Figure 2: Population-level token importance
    # ------------------------------------------------------------------
    def _find_all_drug_positions(self, x_data):
        """Find ALL drug token positions in sequence."""
        drug_min = int(getattr(self.config, 'drug_token_min', 1278))
        drug_max = int(getattr(self.config, 'drug_token_max', 1288))
        tokens = x_data[0].cpu().numpy()
        drug_mask = (tokens >= drug_min) & (tokens <= drug_max)
        return list(np.where(drug_mask)[0])

    def population_attribution(self, dataloader, target='shift',
                                method='grad_input', n_patients=200,
                                drug_positions='all',
                                top_k_tokens=25, save_path=None):
        """
        Aggregate attributions across many patients.

        Parameters
        ----------
        drug_positions : 'all' or 'last'
            - 'all': compute attribution at EVERY drug token position.
              Captures full prescribing history (e.g., initial vs follow-up).
            - 'last': only the last drug token per patient (faster).
        """
        mode_label = 'all drug positions' if drug_positions == 'all' else 'last drug only'
        print(f"[INFO] Population attribution: {target}, {method}, "
              f"n={n_patients}, positions={mode_label}")

        chapter_attr = defaultdict(list)
        token_attr_agg = defaultdict(list)
        n_attributions = 0
        count = 0

        for batch in dataloader:
            if count >= n_patients:
                break
            if len(batch) != 8:
                continue

            bs = batch[0].shape[0]
            for i in range(min(bs, n_patients - count)):
                sample = tuple(b[i:i+1].to(self.device) for b in batch)

                # Determine which positions to explain
                if drug_positions == 'all':
                    positions = self._find_all_drug_positions(sample[0])
                    if not positions:
                        count += 1
                        continue
                else:
                    positions = ['drug']  # _compute resolves to last drug

                for pos in positions:
                    try:
                        attr, actual_pos, pred, true = self._compute(
                            sample, target, method, position=pos,
                            n_steps=20)
                    except Exception:
                        continue

                    toks = sample[0][0].cpu().numpy()
                    for j, tid in enumerate(toks):
                        if tid <= 0:
                            continue
                        tid = int(tid)
                        ch = self.token_chapters.get(tid, '')
                        if ch in ('', 'Technical'):
                            continue
                        chapter_attr[ch].append(abs(attr[j]))
                        token_attr_agg[tid].append(attr[j])

                    n_attributions += 1

                count += 1

            if count % 50 == 0 and count > 0:
                print(f"  {count}/{n_patients} patients, "
                      f"{n_attributions} attributions")

        print(f"[OK]  {count} patients, {n_attributions} total attributions")

        if not chapter_attr:
            print("No attributions collected")
            return None

        # Aggregate
        ch_mean = sorted(
            [(ch, np.mean(v)) for ch, v in chapter_attr.items()],
            key=lambda x: -x[1])

        tok_stats = []
        for tid, vals in token_attr_agg.items():
            if len(vals) >= 3:
                tok_stats.append({
                    'tid': tid,
                    'name': self.token_names.get(tid, f'T{tid}'),
                    'chapter': self.token_chapters.get(tid, ''),
                    'mean_abs': np.mean(np.abs(vals)),
                    'mean': np.mean(vals),
                    'n': len(vals),
                })
        tok_sorted = sorted(tok_stats, key=lambda x: -x['mean_abs'])

        # ---- Plot ----
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Left: chapter-level
        ax = axes[0]
        ch_names = [c[0] for c in ch_mean[:15]]
        ch_vals = [c[1] for c in ch_mean[:15]]
        # Use chapter color of first token in that chapter
        ch_colors = []
        for ch in ch_names:
            matched = [self.token_colors[tid]
                       for tid, c in self.token_chapters.items() if c == ch]
            ch_colors.append(matched[0] if matched else '#999999')

        ax.barh(range(len(ch_names)), ch_vals, color=ch_colors, alpha=0.8)
        ax.set_yticks(range(len(ch_names)))
        ax.set_yticklabels(ch_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |Attribution|')
        ax.set_title('By ICD-10 Chapter', fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x')

        # Right: top tokens
        ax2 = axes[1]
        show = tok_sorted[:top_k_tokens]
        names = [t['name'][:35] for t in show]
        vals = [t['mean'] for t in show]
        cols = ['#d73027' if v > 0 else '#4575b4' for v in vals]

        ax2.barh(range(len(names)), vals, color=cols, alpha=0.8)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=7)
        ax2.invert_yaxis()
        ax2.axvline(0, color='black', lw=0.8)
        ax2.set_xlabel('Mean Attribution (signed)')
        ax2.set_title(f'Top {top_k_tokens} Tokens', fontweight='bold')
        ax2.grid(True, alpha=0.2, axis='x')

        target_lbl = 'SHIFT' if target == 'shift' else 'TOTAL'
        fig.suptitle(
            f'Population-Level {target_lbl} Attribution '
            f'({method}, {mode_label}, '
            f'{count} patients, {n_attributions} attributions)',
            fontsize=13, fontweight='bold')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved → {save_path}")
        plt.show()
        plt.close(fig)

        return {'chapter': dict(ch_mean), 'tokens': tok_sorted,
                'n_patients': count, 'n_attributions': n_attributions}

    # ------------------------------------------------------------------
    # Figure 3: Compare methods side-by-side
    # ------------------------------------------------------------------
    def compare_methods(self, dataloader, patient_idx=0, target='shift',
                        top_k=15, position='drug', save_path=None):
        """
        Side-by-side comparison of IG, Gradient×Input, and Attention Rollout.
        """
        sample = self._get_sample(dataloader, patient_idx)
        x_data = sample[0]
        x_ages = sample[3]

        tokens = x_data[0].cpu().numpy()
        ages = x_ages[0].cpu().numpy() / 365.25
        valid_idx = np.where(tokens > 0)[0]

        # Build labels
        labels = []
        for i in valid_idx:
            tid = int(tokens[i])
            n = self.token_names.get(tid, f'T{tid}')
            if len(n) > 25:
                n = n[:22] + '...'
            labels.append(f"{n} ({ages[i]:.0f})")

        methods = [
            ('Integrated Gradients', 'ig'),
            ('Gradient × Input', 'grad_input'),
            ('Attention Rollout', 'attention'),
        ]

        results = {}
        pred_vals = {}
        for name, key in methods:
            print(f"  Computing {name}...")
            attr, pos, pred, true = self._compute(
                sample, target, key, position,
                n_steps=30)
            # Keep only valid (non-padding) tokens
            results[name] = attr[valid_idx]
            pred_vals[name] = (pred, true)

        # Top-K union across methods
        combined = sum(np.abs(v) for v in results.values())
        top_idx = np.argsort(-combined)[:top_k]

        fig, axes = plt.subplots(1, 3, figsize=(20, max(6, top_k * 0.35)),
                                  sharey=True)

        for ax, (name, _) in zip(axes, methods):
            vals = results[name][top_idx]
            lab = [labels[i] for i in top_idx]
            col = ['#d73027' if v > 0 else '#4575b4' for v in vals]

            ax.barh(range(len(lab)), vals, color=col, alpha=0.8)
            ax.set_yticks(range(len(lab)))
            ax.set_yticklabels(lab, fontsize=7)
            ax.invert_yaxis()
            ax.axvline(0, color='black', lw=0.8)
            p, t = pred_vals[name]
            ax.set_title(f'{name}\npred={p:.2f} true={t:.2f}',
                         fontweight='bold', fontsize=10)
            ax.grid(True, alpha=0.2, axis='x')
            ax.set_xlabel('Attribution')

        target_lbl = 'SHIFT' if target == 'shift' else 'TOTAL'
        fig.suptitle(
            f'{target_lbl} Attribution — Method Comparison '
            f'(Patient {patient_idx})',
            fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved → {save_path}")
        plt.show()
        plt.close(fig)
        return results