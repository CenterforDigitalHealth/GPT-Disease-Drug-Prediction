"""
calibration.py - Discriminative power & calibration analysis
=============================================================
Implements the evaluation methodology from the Delphi paper:

1. **Discriminative Power**: Per-disease ROC-AUC and Average Precision Score
   for all diseases with >= min_cases in the evaluation period.
2. **Calibration**: Predicted rates vs observed incidence in decile bins.

Data split:
  - kr_val.bin  (2002-2010): patient history → model input
  - kr_test.bin (2011-2013): future events  → ground truth observation

Usage
-----
```python
from figutils.calibration import CalibrationAnalyzer
from figutils.common import load_model, load_composite_data

model, _ = load_model('../ckpt.pt', device='cuda')
val_raw, _, val_p2i = load_composite_data('../data/kr_val.bin')
test_raw, _, test_p2i = load_composite_data('../data/kr_test.bin')

cal = CalibrationAnalyzer(
    model,
    val_data=val_raw, val_p2i=val_p2i,
    test_data=test_raw, test_p2i=test_p2i,
    labels_path='../data/labels_chapter.csv',
    device='cuda',
)
cal.run(n_patients=1000, block_size=512)

cal.plot_discrimination(top_k=30)
cal.plot_calibration()
cal.plot_auc_distribution()
cal.plot_per_disease_calibration(top_k=6)
df = cal.get_results_table()
```
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Optional

_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_here))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils import get_batch_composite


class CalibrationAnalyzer:
    """
    Assess discriminative power (ROC-AUC, AP) and calibration of
    predicted disease rates from a CompositeDelphi model.

    Parameters
    ----------
    model : CompositeDelphi
    val_data, val_p2i : history data (2002-2010) — model input
    test_data, test_p2i : future data (2011-2013) — ground truth
    labels_path : str   path to labels_chapter.csv
    device : str
    apply_token_shift : bool
    min_cases : int     minimum observed cases to evaluate a disease
    """

    def __init__(self, model, val_data, val_p2i, test_data, test_p2i, *,
                 labels_path: Optional[str] = None,
                 device: str = 'cuda',
                 apply_token_shift: bool = True,
                 min_cases: int = 25):
        self.model = model
        self.model.eval()
        self.val_data = val_data
        self.val_p2i = val_p2i
        self.test_data = test_data
        self.test_p2i = test_p2i
        self.device = device
        self.apply_token_shift = apply_token_shift
        self.min_cases = min_cases

        # Read extra config from model for get_batch_composite
        config = getattr(model, 'config', None)
        self._shift_continuous = bool(getattr(config, 'shift_continuous', False))
        self._sep_na = bool(getattr(config, 'separate_shift_na_from_padding', False))
        self._na_tok = int(getattr(config, 'shift_na_raw_token', 4))

        # Override apply_token_shift from model config if available
        if config is not None and hasattr(config, 'apply_token_shift'):
            self.apply_token_shift = bool(config.apply_token_shift)

        # Build test patient ID → events lookup
        self._build_test_lookup()

        # Token name mapping + chapter + color
        self.token_names = {}
        self.token_chapters = {}
        self.token_colors = {}
        if labels_path and os.path.exists(labels_path):
            df = pd.read_csv(labels_path)
            for _, row in df.iterrows():
                tid = int(row['token_id'])
                if apply_token_shift:
                    tid += 1
                self.token_names[tid] = row['name']
                self.token_chapters[tid] = row.get('Short Chapter', '')
                self.token_colors[tid] = row.get('color', '#999999')
            print(f"[OK]  Loaded {len(self.token_names)} token labels")

        # Results
        self.results = {}
        self.pred_rates = None
        self.observed = None
        self._ran = False

    def _build_test_lookup(self):
        """Build mapping: patient ID → set of disease tokens in test period,
        and patient ID → dict of {token_id: earliest_age} for follow-up analysis."""
        self.test_events = {}       # patient_id → set of token_ids
        self.test_event_ages = {}   # patient_id → {token_id: earliest_age_in_days}
        for pid in range(len(self.test_p2i)):
            p_start = self.test_p2i[pid, 0]
            p_len = self.test_p2i[pid, 1]
            if p_len == 0:
                continue
            p_slice = slice(p_start, p_start + p_len)
            p_id = int(self.test_data['ID'][p_start])
            p_tokens = self.test_data['DATA'][p_slice].astype(np.int64)
            p_ages = self.test_data['AGE'][p_slice].astype(np.float64)
            if self.apply_token_shift:
                p_tokens = p_tokens + 1
            self.test_events[p_id] = set(p_tokens.tolist())
            # Earliest occurrence age per token
            ages_dict = {}
            for tok, age in zip(p_tokens, p_ages):
                tok = int(tok)
                if tok not in ages_dict or age < ages_dict[tok]:
                    ages_dict[tok] = float(age)
            self.test_event_ages[p_id] = ages_dict

        # Also build val patient index → patient ID mapping
        self.val_pid_map = {}  # val_index → patient_id
        for vid in range(len(self.val_p2i)):
            p_start = self.val_p2i[vid, 0]
            if self.val_p2i[vid, 1] == 0:
                continue
            p_id = int(self.val_data['ID'][p_start])
            self.val_pid_map[vid] = p_id

        # Find overlapping patients
        val_ids = set(self.val_pid_map.values())
        test_ids = set(self.test_events.keys())
        self.overlap_ids = val_ids & test_ids

        # val_index → list for patients that exist in both sets
        self.valid_val_indices = [
            vid for vid, pid in self.val_pid_map.items()
            if pid in self.overlap_ids
        ]

        print(f"[INFO] Val patients: {len(val_ids):,}, "
              f"Test patients: {len(test_ids):,}, "
              f"Overlap: {len(self.overlap_ids):,}")

    # ==================================================================
    # Main pipeline
    # ==================================================================
    def run(self, n_patients: int = 1000, block_size: int = 512,
            eval_years: float = 3.0):
        """
        For each overlapping patient:
        1. Feed val history (2002-2010) into the model.
        2. Extract per-disease hazard rates from time_scale logits.
        3. Convert to 3-year incidence probability via competing exponentials.
        4. Check which diseases occurred in test period (2011-2013).

        Competing-exponentials model:
          lambda_i = exp(time_logit_i)             per-disease hazard rate
          Lambda   = sum_i(lambda_i)               total hazard
          P(disease_i within T) ≈ (lambda_i / Lambda) * (1 - exp(-Lambda * T))

        Parameters
        ----------
        n_patients : int      max patients to evaluate
        block_size : int      context window size
        eval_years : float    observation window (default 3.0 for 2011-2013)
        """
        config = getattr(self.model, 'config', None)
        data_vocab = int(getattr(config, 'data_vocab_size', 1290))
        t_min = float(getattr(config, 't_min', 0.1))

        # Use overlapping patients only
        indices = self.valid_val_indices[:n_patients]
        n_eval = len(indices)

        if n_eval == 0:
            print("[ERROR] No overlapping patients between val and test")
            return

        print(f"[INFO] Running calibration: {n_eval} patients, "
              f"block_size={block_size}, eval={eval_years}yr")

        all_pred = []
        all_obs = []
        all_sex = []           # 0=unknown, 1=female, 2=male
        all_cutoff_age = []    # cutoff age in days

        self.model.eval()
        batch_size = 32

        for start in range(0, n_eval, batch_size):
            end = min(start + batch_size, n_eval)
            batch_indices = indices[start:end]
            bs = len(batch_indices)

            batch = get_batch_composite(
                batch_indices, self.val_data, self.val_p2i,
                select='right', padding='random',
                block_size=block_size, device=self.device,
                apply_token_shift=self.apply_token_shift,
                shift_continuous=self._shift_continuous,
                separate_shift_na_from_padding=self._sep_na,
                shift_na_raw_token=self._na_tok,
            )
            x_data, x_shift, x_total, x_ages = batch[0], batch[1], batch[2], batch[3]
            y_data, y_shift, y_total, y_ages = batch[4], batch[5], batch[6], batch[7]

            with torch.no_grad():
                logits, _, _ = self.model(
                    x_data, x_shift, x_total, x_ages,
                    targets_data=y_data,
                    targets_shift=y_shift,
                    targets_total=y_total,
                    targets_age=y_ages,
                )

            # time_scale logits → per-disease hazard rate
            time_logits = logits.get('time_scale', logits.get('time', None))
            if time_logits is None:
                # Fallback: use data logits with softmax (less accurate)
                print("[WARN] No time_scale logits, falling back to softmax")
                time_logits = logits['data']
                use_softmax_fallback = True
            else:
                use_softmax_fallback = False

            for i in range(bs):
                vid = batch_indices[i]
                pid = self.val_pid_map[vid]

                valid_mask = x_data[i] > 0
                if valid_mask.sum() == 0:
                    continue
                last_pos = valid_mask.nonzero()[-1].item()

                if use_softmax_fallback:
                    probs = torch.softmax(time_logits[i, last_pos], dim=-1).cpu().numpy()
                else:
                    # Competing-exponentials: convert time logits to 3-year rates
                    tl = time_logits[i, last_pos]  # (V,)

                    # Per-event rate: lambda_i (same formula as training loss)
                    # lambda_i = 1 / (exp(-logit_i) + t_min)
                    # In log space: log_lambda_i = logit_i - softplus(logit_i + log(t_min))
                    import math
                    log_t_min = math.log(max(t_min, 1e-8))
                    log_lambda = tl - torch.nn.functional.softplus(tl + log_t_min)
                    lambda_i = torch.exp(log_lambda.clamp(max=20.0))  # (V,)

                    # Total competing hazard
                    Lambda = lambda_i.sum()

                    # Per-disease 3-year incidence:
                    # P_i(T) = (lambda_i / Lambda) * (1 - exp(-Lambda * T))
                    # T is in days (eval_years * 365.25), but lambda is in day^-1
                    T_days = eval_years * 365.25
                    overall_prob = 1.0 - torch.exp(-Lambda * T_days)
                    overall_prob = overall_prob.clamp(max=1.0)
                    probs = (lambda_i / Lambda.clamp(min=1e-10)) * overall_prob
                    probs = probs.cpu().numpy()

                all_pred.append(probs)

                # Detect sex from patient history tokens
                # Female=2+1=3, Male=3+1=4 (after +1 shift)
                tokens_np = x_data[i].cpu().numpy()
                female_tok = 3 if self.apply_token_shift else 2
                male_tok = 4 if self.apply_token_shift else 3
                if female_tok in tokens_np:
                    all_sex.append(1)  # female
                elif male_tok in tokens_np:
                    all_sex.append(2)  # male
                else:
                    all_sex.append(0)  # unknown

                # Store cutoff age (last valid position's age)
                cutoff_age = x_ages[i, last_pos].item()
                all_cutoff_age.append(cutoff_age)

                # Ground truth
                future_tokens = self.test_events.get(pid, set())
                obs = np.zeros(data_vocab, dtype=np.float32)
                for t in future_tokens:
                    if 0 <= t < data_vocab:
                        obs[t] = 1.0
                all_obs.append(obs)

            if (start + batch_size) % 200 < batch_size:
                print(f"  {min(start + batch_size, n_eval)}/{n_eval} patients")

        if not all_pred:
            print("[ERROR] No valid predictions collected")
            return

        self.pred_rates = np.stack(all_pred)  # (N, V)
        self.observed = np.stack(all_obs)     # (N, V)
        self.patient_sex = np.array(all_sex)           # (N,)
        self.patient_cutoff_age = np.array(all_cutoff_age)  # (N,) days
        n = len(all_pred)
        print(f"[OK]  {n} patients processed")

        # Compute per-disease metrics
        self.results = {}
        for token_id in range(data_vocab):
            y_true = self.observed[:, token_id]
            y_score = self.pred_rates[:, token_id]
            n_cases = int(y_true.sum())

            if n_cases < self.min_cases:
                continue
            if n_cases == n:
                continue

            try:
                auc = roc_auc_score(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
            except ValueError:
                continue

            name = self.token_names.get(token_id, f'Token_{token_id}')
            chapter = self.token_chapters.get(token_id, '')
            color = self.token_colors.get(token_id, '#999999')
            self.results[token_id] = {
                'name': name,
                'chapter': chapter,
                'color': color,
                'auc': auc,
                'ap': ap,
                'n_cases': n_cases,
                'n_total': n,
                'prevalence': n_cases / n,
            }

        print(f"[OK]  {len(self.results)} diseases with >= {self.min_cases} cases evaluated")

        # Print summary stats
        if self.results:
            aucs = [d['auc'] for d in self.results.values()]
            aps = [d['ap'] for d in self.results.values()]
            print(f"  Median ROC-AUC: {np.median(aucs):.3f}  "
                  f"Mean: {np.mean(aucs):.3f}")
            print(f"  Median AP:      {np.median(aps):.3f}  "
                  f"Mean: {np.mean(aps):.3f}")

        self._ran = True

    # ==================================================================
    # Discrimination plot
    # ==================================================================
    def plot_discrimination(self, top_k: int = 30, metric: str = 'auc',
                            save_path=None):
        """
        Horizontal bar chart of ROC-AUC or AP for the top-K diseases.
        """
        if not self._ran:
            print("Call .run() first"); return

        items = sorted(self.results.values(),
                       key=lambda x: x[metric], reverse=True)[:top_k]
        if not items:
            print("No diseases to plot"); return

        names = [d['name'] for d in items]
        values = [d[metric] for d in items]
        cases = [d['n_cases'] for d in items]

        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.35)))
        y_pos = np.arange(len(names))

        colors = plt.cm.RdYlGn(np.array(values))
        ax.barh(y_pos, values, color=colors, edgecolor='gray', linewidth=0.5)

        for i, (v, c) in enumerate(zip(values, cases)):
            ax.text(v + 0.005, i, f'{v:.3f} (n={c})', va='center', fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15)
        ax.axvline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)

        title = 'ROC-AUC' if metric == 'auc' else 'Average Precision'
        ax.set_xlabel(title)
        ax.set_title(f'Top {len(items)} Diseases by {title} '
                     f'(>= {self.min_cases} cases, test period 2011-2013)',
                     fontweight='bold')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved -> {save_path}")
        plt.show()
        return fig

    # ==================================================================
    # Calibration plot (aggregated)
    # ==================================================================
    def plot_calibration(self, n_bins: int = 10, save_path=None):
        """
        Calibration plot: predicted rate deciles vs observed incidence.
        Aggregated across all diseases with >= min_cases.
        """
        if not self._ran:
            print("Call .run() first"); return

        all_preds, all_labels = [], []
        for token_id in self.results:
            all_preds.append(self.pred_rates[:, token_id])
            all_labels.append(self.observed[:, token_id])

        if not all_preds:
            print("No data for calibration"); return

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        bin_edges = np.percentile(all_preds, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-10

        bp, bo, bc = [], [], []
        for i in range(n_bins):
            m = (all_preds >= bin_edges[i]) & (all_preds < bin_edges[i + 1])
            if m.sum() == 0:
                continue
            bp.append(all_preds[m].mean())
            bo.append(all_labels[m].mean())
            bc.append(m.sum())

        bp, bo, bc = np.array(bp), np.array(bo), np.array(bc)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Calibration curve
        ax = axes[0]
        ax.plot(bp, bo, 'o-', color='steelblue', markersize=8, linewidth=2,
                label='Model')
        lim = max(bp.max(), bo.max()) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.6,
                label='Perfect calibration')
        ax.set_xlabel('Mean predicted rate (per bin)')
        ax.set_ylabel('Observed incidence rate')
        ax.set_title('Calibration: Predicted vs Observed', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

        # Bin sizes
        ax2 = axes[1]
        x_pos = np.arange(len(bc))
        ax2.bar(x_pos, bc, color='lightcoral', edgecolor='gray')
        ax2.set_xlabel('Decile bin'); ax2.set_ylabel('Count')
        ax2.set_title('Sample Distribution Across Deciles', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'D{i+1}' for i in x_pos])
        ax2.grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'Calibration ({len(self.results)} diseases, '
                     f'>= {self.min_cases} cases)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved -> {save_path}")
        plt.show()
        return fig

    # ==================================================================
    # Per-disease calibration
    # ==================================================================
    def plot_per_disease_calibration(self, token_ids=None, top_k: int = 6,
                                     n_bins: int = 10, save_path=None):
        """Individual calibration curves for selected diseases."""
        if not self._ran:
            print("Call .run() first"); return

        if token_ids is None:
            items = sorted(self.results.items(),
                           key=lambda x: x[1]['auc'], reverse=True)[:top_k]
            token_ids = [t for t, _ in items]

        n = len(token_ids)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if n == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        for idx, tid in enumerate(token_ids):
            if tid not in self.results:
                continue
            info = self.results[tid]
            ax = axes[idx]

            y_pred = self.pred_rates[:, tid]
            y_true = self.observed[:, tid]

            edges = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
            edges[-1] += 1e-10

            bpd, bod = [], []
            for i in range(n_bins):
                m = (y_pred >= edges[i]) & (y_pred < edges[i + 1])
                if m.sum() == 0:
                    continue
                bpd.append(y_pred[m].mean())
                bod.append(y_true[m].mean())

            if bpd:
                ax.plot(bpd, bod, 'o-', color='steelblue', markersize=6)
                lim = max(max(bpd), max(bod)) * 1.2
                ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5)

            ax.set_title(f"{info['name']}\n"
                         f"AUC={info['auc']:.3f}  AP={info['ap']:.3f}  "
                         f"n={info['n_cases']}", fontsize=9)
            ax.set_xlabel('Predicted', fontsize=8)
            ax.set_ylabel('Observed', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        for i in range(len(token_ids), len(axes)):
            axes[i].axis('off')

        fig.suptitle('Per-Disease Calibration Curves', fontsize=14,
                     fontweight='bold')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved -> {save_path}")
        plt.show()
        return fig

    # ==================================================================
    # AUC distribution
    # ==================================================================
    def plot_auc_distribution(self, save_path=None):
        """Histogram of ROC-AUC and AP values across all qualifying diseases."""
        if not self._ran:
            print("Call .run() first"); return

        aucs = [d['auc'] for d in self.results.values()]
        aps = [d['ap'] for d in self.results.values()]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(aucs, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
        axes[0].axvline(np.median(aucs), color='red', ls='--', lw=1.5,
                        label=f'Median: {np.median(aucs):.3f}')
        axes[0].set_xlabel('ROC-AUC'); axes[0].set_ylabel('# diseases')
        axes[0].set_title(f'ROC-AUC Distribution ({len(aucs)} diseases)',
                         fontweight='bold')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].hist(aps, bins=20, color='darkorange', edgecolor='black', alpha=0.8)
        axes[1].axvline(np.median(aps), color='red', ls='--', lw=1.5,
                        label=f'Median: {np.median(aps):.3f}')
        axes[1].set_xlabel('Average Precision'); axes[1].set_ylabel('# diseases')
        axes[1].set_title(f'AP Distribution ({len(aps)} diseases)',
                         fontweight='bold')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved -> {save_path}")
        plt.show()
        return fig

    # ==================================================================
    # Results table
    # ==================================================================
    def get_results_table(self) -> pd.DataFrame:
        """Return a sorted DataFrame of per-disease metrics."""
        if not self._ran:
            print("Call .run() first")
            return pd.DataFrame()

        rows = []
        for tid, info in self.results.items():
            rows.append({
                'token_id': tid,
                'name': info['name'],
                'chapter': info.get('chapter', ''),
                'color': info.get('color', '#999999'),
                'n_cases': info['n_cases'],
                'prevalence': f"{info['prevalence']:.2%}",
                'roc_auc': round(info['auc'], 4),
                'avg_precision': round(info['ap'], 4),
            })
        df = pd.DataFrame(rows).sort_values('roc_auc', ascending=False)
        return df.reset_index(drop=True)

    # ==================================================================
    # Delphi Figure 2b: AUC vs number of disease occurrences (log scale)
    # ==================================================================
    def plot_auc_vs_occurrences(self, save_path=None):
        """
        Scatter: AUC (y) vs number of occurrences (x, log scale),
        colored by ICD-10 chapter.  Reproduces Delphi Fig. 2b.
        """
        if not self._ran:
            print("Call .run() first"); return

        fig, ax = plt.subplots(figsize=(8, 6))

        for tid, info in self.results.items():
            ax.scatter(info['n_cases'], info['auc'],
                       c=info.get('color', '#999999'),
                       s=25, alpha=0.7, edgecolors='none')

        ax.set_xscale('log')
        ax.set_xlabel('Number of disease occurrences')
        ax.set_ylabel('AUC')
        ax.set_ylim(0, 1.02)
        ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.set_title('AUC vs Disease Occurrences', fontweight='bold')
        ax.grid(True, alpha=0.2)

        # Legend by chapter
        chapter_colors = {}
        for info in self.results.values():
            ch = info.get('chapter', '')
            co = info.get('color', '#999999')
            if ch and ch not in chapter_colors:
                chapter_colors[ch] = co
        if chapter_colors:
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=c, markersize=6, label=ch)
                       for ch, c in sorted(chapter_colors.items())]
            ax.legend(handles=handles, fontsize=6, ncol=2,
                      loc='lower right', framealpha=0.8,
                      title='ICD-10 Chapter', title_fontsize=7)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved -> {save_path}")
        plt.show()
        return fig

    # ==================================================================
    # Delphi Figure 2d: AUC by Sex
    # ==================================================================
    def plot_auc_by_sex(self, save_path=None):
        """
        Box plot of AUC aggregated by biological sex.
        Reproduces Delphi Fig. 2d.
        """
        if not self._ran or self.patient_sex is None:
            print("Call .run() first"); return

        female_mask = self.patient_sex == 1
        male_mask = self.patient_sex == 2
        n_female = female_mask.sum()
        n_male = male_mask.sum()
        print(f"[INFO] Sex split: Male={n_male}, Female={n_female}, "
              f"Unknown={(self.patient_sex == 0).sum()}")

        if n_female < 10 or n_male < 10:
            print("[WARN] Not enough patients per sex for meaningful analysis")

        male_aucs, female_aucs = [], []
        for tid, info in self.results.items():
            y_true_m = self.observed[male_mask, tid]
            y_score_m = self.pred_rates[male_mask, tid]
            y_true_f = self.observed[female_mask, tid]
            y_score_f = self.pred_rates[female_mask, tid]

            # Need ≥ 2 classes present
            if y_true_m.sum() >= 5 and y_true_m.sum() < len(y_true_m):
                try:
                    male_aucs.append(roc_auc_score(y_true_m, y_score_m))
                except ValueError:
                    pass
            if y_true_f.sum() >= 5 and y_true_f.sum() < len(y_true_f):
                try:
                    female_aucs.append(roc_auc_score(y_true_f, y_score_f))
                except ValueError:
                    pass

        fig, ax = plt.subplots(figsize=(5, 6))

        data_list = [male_aucs, female_aucs]
        labels = [f'Male\n(n={len(male_aucs)})', f'Female\n(n={len(female_aucs)})']
        colors = ['#6baed6', '#fc9272']

        bp = ax.boxplot(data_list, positions=[0, 1], widths=0.5,
                        patch_artist=True, whis=[2.5, 97.5],
                        showfliers=True,
                        flierprops=dict(markersize=3, alpha=0.5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('AUC')
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.set_title('AUC by Sex', fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')

        # Annotate medians
        for i, aucs in enumerate(data_list):
            if aucs:
                med = np.median(aucs)
                ax.text(i, med + 0.02, f'{med:.3f}', ha='center',
                        fontsize=9, fontweight='bold')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved -> {save_path}")
        plt.show()
        return fig

    # ==================================================================
    # Delphi Figure 2e: AUC by follow-up time
    # ==================================================================
    def plot_auc_by_followup(self, time_bins_months=None, save_path=None):
        """
        Box plot of AUC for different follow-up time windows.
        Reproduces Delphi Fig. 2e.

        For each time window, only count a disease as positive if it
        occurred within that window after the cutoff.

        Parameters
        ----------
        time_bins_months : list of int
            Follow-up windows in months. Default: [6, 12, 24, 36]
        """
        if not self._ran:
            print("Call .run() first"); return

        if time_bins_months is None:
            time_bins_months = [6, 12, 24, 36]

        config = getattr(self.model, 'config', None)
        data_vocab = int(getattr(config, 'data_vocab_size', 1290))

        # For each time window, recompute observed matrix using event ages
        # We need patient IDs to look up test_event_ages
        # Rebuild patient_id list in same order as pred_rates
        indices = self.valid_val_indices[:len(self.pred_rates)]
        patient_ids = [self.val_pid_map[vid] for vid in indices
                       if vid in self.val_pid_map]
        # Align with actual collected data
        if len(patient_ids) != len(self.pred_rates):
            # Rebuild more carefully
            patient_ids = []
            cutoffs = self.patient_cutoff_age
            for idx_i, vid in enumerate(indices):
                if vid in self.val_pid_map:
                    patient_ids.append(self.val_pid_map[vid])
            patient_ids = patient_ids[:len(self.pred_rates)]

        window_aucs = {}
        for months in time_bins_months:
            window_days = months * 30.44  # approximate

            # Build observed matrix for this time window
            obs_window = np.zeros_like(self.observed)
            for pat_i, pid in enumerate(patient_ids):
                if pat_i >= len(self.patient_cutoff_age):
                    break
                cutoff = self.patient_cutoff_age[pat_i]
                event_ages = self.test_event_ages.get(pid, {})
                for tok, age in event_ages.items():
                    if 0 <= tok < data_vocab:
                        # Event within window?
                        if age <= cutoff + window_days:
                            obs_window[pat_i, tok] = 1.0

            # Compute AUC for each disease
            aucs = []
            for tid in self.results:
                y_true = obs_window[:, tid]
                y_score = self.pred_rates[:, tid]
                n_pos = y_true.sum()
                if n_pos < 5 or n_pos >= len(y_true):
                    continue
                try:
                    aucs.append(roc_auc_score(y_true, y_score))
                except ValueError:
                    continue
            window_aucs[months] = aucs

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        data_list = [window_aucs[m] for m in time_bins_months]
        labels = [f'{m}m\n(n={len(window_aucs[m])})' for m in time_bins_months]

        bp = ax.boxplot(data_list, positions=range(len(data_list)),
                        widths=0.5, patch_artist=True,
                        whis=[2.5, 97.5], showfliers=True,
                        flierprops=dict(markersize=3, alpha=0.5))
        for patch in bp['boxes']:
            patch.set_facecolor('#a1d99b'); patch.set_alpha(0.7)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_xlabel('Follow-up time')
        ax.set_ylabel('AUC')
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.set_title('AUC by Follow-up Time Window', fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')

        # Annotate medians
        for i, m in enumerate(time_bins_months):
            aucs = window_aucs[m]
            if aucs:
                med = np.median(aucs)
                ax.text(i, med + 0.02, f'{med:.3f}', ha='center',
                        fontsize=9, fontweight='bold')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved -> {save_path}")
        plt.show()
        return fig

    # ==================================================================
    # Delphi Figure 2 combined (b + d + e + calibration)
    # ==================================================================
    def plot_delphi_figure2(self, time_bins_months=None, save_path=None):
        """
        Combined multi-panel figure inspired by Delphi Fig. 2:
          (a) AUC vs occurrences (log scale, colored by chapter)
          (b) AUC by sex (boxplot)
          (c) AUC by follow-up time window (boxplot)
          (d) Calibration curve (decile binning)

        Returns the figure.
        """
        if not self._ran:
            print("Call .run() first"); return

        if time_bins_months is None:
            time_bins_months = [6, 12, 24, 36]

        config = getattr(self.model, 'config', None)
        data_vocab = int(getattr(config, 'data_vocab_size', 1290))

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

        # ─── (a) AUC vs occurrences ───
        ax_a = fig.add_subplot(gs[0, 0])
        for tid, info in self.results.items():
            ax_a.scatter(info['n_cases'], info['auc'],
                         c=info.get('color', '#999999'),
                         s=20, alpha=0.7, edgecolors='none')
        ax_a.set_xscale('log')
        ax_a.set_xlabel('Number of disease occurrences')
        ax_a.set_ylabel('AUC')
        ax_a.set_ylim(0, 1.02)
        ax_a.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax_a.set_title('(a) AUC vs Disease Occurrences', fontweight='bold')
        ax_a.grid(True, alpha=0.2)

        # Chapter legend
        chapter_colors = {}
        for info in self.results.values():
            ch = info.get('chapter', '')
            co = info.get('color', '#999999')
            if ch and ch not in chapter_colors and ch not in ('Technical', 'Sex'):
                chapter_colors[ch] = co
        if chapter_colors:
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=c, markersize=5, label=ch)
                       for ch, c in sorted(chapter_colors.items())]
            ax_a.legend(handles=handles, fontsize=5, ncol=2,
                        loc='lower right', framealpha=0.8)

        # ─── (b) AUC by sex ───
        ax_b = fig.add_subplot(gs[0, 1])
        female_mask = self.patient_sex == 1
        male_mask = self.patient_sex == 2

        male_aucs, female_aucs = [], []
        for tid in self.results:
            y_true_m = self.observed[male_mask, tid]
            y_score_m = self.pred_rates[male_mask, tid]
            y_true_f = self.observed[female_mask, tid]
            y_score_f = self.pred_rates[female_mask, tid]
            if y_true_m.sum() >= 5 and y_true_m.sum() < len(y_true_m):
                try: male_aucs.append(roc_auc_score(y_true_m, y_score_m))
                except ValueError: pass
            if y_true_f.sum() >= 5 and y_true_f.sum() < len(y_true_f):
                try: female_aucs.append(roc_auc_score(y_true_f, y_score_f))
                except ValueError: pass

        bp_sex = ax_b.boxplot([male_aucs, female_aucs], positions=[0, 1],
                              widths=0.5, patch_artist=True, whis=[2.5, 97.5],
                              showfliers=True,
                              flierprops=dict(markersize=3, alpha=0.5))
        for patch, color in zip(bp_sex['boxes'], ['#6baed6', '#fc9272']):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax_b.set_xticks([0, 1])
        ax_b.set_xticklabels([f'Male\n(n={len(male_aucs)})',
                               f'Female\n(n={len(female_aucs)})'], fontsize=10)
        ax_b.set_ylabel('AUC')
        ax_b.set_ylim(0, 1.05)
        ax_b.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax_b.set_title('(b) AUC by Sex', fontweight='bold')
        ax_b.grid(True, alpha=0.2, axis='y')
        for i, aucs in enumerate([male_aucs, female_aucs]):
            if aucs:
                med = np.median(aucs)
                ax_b.text(i, med + 0.02, f'{med:.3f}', ha='center',
                          fontsize=9, fontweight='bold')

        # ─── (c) AUC by follow-up time ───
        ax_c = fig.add_subplot(gs[1, 0])

        indices = self.valid_val_indices[:len(self.pred_rates)]
        patient_ids = [self.val_pid_map[vid] for vid in indices
                       if vid in self.val_pid_map][:len(self.pred_rates)]

        window_aucs = {}
        for months in time_bins_months:
            window_days = months * 30.44
            obs_window = np.zeros_like(self.observed)
            for pat_i, pid in enumerate(patient_ids):
                if pat_i >= len(self.patient_cutoff_age):
                    break
                cutoff = self.patient_cutoff_age[pat_i]
                event_ages = self.test_event_ages.get(pid, {})
                for tok, age in event_ages.items():
                    if 0 <= tok < data_vocab and age <= cutoff + window_days:
                        obs_window[pat_i, tok] = 1.0

            aucs_w = []
            for tid in self.results:
                y_true = obs_window[:, tid]
                y_score = self.pred_rates[:, tid]
                if y_true.sum() >= 5 and y_true.sum() < len(y_true):
                    try: aucs_w.append(roc_auc_score(y_true, y_score))
                    except ValueError: pass
            window_aucs[months] = aucs_w

        data_list_fu = [window_aucs[m] for m in time_bins_months]
        labels_fu = [f'{m}m\n(n={len(window_aucs[m])})' for m in time_bins_months]

        bp_fu = ax_c.boxplot(data_list_fu, positions=range(len(data_list_fu)),
                             widths=0.5, patch_artist=True, whis=[2.5, 97.5],
                             showfliers=True,
                             flierprops=dict(markersize=3, alpha=0.5))
        for patch in bp_fu['boxes']:
            patch.set_facecolor('#a1d99b'); patch.set_alpha(0.7)
        ax_c.set_xticks(range(len(labels_fu)))
        ax_c.set_xticklabels(labels_fu, fontsize=10)
        ax_c.set_xlabel('Follow-up time')
        ax_c.set_ylabel('AUC')
        ax_c.set_ylim(0, 1.05)
        ax_c.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax_c.set_title('(c) AUC by Follow-up Time', fontweight='bold')
        ax_c.grid(True, alpha=0.2, axis='y')
        for i, m in enumerate(time_bins_months):
            if window_aucs[m]:
                med = np.median(window_aucs[m])
                ax_c.text(i, med + 0.02, f'{med:.3f}', ha='center',
                          fontsize=9, fontweight='bold')

        # ─── (d) Calibration curve ───
        ax_d = fig.add_subplot(gs[1, 1])
        all_preds, all_labels = [], []
        for token_id in self.results:
            all_preds.append(self.pred_rates[:, token_id])
            all_labels.append(self.observed[:, token_id])

        if all_preds:
            all_p = np.concatenate(all_preds)
            all_l = np.concatenate(all_labels)
            n_bins = 10
            edges = np.percentile(all_p, np.linspace(0, 100, n_bins + 1))
            edges[-1] += 1e-10
            bp_cal, bo_cal = [], []
            for i in range(n_bins):
                m = (all_p >= edges[i]) & (all_p < edges[i + 1])
                if m.sum() > 0:
                    bp_cal.append(all_p[m].mean())
                    bo_cal.append(all_l[m].mean())

            if bp_cal:
                bp_cal, bo_cal = np.array(bp_cal), np.array(bo_cal)
                ax_d.plot(bp_cal, bo_cal, 'o-', color='steelblue',
                          markersize=8, linewidth=2, label='Model')
                lim = max(bp_cal.max(), bo_cal.max()) * 1.1
                ax_d.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.6,
                          label='Perfect')

        ax_d.set_xlabel('Predicted rate')
        ax_d.set_ylabel('Observed rate')
        ax_d.set_title('(d) Calibration', fontweight='bold')
        ax_d.legend(fontsize=9); ax_d.grid(True, alpha=0.3)

        fig.suptitle(f'Disease Prediction Performance '
                     f'({len(self.results)} diseases, '
                     f'>= {self.min_cases} cases)',
                     fontsize=16, fontweight='bold')

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK]  Saved -> {save_path}")
        plt.show()
        return fig
