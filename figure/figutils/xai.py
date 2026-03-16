"""
xai.py - Attention, Feature Importance & Prediction Pathway
=============================================================
Provides AdvancedXAIAnalyzer class and run_complete_xai_analysis helper.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class AdvancedXAIAnalyzer:
    """
    Advanced XAI analysis for interpretability
    """
    
    def __init__(self, model, tokenizer=None, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
    def extract_attention_weights(self, batch, layer_idx=-1):
        """
        Extract attention weights from CompositeDelphi model
        
        Args:
            batch: Input batch (x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages)
            layer_idx: Which transformer layer to extract from (-1 for last)
        
        Returns:
            attention_weights: [batch, layers, heads, seq_len, seq_len] or None
        """
        if len(batch) != 8:
            print(f"Warning: Expected 8-element batch, got {len(batch)}")
            return None
        
        x_data, x_shift, x_total, x_ages = batch[:4]
        x_data = x_data.to(self.device)
        x_shift = x_shift.to(self.device)
        x_total = x_total.to(self.device)
        x_ages = x_ages.to(self.device)
        
        # Forward pass WITHOUT targets → model_v2 collects attention weights
        # (model_v2 skips att collection when is_training=True i.e. targets_data is not None)
        with torch.no_grad():
            logits, _, att = self.model(
                x_data, x_shift, x_total, x_ages,
            )
        
        # att is a stacked tensor of attention weights from all layers
        # Shape: [num_layers, batch, num_heads, seq_len, seq_len]
        if att is not None and layer_idx < att.shape[0]:
            return att[layer_idx]  # Return specific layer
        
        return att
    
    def visualize_attention_patterns(self, dataloader, num_samples=5, 
                                    save_path=None):
        """
        Visualize attention patterns across multiple samples
        Shows what the model focuses on when making predictions
        """
        fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
        fig.suptitle('Attention Pattern Analysis', fontsize=16, fontweight='bold')
        
        samples_processed = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if samples_processed >= num_samples:
                break
            
            # Extract attention
            attn = self.extract_attention_weights(batch)
            
            if attn is None:
                continue
            
            # Process attention
            if isinstance(attn, torch.Tensor):
                attn = attn.cpu().numpy()
            
            # Get first sample in batch
            if len(attn.shape) == 4:
                attn_sample = attn[0]  # [heads, seq_len, seq_len]
            else:
                attn_sample = attn
            
            # Average across heads
            attn_avg = attn_sample.mean(axis=0)
            
            # Row 1: Full attention matrix
            im1 = axes[0, samples_processed].imshow(attn_avg, cmap='viridis', aspect='auto')
            axes[0, samples_processed].set_title(f'Sample {samples_processed + 1}')
            axes[0, samples_processed].set_xlabel('Key Position')
            axes[0, samples_processed].set_ylabel('Query Position')
            plt.colorbar(im1, ax=axes[0, samples_processed])
            
            # Row 2: Attention focus (sum over queries)
            attn_focus = attn_avg.sum(axis=0)
            axes[1, samples_processed].bar(range(len(attn_focus)), attn_focus, color='steelblue')
            axes[1, samples_processed].set_xlabel('Token Position')
            axes[1, samples_processed].set_ylabel('Total Attention')
            axes[1, samples_processed].set_title('Attention Distribution')
            
            samples_processed += 1
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention patterns saved to {save_path}")
        plt.show()
    
    def compute_token_importance_shap(self, dataloader, background_size=50,
                                     test_size=10, save_path=None):
        """
        Use SHAP to compute token-level importance
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available")
            return None
        
        print("Computing token-level SHAP values...")
        
        # Collect background data (CompositeDelphi format)
        background_samples = []
        for i, batch in enumerate(dataloader):
            if i >= background_size:
                break
            if len(batch) == 8:
                background_samples.append(batch[0].cpu())  # x_data
        
        # Pad and concatenate
        if not background_samples:
            print("No background samples collected")
            return None
        
        max_len = max(s.shape[1] for s in background_samples)
        padded_bg = []
        for s in background_samples:
            if s.shape[1] < max_len:
                pad_size = max_len - s.shape[1]
                padding = torch.zeros(s.shape[0], pad_size, dtype=s.dtype)
                s = torch.cat([s, padding], dim=1)
            padded_bg.append(s)
        
        background_data = torch.cat(padded_bg, dim=0)[:background_size]
        
        # Model wrapper for shift prediction (CompositeDelphi format)
        def predict_shift(x):
            """x is numpy array of shape (N, T)"""
            x_data = torch.tensor(x, dtype=torch.long).to(self.device)
            batch_size = x_data.shape[0]
            seq_len = x_data.shape[1]
            
            # Create dummy inputs for other fields
            x_shift = torch.zeros_like(x_data)
            x_total = torch.zeros_like(x_data)
            x_ages = torch.zeros(batch_size, seq_len, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                try:
                    logits, _, _ = self.model(
                        x_data, x_shift, x_total, x_ages,
                        targets_data=None,
                        targets_shift=None,
                        targets_total=None,
                        targets_age=None
                    )
                    
                    # Get shift predictions
                    if 'shift_drug_cond' in logits:
                        shift_logits = logits['shift_drug_cond']
                    else:
                        shift_logits = logits['shift']
                    
                    # Return probabilities for each class
                    probs = torch.softmax(shift_logits, dim=-1)
                    # Average over sequence dimension
                    probs = probs.mean(dim=1)
                    return probs.cpu().numpy()
                except Exception as e:
                    print(f"Error in predict_shift: {e}")
                    return None
        
        # Collect test samples
        test_samples = []
        for i, batch in enumerate(dataloader):
            if i >= test_size:
                break
            if len(batch) == 8:
                test_samples.append(batch[0].cpu())  # x_data
        
        if not test_samples:
            print("No test samples collected")
            return None
        
        # Pad test samples
        max_len_test = max(s.shape[1] for s in test_samples)
        max_len_all = max(max_len, max_len_test)
        
        # Pad background to match max
        if background_data.shape[1] < max_len_all:
            pad_size = max_len_all - background_data.shape[1]
            padding = torch.zeros(background_data.shape[0], pad_size, dtype=background_data.dtype)
            background_data = torch.cat([background_data, padding], dim=1)
        
        # Pad test samples
        padded_test = []
        for s in test_samples:
            if s.shape[1] < max_len_all:
                pad_size = max_len_all - s.shape[1]
                padding = torch.zeros(s.shape[0], pad_size, dtype=s.dtype)
                s = torch.cat([s, padding], dim=1)
            padded_test.append(s)
        
        test_data = torch.cat(padded_test, dim=0)[:test_size]
        
        try:
            # Create explainer
            # Use a smaller background set for faster computation
            explainer = shap.KernelExplainer(
                predict_shift,
                background_data.numpy()[:10]
            )
            
            # Compute SHAP values
            shap_values = explainer.shap_values(test_data.numpy()[:5], nsamples=100)
            
            # Visualize
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Token-Level SHAP Analysis', fontsize=16, fontweight='bold')
            
            # SHAP values for each class
            class_names = ['Decrease', 'Maintain', 'Increase']
            
            for i, class_name in enumerate(class_names):
                # Summary plot for this class
                if isinstance(shap_values, list) and i < len(shap_values):
                    shap_class = shap_values[i]
                    
                    # Calculate mean absolute SHAP value per token
                    mean_shap = np.abs(shap_class).mean(axis=0)
                    
                    axes[i].bar(range(len(mean_shap)), mean_shap, color='coral')
                    axes[i].set_title(f'{class_name} Class')
                    axes[i].set_xlabel('Token Position')
                    axes[i].set_ylabel('Mean |SHAP Value|')
                    axes[i].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Token importance saved to {save_path}")
            plt.show()
            
            return shap_values
            
        except Exception as e:
            print(f"Error computing SHAP: {e}")
            return None
    
    def visualize_prediction_pathway(self, sample_batch, sample_idx=0,
                                    save_path=None):
        """
        Visualize the complete prediction pathway for a single patient
        Shows: input -> attention -> prediction
        Supports CompositeDelphi model with drug conditioning
        """
        if len(sample_batch) != 8:
            print(f"Warning: Expected 8-element batch, got {len(sample_batch)}")
            return
        
        x_data, x_shift, x_total, x_ages = sample_batch[:4]
        y_data, y_shift, y_total, y_ages = sample_batch[4:]
        
        # Get single sample
        x_data = x_data[sample_idx:sample_idx+1].to(self.device)
        x_shift = x_shift[sample_idx:sample_idx+1].to(self.device)
        x_total = x_total[sample_idx:sample_idx+1].to(self.device)
        x_ages = x_ages[sample_idx:sample_idx+1].to(self.device)
        y_data = y_data[sample_idx:sample_idx+1].to(self.device)
        y_shift = y_shift[sample_idx:sample_idx+1].to(self.device)
        y_total = y_total[sample_idx:sample_idx+1].to(self.device)
        y_ages = y_ages[sample_idx:sample_idx+1].to(self.device)
        
        # Get predictions (with targets for drug_cond)
        with torch.no_grad():
            logits, _, _ = self.model(
                x_data, x_shift, x_total, x_ages,
                targets_data=y_data,
                targets_shift=y_shift,
                targets_total=y_total,
                targets_age=y_ages
            )
            # Separate pass without targets to get attention weights
            _, _, att = self.model(
                x_data, x_shift, x_total, x_ages,
            )
        
        # Extract predictions from dict
        data_logits = logits['data']  # (B, T, data_vocab_size)
        
        # Use drug-conditioned predictions if available
        if 'shift_drug_cond' in logits:
            shift_logits = logits['shift_drug_cond']
        else:
            shift_logits = logits['shift']
        
        if 'total_drug_cond' in logits:
            total_pred = logits['total_drug_cond']
        else:
            total_pred = logits['total']
        
        # Get predictions for last position
        shift_probs = torch.softmax(shift_logits[:, -1, :], dim=-1)
        shift_pred = torch.argmax(shift_probs, dim=-1)
        total_pred_val = total_pred[:, -1]
        
        # Create visualization
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        fig.suptitle(f'Prediction Pathway Analysis - Patient {sample_idx}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Input sequences
        ax1 = fig.add_subplot(gs[0, :])
        data_np = x_data.cpu().numpy()[0]
        shift_np = x_shift.cpu().numpy()[0]
        total_np = x_total.cpu().numpy()[0]
        
        x_pos = range(len(data_np))
        ax1.bar(x_pos, data_np, alpha=0.4, color='steelblue', label='DATA')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x_pos, shift_np, 'o-', color='orange', label='SHIFT', markersize=4)
        ax1_twin.plot(x_pos, total_np, 's-', color='green', label='TOTAL', markersize=4, alpha=0.6)
        
        ax1.set_title('Input Token Sequences', fontweight='bold')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('DATA Token ID', color='steelblue')
        ax1_twin.set_ylabel('SHIFT/TOTAL Values', color='orange')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Attention pattern (if available)
        if att is not None:
            ax2 = fig.add_subplot(gs[1, 0])
            # att shape: [num_layers, batch, num_heads, seq_len, seq_len]
            att_np = att.cpu().numpy() if isinstance(att, torch.Tensor) else att
            
            # Average across layers and heads for visualization
            if len(att_np.shape) == 5:
                attn_avg = att_np[:, sample_idx, :, :, :].mean(axis=(0, 1))  # Average layers and heads
            elif len(att_np.shape) == 4:
                attn_avg = att_np[sample_idx].mean(axis=0)
            else:
                attn_avg = att_np.mean(axis=0)
            
            im = ax2.imshow(attn_avg, cmap='YlOrRd', aspect='auto')
            ax2.set_title('Attention Heatmap (Avg)', fontweight='bold')
            ax2.set_xlabel('Key Position')
            ax2.set_ylabel('Query Position')
            plt.colorbar(im, ax=ax2)
        
        # 3. Shift prediction probabilities
        ax3 = fig.add_subplot(gs[1, 1])
        shift_probs_np = shift_probs.cpu().numpy()[0]
        
        # Auto-detect: binary (model_v2) vs multi-class
        if len(shift_probs_np) == 2:
            # Binary: Maintain vs Changed
            class_names = ['Maintain', 'Changed']
            colors = ['lightgreen', 'salmon']
        else:
            # Multi-class (5): 0=Pad, 1=Non-Drug, 2=Decrease, 3=Maintain, 4=Increase
            class_names = ['Pad', 'Non-Drug', 'Decrease', 'Maintain', 'Increase']
            class_names = class_names[:len(shift_probs_np)]
            colors = ['gray', 'lightgray', 'salmon', 'lightgreen', 'orange'][:len(shift_probs_np)]
        
        bars = ax3.bar(class_names[:len(shift_probs_np)], shift_probs_np, color=colors, alpha=0.7)
        ax3.set_title('Shift Prediction Probabilities', fontweight='bold')
        ax3.set_ylabel('Probability')
        ax3.set_ylim([0, 1])
        ax3.tick_params(axis='x', rotation=45)
        
        # Highlight predicted class
        predicted_class = shift_pred.cpu().numpy()[0]
        if 0 <= predicted_class < len(bars):
            bars[predicted_class].set_edgecolor('red')
            bars[predicted_class].set_linewidth(3)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 4. Total prediction
        ax4 = fig.add_subplot(gs[1, 2])
        total_pred_np = total_pred_val.cpu().numpy()[0]
        total_true_np = y_total.cpu().numpy()[0, -1]  # Last position target
        
        ax4.bar(['True', 'Predicted'], 
               [total_true_np, total_pred_np],
               color=['steelblue', 'coral'], alpha=0.7)
        ax4.set_title('Total Dosage Prediction', fontweight='bold')
        ax4.set_ylabel('Dosage Amount (Raw Scale)')
        
        # Add values on bars
        for i, v in enumerate([total_true_np, total_pred_np]):
            ax4.text(i, v, f'{v:.1f}',
                    ha='center', va='bottom', fontweight='bold')
        
        # 5. Summary box
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        summary_text = "Prediction Summary:\n\n"
        
        if len(shift_probs_np) == 2:
            class_map = {0: 'Maintain', 1: 'Changed'}
        else:
            class_map = {0: 'Pad', 1: 'Non-Drug', 2: 'Decrease', 3: 'Maintain', 4: 'Increase'}
        predicted_shift_class = predicted_class
        true_shift_class = y_shift.cpu().numpy()[0, -1]
        
        summary_text += f"Shift Prediction:\n"
        summary_text += f"  Predicted: {class_map.get(predicted_shift_class, 'Unknown')} (class {predicted_shift_class})\n"
        summary_text += f"  True: {class_map.get(true_shift_class, 'Unknown')} (class {true_shift_class})\n"
        summary_text += f"  Confidence: {shift_probs_np[predicted_shift_class]:.2%}\n"
        summary_text += f"  Correct: {'✓' if predicted_shift_class == true_shift_class else '✗'}\n\n"
        
        summary_text += f"Total Dosage Prediction:\n"
        summary_text += f"  Predicted: {total_pred_np:.2f}\n"
        summary_text += f"  True: {total_true_np:.2f}\n"
        
        if total_true_np > 0:
            error = abs(total_pred_np - total_true_np)
            summary_text += f"  Absolute Error: {error:.2f}\n"
            summary_text += f"  Relative Error: {(error/total_true_np)*100:.1f}%\n"
        
        # Drug conditioning info
        if 'shift_drug_cond' in logits or 'total_drug_cond' in logits:
            summary_text += f"\n[Drug-Conditioned Predictions Used]\n"
        
        ax5.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction pathway saved to {save_path}")
        plt.show()
    
    def create_feature_importance_analysis(self, dataloader, save_path=None, 
                                           token_type='all', drug_token_range=(1277, 1287),
                                           disease_token_range=(21, 1276)):
        """
        Analyze which features (token positions) are most important
        across the dataset
        
        Args:
            dataloader: DataLoader to analyze
            save_path: Path to save the figure
            token_type: Type of tokens to analyze
                - 'all': All DATA tokens (default)
                - 'drug': Only drug tokens (1277~1287)
                - 'disease': Only disease/diagnosis tokens (21~1276)
            drug_token_range: Tuple of (min, max) for drug tokens
            disease_token_range: Tuple of (min, max) for disease tokens
        """
        print(f"Analyzing feature importance across dataset (token_type='{token_type}')...")
        
        # Collect predictions and analyze patterns
        all_data = []
        all_shift_preds = []
        all_total_preds = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 100:  # Limit for speed
                    break
                
                # CompositeDelphi batch format: (x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages)
                if len(batch) == 8:
                    x_data, x_shift, x_total, x_ages = batch[:4]
                    y_data_b, y_shift_b, y_total_b, y_ages_b = batch[4], batch[5], batch[6], batch[7]
                    x_data = x_data.to(self.device)
                    x_shift = x_shift.to(self.device)
                    x_total = x_total.to(self.device)
                    x_ages = x_ages.to(self.device)
                    y_data_b = y_data_b.to(self.device)
                    y_shift_b = y_shift_b.to(self.device)
                    y_total_b = y_total_b.to(self.device)
                    y_ages_b = y_ages_b.to(self.device)
                    
                    # Forward pass
                    logits, _, _ = self.model(
                        x_data, x_shift, x_total, x_ages,
                        targets_data=y_data_b,
                        targets_shift=y_shift_b,
                        targets_total=y_total_b,
                        targets_age=y_ages_b
                    )
                    
                    # 🔥 Apply token type filter
                    filtered_data = x_data.clone().cpu()
                    
                    if token_type == 'drug':
                        # Keep only drug tokens, mask others to 0
                        drug_mask = (x_data >= drug_token_range[0]) & (x_data <= drug_token_range[1])
                        filtered_data = filtered_data * drug_mask.cpu()
                        
                    elif token_type == 'disease':
                        # Keep only disease tokens, mask others to 0
                        disease_mask = (x_data >= disease_token_range[0]) & (x_data < disease_token_range[1])
                        filtered_data = filtered_data * disease_mask.cpu()
                    
                    # token_type == 'all': no filtering
                    
                    all_data.append(filtered_data)
                    
                    # Extract predictions from dict
                    if 'shift_drug_cond' in logits:
                        shift_logits = logits['shift_drug_cond']
                    else:
                        shift_logits = logits['shift']
                    
                    if 'total_drug_cond' in logits:
                        total_pred = logits['total_drug_cond']
                    else:
                        total_pred = logits['total']
                    
                    all_shift_preds.append(torch.argmax(shift_logits, dim=-1).cpu())
                    all_total_preds.append(total_pred.cpu())
                else:
                    print(f"Warning: Unexpected batch format with {len(batch)} elements")
                    continue
        
        # Concatenate with padding for variable lengths
        # Find max sequence length
        max_len = max(d.shape[1] for d in all_data)
        
        # Pad all tensors
        def pad_to_max(tensor_list, max_len, pad_value=0):
            padded = []
            for t in tensor_list:
                if t.shape[1] < max_len:
                    pad_size = max_len - t.shape[1]
                    if len(t.shape) == 2:  # (B, T)
                        padding = torch.full((t.shape[0], pad_size), pad_value, dtype=t.dtype)
                    else:  # Shouldn't happen but just in case
                        padding = torch.zeros(t.shape[0], pad_size, dtype=t.dtype)
                    t = torch.cat([t, padding], dim=1)
                padded.append(t)
            return padded
        
        all_data = pad_to_max(all_data, max_len, 0)
        all_shift_preds = pad_to_max(all_shift_preds, max_len, 0)
        all_total_preds = pad_to_max(all_total_preds, max_len, 0)
        
        all_data = torch.cat(all_data, dim=0).numpy()
        all_shift_preds = torch.cat(all_shift_preds, dim=0).numpy()
        all_total_preds = torch.cat(all_total_preds, dim=0).numpy()
        
        # Calculate statistics for filtered tokens
        nonzero_mask = all_data > 0
        nonzero_count = nonzero_mask.sum()
        total_count = all_data.size
        coverage = (nonzero_count / total_count) * 100
        
        print(f"  Token coverage: {nonzero_count:,} / {total_count:,} ({coverage:.1f}%)")
        if nonzero_count > 0:
            print(f"  Token value range: {all_data[nonzero_mask].min():.0f} ~ {all_data[nonzero_mask].max():.0f}")
            print(f"  Mean token value: {all_data[nonzero_mask].mean():.1f}")
        
        # Analyze correlation between token positions and predictions
        seq_len = all_data.shape[1]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Update title with token type
        token_type_label = {
            'all': 'All Tokens',
            'drug': 'Drug Tokens Only',
            'disease': 'Disease Tokens Only'
        }
        fig.suptitle(f'Feature Importance Analysis ({token_type_label[token_type]})', 
                     fontsize=16, fontweight='bold')
        
        # 1. Token value distribution by position
        ax1 = axes[0, 0]
        
        # Only calculate stats for non-zero tokens (skip masked positions)
        position_means = []
        position_stds = []
        for pos in range(seq_len):
            pos_data = all_data[:, pos]
            pos_data_nonzero = pos_data[pos_data > 0]
            if len(pos_data_nonzero) > 0:
                position_means.append(pos_data_nonzero.mean())
                position_stds.append(pos_data_nonzero.std())
            else:
                position_means.append(0)
                position_stds.append(0)
        
        position_means = np.array(position_means)
        position_stds = np.array(position_stds)
        
        ax1.fill_between(range(seq_len), 
                         position_means - position_stds,
                         position_means + position_stds,
                         alpha=0.3, color='steelblue')
        ax1.plot(range(seq_len), position_means, color='darkblue', linewidth=2)
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Mean Token Value (non-zero)')
        ax1.set_title('Token Value Distribution by Position')
        ax1.grid(True, alpha=0.3)
        
        # 2. Position importance for shift prediction
        ax2 = axes[0, 1]
        
        # Calculate variance for non-zero tokens only
        position_importance = []
        for pos in range(seq_len):
            pos_data = all_data[:, pos]
            pos_data_nonzero = pos_data[pos_data > 0]
            if len(pos_data_nonzero) > 10:  # Need enough samples
                variance = pos_data_nonzero.var()
            else:
                variance = 0
            position_importance.append(variance)
        
        ax2.bar(range(seq_len), position_importance, color='coral', alpha=0.7)
        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Token Variance (non-zero)')
        ax2.set_title('Position Variability (Proxy for Importance)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Shift prediction distribution by position
        ax3 = axes[1, 0]
        
        # Auto-detect: binary (0/1) vs multi-class (2/3/4)
        unique_preds = np.unique(all_shift_preds[all_shift_preds >= 0])
        if unique_preds.max() <= 1:
            # Binary model: 0=Maintain, 1=Changed
            shift_classes = [0, 1]
            shift_labels = ['Maintain', 'Changed']
            shift_colors = ['lightgreen', 'salmon']
        else:
            # Multi-class: 2=Decrease, 3=Maintain, 4=Increase (or 1,2,3 if no token shift)
            if np.isin([2, 3, 4], unique_preds).any():
                shift_classes = [2, 3, 4]
            else:
                shift_classes = [1, 2, 3]
            shift_labels = ['Decrease', 'Maintain', 'Increase']
            shift_colors = ['salmon', 'lightgreen', 'orange']
        
        shift_by_position = []
        for pos in range(min(seq_len, 20)):
            preds_col = all_shift_preds[:, 0]
            valid_mask = np.isin(preds_col, shift_classes)
            if valid_mask.sum() > 0:
                counts_per_class = [np.sum(preds_col[valid_mask] == c) for c in shift_classes]
                total = sum(counts_per_class)
                shift_by_position.append([c / max(total, 1) for c in counts_per_class])
            else:
                shift_by_position.append([0] * len(shift_classes))
        
        # Plot stacked bar
        if shift_by_position:
            shift_array = np.array(shift_by_position).T
            bottom = np.zeros(len(shift_by_position))
            
            for i, (color, label) in enumerate(zip(shift_colors, shift_labels)):
                ax3.bar(range(len(shift_by_position)), shift_array[i], 
                       bottom=bottom, color=color, alpha=0.7, label=label)
                bottom += shift_array[i]
            
            ax3.set_xlabel('Position (First 20)')
            ax3.set_ylabel('Proportion')
            ax3.set_title('Shift Prediction Distribution by Position')
            ax3.legend()
        
        # 4. Total prediction correlation
        ax4 = axes[1, 1]
        
        # Correlation between token values and total prediction (only non-zero tokens)
        correlations = []
        for pos in range(seq_len):
            # Only use positions with non-zero tokens
            pos_nonzero_mask = all_data[:, pos] > 0
            total_nonzero_mask = all_total_preds[:, 0] > 0
            combined_mask = pos_nonzero_mask & total_nonzero_mask
            
            if combined_mask.sum() > 10:
                corr = np.corrcoef(all_data[combined_mask, pos], 
                                  all_total_preds[combined_mask, 0])[0, 1]
                correlations.append(corr)
            else:
                correlations.append(0)
        
        ax4.bar(range(len(correlations)), correlations, 
               color=['green' if c > 0 else 'red' for c in correlations],
               alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax4.set_xlabel('Token Position')
        ax4.set_ylabel('Correlation with Total Prediction')
        ax4.set_title('Position Correlation with Total Dosage')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance analysis saved to {save_path}")
        plt.show()  # 노트북에서 바로 표시


def run_complete_xai_analysis(model, dataloader, output_dir='./xai_analysis'):
    """
    Run complete XAI analysis pipeline
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Running Complete XAI Analysis...")
    print("=" * 60)
    
    analyzer = AdvancedXAIAnalyzer(model)
    
    # 1. Attention patterns
    print("\n1. Analyzing attention patterns...")
    analyzer.visualize_attention_patterns(
        dataloader, 
        num_samples=5,
        save_path=f'{output_dir}/attention_patterns.png'
    )
    
    # 2. Token importance (SHAP)
    print("\n2. Computing token-level importance...")
    if SHAP_AVAILABLE:
        analyzer.compute_token_importance_shap(
            dataloader,
            save_path=f'{output_dir}/token_importance.png'
        )
    
    # 3. Feature importance
    print("\n3. Analyzing feature importance...")
    analyzer.create_feature_importance_analysis(
        dataloader,
        save_path=f'{output_dir}/feature_importance.png'
    )
    
    # 4. Prediction pathway for sample patients
    print("\n4. Visualizing prediction pathways...")
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Analyze 3 samples
            break
        analyzer.visualize_prediction_pathway(
            batch,
            sample_idx=0,
            save_path=f'{output_dir}/prediction_pathway_sample_{i}.png'
        )
    
    print(f"\n{'='*60}")
    print(f"XAI Analysis Complete!")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("Advanced XAI Analysis for Composite Delphi Model")
    print("=" * 60)
    print("\nThis script provides:")
    print("• Attention pattern visualization")
    print("• SHAP-based token importance")
    print("• Feature importance ranking")
    print("• Patient-level prediction pathways")
    print("\nUsage:")
    print("  run_complete_xai_analysis(model, dataloader)")