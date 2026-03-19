"""
GPT-OSS
- MoE (Mixture of Experts) for domain-specific learning
- Sliding Window Attention for long medical histories
- RoPE (Rotary Position Embedding) 
- AgeEncoding for temporal medical events
- Custom medical loss functions
"""

import math
import inspect
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

import warnings

# Always print (single GPU mode)
def _is_master():
    return True


def _to_dose_model_space(values: torch.Tensor, use_log: bool) -> torch.Tensor:
    """Map raw DOSE values to model space (optionally log1p)."""
    if not use_log:
        return values
    return torch.log1p(torch.clamp(values, min=0.0))


def _from_dose_model_space(values: torch.Tensor, use_log: bool) -> torch.Tensor:
    """Map model-space DOSE values back to raw space."""
    if not use_log:
        return values
    return torch.expm1(values)


def _dose_bounds_for_model_space(dose_min: float, dose_max: float, use_log: bool):
    """Convert raw DOSE bounds to model-space bounds."""
    lo = float(dose_min)
    hi = float(dose_max)
    if not use_log:
        return lo, hi
    lo = max(lo, 0.0)
    hi = max(hi, lo)
    lo = math.log1p(lo)
    hi = math.log1p(hi)
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _transform_targets(values: torch.Tensor, method: str, center: float, scale: float, min_value: float, max_value: float) -> torch.Tensor:
    """Apply configurable label scaling in a numerically safe way."""
    method = str(method).lower()
    if method == 'none':
        return values
    if method in {'zscore', 'robust'}:
        return (values - center) / max(scale, 1e-8)
    if method == 'minmax':
        denom = max(max_value - min_value, 1e-8)
        return (values - min_value) / denom
    raise ValueError(f"Unknown label scaling method: {method}")

def focal_loss_multiclass(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
    ignore_index: Optional[int] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Multi-class focal loss.

    Args:
        logits: (N, C)
        targets: (N,) int64
        gamma: focusing parameter
        alpha: optional per-class weights (C,)
        ignore_index: optional class id to ignore
        reduction: 'mean' | 'sum' | 'none'
    """
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)

    if ignore_index is not None:
        valid = targets != ignore_index
        logits = logits[valid]
        targets = targets[valid]
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

    log_probs = F.log_softmax(logits, dim=-1)  # (N, C)
    log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (N,)
    pt = log_pt.exp()

    focal = (1.0 - pt).clamp(min=0.0, max=1.0).pow(gamma)
    loss = -focal * log_pt

    if alpha is not None:
        alpha_t = alpha.gather(dim=0, index=targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def dice_loss_multiclass(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    eps: float = 1.0,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """
    Multi-class Dice loss that averages only over classes present in targets.
    This is robust to severe class imbalance.
    """
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)

    if ignore_index is not None:
        valid = targets != ignore_index
        logits = logits[valid]
        targets = targets[valid]
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

    num_classes = logits.size(-1)
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(targets.long(), num_classes=num_classes).to(dtype=probs.dtype)

    inter = (probs * one_hot).sum(dim=0)
    denom = probs.sum(dim=0) + one_hot.sum(dim=0)
    dice = (2.0 * inter + eps) / (denom + eps)

    present = one_hot.sum(dim=0) > 0
    if present.any():
        return 1.0 - dice[present].mean()
    return torch.tensor(0.0, device=logits.device)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more efficient than LayerNorm)"""
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding
    
    Args:
        x: (B * n_head, T, head_dim)
        cos, sin: (T, head_dim)
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)  # 각각 (B * n_head, T, head_dim // 2)
    
    # cos, sin을 (1, T, head_dim // 2)로 변환하여 broadcasting 가능하게 함
    half_dim = x1.shape[-1]
    cos = cos[:, :half_dim].unsqueeze(0).to(x.dtype)  # (1, T, head_dim // 2)
    sin = sin[:, :half_dim].unsqueeze(0).to(x.dtype)  # (1, T, head_dim // 2)
    
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE with medical age information"""
    
    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        """
        Args:
            x: input tensor
            seq_len: sequence length
        Returns:
            cos, sin tensors for rotary embedding
        """
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


class AgeEncoding(nn.Module):
    """Delphi's signature age encoding for medical events"""
    
    def __init__(self, config, max_dim: int = 1024):
        super().__init__()
        div_term = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(10000.0) / config.n_embd))
        self.register_buffer('div_term', div_term)
        self.n_embd = config.n_embd
        self.linear = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, age):
        """
        Args:
            age: age tensor in days, shape (B, T)
        Returns:
            age embeddings, shape (B, T, n_embd)
        """
        y = torch.zeros(age.shape[0], age.shape[1], self.n_embd, device=age.device)
        y[..., 0::2] = torch.sin(age.unsqueeze(-1) / 365.25 * self.div_term)
        y[..., 1::2] = torch.cos(age.unsqueeze(-1) / 365.25 * self.div_term)
        y = self.linear(y)
        return y


def swiglu(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
    """SwiGLU activation with optional limiting"""
    alpha = 1.0
    x_glu, x_linear = x.chunk(2, dim=-1)
    if limit > 0:
        x_glu = x_glu.clamp(min=-limit, max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with sliding window support"""
    
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if hasattr(config, 'n_kv_head') else config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.layer_idx = layer_idx
        
        # Sliding window: apply to every other layer
        self.sliding_window = config.sliding_window if (hasattr(config, 'sliding_window') and layer_idx % 2 == 0) else 0
        
        # Q, K, V projections with GQA
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # RoPE
        self.rope = RotaryEmbedding(self.head_dim)
        
        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, age=None, attn_mask=None):
        B, T, C = x.size()
        
        # Compute Q, K, V
        q = self.q_proj(x)  # (B, T, n_embd)
        k = self.k_proj(x)  # (B, T, n_kv_head * head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_head * head_dim)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        
        # Apply RoPE
        cos, sin = self.rope(x, T)
        q_shape = q.shape
        k_shape = k.shape
        q = _apply_rotary_emb(q.reshape(B * self.n_head, T, self.head_dim), cos, sin).reshape(q_shape)
        k = _apply_rotary_emb(k.reshape(B * self.n_kv_head, T, self.head_dim), cos, sin).reshape(k_shape)
        
        # Expand K, V to match number of query heads (GQA)
        n_rep = self.n_head // self.n_kv_head
        if n_rep > 1:
            k = k.unsqueeze(2).repeat(1, 1, n_rep, 1, 1).reshape(B, self.n_head, T, self.head_dim)
            v = v.unsqueeze(2).repeat(1, 1, n_rep, 1, 1).reshape(B, self.n_head, T, self.head_dim)
        
        # Compute attention
        if self.flash and attn_mask is None:
            # Use Flash Attention
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
            att = None
        else:
            # Manual attention with custom mask
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            
            # Causal mask
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att = att.masked_fill(causal_mask.view(1, 1, T, T), float('-inf'))
            
            # Sliding window mask
            if self.sliding_window > 0:
                window_mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=-self.sliding_window).bool()
                att = att.masked_fill(window_mask.view(1, 1, T, T), float('-inf'))
            
            # Custom medical attention mask
            if attn_mask is not None:
                att = att.masked_fill(attn_mask == 0, float('-inf'))
            
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        # Reassemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))
        
        return y, att


class MixtureOfExperts(nn.Module):
    """Lightweight MoE for domain-specific medical knowledge"""
    
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts if hasattr(config, 'num_experts') else 8
        self.experts_per_token = config.experts_per_token if hasattr(config, 'experts_per_token') else 2
        self.n_embd = config.n_embd
        self.intermediate_size = 4 * config.n_embd  # Standard FFN expansion
        
        # Router
        self.gate = nn.Linear(config.n_embd, self.num_experts, bias=False)
        
        # Experts (smaller than in gpt-oss for efficiency)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, self.intermediate_size, bias=config.bias),
                nn.GELU(),
                nn.Linear(self.intermediate_size, config.n_embd, bias=config.bias),
                nn.Dropout(config.dropout)
            )
            for _ in range(self.num_experts)
        ])

    def forward(self, x):
        B, T, C = x.shape
        
        # Route tokens to experts
        router_logits = self.gate(x)  # (B, T, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            router_probs, self.experts_per_token, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)  # Normalize
        
        # ============================================================
        # Load Balancing Auxiliary Loss (Switch Transformer style)
        # Encourages uniform expert utilization
        # L_aux = N * Σ(f_i * P_i)
        #   f_i = fraction of tokens routed to expert i (hard assignment)
        #   P_i = mean routing probability for expert i (soft assignment)
        # ============================================================
        with torch.no_grad():
            # f_i: fraction of tokens dispatched to each expert
            one_hot = F.one_hot(selected_experts, self.num_experts).float()  # (B, T, k, num_experts)
            f = one_hot.sum(dim=(0, 1, 2)) / (B * T * self.experts_per_token)  # (num_experts,)
        
        P = router_probs.mean(dim=(0, 1))  # (num_experts,) - differentiable
        aux_loss = self.num_experts * (f * P).sum()
        
        # Compute expert outputs
        final_output = torch.zeros_like(x)
        
        # Process each expert
        for i in range(self.num_experts):
            # Find all tokens that selected this expert at any position
            expert_mask = (selected_experts == i).any(dim=-1)  # (B, T)
            
            if not expert_mask.any():
                continue
            
            # Process all tokens that use this expert
            expert_input = x[expert_mask]  # (N, C) where N = expert_mask.sum()
            expert_output = self.experts[i](expert_input)  # (N, C)
            
            # Create a mapping to put expert_output back in the right places
            expert_output_full = torch.zeros_like(x)  # (B, T, C)
            expert_output_full[expert_mask] = expert_output
            
            # For each expert position k, add weighted contribution
            for k in range(self.experts_per_token):
                token_mask = (selected_experts[..., k] == i)  # (B, T)
                if token_mask.any():
                    weights = routing_weights[..., k:k+1]  # (B, T, 1)
                    final_output += weights * expert_output_full * token_mask.unsqueeze(-1)
        
        return final_output, aux_loss


class TransformerFFN(nn.Module):
    """FFN with SwiGLU activation"""
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.intermediate_size = 4 * config.n_embd
        
        # SwiGLU requires 2 * intermediate_size for gating
        self.c_fc = nn.Linear(config.n_embd, 2 * self.intermediate_size, bias=config.bias)
        self.c_proj = nn.Linear(self.intermediate_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = swiglu(x, limit=7.0)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block"""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = GroupedQueryAttention(config, layer_idx)
        self.ln_2 = RMSNorm(config.n_embd)
        
        # Use MoE or standard FFN
        if hasattr(config, 'use_moe') and config.use_moe:
            self.mlp = MixtureOfExperts(config)
        else:
            self.mlp = TransformerFFN(config)

    def forward(self, x, age=None, attn_mask=None):
        # Pre-norm architecture (more stable)
        y, att = self.attn(self.ln_1(x), age, attn_mask)
        x = x + y
        mlp_out = self.mlp(self.ln_2(x))
        aux_loss = None
        if isinstance(mlp_out, tuple):
            mlp_out, aux_loss = mlp_out
        x = x + mlp_out
        return x, att, aux_loss

# =============================================================================
# Composite Embedding + Multi-Head Output Architecture
# =============================================================================

class CompositeEmbedding(nn.Module):
    """
    Composite Embedding Layer: 여러 입력 필드를 각각 임베딩하고 투영
    - DATA (약품/질병 코드) -> ID Embedding
    - DOSE (시프트 값) -> Shift Embedding
    - DURATION (기간) -> Duration Embedding
    
    Concatenation + Projection: 각 필드의 정보를 더 잘 보존
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.dose_continuous = bool(getattr(config, 'dose_continuous', False))
        self.dose_log = bool(getattr(config, 'dose_log', False)) and self.dose_continuous
        self.dose_input_scale = float(getattr(config, 'dose_input_scale', 1.0))
        
        # 각 필드별 Embedding
        self.data_emb = nn.Embedding(config.data_vocab_size, config.n_embd)
        self.dose_emb = nn.Embedding(config.dose_vocab_size, config.n_embd)
        self.dur_emb = nn.Embedding(config.dur_vocab_size, config.n_embd)
        if self.dose_continuous:
            # Continuous DOSE encoder (scalar -> dense representation)
            self.dose_value_proj = nn.Sequential(
                nn.Linear(1, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd),
            )
        
        # Concatenation → Projection (3*n_embd → n_embd)
        self.proj = nn.Linear(config.n_embd * 3, config.n_embd, bias=False)
        
    def forward(self, data, dose, dur):
        """
        Args:
            data: (B, T) DATA tokens
            dose: (B, T) DOSE values (정수값)
            dur: (B, T) DURATION tokens
        Returns:
            combined embedding (B, T, n_embd)
        """
        # DATA embedding (clamp to valid range)
        data_idx = torch.clamp(data, min=0, max=self.data_emb.num_embeddings - 1)
        data_emb = self.data_emb(data_idx)
        
        if self.dose_continuous:
            # DOSE continuous value encoding
            scale = self.dose_input_scale if self.dose_input_scale > 0 else 1.0
            dose_value = dose.float()
            dose_value = _to_dose_model_space(dose_value, self.dose_log)
            dose_value = dose_value.unsqueeze(-1) / scale
            dose_emb = self.dose_value_proj(dose_value)
        else:
            # DOSE embedding (legacy discrete mode)
            dose_idx = torch.clamp(dose, min=0, max=self.dose_emb.num_embeddings - 1)
            dose_emb = self.dose_emb(dose_idx)
        
        # DURATION embedding (clamp to valid range)
        dur_idx = torch.clamp(dur, min=0, max=self.dur_emb.num_embeddings - 1)
        dur_emb = self.dur_emb(dur_idx)
        
        # Concatenate + Project (preserves each field's information better than sum)
        combined = torch.cat([data_emb, dose_emb, dur_emb], dim=-1)  # (B, T, 3*n_embd)
        combined = self.proj(combined)  # (B, T, n_embd)
        
        return combined


class MixtureDensityHead(nn.Module):
    """
    Mixture of Logistics head for multi-modal DURATION distribution.
    """

    def __init__(self, n_embd: int, n_components: int, min_value: float, max_value: float):
        super().__init__()
        self.n_components = int(n_components)
        self.min_value = float(min_value)
        self.max_value = float(max_value)

        self.proj = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, self.n_components * 3),
        )

    def reset_mdn_bias(self):
        """
        Initialize MDN output layer:
        - pi logits: uniform (bias=0)
        - mu: spread across [min, max] via inverse-sigmoid
        - log_sigma: moderately wide (~exp(2)=7.4)
        """
        last = self.proj[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.zero_()
            k = self.n_components
            if k > 1:
                grid = torch.linspace(0.05, 0.95, steps=k, device=last.bias.device)
            else:
                grid = torch.tensor([0.5], device=last.bias.device)
            mu_bias = torch.logit(grid, eps=1e-6)
            last.bias[k:2 * k].copy_(mu_bias)
            last.bias[2 * k:3 * k].fill_(2.0)

    def forward(self, x: torch.Tensor):
        params = self.proj(x)  # (B, T, 3K)
        k = self.n_components
        pi_logits, mu_raw, log_s = params.split(k, dim=-1)

        # Keep means in valid DURATION range
        mu = torch.sigmoid(mu_raw) * (self.max_value - self.min_value) + self.min_value
        # Prevent degenerate ultra-narrow components
        log_s = torch.clamp(log_s, min=-1.0, max=5.0)
        # Backward-compatible point estimate
        pi = F.softmax(pi_logits, dim=-1)
        mean = (pi * mu).sum(dim=-1)
        return {
            'pi_logits': pi_logits,
            'mu': mu,
            'log_s': log_s,
            'mean': mean,
        }


class MultiHeadOutput(nn.Module):
    """
    Multi-Head Output Layer: 각 필드별 예측 헤드
    - DATA Head: 다음 DATA 토큰 예측 (Classification)
    - DOSE Head: 다음 DOSE 값 예측 (Regression, MDN)
    - DURATION Head: 다음 DURATION 값 예측 (Regression, 연속값)
    - Time Head: 다음 이벤트까지의 시간 예측
      - Exponential: scale (λ) parameter만 예측
      - Weibull: scale (λ) + shape (k) parameter 예측
    
    Drug-Conditioned Heads (optional):
    - 약물(drug) 정보를 조건으로 DOSE/DURATION 예측 성능 향상
    - FiLM (Feature-wise Linear Modulation) 방식 사용
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.time_distribution = getattr(config, 'time_distribution', 'exponential')
        self.dose_continuous = bool(getattr(config, 'dose_continuous', False))
        self.dose_log = bool(getattr(config, 'dose_log', False)) and self.dose_continuous
        self.mdn_n_components = int(getattr(config, 'mdn_n_components', 8))
        self.dur_min_value = float(getattr(config, 'dur_min_value', 0.0))
        self.dur_max_value = float(getattr(config, 'dur_max_value', 550.0))
        if self.dose_continuous:
            default_dose_min = 0.0
            default_dose_max = float(getattr(config, 'dur_max_value', 550.0))
        else:
            default_dose_min = 2.0 if bool(getattr(config, 'apply_token_shift', False)) else 1.0
            default_dose_max = 4.0 if bool(getattr(config, 'apply_token_shift', False)) else 3.0
        dose_min_cfg = float(getattr(config, 'dose_min_value', -1.0))
        dose_max_cfg = float(getattr(config, 'dose_max_value', -1.0))
        self.dose_min_value = default_dose_min if dose_min_cfg < 0.0 else dose_min_cfg
        self.dose_max_value = default_dose_max if dose_max_cfg < 0.0 else dose_max_cfg
        self.dose_min_value_model, self.dose_max_value_model = _dose_bounds_for_model_space(
            self.dose_min_value,
            self.dose_max_value,
            self.dose_log,
        )
        
        # Drug-conditioning option
        self.use_drug_conditioning = getattr(config, 'use_drug_conditioning', False)
        
        # Heads
        self.data_head = nn.Linear(config.n_embd, config.data_vocab_size, bias=False)
        
        # DOSE head: MDN regression (same structure as DURATION)
        self.dose_head = MixtureDensityHead(
            n_embd=config.n_embd,
            n_components=self.mdn_n_components,
            min_value=self.dose_min_value_model,
            max_value=self.dose_max_value_model,
        )

        # DURATION head: MDN (Mixture of Logistics)
        self.dur_head = MixtureDensityHead(
            n_embd=config.n_embd,
            n_components=self.mdn_n_components,
            min_value=self.dur_min_value,
            max_value=self.dur_max_value,
        )
        
        # ============================================================
        # Drug-Conditioned Heads (FiLM style)
        # 약물 정보로 hidden state를 변조하여 DOSE/DURATION 예측
        # ============================================================
        if self.use_drug_conditioning:
            # FiLM generator: drug_emb → (gamma, beta) for modulation
            # DOSE: drug 조건
            self.dose_film_generator = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd * 2)  # gamma, beta
            )
            self.dose_drug_cond_head = MixtureDensityHead(
                n_embd=config.n_embd,
                n_components=self.mdn_n_components,
                min_value=self.dose_min_value_model,
                max_value=self.dose_max_value_model,
            )
            
            # DURATION: drug 조건만
            self.dur_film_generator = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd * 2)  # gamma, beta
            )
            self.dur_drug_cond_head = MixtureDensityHead(
                n_embd=config.n_embd,
                n_components=self.mdn_n_components,
                min_value=self.dur_min_value,
                max_value=self.dur_max_value,
            )
            # Provisional identity init here; reapplied after parent model init.
            self.reset_film_identity()
        
        # v6 (Delphi-style): DATA logits are shared for event type and time-to-event.
        # Keep only optional Weibull shape head.
        
        # Weibull shape parameter (k) - 전역 또는 per-event
        if self.time_distribution == 'weibull':
            # Shape parameter per event type (more expressive)
            self.time_shape_head = nn.Linear(config.n_embd, config.data_vocab_size, bias=False)

    def reset_film_identity(self):
        """Initialize FiLM generators to identity: gamma=1, beta=0."""
        if self.use_drug_conditioning:
            for film_gen in [self.dose_film_generator, self.dur_film_generator]:
                last_layer = film_gen[-1]  # Last Linear layer
                nn.init.zeros_(last_layer.weight)
                with torch.no_grad():
                    # first half = gamma, second half = beta
                    last_layer.bias[:self.n_embd].fill_(1.0)
                    last_layer.bias[self.n_embd:].zero_()

        # MDN heads are sensitive to init; keep spread-out initial components.
        self.dose_head.reset_mdn_bias()
        self.dur_head.reset_mdn_bias()
        if self.use_drug_conditioning:
            self.dose_drug_cond_head.reset_mdn_bias()
            self.dur_drug_cond_head.reset_mdn_bias()
        
    def forward(self, x, drug_emb=None, drug_token_mask=None):
        """
        Args:
            x: (B, T, n_embd) transformer output
            drug_emb: (B, T, n_embd) drug token embedding for conditioning (optional)
                      - 학습 시: INPUT data의 embedding (과거 정보)
                      - 추론 시: 현재까지의 data의 embedding
            drug_token_mask: (B, T) bool tensor, True where TARGET is a drug token
                             FiLM only applies at these positions
        Returns:
            dict of logits/values for each head
        """
        dose_mdn = self.dose_head(x)
        dur_mdn = self.dur_head(x)
        dose_mean_out = _from_dose_model_space(dose_mdn['mean'], self.dose_log)
        output = {
            'data': self.data_head(x),             # (B, T, data_vocab_size) - classification logits
            'dose': dose_mean_out,               # (B, T) - raw-space point estimate for compatibility
            'dose_mdn': dose_mdn,                # MDN params for NLL training/sampling
            'duration': dur_mdn['mean'],           # (B, T) - point estimate for compatibility
            'dur_mdn': dur_mdn,               # MDN params for NLL training/sampling
        }
        
        # ============================================================
        # Drug-Conditioned Predictions (FiLM modulation)
        # Only applies when target is a drug token (drug_token_mask=True)
        # ============================================================
        if self.use_drug_conditioning and drug_emb is not None:
            # DOSE: FiLM modulation with drug embedding
            dose_film = self.dose_film_generator(drug_emb)  # (B, T, n_embd*2)
            dose_gamma, dose_beta = dose_film.chunk(2, dim=-1)  # 각각 (B, T, n_embd)
            dose_modulated = dose_gamma * x + dose_beta  # FiLM: γ * x + β
            dose_drug_mdn = self.dose_drug_cond_head(dose_modulated)
            dose_drug_cond = _from_dose_model_space(dose_drug_mdn['mean'], self.dose_log)
            
            # DURATION: FiLM modulation with drug embedding
            dur_film = self.dur_film_generator(drug_emb)  # (B, T, n_embd*2)
            dur_gamma, dur_beta = dur_film.chunk(2, dim=-1)  # 각각 (B, T, n_embd)
            dur_modulated = dur_gamma * x + dur_beta  # FiLM: γ * x + β
            dur_drug_mdn = self.dur_drug_cond_head(dur_modulated)
            dur_drug_cond = dur_drug_mdn['mean']  # (B, T)
            
            # Apply drug token masking: only use FiLM output where target is a drug
            if drug_token_mask is not None:
                # Blend: use FiLM output for drug tokens, standard output otherwise
                # DOSE: (B, T)
                dose_drug_cond_masked = torch.where(
                    drug_token_mask,  # (B, T)
                    dose_drug_cond,
                    output['dose']
                )
                output['dose_drug_cond'] = dose_drug_cond_masked
                mask_mdn = drug_token_mask.unsqueeze(-1)
                output['dose_mdn_drug_cond'] = {
                    'pi_logits': torch.where(mask_mdn, dose_drug_mdn['pi_logits'], output['dose_mdn']['pi_logits']),
                    'mu': torch.where(mask_mdn, dose_drug_mdn['mu'], output['dose_mdn']['mu']),
                    'log_s': torch.where(mask_mdn, dose_drug_mdn['log_s'], output['dose_mdn']['log_s']),
                    'mean': torch.where(mask_mdn.squeeze(-1), dose_drug_mdn['mean'], output['dose_mdn']['mean']),
                }
                
                # DURATION: (B, T)
                dur_drug_cond_masked = torch.where(
                    drug_token_mask,  # (B, T)
                    dur_drug_cond,
                    output['duration']
                )
                output['dur_drug_cond'] = dur_drug_cond_masked
                # MDN params: (B, T, K)
                mask_mdn = drug_token_mask.unsqueeze(-1)
                output['dur_mdn_drug_cond'] = {
                    'pi_logits': torch.where(mask_mdn, dur_drug_mdn['pi_logits'], output['dur_mdn']['pi_logits']),
                    'mu': torch.where(mask_mdn, dur_drug_mdn['mu'], output['dur_mdn']['mu']),
                    'log_s': torch.where(mask_mdn, dur_drug_mdn['log_s'], output['dur_mdn']['log_s']),
                    'mean': dur_drug_cond_masked,
                }
            else:
                # No mask: apply FiLM to all positions (backward compatibility)
                output['dose_drug_cond'] = dose_drug_cond
                output['dose_mdn_drug_cond'] = dose_drug_mdn
                output['dur_drug_cond'] = dur_drug_cond
                output['dur_mdn_drug_cond'] = dur_drug_mdn
        
        # v6 keeps Delphi semantics: time scale uses DATA logits.
        # Expose aliases for compatibility with existing tooling.
        output['time_scale'] = output['data']
        output['time'] = output['time_scale']
        
        if self.time_distribution == 'weibull':
            # Weibull shape parameter (k > 0, use softplus to ensure positivity)
            output['time_shape'] = F.softplus(self.time_shape_head(x)) + 0.1  # (B, T, data_vocab_size)
        
        return output


@dataclass
class CompositeDelphiConfig:
    """Configuration for Composite Delphi with Multi-Head Output"""
    block_size: int = 1024
    
    # Vocabulary sizes for each field
    # 
    # Embedding vocab sizes (모든 필드의 embedding에 사용):
    # - DATA: includes drugs (Metformin~Death, raw 1277-1288) → after +1 dose: 1278-1289 → vocab_size = 1290
    # - DOSE: range depends on dataset (need to check actual range)
    # - DURATION: range 0-550 → vocab_size = 551
    #
    # Head 구조:
    # - DATA Head: Linear(n_embd, 1290) → Softmax + Cross-Entropy (Classification)
    # - DOSE Head: MDN regression head (same as DURATION head structure)
    # - DURATION Head: MDN regression head
    # Note: vocab sizes include +1 for the dose in get_batch_composite (0 reserved for padding)
    # Drug tokens: raw 1277~1288 → after +1 dose: 1278~1289 → max token 1289
    data_vocab_size: int = 1290   # DATA embedding & head (Classification) - includes Death token
    # NOTE: dose_vocab_size is used for DOSE embedding input space.
    dose_vocab_size: int = 5
    dur_vocab_size: int = 552   # DURATION embedding only (Regression head dim=1) - max 551 after +1 dose
    
    # Model architecture
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 384
    dropout: float = 0.1
    token_dropout: float = 0.0
    bias: bool = False
    
    # Medical specific
    t_min: float = 0.1
    mask_ties: bool = True
    ignore_tokens: list = field(default_factory=lambda: [0])
    
    # Drug-Conditioning: 약물 정보를 조건으로 DOSE/DURATION 예측 성능 향상
    # FiLM (Feature-wise Linear Modulation) 방식 사용
    use_drug_conditioning: bool = False
    
    # Drug token range defaults for apply_token_shift=False (raw tokens):
    # Metformin(1278) ... Death(1288)
    # If apply_token_shift=True, override via model_args to 1279..1289.
    drug_token_min: int = 1278
    drug_token_max: int = 1288
    apply_token_shift: bool = False
    separate_dose_na_from_padding: bool = False
    dose_na_raw_token: int = 4

    # Architecture features
    use_moe: bool = True
    num_experts: int = 8
    experts_per_token: int = 2
    sliding_window: int = 256
    rope_theta: float = 10000.0
    
    # DURATION MDN options
    mdn_n_components: int = 8
    dose_min_value: float = -1.0  # auto: 1(raw) or 2(shifted)
    dose_max_value: float = -1.0  # auto: 3(raw) or 4(shifted)
    dose_continuous: bool = True
    dose_log: bool = False
    dose_input_scale: float = 1.0
    dose_exclude_na_token: bool = True
    dose_mdn_nll_weight: float = 0.05
    dose_label_scaling: str = 'none'  # 'none' | 'zscore' | 'robust' | 'minmax'
    dose_label_center: float = 0.0
    dose_label_scale: float = 1.0
    dose_label_min: float = 0.0
    dose_label_max: float = 1.0
    dur_min_value: float = 0.0
    dur_max_value: float = 550.0
    dur_label_scaling: str = 'none'  # 'none' | 'zscore' | 'robust' | 'minmax'
    dur_label_center: float = 0.0
    dur_label_scale: float = 1.0
    dur_label_min: float = 0.0
    dur_label_max: float = 1.0
    # Drug-token-only regression: DOSE/DURATION loss를 약물 토큰 위치에서만 계산
    # 비약물 토큰(질병 등)의 무의미한 DURATION/DOSE 값이 MDN 학습을 희석하는 문제 해결
    drug_token_only_regression: bool = False
    # Drug-token loss에 추가 가중치 (drug_token_only_regression=False일 때 사용)
    drug_token_loss_weight: float = 1.0
    loss_normalize_by_variance: bool = False
    dose_loss_variance: float = 1.0
    dur_loss_variance: float = 1.0
    # Backward-compatibility option for legacy non-MDN checkpoints
    dur_log_transform: bool = False
    
    # Loss weights
    loss_weight_data: float = 1.0
    loss_weight_dose: float = 20.0
    # No separate auxiliary change head in legacy versions.
    loss_weight_change: float = 0.0
    loss_weight_total: float = 5.0
    loss_weight_time: float = 1.0

    # Kept for compatibility (unused in legacy DOSE regression)
    # - dose_loss_type: 'dice_focal' | 'focal' | 'ce'
    # - dose_class_weights: legacy class weights
    dose_loss_type: str = 'dice_focal'
    dose_dice_weight: float = 0.5
    dose_ignore_index: int = -1
    dose_maintain_idx: int = 2  # kept for compatibility (unused in legacy)
    dose_change_weight_max: float = 10.0  # kept for compatibility (unused in legacy)
    dose_focal_gamma: float = 2.0
    dose_class_weights: list = field(default_factory=list)
    
    # Time-to-Event distribution: 'exponential' or 'weibull'
    # - exponential: 상수 hazard rate (memoryless)
    # - weibull: 시간에 따라 변하는 hazard rate (shape parameter k로 조절)
    time_distribution: str = 'exponential'


class CompositeDelphi(nn.Module):
    """
    Composite Delphi: Composite Embedding + Multi-Head Output
    
    입력: (DATA, DOSE, DURATION, AGE)
    출력: 각 필드별 예측 + 시간 예측
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Composite Embedding
        self.composite_emb = CompositeEmbedding(config)
        
        # Age Encoding (기존과 동일)
        self.age_encoding = AgeEncoding(config)
        
        # Dropout layers
        self.token_drop = nn.Dropout(config.token_dropout)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks (기존과 동일)
        self.h = nn.ModuleList([TransformerBlock(config, i) for i in range(config.n_layer)])
        
        # Final normalization
        self.ln_f = RMSNorm(config.n_embd)
        
        # Multi-Head Output
        self.multi_head = MultiHeadOutput(config)
        
        # Weight tying: data_head와 data_emb
        self.multi_head.data_head.weight = self.composite_emb.data_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        # Re-apply FiLM identity init after global init to avoid overwrite.
        self.multi_head.reset_film_identity()
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, data, dose, dur, age,
                targets_data=None, targets_dose=None, targets_dur=None,
                targets_age=None, drug_conditioning_data=None,
                validation_loss_mode=False):
        """
        Args:
            data: (B, T) DATA tokens
            dose: (B, T) DOSE values
            dur: (B, T) DURATION tokens
            age: (B, T) AGE values
            targets_*: 각 필드의 타겟 (optional)
        """
        device = data.device
        b, t = data.size()
        
        # 1. Composite Embedding
        composite_emb = self.composite_emb(data, dose, dur)
        
        # 2. Age Encoding
        age_emb = self.age_encoding(age)
        
        # 3. Combine embeddings
        x = self.token_drop(composite_emb) * (1 - self.config.token_dropout)
        x = x + age_emb
        x = self.drop(x)
        
        # 4. Attention mask
        attn_mask = (data > 0).view(b, 1, 1, t) * (data > 0).view(b, 1, t, 1)
        attn_mask *= torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0
        
        if targets_data is not None and self.config.mask_ties:
            attn_mask *= (age.view(b, 1, 1, t) != targets_age.view(b, 1, t, 1))
            attn_mask += (attn_mask.sum(-1, keepdim=True) == 0) * torch.diag(torch.ones(t, device=device)) > 0
        
        attn_mask = attn_mask + (data == 0).view(b, 1, 1, t) * torch.diag(torch.ones(t, device=device)) > 0
        attn_mask *= torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0
        
        # 5. Transformer blocks
        # Skip attention weight collection during training to save ~13GB+ GPU memory
        is_training = targets_data is not None
        att_list = []
        aux_losses = []
        for block in self.h:
            x, att, aux_loss = block(x, age, attn_mask)
            if not is_training:
                att_list.append(att)
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        
        # Average MoE load balancing loss across layers
        moe_aux_loss = sum(aux_losses) / max(len(aux_losses), 1) if aux_losses else None
        
        x = self.ln_f(x)
        att = torch.stack(att_list) if (att_list and att_list[0] is not None) else None
        
        # 6. Multi-Head Output
        # Drug-Conditioning: FiLM modulation for DOSE/DURATION prediction
        #
        # For teacher-forced training/evaluation, condition on target token ids so
        # DOSE/DURATION heads learn p(value | next event token, history).
        # For free-running inference (no targets), fall back to current input tokens.
        drug_emb = None
        drug_token_mask = None
        if self.config.use_drug_conditioning:
            # Priority:
            # 1) explicit conditioning tokens (if provided)
            # 2) teacher-forced target tokens
            # 3) current input tokens (inference fallback)
            if drug_conditioning_data is not None:
                drug_source = drug_conditioning_data
            elif targets_data is not None:
                drug_source = targets_data
            else:
                drug_source = data

            if drug_source is not None:
                # Clamp to valid range
                drug_source_clamped = torch.clamp(
                    drug_source,
                    min=0,
                    max=self.composite_emb.data_emb.num_embeddings - 1,
                )
                drug_emb = self.composite_emb.data_emb(drug_source_clamped)
            
            # Create drug token mask: FiLM only applies when next token is a drug.
            # Drug token range must match data tokenization setup.
            drug_token_min = getattr(self.config, 'drug_token_min', 1278)
            drug_token_max = getattr(self.config, 'drug_token_max', 1288)
            if targets_data is not None:
                drug_token_mask = (targets_data >= drug_token_min) & (targets_data <= drug_token_max)
        
        logits = self.multi_head(x, drug_emb=drug_emb, drug_token_mask=drug_token_mask)
        
        # 7. Compute losses if targets provided
        if targets_data is not None:
            loss = self._compute_loss(
                logits, data, age,
                targets_data, targets_dose, targets_dur, targets_age,
                attn_mask, validation_loss_mode,
                moe_aux_loss=moe_aux_loss,
                drug_token_mask=drug_token_mask
            )
        else:
            loss = None
        
        return logits, loss, att
    
    def _compute_loss(self, logits, data, age,
                      targets_data, targets_dose, targets_dur, targets_age,
                      attn_mask, validation_loss_mode,
                      moe_aux_loss=None, drug_token_mask=None):
        """Compute multi-head losses"""
        device = data.device
        b, t = data.size()

        ignored_tokens = self.config.ignore_tokens.copy()
        if validation_loss_mode:
            ignored_tokens += [1]

        # Valid token mask
        targets_flat = targets_data.reshape(-1)
        pass_tokens = targets_flat != -1
        for k in ignored_tokens:
            pass_tokens = pass_tokens * (targets_flat != k)

        # Drug token mask for regression heads (DOSE/DURATION)
        drug_only_reg = bool(getattr(self.config, 'drug_token_only_regression', False))
        drug_loss_weight = float(getattr(self.config, 'drug_token_loss_weight', 1.0))
        if drug_token_mask is not None:
            drug_mask_flat = drug_token_mask.reshape(-1)
        else:
            # Fallback: construct from config
            drug_token_min = getattr(self.config, 'drug_token_min', 1278)
            drug_token_max = getattr(self.config, 'drug_token_max', 1288)
            drug_mask_flat = (targets_flat >= drug_token_min) & (targets_flat <= drug_token_max)
        
        # Clamp targets to valid vocab range (defensive measure)
        data_vocab_size = self.config.data_vocab_size
        targets_flat_clamped = torch.clamp(targets_flat, min=0, max=data_vocab_size - 1)
        
        # 1. DATA Cross-Entropy Loss
        data_logits = logits['data']
        if validation_loss_mode:
            data_logits[..., ignored_tokens] = -torch.inf
        
        loss_data = F.cross_entropy(
            data_logits.reshape(-1, data_logits.size(-1))[pass_tokens],
            targets_flat_clamped[pass_tokens],  # ← clamp된 값 사용
            ignore_index=-1
        )
        
        # 2. DOSE regression loss (MDN NLL; same style as DURATION head)
        dose_mdn_source = logits.get('dose_mdn', None)
        if 'dose_mdn_drug_cond' in logits and self.config.use_drug_conditioning:
            dose_mdn_source = logits['dose_mdn_drug_cond']

        dose_targets_all = targets_dose.reshape(-1).float()
        dose_pass_tokens = pass_tokens & (dose_targets_all != -1)
        dose_continuous = bool(getattr(self.config, 'dose_continuous', False))
        dose_log = bool(getattr(self.config, 'dose_log', False)) and dose_continuous
        if dose_continuous and bool(getattr(self.config, 'separate_dose_na_from_padding', False)) and bool(getattr(self.config, 'dose_exclude_na_token', True)):
            dose_na_token = int(getattr(self.config, 'dose_na_raw_token', 4))
            if bool(getattr(self.config, 'apply_token_shift', False)):
                dose_na_token += 1
            dose_pass_tokens = dose_pass_tokens & (dose_targets_all != float(dose_na_token))

        if dose_continuous:
            dose_valid_mask = dose_pass_tokens & (dose_targets_all >= 0)
        else:
            dose_targets_discrete = dose_targets_all.long()
            if getattr(self.config, 'apply_token_shift', False):
                # shifted labels: 2=Dec, 3=Maint, 4=Inc
                is_valid_dose = (dose_targets_discrete == 2) | (dose_targets_discrete == 3) | (dose_targets_discrete == 4)
            else:
                # raw labels: 1=Dec, 2=Maint, 3=Inc
                is_valid_dose = (dose_targets_discrete == 1) | (dose_targets_discrete == 2) | (dose_targets_discrete == 3)
            dose_valid_mask = dose_pass_tokens & is_valid_dose

        # Drug-token-only regression: DOSE loss를 약물 토큰에서만 계산
        if drug_only_reg:
            dose_valid_mask = dose_valid_mask & drug_mask_flat

        dose_min = float(getattr(self.config, 'dose_min_value', -1.0))
        dose_max = float(getattr(self.config, 'dose_max_value', -1.0))
        if dose_max <= dose_min:
            if dose_continuous:
                dose_min = 0.0
                dose_max = float(getattr(self.config, 'dur_max_value', 550.0))
            elif bool(getattr(self.config, 'apply_token_shift', False)):
                dose_min, dose_max = 2.0, 4.0
            else:
                dose_min, dose_max = 1.0, 3.0
        dose_min_model, dose_max_model = _dose_bounds_for_model_space(
            dose_min,
            dose_max,
            dose_log,
        )

        loss_dose = torch.tensor(0.0, device=device)
        loss_change = torch.tensor(0.0, device=device)
        dose_scale = max(dose_max_model - dose_min_model, 1.0)
        dose_label_scaling = str(getattr(self.config, 'dose_label_scaling', 'none')).lower()
        dose_label_center = float(getattr(self.config, 'dose_label_center', 0.0))
        dose_label_scale = float(getattr(self.config, 'dose_label_scale', 1.0))
        dose_label_min = float(getattr(self.config, 'dose_label_min', dose_min_model))
        dose_label_max = float(getattr(self.config, 'dose_label_max', dose_max_model))
        if isinstance(dose_mdn_source, dict):
            dose_pi_logits = dose_mdn_source['pi_logits'].reshape(-1, dose_mdn_source['pi_logits'].size(-1))[dose_valid_mask]
            dose_mu = dose_mdn_source['mu'].reshape(-1, dose_mdn_source['mu'].size(-1))[dose_valid_mask]
            dose_log_s = dose_mdn_source['log_s'].reshape(-1, dose_mdn_source['log_s'].size(-1))[dose_valid_mask]
            dose_target = dose_targets_all[dose_valid_mask]

            if dose_target.numel() > 0:
                dose_target = _to_dose_model_space(dose_target, dose_log)
                dose_target = dose_target.clamp(min=dose_min_model, max=dose_max_model)
                dose_mean_src = dose_mdn_source.get('mean', None)
                if dose_mean_src is not None:
                    dose_pred_flat = dose_mean_src.reshape(-1)[dose_valid_mask]
                else:
                    dose_pi = F.softmax(dose_pi_logits, dim=-1)
                    dose_pred_flat = (dose_pi * dose_mu).sum(dim=-1)
                dose_pred_flat = dose_pred_flat.clamp(min=dose_min_model, max=dose_max_model)

                dose_pred_scaled = _transform_targets(
                    dose_pred_flat,
                    dose_label_scaling,
                    dose_label_center,
                    dose_label_scale,
                    dose_label_min,
                    dose_label_max,
                )
                dose_target_scaled = _transform_targets(
                    dose_target,
                    dose_label_scaling,
                    dose_label_center,
                    dose_label_scale,
                    dose_label_min,
                    dose_label_max,
                )

                # Primary objective for stable continuous regression.
                loss_reg = F.smooth_l1_loss(
                    dose_pred_scaled,
                    dose_target_scaled,
                    beta=max(min(0.25 * dose_scale, 5.0), 0.1),
                ) / max(dose_scale, 1.0)

                # Optional MDN NLL auxiliary term (small weight), without hard plateau at 20.
                y_dose = dose_target.unsqueeze(-1)
                z_dose = (y_dose - dose_mu) / torch.exp(dose_log_s)
                log_pdf_dose = -z_dose - dose_log_s - 2.0 * F.softplus(-z_dose)
                log_pi_dose = F.log_softmax(dose_pi_logits, dim=-1)
                log_prob_dose = torch.logsumexp(log_pi_dose + log_pdf_dose, dim=-1)
                nll_dose = -log_prob_dose
                nll_dose = torch.nan_to_num(nll_dose, nan=200.0, posinf=200.0, neginf=0.0)
                loss_nll = torch.clamp(nll_dose, min=0.0, max=200.0).mean() / max(dose_scale, 1.0)
                nll_w = float(getattr(self.config, 'dose_mdn_nll_weight', 0.05))
                nll_w = min(max(nll_w, 0.0), 1.0)
                loss_dose = (1.0 - nll_w) * loss_reg + nll_w * loss_nll
        else:
            dose_pred_source = logits['dose']
            if 'dose_drug_cond' in logits and self.config.use_drug_conditioning:
                dose_pred_source = logits['dose_drug_cond']
            dose_pred_flat = dose_pred_source.reshape(-1)[dose_valid_mask]
            dose_target = dose_targets_all[dose_valid_mask]
            if dose_target.numel() > 0:
                dose_pred_flat = _to_dose_model_space(dose_pred_flat, dose_log)
                dose_target = _to_dose_model_space(dose_target, dose_log)
                dose_pred_flat = torch.clamp(dose_pred_flat, min=dose_min_model, max=dose_max_model)
                dose_target = dose_target.clamp(min=dose_min_model, max=dose_max_model)
                dose_pred_scaled = _transform_targets(
                    dose_pred_flat,
                    dose_label_scaling,
                    dose_label_center,
                    dose_label_scale,
                    dose_label_min,
                    dose_label_max,
                )
                dose_target_scaled = _transform_targets(
                    dose_target,
                    dose_label_scaling,
                    dose_label_center,
                    dose_label_scale,
                    dose_label_min,
                    dose_label_max,
                )
                loss_dose = F.smooth_l1_loss(
                    dose_pred_scaled,
                    dose_target_scaled,
                    beta=max(min(0.25 * dose_scale, 5.0), 0.1),
                ) / max(dose_scale, 1.0)

        # 3. DURATION Loss (Mixture Density NLL preferred)
        # Drug-token-only regression: DURATION loss를 약물 토큰에서만 계산
        dur_pass_tokens = pass_tokens & drug_mask_flat if drug_only_reg else pass_tokens
        dur_target = targets_dur.float().reshape(-1)[dur_pass_tokens]  # (N,)
        dur_mdn_source = logits.get('dur_mdn', None)
        if 'dur_mdn_drug_cond' in logits and self.config.use_drug_conditioning:
            dur_mdn_source = logits['dur_mdn_drug_cond']

        dur_label_scaling = str(getattr(self.config, 'dur_label_scaling', 'none')).lower()
        dur_label_center = float(getattr(self.config, 'dur_label_center', 0.0))
        dur_label_scale = float(getattr(self.config, 'dur_label_scale', 1.0))
        dur_label_min = float(getattr(self.config, 'dur_label_min', 0.0))
        dur_label_max = float(getattr(self.config, 'dur_label_max', float(getattr(self.config, 'dur_max_value', 550.0))))
        if isinstance(dur_mdn_source, dict):
            pi_logits = dur_mdn_source['pi_logits'].reshape(-1, dur_mdn_source['pi_logits'].size(-1))[dur_pass_tokens]
            mu = dur_mdn_source['mu'].reshape(-1, dur_mdn_source['mu'].size(-1))[dur_pass_tokens]
            log_s = dur_mdn_source['log_s'].reshape(-1, dur_mdn_source['log_s'].size(-1))[dur_pass_tokens]

            # Mixture of logistics NLL:
            # log p(y) = logsumexp_k(log pi_k + log Logistic(y|mu_k, s_k))
            dur_target_scaled = _transform_targets(
                dur_target,
                dur_label_scaling,
                dur_label_center,
                dur_label_scale,
                dur_label_min,
                dur_label_max,
            )

            if dur_label_scaling in {'zscore', 'robust'}:
                affine_scale = max(dur_label_scale, 1e-8)
                mu_scaled = (mu - dur_label_center) / affine_scale
                log_s_scaled = log_s - math.log(affine_scale)
            elif dur_label_scaling == 'minmax':
                affine_scale = max(dur_label_max - dur_label_min, 1e-8)
                mu_scaled = (mu - dur_label_min) / affine_scale
                log_s_scaled = log_s - math.log(affine_scale)
            else:
                mu_scaled = mu
                log_s_scaled = log_s

            y = dur_target_scaled.unsqueeze(-1)
            z = (y - mu_scaled) / torch.exp(log_s_scaled)
            log_pdf = -z - log_s_scaled - 2.0 * F.softplus(-z)
            log_pi = F.log_softmax(pi_logits, dim=-1)
            log_prob = torch.logsumexp(log_pi + log_pdf, dim=-1)
            nll = -log_prob
            nll = torch.nan_to_num(nll, nan=50.0, posinf=50.0, neginf=0.0)
            loss_dur = torch.clamp(nll, min=0.0, max=50.0).mean()
        else:
            # Fallback for legacy checkpoints without MDN params
            dur_pred = logits['duration'].reshape(-1)[dur_pass_tokens]
            dur_scale = float(getattr(self.config, 'dur_max_value', 550.0))
            dur_pred = torch.clamp(dur_pred, min=0.0, max=dur_scale)
            dur_pred_scaled = _transform_targets(
                dur_pred,
                dur_label_scaling,
                dur_label_center,
                dur_label_scale,
                dur_label_min,
                dur_label_max,
            )
            dur_target_scaled = _transform_targets(
                dur_target,
                dur_label_scaling,
                dur_label_center,
                dur_label_scale,
                dur_label_min,
                dur_label_max,
            )
            loss_dur = F.mse_loss(dur_pred_scaled, dur_target_scaled) / (dur_scale ** 2)
        
        # 4. Time-to-Event Loss
        # dt = time difference (days until next event)
        dt = torch.clamp(targets_age - age, min=1.0)
        
        if self.config.mask_ties:
            dt = torch.gather(
                dt, -1,
                (attn_mask * torch.arange(0, t, device=device, dtype=torch.float32).view(1, 1, 1, -1))
                .max(-1).indices.squeeze((1, 2))
            )
        
        time_distribution = getattr(self.config, 'time_distribution', 'exponential')
        
        if time_distribution == 'weibull':
            # ============================================================
            # Weibull Distribution Loss (rate parameterization)
            # ============================================================
            # f(t) = k * lambda * t^(k-1) * exp(-lambda * t^k)
            # log f(t) = log(k) + log(lambda) + (k-1)*log(t) - lambda*t^k
            # k=1이면 Exponential(rate=lambda)로 환원된다.
            # Delphi-style: shared logits for event type and time-to-event rate.
            time_logits = logits['data']  # (B, T, data_vocab_size)
            time_shape = logits['time_shape']  # (B, T, data_vocab_size), already positive

            # Stable per-event log-rate with floor (same spirit as exponential branch):
            # log_lambda_i = -log(exp(-logit_i) + t_min)
            #             = logit_i - softplus(logit_i + log(t_min))
            t_min = float(getattr(self.config, 't_min', 0.1))
            t_min = max(t_min, 1e-8)
            log_t_min = math.log(t_min)
            log_lambda_i = time_logits - F.softplus(time_logits + log_t_min)  # (B, T, V)

            # Competing-risk aggregate rate:
            # lambda_total = sum_i p_i * lambda_i
            event_log_probs = F.log_softmax(time_logits, dim=-1)
            log_lambda = torch.logsumexp(event_log_probs + log_lambda_i, dim=-1)  # (B, T)
            log_lambda = torch.clamp(log_lambda, min=-20.0, max=20.0)

            # Shared shape k from event-probability-weighted mean
            event_probs = torch.exp(event_log_probs)
            shape = torch.clamp((event_probs * time_shape).sum(-1), min=0.2, max=5.0)  # (B, T)

            # Train Weibull in years to avoid extreme hazard from day-scale dt.
            dt_flat = torch.clamp(dt.reshape(-1) / 365.25, min=1.0 / 365.25)
            log_lambda_flat = log_lambda.reshape(-1)
            shape_flat = shape.reshape(-1)
            log_dt = torch.log(dt_flat)

            # lambda * t^k = exp(log_lambda + k*log(t))
            hazard_exp = torch.clamp(log_lambda_flat + shape_flat * log_dt, max=30.0)
            lambda_tk = torch.exp(hazard_exp)

            log_likelihood = (
                torch.log(shape_flat) +
                log_lambda_flat +
                (shape_flat - 1.0) * log_dt -
                lambda_tk
            )

            nll = -log_likelihood[pass_tokens]
            nll = torch.nan_to_num(nll, nan=200.0, posinf=200.0, neginf=0.0)
            loss_time = torch.clamp(nll, min=0.0, max=200.0).mean()
            
        else:
            # ============================================================
            # Exponential Distribution Loss (기존 Delphi와 동일)
            # ============================================================
            # PDF: f(t) = λ * exp(-λt)
            # log f(t) = log(λ) - λt
            
            # Delphi-style: use DATA logits directly for time-to-event.
            time_logits = logits['data']  # (B, T, data_vocab_size)
            lse = torch.logsumexp(time_logits, -1)  # (B, T)
            lse = -torch.log(torch.exp(-lse) + self.config.t_min)
            
            ldt = -torch.log(dt + self.config.t_min).view(-1)
            loss_time = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt.reshape(-1)))
            loss_time = torch.mean(loss_time[pass_tokens])
        
        if bool(getattr(self.config, 'loss_normalize_by_variance', False)):
            dose_var = max(float(getattr(self.config, 'dose_loss_variance', 1.0)), 1e-8)
            dur_var = max(float(getattr(self.config, 'dur_loss_variance', 1.0)), 1e-8)
            loss_dose = loss_dose / dose_var
            loss_dur = loss_dur / dur_var

        # Drug-token loss upweighting (drug_only_reg=False일 때만 의미 있음)
        if (not drug_only_reg) and drug_loss_weight > 1.0:
            loss_dose = loss_dose * drug_loss_weight
            loss_dur = loss_dur * drug_loss_weight

        # Weighted sum of losses
        total_loss = (
            self.config.loss_weight_data * loss_data +
            self.config.loss_weight_dose * loss_dose +
            self.config.loss_weight_change * loss_change +
            self.config.loss_weight_total * loss_dur +
            self.config.loss_weight_time * loss_time
        )
        
        # MoE load balancing loss (α=0.01)
        loss_moe = torch.tensor(0.0, device=total_loss.device)
        if moe_aux_loss is not None:
            loss_moe = moe_aux_loss
            total_loss = total_loss + 0.01 * loss_moe
        
        return {
            'loss': total_loss,
            'loss_data': loss_data,
            'loss_dose': loss_dose,
            'loss_change': loss_change,
            'loss_dur': loss_dur,
            'loss_time': loss_time,
            'loss_moe': loss_moe
        }
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure optimizer"""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (RMSNorm, torch.nn.Embedding, torch.nn.LayerNorm)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif isinstance(m, RMSNorm) and pn == 'scale':
                    # RMSNorm scale parameter
                    no_decay.add(fpn)
                elif pn.endswith('weight'):
                    # Any other weight parameter defaults to decay
                    decay.add(fpn)
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        # Handle weight tying: multi_head.data_head.weight is tied to composite_emb.data_emb.weight (Embedding)
        # The actual parameter is composite_emb.data_emb.weight, which is already in no_decay (Embedding)
        # So we just need to remove multi_head.data_head.weight from decay if it exists
        if 'multi_head.data_head.weight' in decay:
            decay.discard('multi_head.data_head.weight')
        # Don't add it to no_decay if it doesn't exist in param_dict (due to weight tying)
        
        # Check for overlapping parameters and resolve
        inter_params = decay & no_decay
        if inter_params:
            if _is_master():
                print(f"Warning: Found {len(inter_params)} parameters in both decay and no_decay, moving to no_decay:")
                for pn in sorted(inter_params):
                    print(f"  {pn}")
            for pn in list(inter_params):
                decay.discard(pn)
        
        union_params = decay | no_decay
        
        # Find missing parameters
        missing_params = param_dict.keys() - union_params
        if missing_params:
            if _is_master():
                print(f"Warning: Found {len(missing_params)} unclassified parameters, adding to no_decay:")
                for pn in sorted(missing_params):
                    print(f"  {pn}")
            for pn in missing_params:
                no_decay.add(pn)
        
        # Final check - only check params that actually exist
        union_params = decay | no_decay
        inter_params = decay & no_decay
        assert len(inter_params) == 0, f"Still have overlapping params: {inter_params}"
        
        # Only check params that exist in param_dict
        missing_in_union = param_dict.keys() - union_params
        assert len(missing_in_union) == 0, f"Missing params: {missing_in_union}"
        
        # Only include params that exist in param_dict
        decay_filtered = [pn for pn in sorted(list(decay)) if pn in param_dict]
        no_decay_filtered = [pn for pn in sorted(list(no_decay)) if pn in param_dict]
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in decay_filtered], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in no_decay_filtered], "weight_decay": 0.0},
        ]
        
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        if _is_master():
            print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        
        return optimizer
    
    @torch.no_grad()
    def generate(self, data, dose, dur, age, 
                 max_new_tokens=100, max_age=85*365.25,
                 no_repeat=True, termination_tokens=None):
        """Generate composite sequences"""
        if termination_tokens is None:
            warnings.warn('Consider setting termination_tokens for your dataset.')
            termination_tokens = [1269]
        
        termination_tokens = torch.tensor(termination_tokens, dtype=torch.int64, device=data.device)
        
        if max_new_tokens == -1:
            max_new_tokens = 128
        
        dose_continuous = bool(getattr(self.config, 'dose_continuous', False))
        dose_log = bool(getattr(self.config, 'dose_log', False)) and dose_continuous
        if dose_continuous and not torch.is_floating_point(dose):
            dose = dose.float()

        for _ in range(max_new_tokens):
            logits, _, _ = self(data, dose, dur, age, drug_conditioning_data=data)
            
            # Get last position logits
            data_logits = logits['data'][:, -1, :]
            dose_pred_base = logits['dose'][:, -1]
            dose_pred_drug = dose_pred_base
            if 'dose_drug_cond' in logits and self.config.use_drug_conditioning:
                dose_pred_drug = logits['dose_drug_cond'][:, -1]
            dose_mdn_base = logits.get('dose_mdn', None)
            dose_mdn_drug = dose_mdn_base
            if 'dose_mdn_drug_cond' in logits and self.config.use_drug_conditioning:
                dose_mdn_drug = logits['dose_mdn_drug_cond']

            dur_pred_base = logits['duration'][:, -1]
            dur_pred_drug = dur_pred_base
            if 'dur_drug_cond' in logits and self.config.use_drug_conditioning:
                dur_pred_drug = logits['dur_drug_cond'][:, -1]
            dur_mdn_base = logits.get('dur_mdn', None)
            dur_mdn_drug = dur_mdn_base
            if 'dur_mdn_drug_cond' in logits and self.config.use_drug_conditioning:
                dur_mdn_drug = logits['dur_mdn_drug_cond']
            # Mask ignored tokens
            data_logits[:, self.config.ignore_tokens] = -torch.inf
            
            if no_repeat:
                fill = data.clone()
                fill[fill == 1] = 0
                data_logits = data_logits.scatter_(1, fill, -torch.inf)
            
            # Sample next tokens from configured time distribution
            time_distribution = getattr(self.config, 'time_distribution', 'exponential')
            time_logits = data_logits  # Delphi-style shared logits (with generation constraints applied)
            if time_distribution == 'weibull' and 'time_shape' in logits:
                # Weibull competing risks:
                # T_i = (-log U / lambda_i)^(1/k), lambda_i = exp(time_logit_i)
                # Use shared k (weighted average) to stay aligned with training loss.
                time_shape_logits = logits['time_shape'][:, -1, :]  # (B, V), already positive
                event_probs = F.softmax(time_logits, dim=-1)
                shape = torch.clamp((event_probs * time_shape_logits).sum(-1, keepdim=True), min=0.2, max=5.0)  # (B, 1)

                t_min = float(getattr(self.config, 't_min', 0.1))
                t_min = max(t_min, 1e-8)
                log_t_min = math.log(t_min)
                log_lambda_i = time_logits - F.softplus(time_logits + log_t_min)
                lambda_i = torch.exp(torch.clamp(log_lambda_i, min=-20.0, max=20.0))  # (B, V)
                u = torch.rand_like(time_logits).clamp_min(1e-12)
                w = -torch.log(u) / torch.clamp(lambda_i, min=1e-8)
                sampled_t_years = torch.pow(torch.clamp(w, min=1e-12), 1.0 / shape)
                sampled_t_days = sampled_t_years * 365.25
                t_next = torch.clamp(sampled_t_days, min=0.0, max=365 * 80).min(1)
            else:
                # Exponential competing risks (original Delphi behavior)
                t_next = torch.clamp(
                    -torch.exp(-time_logits) * torch.rand(time_logits.shape, device=data.device).log(),
                    min=0, max=365*80
                ).min(1)
            
            data_next = t_next[1][:, None]
            age_next = age[..., [-1]] + t_next[0][:, None]

            # Use drug-conditioned DOSE/DURATION only when next DATA token is a drug.
            dose_pred = dose_pred_base
            dose_mdn_selected = dose_mdn_base
            dur_pred = dur_pred_base
            dur_mdn_selected = dur_mdn_base
            if self.config.use_drug_conditioning:
                drug_token_min = getattr(self.config, 'drug_token_min', 1278)
                drug_token_max = getattr(self.config, 'drug_token_max', 1288)
                next_is_drug = (data_next >= drug_token_min) & (data_next <= drug_token_max)  # (B, 1)
                dose_pred = torch.where(next_is_drug.squeeze(-1), dose_pred_drug, dose_pred_base)
                if isinstance(dose_mdn_base, dict) and isinstance(dose_mdn_drug, dict):
                    mask_mdn = next_is_drug
                    dose_mdn_selected = {
                        'pi_logits': torch.where(mask_mdn, dose_mdn_drug['pi_logits'][:, -1, :], dose_mdn_base['pi_logits'][:, -1, :]),
                        'mu': torch.where(mask_mdn, dose_mdn_drug['mu'][:, -1, :], dose_mdn_base['mu'][:, -1, :]),
                        'log_s': torch.where(mask_mdn, dose_mdn_drug['log_s'][:, -1, :], dose_mdn_base['log_s'][:, -1, :]),
                    }
                dur_pred = torch.where(next_is_drug.squeeze(-1), dur_pred_drug, dur_pred_base)
                if isinstance(dur_mdn_base, dict) and isinstance(dur_mdn_drug, dict):
                    mask_mdn = next_is_drug
                    dur_mdn_selected = {
                        'pi_logits': torch.where(mask_mdn, dur_mdn_drug['pi_logits'][:, -1, :], dur_mdn_base['pi_logits'][:, -1, :]),
                        'mu': torch.where(mask_mdn, dur_mdn_drug['mu'][:, -1, :], dur_mdn_base['mu'][:, -1, :]),
                        'log_s': torch.where(mask_mdn, dur_mdn_drug['log_s'][:, -1, :], dur_mdn_base['log_s'][:, -1, :]),
                    }
            elif isinstance(dose_mdn_base, dict):
                dose_mdn_selected = {
                    'pi_logits': dose_mdn_base['pi_logits'][:, -1, :],
                    'mu': dose_mdn_base['mu'][:, -1, :],
                    'log_s': dose_mdn_base['log_s'][:, -1, :],
                }
            elif isinstance(dur_mdn_base, dict):
                dur_mdn_selected = {
                    'pi_logits': dur_mdn_base['pi_logits'][:, -1, :],
                    'mu': dur_mdn_base['mu'][:, -1, :],
                    'log_s': dur_mdn_base['log_s'][:, -1, :],
                }
            if isinstance(dose_mdn_selected, dict) and dose_mdn_selected['pi_logits'].dim() == 3:
                dose_mdn_selected = {
                    'pi_logits': dose_mdn_selected['pi_logits'][:, -1, :],
                    'mu': dose_mdn_selected['mu'][:, -1, :],
                    'log_s': dose_mdn_selected['log_s'][:, -1, :],
                }
            if isinstance(dur_mdn_selected, dict) and dur_mdn_selected['pi_logits'].dim() == 3:
                dur_mdn_selected = {
                    'pi_logits': dur_mdn_selected['pi_logits'][:, -1, :],
                    'mu': dur_mdn_selected['mu'][:, -1, :],
                    'log_s': dur_mdn_selected['log_s'][:, -1, :],
                }
            
            # Sample dose, dur from their distributions
            dose_sample_model = _to_dose_model_space(dose_pred, dose_log)
            if isinstance(dose_mdn_selected, dict):
                dose_pi_logits = dose_mdn_selected['pi_logits']  # (B, K)
                dose_mu = dose_mdn_selected['mu']                # (B, K)
                dose_log_s = dose_mdn_selected['log_s']          # (B, K)
                dose_comp_idx = torch.distributions.Categorical(logits=dose_pi_logits).sample()
                dose_gather_idx = dose_comp_idx.unsqueeze(-1)
                dose_mu_sel = torch.gather(dose_mu, 1, dose_gather_idx).squeeze(-1)
                dose_s_sel = torch.exp(torch.gather(dose_log_s, 1, dose_gather_idx).squeeze(-1))
                dose_u = torch.rand_like(dose_mu_sel).clamp(min=1e-6, max=1.0 - 1e-6)
                dose_sample_model = dose_mu_sel + dose_s_sel * (torch.log(dose_u) - torch.log1p(-dose_u))
            dose_min = float(getattr(self.config, 'dose_min_value', 1.0))
            dose_max = float(getattr(self.config, 'dose_max_value', 3.0))
            if dose_min < 0.0 or dose_max < 0.0:
                if dose_continuous:
                    dose_min, dose_max = 0.0, float(getattr(self.config, 'dur_max_value', 550.0))
                elif bool(getattr(self.config, 'apply_token_shift', False)):
                    dose_min, dose_max = 2.0, 4.0
                else:
                    dose_min, dose_max = 1.0, 3.0
            dose_min_model, dose_max_model = _dose_bounds_for_model_space(
                dose_min,
                dose_max,
                dose_log,
            )
            if dose_continuous:
                dose_sample_model = torch.clamp(dose_sample_model, min=dose_min_model, max=dose_max_model)
                dose_next = _from_dose_model_space(dose_sample_model, dose_log)
                dose_next = torch.clamp(dose_next, min=dose_min, max=dose_max).unsqueeze(-1)
            else:
                dose_next = (
                    torch.clamp(dose_sample_model.round(), min=dose_min, max=dose_max)
                    .long()
                    .unsqueeze(-1)
                )
            
            # DURATION generation:
            # - Prefer MDN sampling (component sampling + logistic sampling)
            # - Fallback to legacy point estimate path for backward compatibility
            dur_raw = dur_pred
            if isinstance(dur_mdn_selected, dict):
                pi_logits = dur_mdn_selected['pi_logits']  # (B, K)
                mu = dur_mdn_selected['mu']                # (B, K)
                log_s = dur_mdn_selected['log_s']          # (B, K)
                comp_idx = torch.distributions.Categorical(logits=pi_logits).sample()  # (B,)
                gather_idx = comp_idx.unsqueeze(-1)
                mu_sel = torch.gather(mu, 1, gather_idx).squeeze(-1)
                s_sel = torch.exp(torch.gather(log_s, 1, gather_idx).squeeze(-1))
                u = torch.rand_like(mu_sel).clamp(min=1e-6, max=1.0 - 1e-6)
                dur_raw = mu_sel + s_sel * (torch.log(u) - torch.log1p(-u))
            elif self.config.dur_log_transform:
                dur_raw = torch.expm1(dur_pred)  # legacy inverse of log1p
            
            dur_next = (
                torch.clamp(
                    dur_raw.round(),
                    min=0,
                    max=self.config.dur_vocab_size - 1,
                )
                .long()
                .unsqueeze(-1)
            )
            
            # Append to sequences
            data = torch.cat((data, data_next), dim=1)
            dose = torch.cat((dose, dose_next), dim=1)
            dur = torch.cat((dur, dur_next), dim=1)
            age = torch.cat((age, age_next), dim=1)
            
            # Check termination
            if torch.logical_or(
                torch.isin(data, termination_tokens).any(-1), 
                age_next > max_age
            ).all():
                break
        
        return data, dose, dur, age, logits
