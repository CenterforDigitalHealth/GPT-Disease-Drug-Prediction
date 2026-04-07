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
    """Standard RoPE over sequence positions; age is injected separately."""
    
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
        # `age` is accepted for interface consistency. Attention itself uses
        # sequence-position RoPE; age enters via additive AgeEncoding upstream.
        
        # Compute Q, K, V
        q = self.q_proj(x)  # (B, T, n_embd)
        k = self.k_proj(x)  # (B, T, n_kv_head * head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_head * head_dim)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        
        # Apply standard sequence-position RoPE (not age-aware RoPE)
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
            
            # Optional externally supplied mask (e.g. padding / tie masking)
            if attn_mask is not None:
                att = att.masked_fill(attn_mask == 0, float('-inf'))
            
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        # Reassemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))
        
        return y, att


class SwiGLUExpert(nn.Module):
    """Single MoE expert with SwiGLU activation."""

    def __init__(self, n_embd, intermediate_size, bias, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 2 * intermediate_size, bias=bias)
        self.c_proj = nn.Linear(intermediate_size, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = swiglu(x, limit=7.0)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class MixtureOfExperts(nn.Module):
    """Lightweight MoE with SwiGLU experts for domain-specific medical knowledge"""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts if hasattr(config, 'num_experts') else 4
        self.experts_per_token = config.experts_per_token if hasattr(config, 'experts_per_token') else 2
        self.n_embd = config.n_embd
<<<<<<< HEAD
        self.intermediate_size = int(getattr(config, 'moe_intermediate_size', 0) or (2 * config.n_embd))
=======
        self.intermediate_size = 2 * config.n_embd
>>>>>>> 3053ef3 (reorg repo with final model)

        # Router
        self.gate = nn.Linear(config.n_embd, self.num_experts, bias=False)

        # SwiGLU experts
        self.experts = nn.ModuleList([
            SwiGLUExpert(config.n_embd, self.intermediate_size, config.bias, config.dropout)
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
<<<<<<< HEAD
        self.intermediate_size = int(getattr(config, 'ffn_intermediate_size', 0) or (2 * config.n_embd))
=======
        self.intermediate_size = 2 * config.n_embd
>>>>>>> 3053ef3 (reorg repo with final model)
        
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

        # Backbone FiLM: drug conditioning inside transformer blocks
        self.backbone_film = None
        if getattr(config, 'film_in_backbone', False):
            self.backbone_film = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd * 2),
            )

    def forward(self, x, age=None, attn_mask=None, drug_emb=None, drug_token_mask=None):
        # Pre-norm architecture (more stable)
        y, att = self.attn(self.ln_1(x), age, attn_mask)
        x = x + y
        mlp_out = self.mlp(self.ln_2(x))
        aux_loss = None
        if isinstance(mlp_out, tuple):
            mlp_out, aux_loss = mlp_out
        x = x + mlp_out

        # Backbone FiLM: modulate hidden state with drug identity
        if self.backbone_film is not None and drug_emb is not None:
            film = self.backbone_film(drug_emb)
            gamma, beta = film.chunk(2, dim=-1)
            x_mod = gamma * x + beta
            if drug_token_mask is not None:
                x = torch.where(drug_token_mask.unsqueeze(-1), x_mod, x)
            else:
                x = x_mod

        return x, att, aux_loss

# =============================================================================
# Composite Embedding + Multi-Head Output Architecture
# =============================================================================

class CompositeEmbedding(nn.Module):
    """
    Composite Embedding Layer: 여러 입력 필드를 각각 임베딩하고 투영
    - DATA (약품/질병 코드) -> ID Embedding
    - SHIFT (시프트 값) -> Shift Embedding
    - TOTAL (기간) -> Duration Embedding
    
    Concatenation + Projection: 각 필드의 정보를 더 잘 보존
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        
        # 각 필드별 Embedding
        self.data_emb = nn.Embedding(config.data_vocab_size, config.n_embd)
        self.shift_emb = nn.Embedding(config.shift_vocab_size, config.n_embd)
        self.total_emb = nn.Embedding(config.total_vocab_size, config.n_embd)
        
        # Concatenation → Projection (3*n_embd → n_embd)
        self.proj = nn.Linear(config.n_embd * 3, config.n_embd, bias=False)
        
    def forward(self, data, shift, total):
        """
        Args:
            data: (B, T) DATA tokens
            shift: (B, T) SHIFT values (정수값)
            total: (B, T) TOTAL tokens
        Returns:
            combined embedding (B, T, n_embd)
        """
        # DATA embedding (clamp to valid range)
        data_idx = torch.clamp(data, min=0, max=self.data_emb.num_embeddings - 1)
        data_emb = self.data_emb(data_idx)
        
        # SHIFT embedding (clamp to valid range)
        shift_idx = torch.clamp(shift, min=0, max=self.shift_emb.num_embeddings - 1)
        shift_emb = self.shift_emb(shift_idx)
        
        # TOTAL embedding (clamp to valid range)
        total_idx = torch.clamp(total, min=0, max=self.total_emb.num_embeddings - 1)
        total_emb = self.total_emb(total_idx)
        
        # Concatenate + Project (preserves each field's information better than sum)
        combined = torch.cat([data_emb, shift_emb, total_emb], dim=-1)  # (B, T, 3*n_embd)
        combined = self.proj(combined)  # (B, T, n_embd)
        
        return combined


class MixtureDensityHead(nn.Module):
    """
    Mixture of Logistics head for multi-modal TOTAL distribution.
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

        # Keep means in valid TOTAL range
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


class ShiftClassificationHead(nn.Module):
    """
    SHIFT classification head with configurable number of classes.
    Default 3-class: decrease(0) / maintain(1) / increase(2)
    """

    def __init__(self, n_embd: int, dropout: float, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            nn.Dropout(dropout),
            nn.Linear(n_embd, num_classes),
        )

    def forward(self, x: torch.Tensor):
        return self.head(x)


class MultiHeadOutput(nn.Module):
    """
    Multi-Head Output Layer: 각 필드별 예측 헤드
    - DATA Head: 다음 DATA 토큰 예측 (Classification)
    - SHIFT Head: 다음 SHIFT 값 예측 (Classification)
    - TOTAL Head: 다음 TOTAL 값 예측 (Regression, 연속값)
    - Time Head: 다음 이벤트까지의 시간 예측
      - Exponential: scale (λ) parameter만 예측
      - Weibull: scale (λ) + shape (k) parameter 예측
    
    Drug-Conditioned Heads (optional):
    - 약물(drug) 정보를 조건으로 SHIFT/TOTAL 예측 성능 향상
    - FiLM (Feature-wise Linear Modulation) 방식 사용
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.time_distribution = getattr(config, 'time_distribution', 'exponential')
        self.mdn_n_components = int(getattr(config, 'mdn_n_components', 8))
        self.total_min_value = float(getattr(config, 'total_min_value', 0.0))
        self.total_max_value = float(getattr(config, 'total_max_value', 550.0))
        
        # Drug-conditioning option
        self.use_drug_conditioning = getattr(config, 'use_drug_conditioning', False)
        self.num_shift_classes = int(getattr(config, 'num_shift_classes', 3))

        # Classification Heads (DATA, SHIFT)
        self.data_head = nn.Linear(config.n_embd, config.data_vocab_size, bias=False)

        # SHIFT head: configurable number of classes (default 3: decrease/maintain/increase)
        self.shift_head = ShiftClassificationHead(
            n_embd=config.n_embd,
            dropout=float(getattr(config, 'dropout', 0.1)),
            num_classes=self.num_shift_classes,
        )

        # TOTAL head: MDN (Mixture of Logistics)
        self.total_head = MixtureDensityHead(
            n_embd=config.n_embd,
            n_components=self.mdn_n_components,
            min_value=self.total_min_value,
            max_value=self.total_max_value,
        )

        # ============================================================
        # Drug-Conditioned Heads (FiLM style)
        # ============================================================
        if self.use_drug_conditioning:
            film_dropout = float(getattr(config, 'film_dropout', 0.0))

            # FiLM generator: drug_emb -> (gamma, beta) for modulation
            shift_film_layers = [
                nn.Linear(config.n_embd, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd * 2),
            ]
            if film_dropout > 0.0:
                shift_film_layers.append(nn.Dropout(film_dropout))
            self.shift_film_generator = nn.Sequential(*shift_film_layers)

            self.shift_drug_cond_head = ShiftClassificationHead(
                n_embd=config.n_embd,
                dropout=float(getattr(config, 'dropout', 0.1)),
                num_classes=self.num_shift_classes,
            )

            # TOTAL FiLM
            total_film_layers = [
                nn.Linear(config.n_embd, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd * 2),
            ]
            if film_dropout > 0.0:
                total_film_layers.append(nn.Dropout(film_dropout))
            self.total_film_generator = nn.Sequential(*total_film_layers)

            self.total_drug_cond_head = MixtureDensityHead(
                n_embd=config.n_embd,
                n_components=self.mdn_n_components,
                min_value=self.total_min_value,
                max_value=self.total_max_value,
            )
            # Provisional identity init here; reapplied after parent model init.
            self.reset_film_identity()
        
        # Time Head: scale parameter (λ) for all event types
        self.time_head = nn.Linear(config.n_embd, config.data_vocab_size, bias=False)
        
        # Weibull shape parameter (k) - 전역 또는 per-event
        if self.time_distribution == 'weibull':
            # Shape parameter per event type (more expressive)
            self.time_shape_head = nn.Linear(config.n_embd, config.data_vocab_size, bias=False)

    def reset_film_identity(self):
        """Initialize FiLM generators to identity: gamma=1, beta=0."""
        if self.use_drug_conditioning:
            for film_gen in [self.shift_film_generator, self.total_film_generator]:
                # Find last Linear robustly (may have trailing Dropout)
                last_layer = next(m for m in reversed(list(film_gen.modules()))
                                  if isinstance(m, nn.Linear))
                nn.init.zeros_(last_layer.weight)
                with torch.no_grad():
                    # first half = gamma, second half = beta
                    last_layer.bias[:self.n_embd].fill_(1.0)
                    last_layer.bias[self.n_embd:].zero_()

        # MDN heads are sensitive to init; keep spread-out initial components.
        self.total_head.reset_mdn_bias()
        if self.use_drug_conditioning:
            self.total_drug_cond_head.reset_mdn_bias()
        
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
        shift_logits = self.shift_head(x)
        total_mdn = self.total_head(x)
        output = {
            'data': self.data_head(x),             # (B, T, data_vocab_size) - classification logits
            'shift': shift_logits,                 # (B, T, 2) - binary change logits
            'total': total_mdn['mean'],           # (B, T) - point estimate for compatibility
            'total_mdn': total_mdn,               # MDN params for NLL training/sampling
            'time_scale': self.time_head(x),      # (B, T, data_vocab_size) - λ parameter
        }
        
        # ============================================================
        # Drug-Conditioned Predictions (FiLM modulation)
        # Only applies when target is a drug token (drug_token_mask=True)
        # ============================================================
        if self.use_drug_conditioning and drug_emb is not None:
            # SHIFT: FiLM modulation with drug embedding
            shift_film = self.shift_film_generator(drug_emb)  # (B, T, n_embd*2)
            shift_gamma, shift_beta = shift_film.chunk(2, dim=-1)  # 각각 (B, T, n_embd)
            shift_modulated = shift_gamma * x + shift_beta  # FiLM: γ * x + β
            shift_drug_cond = self.shift_drug_cond_head(shift_modulated)
            
            # TOTAL: FiLM modulation with drug embedding
            total_film = self.total_film_generator(drug_emb)  # (B, T, n_embd*2)
            total_gamma, total_beta = total_film.chunk(2, dim=-1)  # 각각 (B, T, n_embd)
            total_modulated = total_gamma * x + total_beta  # FiLM: γ * x + β
            total_drug_mdn = self.total_drug_cond_head(total_modulated)
            total_drug_cond = total_drug_mdn['mean']  # (B, T)
            
            # Apply drug token masking: only use FiLM output where target is a drug
            if drug_token_mask is not None:
                # Blend: use FiLM output for drug tokens, standard output otherwise
                # SHIFT: (B, T, shift_vocab_size)
                shift_drug_cond_masked = torch.where(
                    drug_token_mask.unsqueeze(-1),  # (B, T, 1)
                    shift_drug_cond,
                    output['shift']
                )
                output['shift_drug_cond'] = shift_drug_cond_masked
                
                # TOTAL: (B, T)
                total_drug_cond_masked = torch.where(
                    drug_token_mask,  # (B, T)
                    total_drug_cond,
                    output['total']
                )
                output['total_drug_cond'] = total_drug_cond_masked
                # MDN params: (B, T, K)
                mask_mdn = drug_token_mask.unsqueeze(-1)
                output['total_mdn_drug_cond'] = {
                    'pi_logits': torch.where(mask_mdn, total_drug_mdn['pi_logits'], output['total_mdn']['pi_logits']),
                    'mu': torch.where(mask_mdn, total_drug_mdn['mu'], output['total_mdn']['mu']),
                    'log_s': torch.where(mask_mdn, total_drug_mdn['log_s'], output['total_mdn']['log_s']),
                    'mean': total_drug_cond_masked,
                }
            else:
                # No mask: apply FiLM to all positions (backward compatibility)
                output['shift_drug_cond'] = shift_drug_cond
                output['total_drug_cond'] = total_drug_cond
                output['total_mdn_drug_cond'] = total_drug_mdn
        
        # For backward compatibility
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
    # Token space convention: apply_token_shift=False (raw tokens, 0 reserved for padding via clamp_min).
<<<<<<< HEAD
    # - DATA: raw 0-1288. Drug tokens: Metformin(1278)..Death(1288). vocab_size=1290.
    # - SHIFT: raw 0-4. Values 1=decrease, 2=maintain, 3=increase, 0=padding, 4=NA.
    # - TOTAL: raw 0-550. vocab_size=552 (embedding input only).
    #
    # Head outputs:
    # - DATA Head: Linear(n_embd, 1290) → Cross-Entropy (Classification)
    # - SHIFT Head: ShiftClassificationHead → num_shift_classes logits (default 3: dec/maint/inc)
    # - TOTAL Head: MixtureDensityHead → MDN NLL (Regression, continuous)
    data_vocab_size: int = 1290   # DATA embedding & head — max raw token 1288 (Death) + padding overhead
    # NOTE: shift_vocab_size is used for SHIFT embedding input space only.
    # The SHIFT classification head output size is controlled by num_shift_classes.
    shift_vocab_size: int = 5
    total_vocab_size: int = 552   # TOTAL embedding only — raw range 0-550, 0 doubles as padding
=======
    # - DATA: raw 2-1288. Drug tokens: Metformin(1278)..Death(1288). vocab_size=1289.
    # - SHIFT: raw 0-3. Values 0=non-drug, 1=decrease, 2=maintain, 3=increase.
    # - TOTAL: raw 0-550. vocab_size=551 (embedding input only).
    #
    # Head outputs:
    # - DATA Head: Linear(n_embd, 1289) → Cross-Entropy (Classification)
    # - SHIFT Head: ShiftClassificationHead → num_shift_classes logits (default 3: dec/maint/inc)
    # - TOTAL Head: MixtureDensityHead → MDN NLL (Regression, continuous)
    data_vocab_size: int = 1289   # DATA embedding & head — max raw token 1288 (Death)
    # NOTE: shift_vocab_size is used for SHIFT embedding input space only.
    # The SHIFT classification head output size is controlled by num_shift_classes.
    shift_vocab_size: int = 4     # SHIFT raw range 0-3
    total_vocab_size: int = 551   # TOTAL embedding only — raw range 0-550
>>>>>>> 3053ef3 (reorg repo with final model)
    
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
    
    # Drug-Conditioning via FiLM (Feature-wise Linear Modulation)
    use_drug_conditioning: bool = True
    film_dropout: float = 0.0
    
    # Drug token range defaults for apply_token_shift=False (raw tokens):
    # Metformin(1278) ... Death(1288)
    # If apply_token_shift=True, override via model_args to 1279..1289.
    drug_token_min: int = 1278
    drug_token_max: int = 1288
    apply_token_shift: bool = False
    separate_shift_na_from_padding: bool = False
    shift_na_raw_token: int = 4

    # Architecture features
    use_moe: bool = True
    num_experts: int = 4
    experts_per_token: int = 2
<<<<<<< HEAD
    moe_intermediate_size: int = 0  # 0 => derive from current default (2 * n_embd)
    ffn_intermediate_size: int = 0  # 0 => derive from current default (2 * n_embd)
=======
>>>>>>> 3053ef3 (reorg repo with final model)
    sliding_window: int = 512
    rope_theta: float = 10000.0

    # SHIFT head
    num_shift_classes: int = 3  # 3-class: decrease(0) / maintain(1) / increase(2)
    drug_token_only_shift: bool = True  # compute SHIFT loss only at drug token positions
    drug_token_only_total: bool = True  # compute TOTAL loss only at drug token positions

    # TOTAL MDN options
    mdn_n_components: int = 8
    total_min_value: float = 0.0
    total_max_value: float = 550.0
    total_log_transform: bool = False

    # Uncertainty-weighted multi-task learning (Kendall 2018)
    use_uncertainty_weighting: bool = False

    # FiLM conditioning in transformer backbone (not just output layer)
    film_in_backbone: bool = False

    # Teacher forcing: use target tokens for drug conditioning during training
    use_teacher_forcing_drug_cond: bool = False

    # DATA head label smoothing (improves calibration → AUC)
    data_label_smoothing: float = 0.0

    # Loss weights (used when use_uncertainty_weighting=False)
    loss_weight_data: float = 1.0
    loss_weight_shift: float = 20.0
    loss_weight_change: float = 0.0
    loss_weight_total: float = 5.0
    loss_weight_time: float = 1.0

    # SHIFT loss options
    shift_loss_type: str = 'dice_focal'
    shift_dice_weight: float = 0.5
    shift_ignore_index: int = -1
    shift_maintain_idx: int = 2
    shift_change_weight_max: float = 10.0
    shift_focal_gamma: float = 2.0
    shift_class_weights: list = field(default_factory=list)

    # Time-to-Event distribution: 'exponential' or 'weibull'
    time_distribution: str = 'exponential'


class CompositeDelphi(nn.Module):
    """
    Composite Delphi: Composite Embedding + Multi-Head Output
    
    입력: (DATA, SHIFT, TOTAL, AGE)
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
        
        # Uncertainty-weighted multi-task learning (Kendall 2018)
        if getattr(config, 'use_uncertainty_weighting', False):
            self.log_sigma_data = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_shift = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_total = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_time = nn.Parameter(torch.tensor(0.0))

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        # Re-apply FiLM identity init after global init to avoid overwrite.
        self.multi_head.reset_film_identity()

        # Backbone FiLM identity init
        if getattr(config, 'film_in_backbone', False):
            for block in self.h:
                if block.backbone_film is not None:
                    last_layer = block.backbone_film[-1]  # last Linear
                    nn.init.zeros_(last_layer.weight)
                    with torch.no_grad():
                        last_layer.bias[:config.n_embd].fill_(1.0)   # gamma = 1
                        last_layer.bias[config.n_embd:].zero_()      # beta = 0
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, data, shift, total, age,
                targets_data=None, targets_shift=None, targets_total=None,
                targets_age=None, drug_conditioning_data=None,
                validation_loss_mode=False):
        """
        Args:
            data: (B, T) DATA tokens
            shift: (B, T) SHIFT values
            total: (B, T) TOTAL tokens
            age: (B, T) AGE values
            targets_*: 각 필드의 타겟 (optional)
        """
        device = data.device
        b, t = data.size()
        
        # 1. Composite Embedding
        composite_emb = self.composite_emb(data, shift, total)
        
        # 2. Age encoding is added to the hidden states before attention blocks.
        age_emb = self.age_encoding(age)
        
        # 3. Combine embeddings
        x = self.token_drop(composite_emb) * (1 - self.config.token_dropout)
        x = x + age_emb
        x = self.drop(x)
        
        # 4. Build an attention mask. This can enforce age-based exclusions,
        # but attention scores themselves are still standard RoPE attention.
        attn_mask = (data > 0).view(b, 1, 1, t) * (data > 0).view(b, 1, t, 1)
        attn_mask *= torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0
        
        if targets_data is not None and self.config.mask_ties:
            attn_mask *= (age.view(b, 1, 1, t) != targets_age.view(b, 1, t, 1))
            attn_mask += (attn_mask.sum(-1, keepdim=True) == 0) * torch.diag(torch.ones(t, device=device)) > 0
        
        attn_mask = attn_mask + (data == 0).view(b, 1, 1, t) * torch.diag(torch.ones(t, device=device)) > 0
        attn_mask *= torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0
        
        # 5. Drug conditioning (compute before transformer blocks for backbone FiLM)
        drug_emb = None
        drug_token_mask = None
        if self.config.use_drug_conditioning or getattr(self.config, 'film_in_backbone', False):
            # Teacher forcing: use target tokens during training for cleaner drug signal
            if getattr(self.config, 'use_teacher_forcing_drug_cond', False) and targets_data is not None:
                drug_source = targets_data
            else:
                drug_source = data  # Current input tokens (inference / no teacher forcing)
            if drug_source is not None:
                drug_source_clamped = torch.clamp(
                    drug_source,
                    min=0,
                    max=self.composite_emb.data_emb.num_embeddings - 1,
                )
                drug_emb = self.composite_emb.data_emb(drug_source_clamped)

            drug_token_min = getattr(self.config, 'drug_token_min', 1278)
            drug_token_max = getattr(self.config, 'drug_token_max', 1288)
            if targets_data is not None:
                drug_token_mask = (targets_data >= drug_token_min) & (targets_data <= drug_token_max)

        # 6. Transformer blocks
        # Skip attention weight collection during training to save ~13GB+ GPU memory
        is_training = targets_data is not None
        att_list = []
        aux_losses = []
        for block in self.h:
            x, att, aux_loss = block(x, age, attn_mask,
                                     drug_emb=drug_emb, drug_token_mask=drug_token_mask)
            if not is_training:
                att_list.append(att)
            if aux_loss is not None:
                aux_losses.append(aux_loss)

        # Average MoE load balancing loss across layers
        moe_aux_loss = sum(aux_losses) / max(len(aux_losses), 1) if aux_losses else None

        x = self.ln_f(x)
        att = torch.stack(att_list) if (att_list and att_list[0] is not None) else None

        # 7. Multi-Head Output
        logits = self.multi_head(x, drug_emb=drug_emb, drug_token_mask=drug_token_mask)
        
        # 8. Compute losses if targets provided
        if targets_data is not None:
            loss = self._compute_loss(
                logits, data, age,
                targets_data, targets_shift, targets_total, targets_age,
                attn_mask, validation_loss_mode,
                moe_aux_loss=moe_aux_loss
            )
        else:
            loss = None
        
        return logits, loss, att
    
    def _compute_loss(self, logits, data, age,
                      targets_data, targets_shift, targets_total, targets_age,
                      attn_mask, validation_loss_mode,
                      moe_aux_loss=None):
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
        
        # Clamp targets to valid vocab range (defensive measure)
        data_vocab_size = self.config.data_vocab_size
        targets_flat_clamped = torch.clamp(targets_flat, min=0, max=data_vocab_size - 1)
        
        # 1. DATA Cross-Entropy Loss
        data_logits = logits['data']
        if validation_loss_mode:
            data_logits[..., ignored_tokens] = -torch.inf
        
        # Label smoothing distributes mass over *all* classes. Masking any class logits to
        # -inf (validation_loss_mode) makes those terms -inf in log-softmax → CE becomes inf.
        # Training keeps smoothing; eval/estimate_loss must use smoothing=0 here.
        data_label_smoothing = float(getattr(self.config, 'data_label_smoothing', 0.0))
        if validation_loss_mode:
            data_label_smoothing = 0.0

        loss_data = F.cross_entropy(
            data_logits.reshape(-1, data_logits.size(-1))[pass_tokens],
            targets_flat_clamped[pass_tokens],  # ← clamp된 값 사용
            ignore_index=-1,
            label_smoothing=data_label_smoothing,
        )
        
        # 2. SHIFT Loss (3-class: decrease / maintain / increase)
        shift_logits_source = logits['shift']
        if 'shift_drug_cond' in logits and self.config.use_drug_conditioning:
            shift_logits_source = logits['shift_drug_cond']

        shift_targets_all = targets_shift.reshape(-1)
        shift_pass_tokens = shift_targets_all != -1

        shift_logits_flat = shift_logits_source.reshape(-1, shift_logits_source.size(-1))[shift_pass_tokens]
        shift_targets_flat = shift_targets_all[shift_pass_tokens].long()

        # Also get DATA tokens at same positions for drug-token-only filtering
        data_flat_for_shift = targets_data.reshape(-1)[shift_pass_tokens]

        # Remap raw SHIFT labels to class indices
        if getattr(self.config, 'apply_token_shift', False):
            remap = {2: 0, 3: 1, 4: 2}  # shifted: decrease=2, maintain=3, increase=4
        else:
            remap = {1: 0, 2: 1, 3: 2}  # raw: decrease=1, maintain=2, increase=3

        num_shift_classes = int(getattr(self.config, 'num_shift_classes', 3))
        if num_shift_classes == 2:
            # Binary fallback: label1->0, label2/3->1
            if getattr(self.config, 'apply_token_shift', False):
                remap = {2: 0, 3: 1, 4: 1}
            else:
                remap = {1: 0, 2: 1, 3: 1}

        mapped = torch.full_like(shift_targets_flat, -1)
        for raw_val, cls in remap.items():
            mapped[shift_targets_flat == raw_val] = cls
        shift_valid = mapped >= 0

        # Drug-token-only SHIFT loss
        if getattr(self.config, 'drug_token_only_shift', True):
            drug_token_min = getattr(self.config, 'drug_token_min', 1278)
            drug_token_max = getattr(self.config, 'drug_token_max', 1288)
            drug_mask = (data_flat_for_shift >= drug_token_min) & (data_flat_for_shift <= drug_token_max)
            shift_valid = shift_valid & drug_mask

        shift_logits_flat = shift_logits_flat[shift_valid]
        mapped_targets = mapped[shift_valid]

        loss_shift = torch.tensor(0.0, device=device)
        loss_change = torch.tensor(0.0, device=device)
        if shift_logits_flat.numel() > 0:
            shift_loss_type = str(getattr(self.config, 'shift_loss_type', 'dice_focal')).lower()
            weights_list = getattr(self.config, 'shift_class_weights', None)
            weight_t = None
            if isinstance(weights_list, (list, tuple)) and len(weights_list) == num_shift_classes:
                weight_t = torch.tensor(weights_list, device=device, dtype=torch.float32)

            gamma = float(getattr(self.config, 'shift_focal_gamma', 2.0))
            focal_term = focal_loss_multiclass(
                shift_logits_flat,
                mapped_targets,
                gamma=gamma,
                alpha=weight_t,
                ignore_index=None,
                reduction='mean',
            )
            if shift_loss_type == 'dice_focal':
                dice_w = float(getattr(self.config, 'shift_dice_weight', 0.5))
                dice_term = dice_loss_multiclass(
                    shift_logits_flat,
                    mapped_targets,
                    eps=1.0,
                    ignore_index=None,
                )
                loss_shift = dice_w * dice_term + (1.0 - dice_w) * focal_term
            elif shift_loss_type == 'focal':
                loss_shift = focal_term
            else:
                loss_shift = F.cross_entropy(
                    shift_logits_flat,
                    mapped_targets,
                    weight=weight_t,
                    ignore_index=-1,
                )

        # 3. TOTAL Loss (Mixture Density NLL preferred)
        # Drug-token-only TOTAL loss: restrict to drug positions (same pattern as SHIFT)
        total_tokens = pass_tokens
        if getattr(self.config, 'drug_token_only_total', True):
            drug_token_min = getattr(self.config, 'drug_token_min', 1278)
            drug_token_max = getattr(self.config, 'drug_token_max', 1288)
            total_drug_mask = (targets_flat >= drug_token_min) & (targets_flat <= drug_token_max)
            total_tokens = pass_tokens & total_drug_mask

        total_target = targets_total.float().reshape(-1)[total_tokens]  # (N,)
        total_mdn_source = logits.get('total_mdn', None)
        if 'total_mdn_drug_cond' in logits and self.config.use_drug_conditioning:
            total_mdn_source = logits['total_mdn_drug_cond']

        loss_total = torch.tensor(0.0, device=device)
        if total_target.numel() > 0 and isinstance(total_mdn_source, dict):
            pi_logits = total_mdn_source['pi_logits'].reshape(-1, total_mdn_source['pi_logits'].size(-1))[total_tokens]
            mu = total_mdn_source['mu'].reshape(-1, total_mdn_source['mu'].size(-1))[total_tokens]
            log_s = total_mdn_source['log_s'].reshape(-1, total_mdn_source['log_s'].size(-1))[total_tokens]

            # Mixture of logistics NLL:
            # log p(y) = logsumexp_k(log pi_k + log Logistic(y|mu_k, s_k))
            y = total_target.unsqueeze(-1)
            z = (y - mu) / torch.exp(log_s)
            log_pdf = -z - log_s - 2.0 * F.softplus(-z)
            log_pi = F.log_softmax(pi_logits, dim=-1)
            log_prob = torch.logsumexp(log_pi + log_pdf, dim=-1)
            nll = -log_prob
            loss_total = torch.clamp(nll, min=0.0, max=20.0).mean()
        elif total_target.numel() > 0:
            # Fallback for legacy checkpoints without MDN params
            total_pred = logits['total'].reshape(-1)[total_tokens]
            total_scale = float(getattr(self.config, 'total_max_value', 550.0))
            total_pred = torch.clamp(total_pred, min=0.0, max=total_scale)
            loss_total = F.mse_loss(total_pred, total_target) / (total_scale ** 2)
        
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
            time_logits = logits['time_scale']  # (B, T, data_vocab_size)
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
            
            # Use time head logits for time calculation (original Delphi approach)
            time_logits = logits['time_scale']  # (B, T, data_vocab_size)
            lse = torch.logsumexp(time_logits, -1)  # (B, T)
            lse = -torch.log(torch.exp(-lse) + self.config.t_min)
            
            ldt = -torch.log(dt + self.config.t_min).view(-1)
            loss_time = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt.reshape(-1)))
            loss_time = torch.mean(loss_time[pass_tokens])
        
        # Weighted sum of losses
        if getattr(self.config, 'use_uncertainty_weighting', False):
            # Kendall 2018: L = Σ (1/2)*exp(-2*log_σ_i)*L_i + log_σ_i
            # exp(-2*log_σ) = 1/σ² acts as precision weight
            total_loss = (
                0.5 * torch.exp(-2.0 * self.log_sigma_data) * loss_data + self.log_sigma_data +
                0.5 * torch.exp(-2.0 * self.log_sigma_shift) * loss_shift + self.log_sigma_shift +
                0.5 * torch.exp(-2.0 * self.log_sigma_total) * loss_total + self.log_sigma_total +
                0.5 * torch.exp(-2.0 * self.log_sigma_time) * loss_time + self.log_sigma_time
            )
        else:
            total_loss = (
                self.config.loss_weight_data * loss_data +
                self.config.loss_weight_shift * loss_shift +
                self.config.loss_weight_change * loss_change +
                self.config.loss_weight_total * loss_total +
                self.config.loss_weight_time * loss_time
            )
        
        # MoE load balancing loss (α=0.01)
        loss_moe = torch.tensor(0.0, device=total_loss.device)
        if moe_aux_loss is not None:
            loss_moe = moe_aux_loss
            total_loss = total_loss + 0.01 * loss_moe
        
        loss_dict = {
            'loss': total_loss,
            'loss_data': loss_data,
            'loss_shift': loss_shift,
            'loss_change': loss_change,
            'loss_total': loss_total,
            'loss_time': loss_time,
            'loss_moe': loss_moe
        }
        if getattr(self.config, 'use_uncertainty_weighting', False):
            loss_dict['sigma_data'] = torch.exp(self.log_sigma_data).item()
            loss_dict['sigma_shift'] = torch.exp(self.log_sigma_shift).item()
            loss_dict['sigma_total'] = torch.exp(self.log_sigma_total).item()
            loss_dict['sigma_time'] = torch.exp(self.log_sigma_time).item()
        return loss_dict
    
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
    def generate(self, data, shift, total, age, 
                 max_new_tokens=100, max_age=85*365.25,
                 no_repeat=True, termination_tokens=None):
        """Generate composite sequences"""
        if termination_tokens is None:
            warnings.warn('Consider setting termination_tokens for your dataset.')
            termination_tokens = [1269]
        
        termination_tokens = torch.tensor(termination_tokens, dtype=torch.int64, device=data.device)
        
        if max_new_tokens == -1:
            max_new_tokens = 128
        
        for _ in range(max_new_tokens):
            logits, _, _ = self(data, shift, total, age, drug_conditioning_data=data)
            
            # Get last position logits
            data_logits = logits['data'][:, -1, :]
            shift_logits_base = logits['shift'][:, -1, :]
            shift_logits_drug = shift_logits_base
            if 'shift_drug_cond' in logits and self.config.use_drug_conditioning:
                shift_logits_drug = logits['shift_drug_cond'][:, -1, :]

            total_pred_base = logits['total'][:, -1]
            total_pred_drug = total_pred_base
            if 'total_drug_cond' in logits and self.config.use_drug_conditioning:
                total_pred_drug = logits['total_drug_cond'][:, -1]
            total_mdn_base = logits.get('total_mdn', None)
            total_mdn_drug = total_mdn_base
            if 'total_mdn_drug_cond' in logits and self.config.use_drug_conditioning:
                total_mdn_drug = logits['total_mdn_drug_cond']
            time_logits = logits['time'][:, -1, :]
            
            # Mask ignored tokens
            data_logits[:, self.config.ignore_tokens] = -torch.inf
            
            if no_repeat:
                fill = data.clone()
                fill[fill == 1] = 0
                data_logits = data_logits.scatter_(1, fill, -torch.inf)
            
            # Sample next tokens from configured time distribution
            time_distribution = getattr(self.config, 'time_distribution', 'exponential')
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

            # Use drug-conditioned SHIFT/TOTAL only when next DATA token is a drug.
            shift_logits = shift_logits_base
            total_pred = total_pred_base
            total_mdn_selected = total_mdn_base
            if self.config.use_drug_conditioning:
                drug_token_min = getattr(self.config, 'drug_token_min', 1278)
                drug_token_max = getattr(self.config, 'drug_token_max', 1288)
                next_is_drug = (data_next >= drug_token_min) & (data_next <= drug_token_max)  # (B, 1)
                shift_logits = torch.where(next_is_drug, shift_logits_drug, shift_logits_base)
                total_pred = torch.where(next_is_drug.squeeze(-1), total_pred_drug, total_pred_base)
                if isinstance(total_mdn_base, dict) and isinstance(total_mdn_drug, dict):
                    mask_mdn = next_is_drug
                    total_mdn_selected = {
                        'pi_logits': torch.where(mask_mdn, total_mdn_drug['pi_logits'][:, -1, :], total_mdn_base['pi_logits'][:, -1, :]),
                        'mu': torch.where(mask_mdn, total_mdn_drug['mu'][:, -1, :], total_mdn_base['mu'][:, -1, :]),
                        'log_s': torch.where(mask_mdn, total_mdn_drug['log_s'][:, -1, :], total_mdn_base['log_s'][:, -1, :]),
                    }
            elif isinstance(total_mdn_base, dict):
                total_mdn_selected = {
                    'pi_logits': total_mdn_base['pi_logits'][:, -1, :],
                    'mu': total_mdn_base['mu'][:, -1, :],
                    'log_s': total_mdn_base['log_s'][:, -1, :],
                }
            if isinstance(total_mdn_selected, dict) and total_mdn_selected['pi_logits'].dim() == 3:
                total_mdn_selected = {
                    'pi_logits': total_mdn_selected['pi_logits'][:, -1, :],
                    'mu': total_mdn_selected['mu'][:, -1, :],
                    'log_s': total_mdn_selected['log_s'][:, -1, :],
                }
            
            # Sample shift, total from their distributions
<<<<<<< HEAD
            shift_next = torch.argmax(shift_logits, dim=-1, keepdim=True)
=======
            # SHIFT head logits are class indices; convert them back to the
            # dataset token values before autoregressive feedback.
            shift_next = torch.argmax(shift_logits, dim=-1, keepdim=True).long()
            shift_next += 2 if bool(getattr(self.config, 'apply_token_shift', False)) else 1
>>>>>>> 3053ef3 (reorg repo with final model)
            
            # TOTAL generation:
            # - Prefer MDN sampling (component sampling + logistic sampling)
            # - Fallback to legacy point estimate path for backward compatibility
            total_raw = total_pred
            if isinstance(total_mdn_selected, dict):
                pi_logits = total_mdn_selected['pi_logits']  # (B, K)
                mu = total_mdn_selected['mu']                # (B, K)
                log_s = total_mdn_selected['log_s']          # (B, K)
                comp_idx = torch.distributions.Categorical(logits=pi_logits).sample()  # (B,)
                gather_idx = comp_idx.unsqueeze(-1)
                mu_sel = torch.gather(mu, 1, gather_idx).squeeze(-1)
                s_sel = torch.exp(torch.gather(log_s, 1, gather_idx).squeeze(-1))
                u = torch.rand_like(mu_sel).clamp(min=1e-6, max=1.0 - 1e-6)
                total_raw = mu_sel + s_sel * (torch.log(u) - torch.log1p(-u))
            elif self.config.total_log_transform:
                total_raw = torch.expm1(total_pred)  # legacy inverse of log1p
            
            total_next = (
                torch.clamp(
                    total_raw.round(),
                    min=0,
                    max=self.config.total_vocab_size - 1,
                )
                .long()
                .unsqueeze(-1)
            )
            
            # Append to sequences
            data = torch.cat((data, data_next), dim=1)
            shift = torch.cat((shift, shift_next), dim=1)
            total = torch.cat((total, total_next), dim=1)
            age = torch.cat((age, age_next), dim=1)
            
            # Check termination
            if torch.logical_or(
                torch.isin(data, termination_tokens).any(-1), 
                age_next > max_age
            ).all():
                break
        
        return data, shift, total, age, logits
