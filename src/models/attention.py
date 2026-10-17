"""
Multi-Head Self-Attention Module.

Implements: Attention(Q, K, V) = Softmax(QK^T / sqrt(d_k)) V
Supports ablation over 8 / 12 / 16 heads as specified in the README.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for the recursive refinement loop.

    Computes: Attention(Q, K, V) = Softmax(QK^T / sqrt(d_k)) V

    The number of heads can be ablated over {8, 12, 16} for the research study.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            hidden_size: Input/output embedding dimension.
            num_heads: Number of attention heads (ablate over 8, 12, 16).
            dropout: Dropout rate for attention weights.
            use_layer_norm: Whether to apply layer normalization after attention.
        """
        super().__init__()

        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of multi-head self-attention.

        Args:
            x: Input embeddings [batch, seq_len, hidden_size]
            attention_mask: Mask [batch, seq_len] (1 for valid, 0 for padding)
            return_attention_weights: Whether to return attention weight matrices.

        Returns:
            Tuple of:
                - output: [batch, seq_len, hidden_size]
                - attn_weights: [batch, num_heads, seq_len, seq_len] or None
        """
        batch_size, seq_len, _ = x.shape
        residual = x

        # Project to Q, K, V
        Q = self.q_proj(x)  # [batch, seq_len, hidden_size]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head: [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: Softmax(QK^T / sqrt(d_k)) V
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # attn_scores: [batch, num_heads, seq_len, seq_len]

        # Apply attention mask (if provided)
        if attention_mask is not None:
            # Expand mask: [batch, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: [batch, num_heads, seq_len, head_dim]

        # Reshape back: [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )

        # Output projection
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)

        # Residual connection + layer norm
        output = output + residual
        if self.use_layer_norm:
            output = self.layer_norm(output)

        if return_attention_weights:
            return output, attn_weights
        return output, None

    def get_attention_map(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Get attention heatmap for explainability.

        Args:
            x: Input embeddings [batch, seq_len, hidden_size]
            attention_mask: Mask [batch, seq_len]
            head_reduction: How to reduce across heads ("mean", "max", or int for specific head).

        Returns:
            Attention map [batch, seq_len, seq_len]
        """
        _, attn_weights = self.forward(x, attention_mask, return_attention_weights=True)
        # attn_weights: [batch, num_heads, seq_len, seq_len]

        if head_reduction == "mean":
            return attn_weights.mean(dim=1)  # [batch, seq_len, seq_len]
        elif head_reduction == "max":
            return attn_weights.max(dim=1)[0]
        elif isinstance(head_reduction, int):
            return attn_weights[:, head_reduction, :, :]
        else:
            return attn_weights.mean(dim=1)
