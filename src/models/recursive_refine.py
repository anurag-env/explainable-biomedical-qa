"""
Recursive Embedding Refinement Layer.

Novel component: E_{t+1} = Attention(E_t) + GRU(E_t)
Iterates until ||E_{t+1} - E_t|| < epsilon.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from .attention import MultiHeadSelfAttention


class RecursiveEmbeddingRefinement(nn.Module):
    """
    Recursive Embedding Refinement (RER) module.
    E_{t+1} = gate * Attention(E_t) + (1-gate) * GRU(E_t)
    Converges when ||E_{t+1} - E_t|| < epsilon.
    """

    def __init__(self, hidden_size=768, num_heads=12, max_iterations=5,
                 convergence_epsilon=1e-3, gru_num_layers=1, dropout=0.1,
                 residual_connection=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_iterations = max_iterations
        self.convergence_epsilon = float(convergence_epsilon)
        self.residual_connection = residual_connection

        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_size, num_heads=num_heads,
            dropout=dropout, use_layer_norm=True,
        )
        self.gru = nn.GRU(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=gru_num_layers, batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, E_0, attention_mask=None, return_all_iterations=False):
        E_t = E_0
        convergence_norms = []
        all_embeddings = [E_0] if return_all_iterations else []
        attention_weights = None

        for t in range(self.max_iterations):
            attn_out, attn_w = self.attention(E_t, attention_mask, return_attention_weights=True)
            attention_weights = attn_w
            gru_out, _ = self.gru(E_t)

            gate = self.fusion_gate(torch.cat([attn_out, gru_out], dim=-1))
            E_next = gate * attn_out + (1 - gate) * gru_out

            if self.residual_connection:
                E_next = E_next + E_t
            E_next = self.layer_norm(E_next)
            E_next = self.dropout(E_next)

            diff_norm = torch.norm(E_next - E_t, p=2, dim=-1).mean().item()
            convergence_norms.append(diff_norm)

            if return_all_iterations:
                all_embeddings.append(E_next)
            if diff_norm < self.convergence_epsilon:
                E_t = E_next
                break
            E_t = E_next

        return {
            "refined_embedding": E_t,
            "num_iterations": t + 1,
            "convergence_norms": convergence_norms,
            "all_embeddings": all_embeddings if return_all_iterations else None,
            "attention_weights": attention_weights,
        }

    def get_convergence_info(self, E_0, attention_mask=None):
        result = self.forward(E_0, attention_mask, return_all_iterations=True)
        return {
            "num_iterations": result["num_iterations"],
            "convergence_norms": result["convergence_norms"],
            "converged": result["convergence_norms"][-1] < self.convergence_epsilon if result["convergence_norms"] else False,
            "final_norm": result["convergence_norms"][-1] if result["convergence_norms"] else float("inf"),
        }
