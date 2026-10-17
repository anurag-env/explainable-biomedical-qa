"""
Explainability Module.

Provides SHAP values, Integrated Gradients, and attention heatmaps
for explaining model predictions.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import numpy as np


class ExplainabilityModule(nn.Module):
    """
    Explainability module providing:
    - Attention heatmaps H = {h1, h2, ..., hn} over biomedical entities
    - SHAP(x_i) for feature importance
    - IG(x) (Integrated Gradients) for prediction-path explanation
    """

    def __init__(self, hidden_size=768, num_labels=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels

    def compute_attention_heatmap(self, attention_weights, tokens=None,
                                   layer_index=-1, head_reduction="mean"):
        """
        Generate attention heatmap from model attention weights.

        Args:
            attention_weights: Tuple of attention matrices per layer,
                each [batch, num_heads, seq_len, seq_len]
            tokens: List of token strings for labeling.
            layer_index: Which layer to visualize (-1 for last).
            head_reduction: "mean", "max", or int for specific head.
        Returns:
            dict with heatmap tensor and metadata.
        """
        if attention_weights is None:
            return {"heatmap": None, "tokens": tokens}

        if isinstance(attention_weights, (list, tuple)):
            attn = attention_weights[layer_index]
        else:
            attn = attention_weights

        # attn: [batch, num_heads, seq_len, seq_len]
        if head_reduction == "mean":
            heatmap = attn.mean(dim=1)
        elif head_reduction == "max":
            heatmap = attn.max(dim=1)[0]
        elif isinstance(head_reduction, int):
            heatmap = attn[:, head_reduction]
        else:
            heatmap = attn.mean(dim=1)

        return {"heatmap": heatmap.detach().cpu(), "tokens": tokens}

    def compute_integrated_gradients(self, model, input_ids, attention_mask,
                                      target_class, embedding_layer,
                                      n_steps=50, internal_batch_size=32):
        """
        Compute Integrated Gradients for a prediction.

        IG(x) = (x - x') * integral_0^1 dF/dx(x' + alpha*(x-x')) dalpha

        Args:
            model: The full QA model.
            input_ids: [1, seq_len]
            attention_mask: [1, seq_len]
            target_class: Target class index.
            embedding_layer: The embedding layer to compute gradients w.r.t.
            n_steps: Number of interpolation steps.
            internal_batch_size: Batch size for gradient computation.
        Returns:
            dict with attributions and convergence delta.
        """
        model.eval()

        # Get input embeddings
        with torch.no_grad():
            input_embeds = embedding_layer(input_ids)  # [1, seq_len, hidden]

        # Baseline: zero embeddings
        baseline = torch.zeros_like(input_embeds)

        # Interpolate between baseline and input
        alphas = torch.linspace(0, 1, n_steps + 1, device=input_embeds.device)
        total_grads = torch.zeros_like(input_embeds)

        for i in range(0, len(alphas), internal_batch_size):
            batch_alphas = alphas[i:i + internal_batch_size]
            # [batch, 1, 1] * [1, seq_len, hidden]
            scaled = baseline + batch_alphas.view(-1, 1, 1) * (input_embeds - baseline)
            scaled.requires_grad_(True)

            # Expand attention mask for batch
            mask_batch = attention_mask.expand(len(batch_alphas), -1)

            outputs = model.forward_from_embeddings(scaled, mask_batch)
            logits = outputs["logits"]

            # Get target class logits
            target_logits = logits[:, target_class].sum()
            target_logits.backward()

            total_grads += scaled.grad.sum(dim=0, keepdim=True)
            model.zero_grad()

        # IG = (input - baseline) * avg_gradients
        avg_grads = total_grads / n_steps
        attributions = (input_embeds - baseline) * avg_grads

        # Reduce to per-token attribution: sum over hidden dimension
        token_attributions = attributions.sum(dim=-1).squeeze(0)  # [seq_len]

        # Convergence check
        with torch.no_grad():
            pred = model.forward_from_embeddings(input_embeds, attention_mask)
            base_pred = model.forward_from_embeddings(baseline, attention_mask)
            delta = (pred["logits"][0, target_class] - base_pred["logits"][0, target_class]).item()
            ig_sum = token_attributions.sum().item()
            convergence_delta = abs(delta - ig_sum)

        return {
            "attributions": token_attributions.detach().cpu(),
            "convergence_delta": convergence_delta,
            "raw_attributions": attributions.detach().cpu(),
        }

    def compute_shap_values(self, model, input_ids, attention_mask, tokenizer,
                             max_evals=500, batch_size=50):
        """
        Compute SHAP values for feature importance.

        Uses a simplified kernel SHAP approach suitable for transformer models.

        Args:
            model: The QA model.
            input_ids: [1, seq_len]
            attention_mask: [1, seq_len]
            tokenizer: Tokenizer for decoding tokens.
            max_evals: Maximum number of evaluations.
            batch_size: Batch size for evaluation.
        Returns:
            dict with shap_values per token and per class.
        """
        model.eval()
        seq_len = input_ids.size(1)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # Get baseline prediction (all masked)
        with torch.no_grad():
            baseline_ids = torch.full_like(input_ids, pad_id)
            baseline_mask = torch.zeros_like(attention_mask)
            base_output = model(baseline_ids, baseline_mask)
            base_probs = torch.softmax(base_output["logits"], dim=-1)[0]

            # Full prediction
            full_output = model(input_ids, attention_mask)
            full_probs = torch.softmax(full_output["logits"], dim=-1)[0]

        # Approximate SHAP via perturbation
        n_samples = min(max_evals, seq_len * 10)
        shap_values = torch.zeros(seq_len, self.num_labels)

        for _ in range(0, n_samples, batch_size):
            bs = min(batch_size, n_samples)
            # Random masks
            masks = torch.bernoulli(torch.ones(bs, seq_len) * 0.5).to(input_ids.device)
            perturbed_ids = input_ids.expand(bs, -1).clone()
            perturbed_ids[masks == 0] = pad_id
            perturbed_mask = masks.long()

            with torch.no_grad():
                outputs = model(perturbed_ids, perturbed_mask)
                probs = torch.softmax(outputs["logits"], dim=-1)

            # Attribute change to each token
            for j in range(seq_len):
                present = masks[:, j] == 1
                absent = masks[:, j] == 0
                if present.sum() > 0 and absent.sum() > 0:
                    diff = probs[present].mean(0) - probs[absent].mean(0)
                    shap_values[j] += diff.cpu()

        shap_values /= (n_samples / batch_size)

        # Get token strings
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

        return {
            "shap_values": shap_values,
            "tokens": tokens,
            "base_probs": base_probs.cpu(),
            "full_probs": full_probs.cpu(),
        }

    def generate_explanation(self, attention_heatmap=None, ig_attributions=None,
                              shap_values=None, tokens=None, prediction=None,
                              confidence=None, top_k=10):
        """
        Generate a human-readable explanation combining all explainability signals.
        """
        explanation = {"prediction": prediction, "confidence": confidence, "evidence": []}

        if tokens is None:
            return explanation

        # Combine signals
        scores = torch.zeros(len(tokens))
        n_signals = 0

        if ig_attributions is not None:
            ig_norm = ig_attributions / (ig_attributions.abs().max() + 1e-8)
            scores += ig_norm
            n_signals += 1

        if shap_values is not None:
            shap_norm = shap_values.abs().sum(dim=-1)
            shap_norm = shap_norm / (shap_norm.max() + 1e-8)
            scores += shap_norm
            n_signals += 1

        if attention_heatmap is not None and attention_heatmap.dim() >= 2:
            attn_scores = attention_heatmap[0].mean(dim=0) if attention_heatmap.dim() == 3 else attention_heatmap.mean(dim=0)
            attn_scores = attn_scores / (attn_scores.max() + 1e-8)
            min_len = min(len(scores), len(attn_scores))
            scores[:min_len] += attn_scores[:min_len]
            n_signals += 1

        if n_signals > 0:
            scores /= n_signals

        # Get top-k important tokens
        top_indices = scores.abs().topk(min(top_k, len(tokens))).indices
        for idx in top_indices:
            explanation["evidence"].append({
                "token": tokens[idx],
                "importance": scores[idx].item(),
                "position": idx.item(),
            })

        return explanation
