"""
RERBQAModel — Full assembled model.

Pipeline:
  Input Question
  -> Biomedical Preprocessing
  -> RoBERTa Biomedical Encoder
  -> Multi-head Self-Attention
  -> Recursive Embedding Refinement Layer
  -> [Evidence Graph] + [Confidence Layer]
  -> Explainable Answer Generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import BiomedicalRoBERTaEncoder
from .attention import MultiHeadSelfAttention
from .recursive_refine import RecursiveEmbeddingRefinement
from .evidence_graph import EvidenceGraph
from .confidence import ConfidenceCalibrationHead
from .explainability import ExplainabilityModule


class RERBQAModel(nn.Module):
    """
    Recursive Embedding Refinement Biomedical QA Model.

    Assembles: Encoder -> Attention -> Recursive Refinement ->
               Evidence Graph -> Confidence -> Classifier
    """

    def __init__(
        self,
        encoder_name="allenai/biomed_roberta_base",
        hidden_size=768,
        num_labels=3,
        num_heads=12,
        max_iterations=5,
        convergence_epsilon=1e-3,
        dropout=0.3,
        freeze_layers=0,
        use_evidence_graph=True,
        use_confidence=True,
        max_options=5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.use_evidence_graph = use_evidence_graph
        self.use_confidence = use_confidence
        self.max_options = max_options

        # 1. Biomedical RoBERTa Encoder: E0 = RoBERTa(Q, A)
        self.encoder = BiomedicalRoBERTaEncoder(
            model_name=encoder_name,
            hidden_size=hidden_size,
            freeze_layers=freeze_layers,
        )

        # 2. Recursive Embedding Refinement
        self.recursive_refinement = RecursiveEmbeddingRefinement(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_iterations=max_iterations,
            convergence_epsilon=convergence_epsilon,
            dropout=dropout,
        )

        # 3. Evidence Graph (optional)
        if use_evidence_graph:
            self.evidence_graph = EvidenceGraph(
                node_embedding_dim=hidden_size,
                hidden_channels=hidden_size // 3,
                num_heads=4,
                num_layers=2,
                dropout=dropout,
            )
            classifier_input_size = hidden_size * 2  # refined + graph
        else:
            self.evidence_graph = None
            classifier_input_size = hidden_size

        # 4. Confidence Calibration
        if use_confidence:
            self.confidence_head = ConfidenceCalibrationHead(
                hidden_size=classifier_input_size,
                num_labels=num_labels,
            )

        # 5. Classification Head (Yes/No/Maybe)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, num_labels),
        )

        # 6. Multi-Choice Scoring Head (selects best option from A/B/C/D)
        self.mc_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        # 7. Explainability module (inference only)
        self.explainability = ExplainabilityModule(
            hidden_size=hidden_size, num_labels=num_labels,
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                labels=None, graph_data=None, return_explanations=False,
                task_type="classification", num_options=None,
                label_smoothing=None, calibration_weight=0.1):
        """
        Full forward pass.

        Args:
            input_ids: [batch, seq_len] for classification,
                       [batch, num_options, seq_len] for multi_choice.
            attention_mask: [batch, seq_len] or [batch, num_options, seq_len].
            token_type_ids: [batch, seq_len] (optional)
            labels: [batch] ground truth labels (optional, for loss)
            graph_data: dict with node_features, edge_index, node_types (optional)
            return_explanations: Whether to return explainability info.
            task_type: "classification" (Yes/No/Maybe) or "multi_choice" (A/B/C/D).
            num_options: [batch] valid number of options per sample (optional)
        Returns:
            dict with logits, loss, confidence, explanations, convergence info.
        """
        # Route to multi-choice head if applicable
        if task_type == "multi_choice":
            return self._forward_multi_choice(
                input_ids, attention_mask, labels=labels,
                return_explanations=return_explanations,
                valid_num_options=num_options,
                label_smoothing=0.05 if label_smoothing is None else label_smoothing,
                calibration_weight=calibration_weight,
            )

        # Step 1: Encode — E0 = RoBERTa(Q, A)
        encoder_output = self.encoder(input_ids, attention_mask, token_type_ids)
        E_0 = encoder_output["last_hidden_state"]  # [batch, seq_len, hidden]

        # Step 2: Recursive Refinement — E* = refine(E0)
        refinement_output = self.recursive_refinement(
            E_0, attention_mask, return_attention_weights=return_explanations
        )
        E_star = refinement_output["refined_embedding"]  # [batch, seq_len, hidden]

        # Pool: take CLS token representation
        cls_repr = E_star[:, 0, :]  # [batch, hidden]

        # Step 3: Evidence Graph (optional)
        if self.use_evidence_graph and graph_data is not None:
            graph_output = self.evidence_graph(
                graph_data["node_features"],
                graph_data["edge_index"],
                graph_data.get("node_types"),
            )
            graph_emb = graph_output["graph_embedding"]  # [1, hidden]
            graph_emb = graph_emb.expand(cls_repr.size(0), -1)
            combined = torch.cat([cls_repr, graph_emb], dim=-1)
        elif self.use_evidence_graph:
            # No graph data provided — use zero padding
            dummy_graph = torch.zeros_like(cls_repr)
            combined = torch.cat([cls_repr, dummy_graph], dim=-1)
        else:
            combined = cls_repr

        # Step 4: Classification
        logits = self.classifier(combined)  # [batch, num_labels]

        result = {
            "logits": logits,
            "refined_embedding": E_star,
            "convergence_norms": refinement_output["convergence_norms"],
            "num_iterations": refinement_output["num_iterations"],
        }

        # Step 5: Confidence
        if self.use_confidence:
            conf_output = self.confidence_head(combined, logits)
            result["confidence"] = conf_output["confidence_score"]
            result["calibrated_probs"] = conf_output.get("calibrated_probs")

        # Step 6: Loss computation (with label smoothing to prevent false overconfidence)
        if labels is not None:
            valid_mask = labels >= 0  # Filter out -1 (unlabeled)
            if valid_mask.sum() > 0:
                loss = F.cross_entropy(logits[valid_mask], labels[valid_mask], label_smoothing=0.1)
                if self.use_confidence:
                    loss += 0.1 * self.confidence_head.calibration_loss(
                        logits[valid_mask], labels[valid_mask]
                    )
                result["loss"] = loss
            else:
                result["loss"] = torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Step 7: Explanations (optional)
        if return_explanations:
            attn_weights = refinement_output.get("attention_weights")
            result["attention_heatmap"] = self.explainability.compute_attention_heatmap(
                attn_weights
            )

        return result

    def _forward_multi_choice(self, input_ids, attention_mask, labels=None,
                               return_explanations=False, valid_num_options=None,
                               label_smoothing=0.05, calibration_weight=0.1):
        """
        Multi-choice forward pass. Scores each option independently
        and selects the best diagnosis from the available options.

        Args:
            input_ids: [batch, num_options, seq_len]
            attention_mask: [batch, num_options, seq_len]
            labels: [batch] index of the correct option (0-based)
            return_explanations: Whether to return explainability info.
            valid_num_options: [batch] actual number of valid options (masks padding)

        Returns:
            dict with logits [batch, num_options], loss, confidence, etc.
        """
        batch_size, num_options, seq_len = input_ids.shape

        # Flatten options into batch dimension: [batch * num_options, seq_len]
        flat_ids = input_ids.view(-1, seq_len)
        flat_mask = attention_mask.view(-1, seq_len)

        # Step 1: Encode each (question, option_i) pair
        encoder_output = self.encoder(flat_ids, flat_mask)
        E_0 = encoder_output["last_hidden_state"]

        # Step 2: Recursive Refinement
        refinement_output = self.recursive_refinement(
            E_0, flat_mask, return_attention_weights=return_explanations
        )
        E_star = refinement_output["refined_embedding"]

        # Pool: CLS token representation
        cls_repr = E_star[:, 0, :]  # [B*N, hidden]

        # Step 3: Score each option via the multi-choice head
        scores = self.mc_scorer(cls_repr).squeeze(-1)  # [B*N]
        unmasked_logits = scores.view(batch_size, num_options)   # [B, N]
        logits = unmasked_logits

        # Step 3.1: Mask out padded/invalid options with -1e9 so they are never chosen
        if valid_num_options is not None:
            opt_range = torch.arange(num_options, device=logits.device).unsqueeze(0)
            valid_opts = opt_range < valid_num_options.unsqueeze(1)
            logits = logits.masked_fill(~valid_opts, -1e9)
        else:
            valid_opts = torch.ones_like(logits, dtype=torch.bool)

        result = {
            "logits": logits,
            # Kept for training diagnostics and regression tests.  Consumers
            # should use ``logits`` for probabilities and loss.
            "unmasked_logits": unmasked_logits,
            "valid_options_mask": valid_opts,
            "option_logit_variance": unmasked_logits.var(dim=-1, unbiased=False),
            "convergence_norms": refinement_output["convergence_norms"],
            "num_iterations": refinement_output["num_iterations"],
        }

        # Step 4: Confidence calibration on the selected option
        if self.use_confidence:
            cls_reshaped = cls_repr.view(batch_size, num_options, -1)
            best_idx = logits.argmax(dim=-1)
            best_cls = cls_reshaped[
                torch.arange(batch_size, device=logits.device), best_idx
            ]
            if self.use_evidence_graph:
                dummy = torch.zeros_like(best_cls)
                combined_for_conf = torch.cat([best_cls, dummy], dim=-1)
            else:
                combined_for_conf = best_cls
            conf_output = self.confidence_head(combined_for_conf, logits)
            result["confidence"] = conf_output["confidence_score"]
            result["calibrated_probs"] = conf_output.get("calibrated_probs")

        # Step 5: Loss (with label smoothing for regularization)
        if labels is not None:
            valid_mask = labels >= 0
            if valid_mask.sum() > 0:
                loss = F.cross_entropy(logits[valid_mask], labels[valid_mask], label_smoothing=label_smoothing)
                if self.use_confidence and calibration_weight:
                    loss += calibration_weight * self.confidence_head.calibration_loss(
                        logits[valid_mask], labels[valid_mask]
                    )
                result["loss"] = loss
            else:
                result["loss"] = torch.tensor(
                    0.0, device=logits.device, requires_grad=True
                )

        # Step 6: Explanations
        if return_explanations:
            attn_weights = refinement_output.get("attention_weights")
            result["attention_heatmap"] = (
                self.explainability.compute_attention_heatmap(attn_weights)
            )

        return result

    def forward_from_embeddings(self, embeddings, attention_mask):
        """
        Forward pass starting from embeddings (for Integrated Gradients).
        """
        refinement_output = self.recursive_refinement(embeddings, attention_mask)
        E_star = refinement_output["refined_embedding"]
        cls_repr = E_star[:, 0, :]

        if self.use_evidence_graph:
            dummy_graph = torch.zeros_like(cls_repr)
            combined = torch.cat([cls_repr, dummy_graph], dim=-1)
        else:
            combined = cls_repr

        logits = self.classifier(combined)
        return {"logits": logits}

    def predict(self, input_ids, attention_mask, token_type_ids=None):
        """Get predictions with confidence scores."""
        self.eval()
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask, token_type_ids)
            probs = F.softmax(output["logits"], dim=-1)
            predictions = probs.argmax(dim=-1)

        label_map = {0: "Yes", 1: "No", 2: "Maybe"}
        results = []
        for i in range(predictions.size(0)):
            results.append({
                "prediction": label_map[predictions[i].item()],
                "probabilities": {label_map[j]: probs[i, j].item() for j in range(3)},
                "confidence": output.get("confidence", torch.zeros(1))[i].item() if "confidence" in output else probs[i].max().item(),
            })
        return results

    def predict_multi_choice(self, input_ids, attention_mask, option_texts=None):
        """Get multi-choice predictions with probabilities for all options.

        Args:
            input_ids: [batch, num_options, seq_len]
            attention_mask: [batch, num_options, seq_len]
            option_texts: Optional list of option text strings for labeling.

        Returns:
            List of dicts with prediction index, option probabilities, and confidence.
        """
        self.eval()
        with torch.no_grad():
            output = self._forward_multi_choice(input_ids, attention_mask)
            probs = F.softmax(output["logits"], dim=-1)
            predictions = probs.argmax(dim=-1)

        results = []
        for i in range(predictions.size(0)):
            num_opts = probs.size(1)
            option_probs = {}
            for j in range(num_opts):
                label = f"Option {chr(65 + j)}"
                if option_texts and j < len(option_texts):
                    label += f" ({option_texts[j]})"
                option_probs[label] = probs[i, j].item()

            results.append({
                "prediction": predictions[i].item(),
                "prediction_label": f"Option {chr(65 + predictions[i].item())}",
                "probabilities": option_probs,
                "confidence": output.get("confidence", torch.zeros(1))[i].item()
                    if "confidence" in output else probs[i].max().item(),
            })
        return results
