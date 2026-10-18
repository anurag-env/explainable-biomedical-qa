"""
Confidence Calibration Head.

Produces calibrated confidence scores for each prediction
using temperature scaling and an auxiliary confidence MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceCalibrationHead(nn.Module):
    """
    Confidence calibration head that produces calibrated probability estimates.
    
    Uses:
    1. Temperature scaling for post-hoc calibration
    2. An auxiliary MLP that learns to predict confidence from hidden states
    """

    def __init__(self, hidden_size=768, num_labels=3, confidence_hidden=256,
                 temperature_scaling=True, num_bins=15):
        super().__init__()
        self.num_labels = num_labels
        self.num_bins = num_bins
        self.use_temperature = temperature_scaling

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        # Auxiliary confidence prediction MLP
        self.confidence_mlp = nn.Sequential(
            nn.Linear(hidden_size, confidence_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(confidence_hidden, confidence_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(confidence_hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states, logits=None):
        """
        Args:
            hidden_states: [batch, hidden_size] (CLS representation)
            logits: [batch, num_labels] (raw logits from classifier)
        Returns:
            dict with calibrated_probs, confidence_score, temperature
        """
        # Auxiliary confidence score from hidden states
        confidence_score = self.confidence_mlp(hidden_states).squeeze(-1)  # [batch]

        result = {"confidence_score": confidence_score, "temperature": self.temperature.item()}

        if logits is not None:
            # Temperature-scaled probabilities (clamp temperature >= 1.0 to prevent artificial inflation)
            if self.use_temperature:
                temp = torch.clamp(self.temperature, min=1.0)
                calibrated_logits = logits / temp
            else:
                calibrated_logits = logits
            calibrated_probs = F.softmax(calibrated_logits, dim=-1)
            result["calibrated_probs"] = calibrated_probs
            result["calibrated_logits"] = calibrated_logits

        return result

    def compute_ece(self, probs, labels):
        """
        Compute Expected Calibration Error (ECE).

        Args:
            probs: Predicted probabilities [N, num_labels]
            labels: True labels [N]
        Returns:
            ECE scalar value
        """
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=probs.device)
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1, device=probs.device)

        for i in range(self.num_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].float().mean()
                ece += (in_bin.sum().float() / len(probs)) * torch.abs(avg_accuracy - avg_confidence)

        return ece.item()

    def calibration_loss(self, logits, labels):
        """NLL loss on temperature-scaled logits for calibration."""
        if self.use_temperature:
            scaled = logits / self.temperature
        else:
            scaled = logits
        return F.cross_entropy(scaled, labels)
