"""
Biomedical RoBERTa Encoder.

Wraps a pretrained biomedical RoBERTa model (e.g., allenai/biomed_roberta_base)
as the foundation encoder: E0 = RoBERTa(Q, A).
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BiomedicalRoBERTaEncoder(nn.Module):
    """
    Biomedical RoBERTa encoder that produces contextual embeddings
    from question-context pairs.

    Produces:
        E0 = RoBERTa(Q, A)  (shape: [batch, seq_len, hidden_size])
    """

    def __init__(
        self,
        model_name: str = "allenai/biomed_roberta_base",
        hidden_size: int = 768,
        output_hidden_states: bool = True,
        freeze_layers: int = 0,
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            hidden_size: Expected hidden dimension.
            output_hidden_states: Whether to return all hidden states.
            freeze_layers: Number of bottom transformer layers to freeze.
        """
        super().__init__()

        self.config = AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=output_hidden_states,
        )
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.hidden_size = self.config.hidden_size

        # Project to desired hidden size if different from model's native size
        if self.config.hidden_size != hidden_size:
            self.projection = nn.Linear(self.config.hidden_size, hidden_size)
            self.hidden_size = hidden_size
        else:
            self.projection = None

        # Freeze bottom layers if specified
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

    def _freeze_layers(self, num_layers: int):
        """Freeze the embeddings and the first `num_layers` transformer layers."""
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified number of encoder layers
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            for i, layer in enumerate(self.encoder.encoder.layer):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass through the encoder.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Token type IDs [batch, seq_len] (optional)

        Returns:
            dict with:
                - last_hidden_state: [batch, seq_len, hidden_size]
                - pooler_output: [batch, hidden_size]
                - all_hidden_states: tuple of [batch, seq_len, hidden_size] (if enabled)
                - attentions: attention weights (if available)
        """
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)

        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, native_hidden]

        # Project if necessary
        if self.projection is not None:
            last_hidden_state = self.projection(last_hidden_state)

        # Get pooler output (CLS token representation)
        pooler_output = last_hidden_state[:, 0, :]  # [batch, hidden_size]

        result = {
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooler_output,
        }

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            result["all_hidden_states"] = outputs.hidden_states

        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            result["attentions"] = outputs.attentions

        return result

    def get_output_dim(self) -> int:
        """Return the output dimension of the encoder."""
        return self.hidden_size
