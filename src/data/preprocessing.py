"""
Biomedical text preprocessing utilities.

Handles tokenization, text cleaning, and biomedical-specific
preprocessing for the QA pipeline.
"""

import re
from typing import List, Optional, Tuple

from transformers import AutoTokenizer


class BiomedicalPreprocessor:
    """
    Preprocessor for biomedical text that handles:
    - Tokenization via a biomedical RoBERTa tokenizer
    - Text normalization (whitespace, encoding)
    - Biomedical entity preservation (drug names, gene symbols, etc.)
    """

    def __init__(
        self,
        tokenizer_name: str = "allenai/biomed_roberta_base",
        max_question_length: int = 512,
        max_context_length: int = 512,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_question_length = max_question_length
        self.max_context_length = max_context_length

        # Label mapping for Yes/No/Maybe classification
        self.label2id = {"Yes": 0, "No": 1, "Maybe": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize biomedical text.
        Preserves case (biomedical terms are case-sensitive).
        """
        if not text:
            return ""

        # Normalize unicode characters
        text = text.strip()

        # Normalize whitespace (collapse multiple spaces/newlines)
        text = re.sub(r"\s+", " ", text)

        # Remove non-printable characters but keep standard punctuation
        text = re.sub(r"[^\x20-\x7E\u00A0-\u024F]", "", text)

        return text

    def tokenize_qa_pair(
        self,
        question: str,
        context: str,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> dict:
        """
        Tokenize a question-context pair for the QA model.

        Args:
            question: The biomedical question.
            context: The evidence/abstract context.
            max_length: Maximum sequence length (defaults to max_question_length + max_context_length).
            return_tensors: Return format ("pt" for PyTorch, "np" for NumPy).

        Returns:
            Dict with input_ids, attention_mask, token_type_ids.
        """
        question = self.preprocess_text(question)
        context = self.preprocess_text(context)

        if max_length is None:
            max_length = min(
                self.max_question_length + self.max_context_length,
                self.tokenizer.model_max_length,
            )

        encoded = self.tokenizer(
            question,
            context,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
        )

        return encoded

    def encode_label(self, label: str) -> int:
        """Convert label string to integer ID."""
        if not label or label not in self.label2id:
            return -1  # Unknown/missing label
        return self.label2id[label]

    def decode_label(self, label_id: int) -> str:
        """Convert label ID back to string."""
        return self.id2label.get(label_id, "Unknown")

    def batch_tokenize(
        self,
        questions: List[str],
        contexts: List[str],
        max_length: Optional[int] = None,
    ) -> dict:
        """
        Tokenize a batch of question-context pairs.
        """
        questions = [self.preprocess_text(q) for q in questions]
        contexts = [self.preprocess_text(c) for c in contexts]

        if max_length is None:
            max_length = min(
                self.max_question_length + self.max_context_length,
                self.tokenizer.model_max_length,
            )

        encoded = self.tokenizer(
            questions,
            contexts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return encoded

    def extract_biomedical_entities(self, text: str) -> List[str]:
        """
        Extract potential biomedical entity mentions using regex patterns.
        This is a lightweight extraction — for full NER, use a biomedical NER model.

        Catches patterns like:
        - Drug names (capitalized words followed by dosages)
        - Gene symbols (uppercase 2-5 letter codes with optional numbers)
        - Disease abbreviations
        """
        entities = []

        # Gene-like patterns: 2-5 uppercase letters optionally followed by digits
        gene_pattern = re.findall(r"\b[A-Z]{2,5}\d{0,2}\b", text)
        entities.extend(gene_pattern)

        # Drug dosage patterns: word + number + unit
        drug_pattern = re.findall(
            r"\b[A-Z][a-z]+(?:\s+\d+\s*(?:mg|mcg|ml|g|IU))", text
        )
        entities.extend(drug_pattern)

        # Percentage/statistical values
        stat_pattern = re.findall(r"\b\d+\.?\d*\s*%", text)
        entities.extend(stat_pattern)

        return list(set(entities))
