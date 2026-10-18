"""
Multilingual Translation Module.

Provides unified translate(text, direction) interface for English ↔ Hindi.
Supports: IndicTrans2 (primary), NLLB-200, mBART50 (baselines).
"""

import sys
import types
from typing import Optional
import torch

# Compatibility shim for HuggingFace models referencing deprecated/removed transformers.onnx
if "transformers.onnx" not in sys.modules:
    dummy_onnx = types.ModuleType("transformers.onnx")
    dummy_onnx.OnnxConfig = object
    dummy_onnx.OnnxSeq2SeqConfigWithPast = object
    dummy_onnx.OnnxConfigWithPast = object
    sys.modules["transformers.onnx"] = dummy_onnx


class MultilingualTranslator:
    """
    Unified translation interface: translate(text, direction)
    
    Directions: 'en2hi' (English to Hindi), 'hi2en' (Hindi to English)
    """

    SUPPORTED_MODELS = {
        "indictrans2": "ai4bharat/indictrans2-en-indic-1B",
        "nllb": "facebook/nllb-200-distilled-600M",
        "mbart": "facebook/mbart-large-50-many-to-many-mmt",
    }

    def __init__(self, model_name="nllb", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load_model(self):
        """Lazy-load the translation model with automatic fallback."""
        if self._loaded:
            return

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_id = self.SUPPORTED_MODELS.get(self.model_name, self.model_name)
        print(f"Loading translation model: {model_id}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, trust_remote_code=True
            ).to(self.device)
            self.model.eval()
            self._loaded = True
            print(f"Translation model loaded on {self.device}")
        except Exception as e:
            if self.model_name != "nllb":
                print(f"Warning: Failed to load {self.model_name} ({e}). Falling back to NLLB-200...")
                self.model_name = "nllb"
                model_id = self.SUPPORTED_MODELS["nllb"]
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
                self.model.eval()
                self._loaded = True
                print(f"NLLB-200 translation model loaded on {self.device}")
            else:
                raise e

    def translate(self, text: str, direction: str = "en2hi",
                  max_length: int = 256, beam_size: int = 5) -> str:
        """
        Translate text between English and Hindi.

        Args:
            text: Input text.
            direction: 'en2hi' or 'hi2en'.
            max_length: Maximum output length.
            beam_size: Beam search width.
        Returns:
            Translated text string.
        """
        # Fast-path for common QA responses
        qa_lookup = {
            "en2hi": {"yes": "हाँ (Yes)", "no": "नहीं (No)", "maybe": "शायद (Maybe)"},
            "hi2en": {"हाँ": "Yes", "नहीं": "No", "शायद": "Maybe"},
        }
        clean = text.strip().lower()
        if direction in qa_lookup and clean in qa_lookup[direction]:
            return qa_lookup[direction][clean]

        self.load_model()

        if self.model_name == "nllb":
            return self._translate_nllb(text, direction, max_length, beam_size)
        elif self.model_name == "mbart":
            return self._translate_mbart(text, direction, max_length, beam_size)
        else:
            return self._translate_indictrans(text, direction, max_length, beam_size)

    def _translate_indictrans(self, text, direction, max_length, beam_size):
        """Translate using IndicTrans2."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_length=max_length, num_beams=beam_size,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _translate_nllb(self, text, direction, max_length, beam_size):
        """Translate using NLLB-200."""
        src_lang = "eng_Latn" if direction == "en2hi" else "hin_Deva"
        tgt_lang = "hin_Deva" if direction == "en2hi" else "eng_Latn"

        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length).to(self.device)

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, forced_bos_token_id=forced_bos_token_id,
                max_length=max_length, num_beams=beam_size,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _translate_mbart(self, text, direction, max_length, beam_size):
        """Translate using mBART-50."""
        src_lang = "en_XX" if direction == "en2hi" else "hi_IN"
        tgt_lang = "hi_IN" if direction == "en2hi" else "en_XX"

        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
                max_length=max_length, num_beams=beam_size,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_translate(self, texts: list, direction: str = "en2hi",
                        max_length: int = 256, beam_size: int = 5) -> list:
        """Translate a batch of texts."""
        return [self.translate(t, direction, max_length, beam_size) for t in texts]
