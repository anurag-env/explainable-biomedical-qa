"""
Multilingual Text-to-Speech Module.

Provides unified speak(text, lang) interface.
Supports English (VITS, FastSpeech2, XTTS-v2, Bark) and Hindi TTS.
"""

import os
from pathlib import Path
from typing import Optional
import torch


class MultilingualSpeaker:
    """
    Unified TTS interface: speak(text, lang)
    
    Languages: 'en' (English), 'hi' (Hindi)
    """

    DEFAULT_MODELS = {
        "en": "tts_models/en/ljspeech/vits",
        "hi": "tts_models/hi/custom/vits",
    }

    def __init__(self, output_dir="outputs/audio", device=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._models = {}

    def _load_model(self, lang: str, model_name: Optional[str] = None):
        """Lazy-load a TTS model."""
        if lang in self._models:
            return self._models[lang]

        model_id = model_name or self.DEFAULT_MODELS.get(lang)
        if not model_id:
            raise ValueError(f"No TTS model configured for language: {lang}")

        try:
            from TTS.api import TTS
            print(f"Loading TTS model: {model_id}")
            tts = TTS(model_name=model_id)
            if self.device == "cuda" and torch.cuda.is_available():
                tts = tts.to(self.device)
            self._models[lang] = tts
            print(f"TTS model loaded for {lang}")
            return tts
        except ImportError:
            print("Warning: TTS library not installed. Install with: pip install TTS")
            return None
        except Exception as e:
            print(f"Warning: Could not load TTS model {model_id}: {e}")
            return None

    def speak(self, text: str, lang: str = "en",
              output_path: Optional[str] = None,
              model_name: Optional[str] = None) -> Optional[str]:
        """
        Convert text to speech and save as audio file.

        Args:
            text: Input text to speak.
            lang: Language ('en' or 'hi').
            output_path: Custom output file path.
            model_name: Override default model.
        Returns:
            Path to the generated audio file, or None if TTS unavailable.
        """
        tts = self._load_model(lang, model_name)
        if tts is None:
            print(f"TTS not available for {lang}. Skipping speech generation.")
            return None

        if output_path is None:
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            output_path = str(self.output_dir / f"speech_{lang}_{text_hash}.wav")

        try:
            tts.tts_to_file(text=text, file_path=output_path)
            print(f"Speech saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None

    def speak_answer(self, answer: str, confidence: float,
                     evidence_source: str = "", citation: str = "",
                     lang: str = "en", output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate spoken answer in the template format:
            Answer: <text>
            Confidence: <calibrated %>
            Evidence Source: <PubMed ID>
            Citation: <formatted citation>
        """
        speech_text = f"Answer: {answer}. "
        speech_text += f"Confidence: {confidence:.0f} percent. "
        if evidence_source:
            speech_text += f"Evidence source: {evidence_source}. "
        if citation:
            speech_text += f"Citation: {citation}."

        return self.speak(speech_text, lang=lang, output_path=output_path)

    def batch_speak(self, texts: list, lang: str = "en") -> list:
        """Generate speech for a batch of texts."""
        paths = []
        for i, text in enumerate(texts):
            path = str(self.output_dir / f"batch_{lang}_{i:04d}.wav")
            result = self.speak(text, lang=lang, output_path=path)
            paths.append(result)
        return paths
