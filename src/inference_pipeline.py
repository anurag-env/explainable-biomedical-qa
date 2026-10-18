"""
End-to-end Inference Pipeline.

Takes a biomedical question, retrieves/processes context, generates an
explainable answer, translates it to Hindi, and outputs spoken audio.

Pipeline:
Question -> RoBERTa -> Refinement -> Graph -> Confidence -> SHAP/IG -> Translator -> TTS
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.models.rerb_qa import RERBQAModel
from src.data.preprocessing import BiomedicalPreprocessor
from src.translation.translator import MultilingualTranslator
from src.tts.speaker import MultilingualSpeaker


class RERBQAPipeline:
    """
    End-to-end inference pipeline for RAG-RERoBERTa-XAI-TTS.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        use_translation: bool = True,
        use_tts: bool = True,
        config_overrides: Optional[dict] = None,
    ):
        self.device = (
            "cuda" if torch.cuda.is_available() and device == "auto" else
            (device if device != "auto" else "cpu")
        )
        print(f"Initializing pipeline on {self.device}...")

        # 1. Preprocessor
        self.preprocessor = BiomedicalPreprocessor()
        self.tokenizer = self.preprocessor.tokenizer

        # 2. Model
        model_kwargs = {
            "encoder_name": "allenai/biomed_roberta_base",
            "hidden_size": 768,
            "num_labels": 3,
            "use_evidence_graph": True,
        }
        if config_overrides:
            model_kwargs.update(config_overrides)

        state_dict = None
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            # Auto-detect whether checkpoint was trained with evidence graph (1536 vs 768)
            if "classifier.0.weight" in state_dict:
                cls_in_dim = state_dict["classifier.0.weight"].shape[-1]
                model_kwargs["use_evidence_graph"] = (cls_in_dim == 1536)

        self.model = RERBQAModel(**model_kwargs)

        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Using untrained weights.")

        self.model = self.model.to(self.device)
        self.model.eval()

        # 3. Translator
        self.use_translation = use_translation
        if use_translation:
            self.translator = MultilingualTranslator(device=self.device)

        # 4. Text-to-Speech
        self.use_tts = use_tts
        if use_tts:
            self.speaker = MultilingualSpeaker(device=self.device)

        self.label_map = {0: "Yes", 1: "No", 2: "Maybe"}

    def __call__(
        self,
        question: str,
        context: str = "",
        target_lang: str = "en",
        generate_audio: bool = True,
        explain: bool = True,
        evidence_source: str = "",
    ) -> Dict:
        """
        Process a question and return the answer, explanations, translation, and audio.
        """
        print(f"\nProcessing question: {question}")

        # 1. Tokenize
        encoded = self.preprocessor.tokenize_qa_pair(
            question, context, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # 2. Forward Pass (with explanations)
        with torch.no_grad():
            output = self.model(
                input_ids, attention_mask,
                return_explanations=explain
            )

            # Get predictions
            logits = output["logits"]
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = probs.argmax(dim=-1).item()
            answer_text = self.label_map[pred_idx]

            # Get confidence from calibrated probability of the predicted class
            if "calibrated_probs" in output and output["calibrated_probs"] is not None:
                confidence = output["calibrated_probs"][0][pred_idx].item() * 100
            else:
                confidence = probs[pred_idx].item() * 100

        # 3. Explanations & Clinical Safety Guardrail
        explanations = None
        safety_warning = None
        if explain:
            heatmap = output.get("attention_heatmap", {}).get("heatmap")
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

            # Generate structured explanation
            explanations = self.model.explainability.generate_explanation(
                attention_heatmap=heatmap,
                tokens=tokens,
                prediction=answer_text,
                confidence=confidence,
                top_k=5,
            )

            # Safety Guardrail 1: Context Abstention (prevent naked question overconfidence)
            if not context or len(context.strip()) == 0:
                confidence = min(confidence, 65.0)
                safety_warning = "Ungrounded query: No clinical evidence context provided."
            
            # Safety Guardrail 2: Entity Grounding Verification
            if "Candidate:" in question:
                candidate_name = question.split("Candidate:")[-1].strip().lower()
                evidence_tokens = [
                    ev["token"].replace("Ġ", "").lower()
                    for ev in explanations.get("evidence", [])
                ]
                # Check if candidate entity was actually attended to
                has_entity_attention = any(
                    (tok in candidate_name or candidate_name in tok)
                    for tok in evidence_tokens if len(tok) >= 3
                )
                if not has_entity_attention and context:
                    confidence = min(confidence, 60.0)
                    safety_warning = f"Entity Grounding Warning: Candidate '{candidate_name}' was not directly supported by top evidence tokens."

        # 4. Translation
        translated_answer = None
        if self.use_translation and target_lang != "en":
            print(f"Translating answer to {target_lang}...")
            # Translate the templated English answer
            full_text = f"The answer is {answer_text} with {confidence:.0f}% confidence."
            translated_answer = self.translator.translate(
                full_text, direction=f"en2{target_lang}"
            )

        # 5. Speech Generation
        audio_path = None
        if self.use_tts and generate_audio:
            print(f"Generating speech in {target_lang}...")
            if target_lang == "en":
                audio_path = self.speaker.speak_answer(
                    answer=answer_text,
                    confidence=confidence,
                    evidence_source=evidence_source,
                    lang="en"
                )
            elif target_lang == "hi" and translated_answer:
                audio_path = self.speaker.speak(
                    text=translated_answer,
                    lang="hi"
                )

        result = {
            "question": question,
            "context": context,
            "prediction": answer_text,
            "probabilities": {self.label_map[i]: probs[i].item() for i in range(3)},
            "confidence": confidence,
            "safety_warning": safety_warning,
            "num_refinement_iterations": output["num_iterations"],
            "explanations": explanations,
            "translation": translated_answer,
            "audio_path": audio_path,
        }

        return result


def main():
    parser = argparse.ArgumentParser(description="RAG-RERoBERTa Inference")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--context", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/stage2/best.pt")
    parser.add_argument("--lang", type=str, choices=["en", "hi"], default="en")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--no-explain", action="store_true")
    args = parser.parse_args()

    pipeline = RERBQAPipeline(
        checkpoint_path=args.checkpoint,
        use_translation=args.lang != "en",
        use_tts=not args.no_tts,
    )

    result = pipeline(
        question=args.question,
        context=args.context,
        target_lang=args.lang,
        generate_audio=not args.no_tts,
        explain=not args.no_explain,
    )

    print("\n" + "="*50)
    print(f"Question: {result['question']}")
    print(f"Answer:   {result['prediction']} (Confidence: {result['confidence']:.1f}%)")
    print("Class Probabilities:")
    for lbl, pr in result["probabilities"].items():
        print(f"  - {lbl:>5}: {pr * 100:.1f}%")
        
    if result.get("safety_warning"):
        print(f"\n⚠️  Clinical Safety Alert: {result['safety_warning']}")
    
    if result["translation"]:
        print(f"Hindi:    {result['translation']}")
        
    if result["audio_path"]:
        print(f"Audio:    {result['audio_path']}")
        
    if result["explanations"]:
        print("\nTop Evidence Tokens:")
        for ev in result["explanations"]["evidence"]:
            token = ev["token"].replace("Ġ", "")
            if token and token not in ["[CLS]", "[SEP]", "[PAD]"]:
                print(f"  - '{token}' (importance: {ev['importance']:.3f})")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
