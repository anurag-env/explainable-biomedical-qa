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
        options: Optional[List[str]] = None,
        target_lang: str = "en",
        generate_audio: bool = True,
        explain: bool = True,
        evidence_source: str = "",
    ) -> Dict:
        """
        Process a question and return the answer, explanations, translation, and audio.

        Supports two modes:
        - Multi-choice diagnosis selection: When `options` is provided (e.g., 4 options),
          evaluates all options simultaneously and shows probabilities for each.
        - Yes/No/Maybe classification: When `options` is None (default).
        """
        if options is not None and len(options) > 0:
            return self.predict_diagnosis(
                question=question,
                options=options,
                context=context,
                target_lang=target_lang,
                generate_audio=generate_audio,
                explain=explain,
                evidence_source=evidence_source,
            )

        print(f"\nProcessing question (Yes/No/Maybe mode): {question}")

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
                probs = torch.clamp(probs, min=0.15, max=0.65)
                probs = probs / probs.sum()

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
                    safety_warning = f"Entity Grounding Warning: Candidate '{candidate_name}' was not directly supported by top evidence tokens."
                    answer_text = "Maybe (Unverified)"
                    confidence = 50.0
                    probs = torch.tensor([0.35, 0.35, 0.30])

        # 4. Translation
        translated_answer = None
        if self.use_translation and target_lang != "en":
            print(f"Translating answer to {target_lang}...")
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
            "mode": "classification",
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

    def predict_diagnosis(
        self,
        question: str,
        options: List[str],
        context: str = "",
        target_lang: str = "en",
        generate_audio: bool = True,
        explain: bool = True,
        evidence_source: str = "",
    ) -> Dict:
        """
        Select the best diagnosis from multiple options and compute probabilities for all options.

        Args:
            question: Patient case description / clinical vignette.
            options: List or dict of candidate diagnoses (e.g., 4 options).
            context: Clinical context or background knowledge (optional).
            target_lang: Target language ("en" or "hi").
            generate_audio: Whether to generate TTS audio.
            explain: Whether to compute attention heatmaps and evidence tokens.
            evidence_source: Source of context for speech announcement.

        Returns:
            Dict containing selected diagnosis, probabilities for all options,
            confidence score, explanations, translation, and audio path.
        """
        # Resolve options keys & values
        if isinstance(options, dict):
            option_keys = list(options.keys())
            option_texts = [str(v) for v in options.values()]
        else:
            option_keys = [chr(65 + i) for i in range(len(options))]
            option_texts = [str(o) for o in options]

        num_options = len(option_texts)
        full_question = f"{question} Context: {context}" if context else question
        print(f"\nProcessing multi-choice diagnosis ({num_options} candidate options): {question}")

        # 1. Tokenize multi-choice options
        encoded = self.preprocessor.tokenize_multi_choice(full_question, option_texts)
        input_ids = encoded["input_ids"].unsqueeze(0).to(self.device)          # [1, num_options, seq_len]
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(self.device)  # [1, num_options, seq_len]

        # 2. Forward pass with multi-choice head
        with torch.no_grad():
            output = self.model(
                input_ids,
                attention_mask,
                task_type="multi_choice",
                return_explanations=explain,
            )
            logits = output["logits"][0]  # [num_options]
            probs = torch.softmax(logits, dim=-1)
            pred_idx = probs.argmax(dim=-1).item()

            pred_key = option_keys[pred_idx]
            pred_text = option_texts[pred_idx]
            answer_text = f"Option {pred_key}: {pred_text}"

            # Confidence
            if "confidence" in output and output["confidence"] is not None:
                confidence = output["confidence"][0].item() * 100
            else:
                confidence = probs[pred_idx].item() * 100

        # 3. Probabilities for all candidate options
        all_option_probs = {}
        ranked_options = []
        for i in range(num_options):
            k = option_keys[i]
            t = option_texts[i]
            prob_val = probs[i].item()
            label = f"Option {k} ({t})"
            all_option_probs[label] = prob_val
            ranked_options.append({
                "option": k,
                "text": t,
                "probability": prob_val,
                "percentage": round(prob_val * 100, 2),
            })
        ranked_options.sort(key=lambda x: x["probability"], reverse=True)

        # 4. Explanations & Clinical Safety Alert
        explanations = None
        safety_warning = None
        if explain:
            heatmap_data = output.get("attention_heatmap", {})
            heatmap = heatmap_data.get("heatmap")
            
            # Extract tokens corresponding to the chosen option
            winning_token_ids = input_ids[0, pred_idx].cpu().tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(winning_token_ids)

            # Option-specific heatmap slice [1, seq_len, seq_len]
            opt_heatmap = (
                heatmap[pred_idx:pred_idx + 1]
                if (heatmap is not None and heatmap.dim() == 3)
                else heatmap
            )

            explanations = self.model.explainability.generate_explanation(
                attention_heatmap=opt_heatmap,
                tokens=tokens,
                prediction=answer_text,
                confidence=confidence,
                top_k=5,
            )

            # Differential diagnosis caution if margin is narrow
            if len(ranked_options) > 1:
                margin = (ranked_options[0]["probability"] - ranked_options[1]["probability"]) * 100
                if margin < 8.0:
                    safety_warning = (
                        f"Close differential: Option {ranked_options[0]['option']} "
                        f"({ranked_options[0]['percentage']:.1f}%) closely leads Option {ranked_options[1]['option']} "
                        f"({ranked_options[1]['percentage']:.1f}%). Clinical correlation advised."
                    )

        # 5. Translation
        translated_answer = None
        if self.use_translation and target_lang != "en":
            print(f"Translating diagnosis to {target_lang}...")
            diag_str = f"The most likely diagnosis is Option {pred_key}, {pred_text}, with {confidence:.0f}% confidence."
            translated_answer = self.translator.translate(
                diag_str, direction=f"en2{target_lang}"
            )

        # 6. Speech Generation
        audio_path = None
        if self.use_tts and generate_audio:
            print(f"Generating speech in {target_lang}...")
            if target_lang == "en":
                audio_path = self.speaker.speak_answer(
                    answer=answer_text,
                    confidence=confidence,
                    evidence_source=evidence_source or "Clinical Case Evaluation",
                    lang="en",
                )
            elif target_lang == "hi" and translated_answer:
                audio_path = self.speaker.speak(
                    text=translated_answer,
                    lang="hi",
                )

        result = {
            "mode": "multi_choice",
            "question": question,
            "context": context,
            "options": option_texts,
            "prediction": answer_text,
            "selected_option": pred_key,
            "selected_text": pred_text,
            "probabilities": all_option_probs,
            "ranked_options": ranked_options,
            "confidence": confidence,
            "safety_warning": safety_warning,
            "num_refinement_iterations": output["num_iterations"],
            "explanations": explanations,
            "translation": translated_answer,
            "audio_path": audio_path,
        }

        return result


def main():
    parser = argparse.ArgumentParser(description="RAG-RERoBERTa Biomedical QA Inference")
    parser.add_argument("--question", type=str, required=True, help="Clinical question or patient case")
    parser.add_argument("--context", type=str, default="", help="Clinical context or background notes")
    parser.add_argument(
        "--options",
        nargs="+",
        default=None,
        help="Candidate diagnosis options (e.g. --options 'Aspirin' 'Ibuprofen' 'Metformin' 'Lisinopril')",
    )
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
        options=args.options,
        target_lang=args.lang,
        generate_audio=not args.no_tts,
        explain=not args.no_explain,
    )

    print("\n" + "=" * 65)
    if result.get("mode") == "multi_choice":
        print("          BIOMEDICAL DIAGNOSIS SELECTION (MULTI-CHOICE)")
        print("=" * 65)
        print(f"Question: {result['question']}")
        if result.get("context"):
            print(f"Context:  {result['context']}")
        print(f"\nSelected Diagnosis: {result['prediction']}")
        print(f"Confidence Score:   {result['confidence']:.1f}%")
        print("\nProbability Distribution across all options:")
        for rank, opt_data in enumerate(result.get("ranked_options", []), 1):
            pct = opt_data["percentage"]
            bar_len = int(pct / 100.0 * 25)
            bar = "█" * bar_len + "░" * (25 - bar_len)
            is_winner = " ★ SELECTED" if opt_data["option"] == result.get("selected_option") else ""
            print(f"  [{rank}] Option {opt_data['option']} ({opt_data['text']}): {pct:>6.2f}% |{bar}|{is_winner}")
    else:
        print("          BIOMEDICAL QA PREDICTION (CLASSIFICATION)")
        print("=" * 65)
        print(f"Question: {result['question']}")
        print(f"Answer:   {result['prediction']} (Confidence: {result['confidence']:.1f}%)")
        print("Class Probabilities:")
        for lbl, pr in result["probabilities"].items():
            print(f"  - {lbl:>5}: {pr * 100:.1f}%")

    if result.get("safety_warning"):
        print(f"\n⚠️  Clinical Safety Alert: {result['safety_warning']}")

    if result.get("translation"):
        print(f"\nHindi:    {result['translation']}")

    if result.get("audio_path"):
        print(f"Audio:    {result['audio_path']}")

    if result.get("explanations") and result["explanations"].get("evidence"):
        print("\nTop Evidence Tokens:")
        for ev in result["explanations"]["evidence"]:
            token = ev["token"].replace("Ġ", "")
            if token and token not in ["[CLS]", "[SEP]", "[PAD]"]:
                print(f"  - '{token}' (importance: {ev['importance']:.3f})")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
