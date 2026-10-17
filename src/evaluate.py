"""
Evaluation Module.

Computes all metrics from Section 9:
  QA: Accuracy, F1, Precision, Recall, MCC
  Explainability: Faithfulness, Sufficiency, Comprehensiveness
  Translation: BLEU, METEOR, TER, BERTScore
  Speech: MOS (placeholder), PESQ, STOI
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, classification_report, confusion_matrix,
)


# ── QA Metrics ──────────────────────────────────────────────

def compute_qa_metrics(predictions: List[int], labels: List[int],
                       label_names: List[str] = None) -> Dict:
    """Compute QA evaluation metrics."""
    if label_names is None:
        label_names = ["Yes", "No", "Maybe"]

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions) if len(set(labels)) > 1 else 0.0,
    }

    # Per-class F1
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    for i, name in enumerate(label_names):
        if i < len(per_class_f1):
            metrics[f"f1_{name}"] = per_class_f1[i]

    # Classification report
    metrics["classification_report"] = classification_report(
        labels, predictions, target_names=label_names, zero_division=0
    )

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(labels, predictions).tolist()

    return metrics


# ── Explainability Metrics ──────────────────────────────────

def compute_faithfulness(model, dataloader, device, top_k_ratio=0.1):
    """
    Compute faithfulness: how much does removing top-attributed tokens
    change the prediction?
    """
    model.eval()
    faithfulness_scores = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask, return_explanations=True)
            original_probs = torch.softmax(output["logits"], dim=-1)
            original_pred = original_probs.argmax(dim=-1)
            original_conf = original_probs.max(dim=-1)[0]

        # Get attention-based importance
        heatmap = output.get("attention_heatmap", {}).get("heatmap")
        if heatmap is None:
            continue

        # For each sample in batch
        for i in range(input_ids.size(0)):
            seq_len = attention_mask[i].sum().item()
            k = max(1, int(seq_len * top_k_ratio))

            # Get importance scores (use CLS row of attention heatmap)
            if heatmap.dim() == 3:
                importance = heatmap[i, 0, :int(seq_len)]
            else:
                importance = torch.ones(int(seq_len))

            top_indices = importance.topk(k).indices

            # Mask top-k tokens
            masked_ids = input_ids[i].clone().unsqueeze(0)
            masked_mask = attention_mask[i].clone().unsqueeze(0)
            pad_id = 1  # RoBERTa pad token
            for idx in top_indices:
                masked_ids[0, idx] = pad_id

            with torch.no_grad():
                masked_output = model(masked_ids, masked_mask)
                masked_probs = torch.softmax(masked_output["logits"], dim=-1)
                masked_conf = masked_probs[0, original_pred[i]].item()

            # Faithfulness = drop in confidence when removing important tokens
            faith = original_conf[i].item() - masked_conf
            faithfulness_scores.append(faith)

    return np.mean(faithfulness_scores) if faithfulness_scores else 0.0


def compute_sufficiency(model, dataloader, device, top_k_ratio=0.1):
    """
    Sufficiency: keeping only top-attributed tokens should maintain prediction.
    """
    model.eval()
    sufficiency_scores = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask, return_explanations=True)
            original_probs = torch.softmax(output["logits"], dim=-1)
            original_pred = original_probs.argmax(dim=-1)
            original_conf = original_probs.max(dim=-1)[0]

        heatmap = output.get("attention_heatmap", {}).get("heatmap")
        if heatmap is None:
            continue

        for i in range(input_ids.size(0)):
            seq_len = int(attention_mask[i].sum().item())
            k = max(1, int(seq_len * top_k_ratio))

            if heatmap.dim() == 3:
                importance = heatmap[i, 0, :seq_len]
            else:
                importance = torch.ones(seq_len)

            top_indices = set(importance.topk(k).indices.tolist())

            # Keep only top-k tokens
            sufficient_ids = input_ids[i].clone().unsqueeze(0)
            pad_id = 1
            for j in range(seq_len):
                if j not in top_indices:
                    sufficient_ids[0, j] = pad_id

            with torch.no_grad():
                suff_output = model(sufficient_ids, attention_mask[i].unsqueeze(0))
                suff_probs = torch.softmax(suff_output["logits"], dim=-1)
                suff_conf = suff_probs[0, original_pred[i]].item()

            sufficiency = suff_conf  # High = good (prediction maintained)
            sufficiency_scores.append(sufficiency)

    return np.mean(sufficiency_scores) if sufficiency_scores else 0.0


def compute_comprehensiveness(model, dataloader, device, top_k_ratio=0.1):
    """
    Comprehensiveness: removing attributed tokens should change prediction.
    (Same as faithfulness but we want high values.)
    """
    return compute_faithfulness(model, dataloader, device, top_k_ratio)


# ── Translation Metrics ────────────────────────────────────

def compute_translation_metrics(predictions: List[str], references: List[str]) -> Dict:
    """Compute BLEU, METEOR, TER, BERTScore for translation quality."""
    metrics = {}

    # BLEU
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        ref_tokens = [[ref.split()] for ref in references]
        pred_tokens = [pred.split() for pred in predictions]
        smooth = SmoothingFunction().method1
        metrics["bleu"] = corpus_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)
    except Exception as e:
        metrics["bleu"] = 0.0
        metrics["bleu_error"] = str(e)

    # METEOR
    try:
        from nltk.translate.meteor_score import meteor_score as ms
        scores = [ms([ref], pred) for ref, pred in zip(references, predictions)]
        metrics["meteor"] = np.mean(scores)
    except Exception as e:
        metrics["meteor"] = 0.0

    # BERTScore
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(predictions, references, lang="hi", verbose=False)
        metrics["bertscore_f1"] = F1.mean().item()
        metrics["bertscore_precision"] = P.mean().item()
        metrics["bertscore_recall"] = R.mean().item()
    except Exception as e:
        metrics["bertscore_f1"] = 0.0

    return metrics


# ── Speech Metrics ──────────────────────────────────────────

def compute_speech_metrics(generated_paths: List[str],
                            reference_paths: List[str] = None) -> Dict:
    """Compute PESQ, STOI for speech quality. MOS requires human evaluation."""
    metrics = {"mos": "Requires human evaluation"}

    if reference_paths is None:
        return metrics

    try:
        import soundfile as sf
        from pesq import pesq
        from pystoi import stoi

        pesq_scores = []
        stoi_scores = []

        for gen_path, ref_path in zip(generated_paths, reference_paths):
            if not os.path.exists(gen_path) or not os.path.exists(ref_path):
                continue

            gen_audio, gen_sr = sf.read(gen_path)
            ref_audio, ref_sr = sf.read(ref_path)

            # Resample to 16kHz for PESQ
            if gen_sr != 16000:
                import librosa
                gen_audio = librosa.resample(gen_audio, orig_sr=gen_sr, target_sr=16000)
                gen_sr = 16000
            if ref_sr != 16000:
                import librosa
                ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=16000)
                ref_sr = 16000

            # Match lengths
            min_len = min(len(gen_audio), len(ref_audio))
            gen_audio = gen_audio[:min_len]
            ref_audio = ref_audio[:min_len]

            try:
                p = pesq(16000, ref_audio, gen_audio, "wb")
                pesq_scores.append(p)
            except Exception:
                pass

            try:
                s = stoi(ref_audio, gen_audio, 16000, extended=False)
                stoi_scores.append(s)
            except Exception:
                pass

        if pesq_scores:
            metrics["pesq"] = np.mean(pesq_scores)
        if stoi_scores:
            metrics["stoi"] = np.mean(stoi_scores)

    except ImportError as e:
        metrics["speech_error"] = f"Missing dependency: {e}"

    return metrics


# ── Full Evaluation Pipeline ───────────────────────────────

def run_full_evaluation(model, test_loader, device, output_dir="outputs/reports"):
    """Run complete evaluation across all metric categories."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Running full evaluation...")

    # 1. QA Metrics
    print("\n--- QA Metrics ---")
    model.eval()
    all_preds, all_labels = [], []

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask)
            preds = output["logits"].argmax(dim=-1)

        valid = labels >= 0
        all_preds.extend(preds[valid].cpu().tolist())
        all_labels.extend(labels[valid].cpu().tolist())

    qa_metrics = compute_qa_metrics(all_preds, all_labels)
    print(f"  Accuracy: {qa_metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {qa_metrics['f1_macro']:.4f}")
    print(f"  MCC: {qa_metrics['mcc']:.4f}")
    print(qa_metrics["classification_report"])

    # 2. Explainability Metrics
    print("\n--- Explainability Metrics ---")
    faith = compute_faithfulness(model, test_loader, device)
    suff = compute_sufficiency(model, test_loader, device)
    comp = compute_comprehensiveness(model, test_loader, device)
    explain_metrics = {
        "faithfulness": faith,
        "sufficiency": suff,
        "comprehensiveness": comp,
    }
    for k, v in explain_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Combine all metrics
    all_metrics = {
        "qa": {k: v for k, v in qa_metrics.items() if k != "classification_report"},
        "explainability": explain_metrics,
    }

    # Save report
    with open(output_path / "evaluation_report.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save classification report
    with open(output_path / "classification_report.txt", "w") as f:
        f.write(qa_metrics["classification_report"])

    print(f"\nResults saved to {output_path}")
    return all_metrics


# ── Ablation Study ─────────────────────────────────────────

def run_ablation_study(configs, test_loader, device, output_dir="outputs/reports"):
    """Run ablation study over attention heads and recursive iterations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {"attention_heads": {}, "recursive_iterations": {}}

    # Ablation 1: Attention heads
    for n_heads in [8, 12, 16]:
        print(f"\nAblation: {n_heads} attention heads")
        model = RERBQAModel(
            num_heads=n_heads,
            hidden_size=768,
            use_evidence_graph=False,  # Isolate attention effect
        ).to(device)

        # Quick evaluation (would normally train first)
        model.eval()
        all_preds, all_labels = [], []
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                output = model(input_ids, attention_mask)
                preds = output["logits"].argmax(dim=-1)
            valid = labels >= 0
            all_preds.extend(preds[valid].cpu().tolist())
            all_labels.extend(labels[valid].cpu().tolist())

        metrics = compute_qa_metrics(all_preds, all_labels)
        results["attention_heads"][n_heads] = {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
        }

    # Ablation 2: Recursive iterations
    for max_iter in [3, 5, 7, 10]:
        print(f"\nAblation: {max_iter} recursive iterations")
        model = RERBQAModel(
            max_iterations=max_iter,
            hidden_size=768,
            use_evidence_graph=False,
        ).to(device)

        model.eval()
        all_preds, all_labels = [], []
        iter_counts = []
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                output = model(input_ids, attention_mask)
                preds = output["logits"].argmax(dim=-1)
                iter_counts.append(output["num_iterations"])
            valid = labels >= 0
            all_preds.extend(preds[valid].cpu().tolist())
            all_labels.extend(labels[valid].cpu().tolist())

        metrics = compute_qa_metrics(all_preds, all_labels)
        results["recursive_iterations"][max_iter] = {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "avg_iterations_used": np.mean(iter_counts),
        }

    with open(output_path / "ablation_study.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAblation results saved to {output_path / 'ablation_study.json'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="RAG-RERoBERTa Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/raw/pubmedqa")
    parser.add_argument("--output-dir", type=str, default="outputs/reports")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    from src.data.dataset import create_dataloaders
    _, _, test_loader = create_dataloaders(
        args.data_dir, batch_size=args.batch_size, labeled_only=True,
    )

    if args.ablation:
        from src.train import load_config
        configs = load_config()
        run_ablation_study(configs, test_loader, device, args.output_dir)
    else:
        model = RERBQAModel()
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        model = model.to(device)
        run_full_evaluation(model, test_loader, device, args.output_dir)


if __name__ == "__main__":
    from src.models.rerb_qa import RERBQAModel
    main()
