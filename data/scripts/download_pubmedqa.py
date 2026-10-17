#!/usr/bin/env python3
"""
Download and preprocess the PubMedQA dataset.

PubMedQA: A Dataset for Biomedical Research Question Answering
- 211,300 artificial samples (PQA-A)
- 61,200 unlabeled samples (PQA-U)
- 1,000 expert-labeled samples (PQA-L)
Total: ~273,500 samples

Reference: Jin et al., EMNLP-IJCNLP 2019
"""

import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def download_pubmedqa(output_dir: str = "data/raw/pubmedqa") -> dict:
    """
    Download PubMedQA from Hugging Face and save to disk.

    Returns:
        dict with counts per subset.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    counts = {}

    # PubMedQA labeled (expert-annotated, 1k samples)
    print("Downloading PubMedQA labeled (expert) subset...")
    try:
        ds_labeled = load_dataset("qiaojin/PubMedQA", "pqa_labeled", trust_remote_code=True)
        labeled_records = []
        for split_name in ds_labeled:
            for item in tqdm(ds_labeled[split_name], desc=f"Processing labeled/{split_name}"):
                record = normalize_pubmedqa_record(item, subset="expert")
                labeled_records.append(record)
        counts["expert"] = len(labeled_records)
        save_jsonl(labeled_records, output_path / "pubmedqa_labeled.jsonl")
        print(f"  -> {counts['expert']} expert-labeled samples saved.")
    except Exception as e:
        print(f"  -> Error downloading labeled subset: {e}")
        counts["expert"] = 0

    # PubMedQA artificial (211k samples)
    print("Downloading PubMedQA artificial subset...")
    try:
        ds_artificial = load_dataset("qiaojin/PubMedQA", "pqa_artificial", trust_remote_code=True)
        artificial_records = []
        for split_name in ds_artificial:
            for item in tqdm(ds_artificial[split_name], desc=f"Processing artificial/{split_name}"):
                record = normalize_pubmedqa_record(item, subset="artificial")
                artificial_records.append(record)
        counts["artificial"] = len(artificial_records)
        save_jsonl(artificial_records, output_path / "pubmedqa_artificial.jsonl")
        print(f"  -> {counts['artificial']} artificial samples saved.")
    except Exception as e:
        print(f"  -> Error downloading artificial subset: {e}")
        counts["artificial"] = 0

    # PubMedQA unlabeled (61k samples)
    print("Downloading PubMedQA unlabeled subset...")
    try:
        ds_unlabeled = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", trust_remote_code=True)
        unlabeled_records = []
        for split_name in ds_unlabeled:
            for item in tqdm(ds_unlabeled[split_name], desc=f"Processing unlabeled/{split_name}"):
                record = normalize_pubmedqa_record(item, subset="unlabeled")
                unlabeled_records.append(record)
        counts["unlabeled"] = len(unlabeled_records)
        save_jsonl(unlabeled_records, output_path / "pubmedqa_unlabeled.jsonl")
        print(f"  -> {counts['unlabeled']} unlabeled samples saved.")
    except Exception as e:
        print(f"  -> Error downloading unlabeled subset: {e}")
        counts["unlabeled"] = 0

    # Summary
    total = sum(counts.values())
    counts["total"] = total
    print(f"\nPubMedQA Download Summary:")
    print(f"  Expert:     {counts.get('expert', 0):>10,}")
    print(f"  Artificial: {counts.get('artificial', 0):>10,}")
    print(f"  Unlabeled:  {counts.get('unlabeled', 0):>10,}")
    print(f"  Total:      {total:>10,}")

    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(counts, f, indent=2)

    return counts


def normalize_pubmedqa_record(item: dict, subset: str) -> dict:
    """
    Normalize a PubMedQA record into the unified schema:
    {question, context, answer, label}
    """
    # Extract context from the contexts field
    context = ""
    if "context" in item and item["context"]:
        if isinstance(item["context"], dict):
            # PubMedQA stores contexts as dict with 'contexts' and 'labels' keys
            contexts = item["context"].get("contexts", [])
            if isinstance(contexts, list):
                context = " ".join(contexts)
            else:
                context = str(contexts)
        elif isinstance(item["context"], str):
            context = item["context"]
        elif isinstance(item["context"], list):
            context = " ".join(item["context"])

    # Extract the answer
    answer = item.get("long_answer", item.get("final_decision", ""))
    if isinstance(answer, list):
        answer = " ".join(answer)

    # Extract label (Yes/No/Maybe)
    label = item.get("final_decision", "")
    if label and isinstance(label, str):
        label = label.strip().lower()
        if label in ("yes", "no", "maybe"):
            label = label.capitalize()
        else:
            label = ""

    return {
        "question": item.get("question", ""),
        "context": context,
        "answer": str(answer) if answer else "",
        "label": label,
        "source": "pubmedqa",
        "subset": subset,
        "pubid": str(item.get("pubid", "")),
    }


def save_jsonl(records: list, filepath: Path):
    """Save records as JSONL format."""
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/pubmedqa"
    download_pubmedqa(output_dir)
