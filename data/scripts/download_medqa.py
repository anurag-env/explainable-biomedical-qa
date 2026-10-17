#!/usr/bin/env python3
"""
Download and preprocess the MedQA dataset.

MedQA: A Large-scale Open Domain Question Answering Dataset from Medical Exams
- 12,723 samples (US Medical License Exam questions)
- Multiple-choice format with 4-5 options

Reference: Jin et al., 2020
Source: https://github.com/jind11/MedQA
"""

import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def download_medqa(output_dir: str = "data/raw/medqa") -> dict:
    """
    Download MedQA from Hugging Face and save to disk.

    Returns:
        dict with counts per split.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    counts = {}

    print("Downloading MedQA dataset...")
    try:
        # Try loading from Hugging Face (GBaker/MedQA-USMLE-4-options is commonly used)
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", trust_remote_code=True)

        all_records = []
        for split_name in ds:
            split_records = []
            for item in tqdm(ds[split_name], desc=f"Processing {split_name}"):
                record = normalize_medqa_record(item)
                split_records.append(record)

            counts[split_name] = len(split_records)
            save_jsonl(split_records, output_path / f"medqa_{split_name}.jsonl")
            all_records.extend(split_records)
            print(f"  -> {split_name}: {len(split_records)} samples saved.")

        # Also save the combined dataset
        save_jsonl(all_records, output_path / "medqa_all.jsonl")

    except Exception as e:
        print(f"  -> Error downloading from HuggingFace: {e}")
        print("  -> Trying alternative source...")

        try:
            # Alternative: bigbio/med_qa
            ds = load_dataset("bigbio/med_qa", "med_qa_en_source", trust_remote_code=True)
            all_records = []
            for split_name in ds:
                split_records = []
                for item in tqdm(ds[split_name], desc=f"Processing {split_name}"):
                    record = normalize_medqa_record_bigbio(item)
                    split_records.append(record)
                counts[split_name] = len(split_records)
                save_jsonl(split_records, output_path / f"medqa_{split_name}.jsonl")
                all_records.extend(split_records)
            save_jsonl(all_records, output_path / "medqa_all.jsonl")
        except Exception as e2:
            print(f"  -> Alternative source also failed: {e2}")
            counts["total"] = 0
            return counts

    total = sum(counts.values())
    counts["total"] = total
    print(f"\nMedQA Download Summary:")
    for split_name, count in counts.items():
        if split_name != "total":
            print(f"  {split_name}: {count:>8,}")
    print(f"  Total:    {total:>8,}")

    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(counts, f, indent=2)

    return counts


def normalize_medqa_record(item: dict) -> dict:
    """
    Normalize a MedQA record (from GBaker/MedQA-USMLE-4-options) into unified schema.
    """
    question = item.get("question", item.get("sent1", ""))

    # Build context from the options
    options = item.get("options", {})
    if isinstance(options, dict):
        context_parts = [f"{k}: {v}" for k, v in options.items()]
        context = " | ".join(context_parts)
    elif isinstance(options, list):
        context = " | ".join(options)
    else:
        context = str(options) if options else ""

    # Get the correct answer
    answer_idx = item.get("answer_idx", item.get("answer", ""))
    answer = item.get("answer", "")

    # If answer is an index into options
    if isinstance(answer_idx, str) and isinstance(options, dict) and answer_idx in options:
        answer = options[answer_idx]
    elif isinstance(answer, int) and isinstance(options, list) and answer < len(options):
        answer = options[answer]

    # Map to Yes/No/Maybe where applicable
    label = ""
    answer_lower = str(answer).strip().lower() if answer else ""
    if answer_lower in ("yes", "no", "maybe"):
        label = answer_lower.capitalize()

    return {
        "question": question,
        "context": context,
        "answer": str(answer),
        "label": label,
        "source": "medqa",
        "subset": "usmle",
    }


def normalize_medqa_record_bigbio(item: dict) -> dict:
    """
    Normalize from bigbio/med_qa format.
    """
    question = item.get("question", "")
    choices = item.get("choices", [])

    context = ""
    if choices:
        if isinstance(choices, list):
            context = " | ".join(str(c) for c in choices)

    answer = item.get("answer", "")
    label = ""

    return {
        "question": question,
        "context": context,
        "answer": str(answer),
        "label": label,
        "source": "medqa",
        "subset": "usmle",
    }


def save_jsonl(records: list, filepath: Path):
    """Save records as JSONL format."""
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/medqa"
    download_medqa(output_dir)
