#!/usr/bin/env python3
"""
Download and preprocess the HealthSearchQA dataset.

HealthSearchQA: Health-related search questions
- 4,440 samples
- Commonly searched medical questions

Reference: Singhal et al., Nature 2023
Source: https://huggingface.co/datasets/katielink/healthsearchqa
"""

import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def download_healthsearchqa(output_dir: str = "data/raw/healthsearchqa") -> dict:
    """
    Download HealthSearchQA from Hugging Face and save to disk.

    Returns:
        dict with sample counts.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    counts = {}

    print("Downloading HealthSearchQA dataset...")
    try:
        ds = load_dataset("katielink/healthsearchqa", "all_data", trust_remote_code=True)

        all_records = []
        for split_name in ds:
            split_records = []
            for item in tqdm(ds[split_name], desc=f"Processing {split_name}"):
                record = normalize_healthsearchqa_record(item)
                split_records.append(record)

            counts[split_name] = len(split_records)
            save_jsonl(split_records, output_path / f"healthsearchqa_{split_name}.jsonl")
            all_records.extend(split_records)
            print(f"  -> {split_name}: {len(split_records)} samples saved.")

        # Save combined
        save_jsonl(all_records, output_path / "healthsearchqa_all.jsonl")

    except Exception as e:
        print(f"  -> Error downloading HealthSearchQA: {e}")
        counts["total"] = 0
        return counts

    total = sum(counts.values())
    counts["total"] = total
    print(f"\nHealthSearchQA Download Summary:")
    for split_name, count in counts.items():
        if split_name != "total":
            print(f"  {split_name}: {count:>8,}")
    print(f"  Total:    {total:>8,}")

    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(counts, f, indent=2)

    return counts


def normalize_healthsearchqa_record(item: dict) -> dict:
    """
    Normalize a HealthSearchQA record into the unified schema.
    HealthSearchQA typically has 'question' as the main field.
    """
    question = item.get("question", item.get("text", ""))

    # HealthSearchQA may not have context/abstracts — set empty
    context = item.get("context", item.get("answer", ""))
    if context is None:
        context = ""

    answer = item.get("answer", "")
    if answer is None:
        answer = ""

    label = ""
    if answer:
        answer_lower = str(answer).strip().lower()
        if answer_lower in ("yes", "no", "maybe"):
            label = answer_lower.capitalize()

    return {
        "question": str(question),
        "context": str(context),
        "answer": str(answer),
        "label": label,
        "source": "healthsearchqa",
        "subset": "default",
    }


def save_jsonl(records: list, filepath: Path):
    """Save records as JSONL format."""
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/healthsearchqa"
    download_healthsearchqa(output_dir)
