"""
Unified Biomedical QA Dataset and DataLoader.

Loads data from PubMedQA, MedQA, and HealthSearchQA into a common schema
and provides PyTorch Dataset/DataLoader interfaces for training.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, Subset
from transformers import AutoTokenizer

from .preprocessing import BiomedicalPreprocessor


class BiomedicalQADataset(Dataset):
    """
    PyTorch Dataset for biomedical QA data.

    Expected schema per record:
    {
        "question": str,
        "context": str,
        "answer": str,
        "label": str (Yes/No/Maybe or empty),
        "source": str (pubmedqa/medqa/healthsearchqa),
        "subset": str
    }
    """

    def __init__(
        self,
        data_path: Union[str, Path, List[str]],
        preprocessor: Optional[BiomedicalPreprocessor] = None,
        max_length: int = 512,
        tokenizer_name: str = "allenai/biomed_roberta_base",
    ):
        """
        Args:
            data_path: Path to JSONL file(s) or directory containing JSONL files.
            preprocessor: BiomedicalPreprocessor instance (created if None).
            max_length: Maximum token sequence length.
            tokenizer_name: HuggingFace tokenizer to use.
        """
        self.max_length = max_length

        if preprocessor is None:
            self.preprocessor = BiomedicalPreprocessor(
                tokenizer_name=tokenizer_name,
                max_question_length=max_length,
                max_context_length=max_length,
            )
        else:
            self.preprocessor = preprocessor

        # Load records
        self.records = self._load_records(data_path)
        if len(self.records) == 0:
            raise FileNotFoundError(
                f"No records found in '{data_path}'. "
                f"Please verify that the dataset was downloaded by running the corresponding script in data/scripts/."
            )

    def _load_records(self, data_path: Union[str, Path, List[str]]) -> List[dict]:
        """Load records from JSONL file(s) or directory."""
        records = []

        if isinstance(data_path, list):
            for p in data_path:
                records.extend(self._load_single(p))
        elif isinstance(data_path, (str, Path)):
            path = Path(data_path)
            if path.is_dir():
                jsonl_files = sorted(path.glob("**/*.jsonl"))
                # If specific split files exist alongside an *_all.jsonl, use only the split files
                split_files = [f for f in jsonl_files if not f.name.endswith("_all.jsonl")]
                files_to_load = split_files if split_files else jsonl_files
                for jsonl_file in files_to_load:
                    records.extend(self._load_single(jsonl_file))
            else:
                records.extend(self._load_single(path))

        return records

    def _load_single(self, filepath: Union[str, Path]) -> List[dict]:
        """Load records from a single JSONL file."""
        records = []
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"Warning: {filepath} does not exist, skipping.")
            return records

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        # Transform MedQA multiple choice into verification pairs (Yes/No)
                        if record.get("source") == "medqa" and not record.get("label"):
                            q = record.get("question", "")
                            ans = record.get("answer", "")
                            ctx = record.get("context", "")
                            options = [
                                opt.split(":", 1)[-1].strip()
                                for opt in ctx.split(" | ")
                                if ":" in opt
                            ]
                            distractors = [opt for opt in options if opt.lower() != ans.lower()]

                            # Positive sample (Correct answer -> Yes)
                            records.append({
                                "question": f"{q} Candidate: {ans}",
                                "context": ctx,
                                "answer": ans,
                                "label": "Yes",
                                "source": "medqa",
                                "subset": record.get("subset", "usmle"),
                            })
                            # Negative sample (Distractor -> No)
                            if distractors:
                                records.append({
                                    "question": f"{q} Candidate: {distractors[0]}",
                                    "context": ctx,
                                    "answer": distractors[0],
                                    "label": "No",
                                    "source": "medqa",
                                    "subset": record.get("subset", "usmle"),
                                })
                        else:
                            records.append(record)
                    except json.JSONDecodeError:
                        continue
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]

        question = record.get("question", "")
        context = record.get("context", "")
        label_str = record.get("label", "")

        # Tokenize the question-context pair
        encoded = self.preprocessor.tokenize_qa_pair(
            question=question,
            context=context,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Squeeze batch dimension (tokenizer returns [1, seq_len])
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

        # Add token_type_ids if present
        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        # Encode label
        label_id = self.preprocessor.encode_label(label_str)
        item["labels"] = torch.tensor(label_id, dtype=torch.long)

        return item

    def get_label_distribution(self) -> Dict[str, int]:
        """Get the distribution of labels in the dataset."""
        dist = {}
        for record in self.records:
            label = record.get("label", "Unknown")
            if not label:
                label = "Unknown"
            dist[label] = dist.get(label, 0) + 1
        return dist

    def filter_labeled(self) -> "BiomedicalQADataset":
        """Return a new dataset containing only records with valid labels."""
        valid_labels = {"Yes", "No", "Maybe"}
        filtered = [r for r in self.records if r.get("label", "") in valid_labels]

        new_dataset = BiomedicalQADataset.__new__(BiomedicalQADataset)
        new_dataset.max_length = self.max_length
        new_dataset.preprocessor = self.preprocessor
        new_dataset.records = filtered
        return new_dataset


def create_dataloaders(
    data_path: Union[str, Path, List[str]],
    batch_size: int = 16,
    max_length: int = 512,
    tokenizer_name: str = "allenai/biomed_roberta_base",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    labeled_only: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders from dataset files.

    Args:
        data_path: Path to data file(s) or directory.
        batch_size: Batch size for DataLoaders.
        max_length: Maximum token sequence length.
        tokenizer_name: HuggingFace tokenizer name.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        num_workers: Number of DataLoader workers.
        seed: Random seed for reproducible splits.
        labeled_only: If True, filter to only records with Yes/No/Maybe labels.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    preprocessor = BiomedicalPreprocessor(
        tokenizer_name=tokenizer_name,
        max_question_length=max_length,
        max_context_length=max_length,
    )

    dataset = BiomedicalQADataset(
        data_path=data_path,
        preprocessor=preprocessor,
        max_length=max_length,
    )

    if labeled_only:
        dataset = dataset.filter_labeled()

    # Split dataset
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Dataset splits: train={train_size}, val={val_size}, test={test_size}")
    return train_loader, val_loader, test_loader


class MultiChoiceQADataset(Dataset):
    """
    PyTorch Dataset for multi-choice biomedical QA data (MedQA style).

    Instead of converting options to Yes/No pairs, this dataset preserves
    the original multiple-choice structure so the model can compare all
    options simultaneously and pick the best diagnosis.

    Expected schema per record:
    {
        "question": str,
        "options": list[str] or dict[str, str],
        "answer_idx": int (0-based index of correct option),
        "context": str (legacy format, used as fallback),
        "answer": str,
        "source": str
    }
    """

    def __init__(
        self,
        data_path: Union[str, Path, List[str]],
        preprocessor: Optional[BiomedicalPreprocessor] = None,
        max_length: int = 512,
        tokenizer_name: str = "allenai/biomed_roberta_base",
        max_options: int = 4,
    ):
        """
        Args:
            data_path: Path to JSONL file(s) or directory containing JSONL files.
            preprocessor: BiomedicalPreprocessor instance (created if None).
            max_length: Maximum token sequence length per option pair.
            tokenizer_name: HuggingFace tokenizer to use.
            max_options: Maximum number of options to support (padding applied).
        """
        self.max_length = max_length
        self.max_options = max_options

        if preprocessor is None:
            self.preprocessor = BiomedicalPreprocessor(
                tokenizer_name=tokenizer_name,
                max_question_length=max_length,
                max_context_length=max_length,
            )
        else:
            self.preprocessor = preprocessor

        self.records = self._load_records(data_path)
        if len(self.records) == 0:
            raise FileNotFoundError(
                f"No multi-choice records found in '{data_path}'. "
                f"Please verify that the dataset was downloaded with multi-choice "
                f"fields by re-running data/scripts/download_medqa.py."
            )

    def _load_records(self, data_path: Union[str, Path, List[str]]) -> List[dict]:
        """Load multi-choice records from JSONL file(s) or directory."""
        records = []

        if isinstance(data_path, list):
            for p in data_path:
                records.extend(self._load_single(p))
        elif isinstance(data_path, (str, Path)):
            path = Path(data_path)
            if path.is_dir():
                jsonl_files = sorted(path.glob("**/*.jsonl"))
                split_files = [f for f in jsonl_files if not f.name.endswith("_all.jsonl")]
                files_to_load = split_files if split_files else jsonl_files
                for jsonl_file in files_to_load:
                    records.extend(self._load_single(jsonl_file))
            else:
                records.extend(self._load_single(path))

        return records

    def _load_single(self, filepath: Union[str, Path]) -> List[dict]:
        """Load multi-choice records from a single JSONL file."""
        records = []
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"Warning: {filepath} does not exist, skipping.")
            return records

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    option_keys = None

                    # Extract options
                    options = record.get("options")
                    if options is None:
                        # Parse from legacy context field ("A: X | B: Y | ...")
                        options = self._parse_options_from_context(
                            record.get("context", "")
                        )
                    elif isinstance(options, dict):
                        # Preserve the answer-key order, rather than relying on
                        # arbitrary mapping insertion order.
                        option_keys = sorted(options.keys())
                        options = [options[key] for key in option_keys]
                    else:
                        option_keys = None

                    if len(options) < 2:
                        continue  # Need at least 2 options

                    # Determine correct answer index
                    answer_idx = record.get("answer_idx")
                    if answer_idx is None:
                        answer_text = record.get("answer", "")
                        answer_idx = self._find_answer_index(
                            options, answer_text, record.get("options")
                        )
                    elif isinstance(answer_idx, str):
                        if option_keys is not None and answer_idx in option_keys:
                            answer_idx = option_keys.index(answer_idx)
                        else:
                            answer_idx = self.preprocessor.encode_multi_choice_label(answer_idx)

                    if (answer_idx is None or answer_idx < 0
                            or answer_idx >= len(options)
                            or answer_idx >= self.max_options):
                        continue  # Skip records without valid answer index

                    records.append({
                        "question": record.get("question", ""),
                        "options": options,
                        "answer_idx": answer_idx,
                        "source": record.get("source", "medqa"),
                    })
                except json.JSONDecodeError:
                    continue

        return records

    @staticmethod
    def _parse_options_from_context(context: str) -> List[str]:
        """Parse 'A: X | B: Y | ...' format into list of option texts."""
        if not context:
            return []
        parts = context.split(" | ")
        options = []
        for part in parts:
            if ":" in part:
                options.append(part.split(":", 1)[-1].strip())
            else:
                options.append(part.strip())
        return options

    @staticmethod
    def _find_answer_index(option_texts, answer_text, original_options):
        """Find the 0-based index of the correct answer in the options list."""
        answer_lower = str(answer_text).strip().lower()

        # Exact match
        for i, opt in enumerate(option_texts):
            if opt.strip().lower() == answer_lower:
                return i

        # Match via original option keys (dict)
        if isinstance(original_options, dict):
            for i, (key, val) in enumerate(original_options.items()):
                if str(val).strip().lower() == answer_lower:
                    return i

        # Substring match as fallback
        for i, opt in enumerate(option_texts):
            if answer_lower in opt.lower() or opt.lower() in answer_lower:
                return i

        return None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        question = record["question"]
        options = record["options"][:self.max_options]
        answer_idx = record["answer_idx"]
        num_real_options = len(options)

        # Pad to max_options with empty strings
        while len(options) < self.max_options:
            options.append("")

        # Tokenize each option paired with the question
        encoded = self.preprocessor.tokenize_multi_choice(
            question=question,
            options=options,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"],            # [max_options, seq_len]
            "attention_mask": encoded["attention_mask"],   # [max_options, seq_len]
            "labels": torch.tensor(answer_idx, dtype=torch.long),
            "num_options": torch.tensor(num_real_options, dtype=torch.long),
        }

        return item

    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of correct answer positions."""
        dist = {}
        for record in self.records:
            key = f"Option {chr(65 + record['answer_idx'])}"
            dist[key] = dist.get(key, 0) + 1
        return dist


def create_multi_choice_dataloaders(
    data_path: Union[str, Path, List[str]],
    batch_size: int = 16,
    max_length: int = 512,
    tokenizer_name: str = "allenai/biomed_roberta_base",
    preprocessor: Optional[BiomedicalPreprocessor] = None,
    max_options: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for multi-choice QA datasets.

    Args:
        data_path: Path to data file(s) or directory.
        batch_size: Batch size for DataLoaders.
        max_length: Maximum token sequence length per option pair.
        tokenizer_name: HuggingFace tokenizer name.
        max_options: Maximum number of options to support.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        num_workers: Number of DataLoader workers.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    if preprocessor is None:
        preprocessor = BiomedicalPreprocessor(
            tokenizer_name=tokenizer_name,
            max_question_length=max_length,
            max_context_length=max_length,
        )

    dataset = MultiChoiceQADataset(
        data_path=data_path,
        preprocessor=preprocessor,
        max_length=max_length,
        max_options=max_options,
    )

    # Split groups of identical questions, not individual rows.  This prevents
    # duplicate questions from leaking between train, validation, and test.
    groups = {}
    for index, record in enumerate(dataset.records):
        groups.setdefault(record["question"], []).append(index)
    group_items = list(groups.values())
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(group_items), generator=generator).tolist()
    ordered_groups = [group_items[index] for index in order]
    total = len(dataset)
    train_target = int(total * train_ratio)
    val_target = int(total * val_ratio)
    split_indices = ([], [], [])
    for group in ordered_groups:
        # Fill train then validation to their targets; the remainder is test.
        target = 0 if len(split_indices[0]) < train_target else (1 if len(split_indices[1]) < val_target else 2)
        split_indices[target].extend(group)
    train_dataset, val_dataset, test_dataset = (Subset(dataset, indices) for indices in split_indices)
    train_size, val_size, test_size = map(len, split_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Multi-choice dataset splits: train={train_size}, val={val_size}, test={test_size}")
    print(f"Options per question: max={max_options}")
    return train_loader, val_loader, test_loader
