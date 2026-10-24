"""Fast regression tests for the multi-choice data and scoring contract.

These do not download a checkpoint: the encoder/refinement layers are tiny
deterministic stand-ins so they can run in CI as well as on a GPU host.
"""

import json

import torch
import torch.nn as nn

from src.data.dataset import MultiChoiceQADataset, create_multi_choice_dataloaders
from src.models.rerb_qa import RERBQAModel


class _Preprocessor:
    def encode_multi_choice_label(self, value):
        return ord(value.upper()) - ord("A")

    def tokenize_multi_choice(self, question, options, max_length, return_tensors):
        # Option identity is deliberately retained in token 0.
        ids = torch.tensor([[i + 1, 9, 0, 0] for i, _ in enumerate(options)])
        return {"input_ids": ids, "attention_mask": ids.ne(0).long()}


class _Encoder(nn.Module):
    def forward(self, input_ids, attention_mask):
        values = input_ids[:, :1].float().unsqueeze(-1)
        return {"last_hidden_state": values.expand(-1, input_ids.size(1), 2)}


class _Refinement(nn.Module):
    def forward(self, embeddings, attention_mask, **_):
        return {"refined_embedding": embeddings, "convergence_norms": [], "num_iterations": 1}


def _tiny_mc_model():
    model = RERBQAModel.__new__(RERBQAModel)
    nn.Module.__init__(model)
    model.encoder = _Encoder()
    model.recursive_refinement = _Refinement()
    model.mc_scorer = nn.Linear(2, 1, bias=False)
    model.mc_scorer.weight.data.fill_(1.0)
    model.use_confidence = False
    return model


def test_mc_label_alignment_and_question_group_split(tmp_path):
    rows = [
        {"question": "duplicate", "options": {"D": "d", "A": "a", "C": "c", "B": "b"}, "answer_idx": "B"},
        {"question": "duplicate", "context": "A: a | B: b | C: c | D: d", "answer": "b"},
    ] + [
        {"question": f"q{i}", "context": "A: a | B: b | C: c | D: d", "answer": "c"}
        for i in range(20)
    ]
    path = tmp_path / "medqa.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    dataset = MultiChoiceQADataset(path, preprocessor=_Preprocessor(), max_options=4)
    assert dataset.records[0]["options"] == ["a", "b", "c", "d"]
    assert dataset.records[0]["answer_idx"] == 1
    for record in dataset.records:
        assert record["options"][record["answer_idx"]] in {"b", "c"}

    train, val, test = create_multi_choice_dataloaders(
        path, batch_size=2, preprocessor=_Preprocessor(), num_workers=0, seed=3,
    )
    memberships = []
    for loader in (train, val, test):
        memberships.append({loader.dataset.dataset.records[i]["question"] for i in loader.dataset.indices})
    assert not (memberships[0] & memberships[1])
    assert not (memberships[0] & memberships[2])
    assert not (memberships[1] & memberships[2])


def test_mask_is_valid_and_logits_vary_across_options():
    model = _tiny_mc_model()
    ids = torch.tensor([[[1, 9], [2, 9], [3, 9], [4, 9]]])
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    output = model._forward_multi_choice(
        ids, ids.ne(0).long(), labels=torch.tensor([3]),
        valid_num_options=torch.tensor([4]), label_smoothing=0.0,
    )
    output["loss"].backward()
    optimizer.step()
    output = model._forward_multi_choice(ids, ids.ne(0).long(), valid_num_options=torch.tensor([4]))
    assert output["valid_options_mask"].tolist() == [[True, True, True, True]]
    assert output["option_logit_variance"].item() > 0


def test_fixed_64_example_mc_overfits_without_smoothing_or_calibration():
    """The scorer must learn a tiny fixed MC batch well above chance."""
    torch.manual_seed(0)
    model = _tiny_mc_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    labels = torch.arange(64) % 4
    # Each correct candidate has a learnable positive cue; label positions vary.
    ids = torch.ones(64, 4, 2, dtype=torch.long)
    ids[torch.arange(64), labels, 0] = 2
    mask = ids.ne(0).long()
    initial = None
    history = []
    for step in range(300):
        output = model._forward_multi_choice(
            ids, mask, labels=labels, valid_num_options=torch.full((64,), 4),
            label_smoothing=0.0, calibration_weight=0.0,
        )
        if initial is None:
            initial = output["loss"].item()
        optimizer.zero_grad()
        output["loss"].backward()
        optimizer.step()
        if (step + 1) % 20 == 0:
            accuracy = (output["logits"].argmax(-1) == labels).float().mean().item()
            history.append(accuracy)
            print(f"step={step + 1} train_accuracy={accuracy:.4f}")
    final = model._forward_multi_choice(
        ids, mask, labels=labels, valid_num_options=torch.full((64,), 4),
        label_smoothing=0.0, calibration_weight=0.0,
    )
    accuracy = (final["logits"].argmax(-1) == labels).float().mean().item()
    assert final["loss"].item() < initial
    assert accuracy > 0.90
    assert history[-1] > 0.90
