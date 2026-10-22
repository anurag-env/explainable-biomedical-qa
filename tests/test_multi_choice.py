import pytest
import torch
import torch.nn.functional as F
from src.data.preprocessing import BiomedicalPreprocessor
from src.models.rerb_qa import RERBQAModel
from src.inference_pipeline import RERBQAPipeline


def test_tokenize_multi_choice():
    """Test multi-choice tokenization across multiple options."""
    preprocessor = BiomedicalPreprocessor(max_question_length=32, max_context_length=32)
    question = "A 45-year-old male presents with persistent cough and fever. What is the most likely diagnosis?"
    options = [
        "Streptococcal Pharyngitis",
        "Community-Acquired Pneumonia",
        "Acute Bronchitis",
        "Pulmonary Tuberculosis",
    ]

    encoded = preprocessor.tokenize_multi_choice(question, options, max_length=64)

    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert encoded["input_ids"].shape[0] == 4  # 4 options
    assert encoded["attention_mask"].shape[0] == 4


def test_multi_choice_model_forward():
    """Test model forward pass with task_type='multi_choice'."""
    torch.manual_seed(42)
    model = RERBQAModel(
        encoder_name="allenai/biomed_roberta_base",
        hidden_size=768,
        num_labels=3,
        num_heads=12,
        max_iterations=3,
        use_evidence_graph=False,
    )
    model.eval()

    # Synthetic batch: batch_size=2, num_options=4, seq_len=16
    batch_size = 2
    num_options = 4
    seq_len = 16

    input_ids = torch.randint(10, 1000, (batch_size, num_options, seq_len))
    attention_mask = torch.ones((batch_size, num_options, seq_len))
    labels = torch.tensor([1, 3])  # Correct option indices

    with torch.no_grad():
        output = model(
            input_ids,
            attention_mask,
            labels=labels,
            task_type="multi_choice",
        )

    assert "logits" in output
    assert output["logits"].shape == (batch_size, num_options)
    assert "loss" in output
    assert output["loss"].item() > 0

    probs = F.softmax(output["logits"], dim=-1)
    # Check that probabilities for each question sum to 1 across all 4 options
    for b in range(batch_size):
        assert torch.isclose(probs[b].sum(), torch.tensor(1.0), atol=1e-5)


def test_predict_multi_choice():
    """Test predict_multi_choice helper method."""
    torch.manual_seed(42)
    model = RERBQAModel(
        encoder_name="allenai/biomed_roberta_base",
        hidden_size=768,
        use_evidence_graph=False,
    )
    model.eval()

    batch_size = 1
    num_options = 4
    seq_len = 16

    input_ids = torch.randint(10, 1000, (batch_size, num_options, seq_len))
    attention_mask = torch.ones((batch_size, num_options, seq_len))
    option_texts = ["Aspirin", "Metformin", "Lisinopril", "Atorvastatin"]

    results = model.predict_multi_choice(input_ids, attention_mask, option_texts=option_texts)

    assert len(results) == 1
    res = results[0]
    assert "prediction" in res
    assert 0 <= res["prediction"] < 4
    assert "prediction_label" in res
    assert "probabilities" in res
    assert len(res["probabilities"]) == 4

    prob_sum = sum(res["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-4


def test_inference_pipeline_multi_choice():
    """Test full inference pipeline with options provided."""
    pipeline = RERBQAPipeline(
        checkpoint_path="outputs/checkpoints/stage2/best.pt",
        use_translation=False,
        use_tts=False,
    )

    question = "Which medication is the first-line therapy for type 2 diabetes mellitus?"
    options = ["Metformin", "Glipizide", "Insulin glargine", "Acarbose"]

    result = pipeline(
        question=question,
        options=options,
        explain=False,
        generate_audio=False,
    )

    assert result["mode"] == "multi_choice"
    assert "Option " in result["prediction"]
    assert len(result["probabilities"]) == 4
    assert len(result["ranked_options"]) == 4

    # Verify probability distribution
    total_pct = sum(opt["percentage"] for opt in result["ranked_options"])
    assert 99.0 <= total_pct <= 101.0
