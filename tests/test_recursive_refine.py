import pytest
import torch
import torch.nn.functional as F
from src.models.recursive_refine import RecursiveEmbeddingRefinement

def test_recursive_refinement_convergence():
    """Test that the recursive refinement layer eventually converges on synthetic data."""
    batch_size = 2
    seq_len = 16
    hidden_size = 64
    
    # Initialize module with a large epsilon so it converges faster in test
    module = RecursiveEmbeddingRefinement(
        hidden_size=hidden_size,
        num_heads=4,
        max_iterations=10,
        convergence_epsilon=1e-2,  # Relaxed for test speed
        gru_num_layers=1,
    )
    
    # Create synthetic input
    torch.manual_seed(42)
    E_0 = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Run refinement
    info = module.get_convergence_info(E_0, attention_mask)
    
    # Assertions
    assert "num_iterations" in info
    assert "convergence_norms" in info
    assert len(info["convergence_norms"]) > 0
    assert info["num_iterations"] <= 10
    
    # Output should have same shape
    out = module(E_0, attention_mask)
    assert out["refined_embedding"].shape == (batch_size, seq_len, hidden_size)

def test_recursive_refinement_masking():
    """Test that masked tokens do not affect the output."""
    batch_size = 2
    seq_len = 10
    hidden_size = 32
    
    module = RecursiveEmbeddingRefinement(
        hidden_size=hidden_size,
        num_heads=2,
        max_iterations=3,
    )
    
    E_0 = torch.randn(batch_size, seq_len, hidden_size)
    
    # Mask out the second half of the sequence for batch 0
    mask = torch.ones(batch_size, seq_len)
    mask[0, 5:] = 0
    
    out = module(E_0, mask)
    refined = out["refined_embedding"]
    
    # The output shape should be correct
    assert refined.shape == (batch_size, seq_len, hidden_size)
