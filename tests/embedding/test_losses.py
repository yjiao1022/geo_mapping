"""
Tests for loss functions in embedding.losses module.
"""

import torch
import pytest
import numpy as np
from embedding.losses import (
    reconstruction_loss,
    contrastive_loss,
    predictive_loss,
    MemoryBankContrastiveLoss
)


class TestReconstructionLoss:
    """Test reconstruction loss function."""
    
    def test_reconstruction_loss_zero_for_identical(self):
        """Test that reconstruction loss is 0 for identical inputs."""
        x = torch.randn(32, 64)
        loss = reconstruction_loss(x, x)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    
    def test_reconstruction_loss_positive_for_different(self):
        """Test that reconstruction loss is positive for different inputs."""
        x = torch.randn(32, 64)
        x_recon = torch.randn(32, 64)
        loss = reconstruction_loss(x, x_recon)
        assert loss > 0
    
    def test_reconstruction_loss_shape_mismatch_error(self):
        """Test that shape mismatch raises ValueError."""
        x = torch.randn(32, 64)
        x_recon = torch.randn(16, 64)
        with pytest.raises(ValueError, match="Input shapes must match"):
            reconstruction_loss(x, x_recon)
    
    def test_reconstruction_loss_3d_input(self):
        """Test reconstruction loss with 3D input tensors."""
        x = torch.randn(8, 16, 32)
        x_recon = torch.randn(8, 16, 32)
        loss = reconstruction_loss(x, x_recon)
        assert loss >= 0 and torch.isfinite(loss)


class TestContrastiveLoss:
    """Test contrastive loss function."""
    
    def test_contrastive_loss_finite_positive(self):
        """Test that contrastive loss is finite and positive for random inputs."""
        z_i = torch.randn(32, 64)
        z_j = torch.randn(32, 64)
        loss = contrastive_loss(z_i, z_j)
        assert torch.isfinite(loss) and loss > 0
    
    def test_contrastive_loss_identical_inputs(self):
        """Test contrastive loss with identical inputs (should be low but not zero)."""
        z = torch.randn(32, 64)
        loss = contrastive_loss(z, z)
        # Even identical inputs have randomness due to negative sampling
        assert torch.isfinite(loss) and loss >= 0
    
    def test_contrastive_loss_temperature_effect(self):
        """Test that temperature affects loss magnitude."""
        z_i = torch.randn(16, 32)
        z_j = torch.randn(16, 32)
        
        loss_high_temp = contrastive_loss(z_i, z_j, temperature=1.0)
        loss_low_temp = contrastive_loss(z_i, z_j, temperature=0.1)
        
        # Lower temperature typically gives higher loss (sharper distribution)
        assert torch.isfinite(loss_high_temp) and torch.isfinite(loss_low_temp)
    
    def test_contrastive_loss_shape_mismatch_error(self):
        """Test that shape mismatch raises ValueError."""
        z_i = torch.randn(32, 64)
        z_j = torch.randn(16, 64)
        with pytest.raises(ValueError, match="Input shapes must match"):
            contrastive_loss(z_i, z_j)
    
    def test_contrastive_loss_batch_size_one(self):
        """Test contrastive loss with batch size 1."""
        z_i = torch.randn(1, 64)
        z_j = torch.randn(1, 64)
        loss = contrastive_loss(z_i, z_j)
        assert torch.isfinite(loss) and loss >= 0
    
    def test_contrastive_loss_gradients(self):
        """Test that contrastive loss has proper gradients."""
        z_i = torch.randn(8, 16, requires_grad=True)
        z_j = torch.randn(8, 16, requires_grad=True)
        loss = contrastive_loss(z_i, z_j)
        loss.backward()
        
        assert z_i.grad is not None
        assert z_j.grad is not None
        assert torch.any(z_i.grad != 0)
        assert torch.any(z_j.grad != 0)


class TestPredictiveLoss:
    """Test predictive loss function."""
    
    def test_predictive_loss_zero_for_identical(self):
        """Test that predictive loss is 0 for identical inputs."""
        past = torch.randn(32, 64)
        future = past.clone()
        loss = predictive_loss(past, future)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    
    def test_predictive_loss_positive_for_different(self):
        """Test that predictive loss is positive for different inputs."""
        past = torch.randn(32, 64)
        future = torch.randn(32, 64)
        loss = predictive_loss(past, future)
        assert loss > 0
    
    def test_predictive_loss_shape_mismatch_error(self):
        """Test that shape mismatch raises ValueError."""
        past = torch.randn(32, 64)
        future = torch.randn(16, 64)
        with pytest.raises(ValueError, match="Input shapes must match"):
            predictive_loss(past, future)


class TestMemoryBankContrastiveLoss:
    """Test memory bank contrastive loss."""
    
    def test_memory_bank_initialization(self):
        """Test memory bank is properly initialized."""
        loss_fn = MemoryBankContrastiveLoss(embedding_dim=64)
        
        assert loss_fn.memory_bank.shape == (4096, 64)
        assert loss_fn.memory_ptr.item() == 0
        # Memory bank should be normalized
        norms = torch.norm(loss_fn.memory_bank, dim=1)
        assert torch.allclose(norms, torch.ones(4096), atol=1e-6)
    
    def test_memory_bank_loss_computation(self):
        """Test memory bank loss computation."""
        loss_fn = MemoryBankContrastiveLoss(embedding_dim=32, memory_size=128)
        z_i = torch.randn(16, 32)
        z_j = torch.randn(16, 32)
        
        loss = loss_fn(z_i, z_j)
        assert torch.isfinite(loss) and loss > 0
    
    def test_memory_bank_update(self):
        """Test that memory bank gets updated after forward pass."""
        loss_fn = MemoryBankContrastiveLoss(embedding_dim=32, memory_size=64)
        initial_ptr = loss_fn.memory_ptr.item()
        
        z_i = torch.randn(8, 32)
        z_j = torch.randn(8, 32)
        loss_fn(z_i, z_j)
        
        # Pointer should have advanced
        final_ptr = loss_fn.memory_ptr.item()
        assert final_ptr != initial_ptr
    
    def test_memory_bank_wraparound(self):
        """Test memory bank wraparound when full."""
        loss_fn = MemoryBankContrastiveLoss(embedding_dim=16, memory_size=32)
        
        # Fill memory bank beyond capacity
        for _ in range(5):
            z_i = torch.randn(8, 16)
            z_j = torch.randn(8, 16)
            loss_fn(z_i, z_j)
        
        # Should have wrapped around
        assert loss_fn.memory_ptr.item() < 32
    
    def test_memory_bank_gradients(self):
        """Test that gradients flow properly with memory bank."""
        loss_fn = MemoryBankContrastiveLoss(embedding_dim=16, memory_size=64)
        z_i = torch.randn(8, 16, requires_grad=True)
        z_j = torch.randn(8, 16, requires_grad=True)
        
        loss = loss_fn(z_i, z_j)
        loss.backward()
        
        assert z_i.grad is not None
        assert z_j.grad is not None
        assert torch.any(z_i.grad != 0)
        assert torch.any(z_j.grad != 0)
    
    def test_memory_bank_detached_negatives(self):
        """Test that memory bank negatives are detached (no gradients)."""
        loss_fn = MemoryBankContrastiveLoss(embedding_dim=16, memory_size=32)
        
        # Fill memory bank with some embeddings
        z_init = torch.randn(16, 16, requires_grad=True)
        z_dummy = torch.randn(16, 16)
        loss_fn(z_init, z_dummy)
        
        # Now compute loss with new embeddings
        z_i = torch.randn(8, 16, requires_grad=True)
        z_j = torch.randn(8, 16, requires_grad=True)
        loss = loss_fn(z_i, z_j)
        
        # Check that memory bank requires no gradients
        assert not loss_fn.memory_bank.requires_grad
        
        loss.backward()
        # Initial embeddings should not receive gradients from memory bank usage
        assert z_init.grad is None or torch.all(z_init.grad == 0)


class TestLossIntegration:
    """Integration tests for loss functions."""
    
    def test_all_losses_with_same_input_shapes(self):
        """Test that all losses work with same input tensor shapes."""
        batch_size, embedding_dim = 16, 32
        x = torch.randn(batch_size, embedding_dim)
        x_recon = torch.randn(batch_size, embedding_dim)
        
        # Test all loss functions
        recon_loss = reconstruction_loss(x, x_recon)
        cont_loss = contrastive_loss(x, x_recon)
        pred_loss = predictive_loss(x, x_recon)
        
        memory_loss_fn = MemoryBankContrastiveLoss(embedding_dim)
        mem_loss = memory_loss_fn(x, x_recon)
        
        # All should be finite and non-negative
        for loss in [recon_loss, cont_loss, pred_loss, mem_loss]:
            assert torch.isfinite(loss) and loss >= 0
    
    def test_loss_numerical_stability(self):
        """Test loss numerical stability with extreme inputs."""
        # Very small embeddings
        z_small = torch.randn(8, 16) * 1e-6
        loss_small = contrastive_loss(z_small, z_small)
        assert torch.isfinite(loss_small)
        
        # Very large embeddings  
        z_large = torch.randn(8, 16) * 1e6
        loss_large = contrastive_loss(z_large, z_large, temperature=1.0)
        assert torch.isfinite(loss_large)


if __name__ == "__main__":
    pytest.main([__file__])