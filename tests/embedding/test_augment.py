"""
Tests for data augmentation functions in embedding.augment module.
"""

import torch
import pytest
import numpy as np
from embedding.augment import (
    jitter_series,
    mask_series,
    crop_series,
    augment_series,
    create_augmented_pair,
    get_augmentation_logger,
    AugmentationLogger
)


class TestJitterSeries:
    """Test jitter augmentation function."""
    
    def test_jitter_preserves_shape(self):
        """Test that jitter preserves input shape."""
        x = torch.randn(32, 50)
        augmented = jitter_series(x, noise_std=0.01)
        assert augmented.shape == x.shape
    
    def test_jitter_preserves_dtype(self):
        """Test that jitter preserves data type."""
        x = torch.randn(16, 30, dtype=torch.float64)
        augmented = jitter_series(x, noise_std=0.01)
        assert augmented.dtype == x.dtype
    
    def test_jitter_adds_noise(self):
        """Test that jitter actually adds noise."""
        x = torch.zeros(10, 20)
        augmented = jitter_series(x, noise_std=0.1)
        # Should not be identical to zeros (with very high probability)
        assert not torch.allclose(augmented, x)
    
    def test_jitter_zero_noise(self):
        """Test jitter with zero noise std."""
        x = torch.randn(8, 25)
        augmented = jitter_series(x, noise_std=0.0)
        assert torch.allclose(augmented, x)
    
    def test_jitter_negative_noise_error(self):
        """Test that negative noise std raises error."""
        x = torch.randn(8, 25)
        with pytest.raises(ValueError, match="noise_std must be non-negative"):
            jitter_series(x, noise_std=-0.1)
    
    def test_jitter_reproducibility_with_generator(self):
        """Test that jitter is reproducible with generator."""
        x = torch.randn(8, 20)
        
        gen1 = torch.Generator()
        gen1.manual_seed(42)
        aug1 = jitter_series(x, noise_std=0.1, generator=gen1)
        
        gen2 = torch.Generator()
        gen2.manual_seed(42)
        aug2 = jitter_series(x, noise_std=0.1, generator=gen2)
        
        assert torch.allclose(aug1, aug2)
    
    def test_jitter_3d_input(self):
        """Test jitter with 3D input tensors."""
        x = torch.randn(4, 8, 30)
        augmented = jitter_series(x, noise_std=0.05)
        assert augmented.shape == x.shape
        assert not torch.allclose(augmented, x)


class TestMaskSeries:
    """Test mask augmentation function."""
    
    def test_mask_preserves_shape(self):
        """Test that masking preserves input shape."""
        x = torch.randn(16, 40)
        augmented = mask_series(x, mask_prob=0.2)
        assert augmented.shape == x.shape
    
    def test_mask_preserves_dtype(self):
        """Test that masking preserves data type."""
        x = torch.randn(8, 20, dtype=torch.float64)
        augmented = mask_series(x, mask_prob=0.1)
        assert augmented.dtype == x.dtype
    
    def test_mask_zero_probability(self):
        """Test masking with zero probability."""
        x = torch.randn(8, 30)
        augmented = mask_series(x, mask_prob=0.0)
        assert torch.allclose(augmented, x)
    
    def test_mask_one_probability(self):
        """Test masking with probability 1."""
        x = torch.randn(8, 30)
        augmented = mask_series(x, mask_prob=1.0, mask_value=999.0)
        expected = torch.full_like(x, 999.0)
        assert torch.allclose(augmented, expected)
    
    def test_mask_invalid_probability_error(self):
        """Test that invalid probabilities raise error."""
        x = torch.randn(8, 20)
        
        with pytest.raises(ValueError, match="mask_prob must be in"):
            mask_series(x, mask_prob=-0.1)
        
        with pytest.raises(ValueError, match="mask_prob must be in"):
            mask_series(x, mask_prob=1.1)
    
    def test_mask_custom_value(self):
        """Test masking with custom mask value."""
        x = torch.ones(4, 10)
        augmented = mask_series(x, mask_prob=0.5, mask_value=-999.0)
        
        # Should contain both original values (1.0) and mask values (-999.0)
        unique_vals = torch.unique(augmented)
        assert 1.0 in unique_vals or -999.0 in unique_vals
    
    def test_mask_reproducibility_with_generator(self):
        """Test that masking is reproducible with generator."""
        x = torch.randn(8, 20)
        
        gen1 = torch.Generator()
        gen1.manual_seed(123)
        aug1 = mask_series(x, mask_prob=0.3, generator=gen1)
        
        gen2 = torch.Generator()
        gen2.manual_seed(123)
        aug2 = mask_series(x, mask_prob=0.3, generator=gen2)
        
        assert torch.allclose(aug1, aug2)
    
    def test_mask_contiguous_mode(self):
        """Test contiguous masking mode."""
        x = torch.randn(4, 20)
        augmented = mask_series(x, mask_prob=0.2, mask_mode='contiguous')
        assert augmented.shape == x.shape
    
    def test_mask_invalid_mode_error(self):
        """Test that invalid mask mode raises error."""
        x = torch.randn(8, 20)
        with pytest.raises(ValueError, match="mask_mode must be"):
            mask_series(x, mask_prob=0.1, mask_mode='invalid')


class TestCropSeries:
    """Test crop augmentation function."""
    
    def test_crop_preserves_shape(self):
        """Test that cropping preserves input shape."""
        x = torch.randn(16, 50)
        augmented = crop_series(x, crop_ratio=0.8)
        assert augmented.shape == x.shape
    
    def test_crop_preserves_dtype(self):
        """Test that cropping preserves data type."""
        x = torch.randn(8, 30, dtype=torch.float64)
        augmented = crop_series(x, crop_ratio=0.7)
        assert augmented.dtype == x.dtype
    
    def test_crop_no_change_full_ratio(self):
        """Test that crop ratio of 1.0 returns original."""
        x = torch.randn(8, 25)
        augmented = crop_series(x, crop_ratio=1.0)
        assert torch.allclose(augmented, x)
    
    def test_crop_invalid_ratio_error(self):
        """Test that invalid crop ratios raise error."""
        x = torch.randn(8, 20)
        
        with pytest.raises(ValueError, match="crop_ratio must be in"):
            crop_series(x, crop_ratio=0.0)
        
        with pytest.raises(ValueError, match="crop_ratio must be in"):
            crop_series(x, crop_ratio=1.1)
    
    def test_crop_small_sequence(self):
        """Test cropping very small sequences."""
        x = torch.randn(4, 3)  # Very short sequence
        augmented = crop_series(x, crop_ratio=0.5)
        assert augmented.shape == x.shape
    
    def test_crop_reproducibility_with_generator(self):
        """Test that cropping is reproducible with generator."""
        x = torch.randn(8, 30)
        
        gen1 = torch.Generator()
        gen1.manual_seed(456)
        aug1 = crop_series(x, crop_ratio=0.6, generator=gen1)
        
        gen2 = torch.Generator()
        gen2.manual_seed(456)
        aug2 = crop_series(x, crop_ratio=0.6, generator=gen2)
        
        assert torch.allclose(aug1, aug2)
    
    def test_crop_3d_input(self):
        """Test cropping with 3D input tensors."""
        x = torch.randn(4, 8, 40)
        augmented = crop_series(x, crop_ratio=0.75)
        assert augmented.shape == x.shape


class TestAugmentSeries:
    """Test unified augment_series function."""
    
    def test_augment_jitter_method(self):
        """Test augment_series with jitter method."""
        x = torch.randn(8, 30)
        augmented = augment_series(x, method='jitter', noise_std=0.05)
        assert augmented.shape == x.shape
        assert not torch.allclose(augmented, x)
    
    def test_augment_mask_method(self):
        """Test augment_series with mask method."""
        x = torch.randn(8, 30)
        augmented = augment_series(x, method='mask', mask_prob=0.2)
        assert augmented.shape == x.shape
    
    def test_augment_crop_method(self):
        """Test augment_series with crop method."""
        x = torch.randn(8, 30)
        augmented = augment_series(x, method='crop', crop_ratio=0.8)
        assert augmented.shape == x.shape
    
    def test_augment_invalid_method_error(self):
        """Test that invalid method raises error."""
        x = torch.randn(8, 20)
        with pytest.raises(ValueError, match="Unsupported augmentation method"):
            augment_series(x, method='invalid')
    
    def test_augment_with_generator(self):
        """Test augment_series with generator."""
        x = torch.randn(8, 25)
        gen = torch.Generator()
        gen.manual_seed(789)
        
        augmented = augment_series(x, method='jitter', generator=gen, noise_std=0.1)
        assert augmented.shape == x.shape


class TestCreateAugmentedPair:
    """Test augmented pair creation for contrastive learning."""
    
    def test_create_pair_returns_two_tensors(self):
        """Test that create_augmented_pair returns two tensors."""
        x = torch.randn(16, 40)
        aug1, aug2 = create_augmented_pair(x)
        
        assert isinstance(aug1, torch.Tensor)
        assert isinstance(aug2, torch.Tensor)
        assert aug1.shape == x.shape
        assert aug2.shape == x.shape
    
    def test_create_pair_different_augmentations(self):
        """Test that the two augmentations are different."""
        x = torch.randn(8, 30)
        aug1, aug2 = create_augmented_pair(x, 'jitter', 'mask')
        
        # The two augmentations should be different
        assert not torch.allclose(aug1, aug2)
    
    def test_create_pair_custom_parameters(self):
        """Test create_augmented_pair with custom parameters."""
        x = torch.randn(8, 25)
        aug1, aug2 = create_augmented_pair(
            x, 'jitter', 'crop',
            jitter_noise_std=0.2,
            crop_crop_ratio=0.6
        )
        
        assert aug1.shape == x.shape
        assert aug2.shape == x.shape
    
    def test_create_pair_reproducibility(self):
        """Test pair creation reproducibility with generator."""
        x = torch.randn(8, 20)
        
        gen1 = torch.Generator()
        gen1.manual_seed(999)
        aug1_a, aug2_a = create_augmented_pair(x, generator=gen1)
        
        gen2 = torch.Generator()
        gen2.manual_seed(999)
        aug1_b, aug2_b = create_augmented_pair(x, generator=gen2)
        
        assert torch.allclose(aug1_a, aug1_b)
        assert torch.allclose(aug2_a, aug2_b)


class TestAugmentationLogger:
    """Test augmentation logging functionality."""
    
    def test_logger_initialization(self):
        """Test logger initializes correctly."""
        logger = AugmentationLogger()
        assert logger.get_logs() == []
    
    def test_logger_records_augmentation(self):
        """Test that logger records augmentation calls."""
        logger = AugmentationLogger()
        logger.log_augmentation(
            method='jitter',
            params={'noise_std': 0.05},
            input_shape=(8, 30),
            output_shape=(8, 30)
        )
        
        logs = logger.get_logs()
        assert len(logs) == 1
        assert logs[0]['method'] == 'jitter'
        assert logs[0]['params']['noise_std'] == 0.05
    
    def test_logger_multiple_entries(self):
        """Test logger with multiple augmentation entries."""
        logger = AugmentationLogger()
        
        # Log multiple augmentations
        for i in range(3):
            logger.log_augmentation(
                method=f'method_{i}',
                params={'param': i},
                input_shape=(8, 20),
                output_shape=(8, 20)
            )
        
        logs = logger.get_logs()
        assert len(logs) == 3
        assert logs[2]['params']['param'] == 2
    
    def test_logger_clear_logs(self):
        """Test clearing logger logs."""
        logger = AugmentationLogger()
        logger.log_augmentation('test', {}, (1,), (1,))
        
        assert len(logger.get_logs()) == 1
        logger.clear_logs()
        assert len(logger.get_logs()) == 0
    
    def test_global_logger_integration(self):
        """Test that augmentation functions use global logger."""
        # Clear any existing logs
        global_logger = get_augmentation_logger()
        global_logger.clear_logs()
        
        # Perform augmentation
        x = torch.randn(4, 20)
        jitter_series(x, noise_std=0.01)
        
        # Check that global logger recorded it
        logs = global_logger.get_logs()
        assert len(logs) >= 1
        assert any(log['method'] == 'jitter' for log in logs)


class TestAugmentationIntegration:
    """Integration tests for augmentation functions."""
    
    def test_all_augmentations_preserve_properties(self):
        """Test that all augmentations preserve basic tensor properties."""
        x = torch.randn(8, 30, dtype=torch.float32)
        
        methods = [
            ('jitter', {'noise_std': 0.1}),
            ('mask', {'mask_prob': 0.2}),
            ('crop', {'crop_ratio': 0.8})
        ]
        
        for method, kwargs in methods:
            augmented = augment_series(x, method, **kwargs)
            assert augmented.shape == x.shape
            assert augmented.dtype == x.dtype
            assert torch.isfinite(augmented).all()
    
    def test_augmentation_gradient_flow(self):
        """Test that augmentations preserve gradient flow."""
        x = torch.randn(4, 20, requires_grad=True)
        
        # Test each augmentation method
        for method in ['jitter', 'mask', 'crop']:
            x_aug = augment_series(x, method)
            loss = x_aug.sum()
            loss.backward(retain_graph=True)
            
            assert x.grad is not None
            assert torch.any(x.grad != 0)
            x.grad.zero_()  # Clear gradients for next test
    
    def test_augmentation_device_consistency(self):
        """Test that augmentations work on different devices."""
        x_cpu = torch.randn(4, 15)
        
        # Test on CPU
        aug_cpu = augment_series(x_cpu, 'jitter')
        assert aug_cpu.device == x_cpu.device
        
        # Test on GPU if available
        if torch.cuda.is_available():
            x_gpu = x_cpu.cuda()
            aug_gpu = augment_series(x_gpu, 'jitter')
            assert aug_gpu.device == x_gpu.device


if __name__ == "__main__":
    pytest.main([__file__])