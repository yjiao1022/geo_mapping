"""
Data augmentation utilities for time series data.

This module provides vectorized augmentation functions for geographic time series:
- Temporal jitter with gaussian noise
- Random masking with different strategies
- Temporal cropping and scaling
- Parameter tracking for reproducibility
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union, Literal
import logging


class AugmentationLogger:
    """
    Logger for augmentation parameters to aid reproducibility and debugging.
    
    Tracks parameters used in each augmentation call for analysis and 
    reproducibility purposes.
    """
    
    def __init__(self):
        self.logs: list = []
        self.logger = logging.getLogger(__name__)
    
    def log_augmentation(self, method: str, params: Dict[str, Any], 
                        input_shape: tuple, output_shape: tuple) -> None:
        """Log augmentation parameters and shapes."""
        log_entry = {
            'method': method,
            'params': params.copy(),
            'input_shape': input_shape,
            'output_shape': output_shape
        }
        self.logs.append(log_entry)
        
        self.logger.debug(f"Applied {method} with params {params}, "
                         f"shape {input_shape} -> {output_shape}")
    
    def get_logs(self) -> list:
        """Return all logged augmentation operations."""
        return self.logs.copy()
    
    def clear_logs(self) -> None:
        """Clear augmentation logs."""
        self.logs.clear()


# Global augmentation logger instance
_aug_logger = AugmentationLogger()


def get_augmentation_logger() -> AugmentationLogger:
    """Get the global augmentation logger instance."""
    return _aug_logger


def jitter_series(x: torch.Tensor, noise_std: float = 0.01, 
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Apply Gaussian jitter to time series.
    
    Adds Gaussian noise to the input time series for data augmentation.
    Fully vectorized operation supporting batch processing.
    
    Args:
        x (torch.Tensor): Input time series of shape (..., T) 
        noise_std (float): Standard deviation of Gaussian noise (default: 0.01)
        generator (Optional[torch.Generator]): RNG generator for reproducibility
        
    Returns:
        torch.Tensor: Jittered time series with same shape as input
    """
    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    
    original_shape = x.shape
    if generator is not None:
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) * noise_std
    else:
        noise = torch.randn_like(x) * noise_std
    augmented = x + noise
    
    # Log augmentation parameters
    _aug_logger.log_augmentation(
        method='jitter',
        params={'noise_std': noise_std},
        input_shape=original_shape,
        output_shape=augmented.shape
    )
    
    return augmented


def mask_series(x: torch.Tensor, mask_prob: float = 0.1, 
               mask_value: float = 0.0,
               mask_mode: Literal['random', 'contiguous'] = 'random',
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Apply random masking to time series.
    
    Masks random time steps with a specified probability. Supports both
    random masking and contiguous block masking strategies.
    
    Args:
        x (torch.Tensor): Input time series of shape (..., T)
        mask_prob (float): Probability of masking each time step (default: 0.1)
        mask_value (float): Value to use for masked positions (default: 0.0)
        mask_mode (str): 'random' for random masking, 'contiguous' for block masking
        generator (Optional[torch.Generator]): RNG generator for reproducibility
        
    Returns:
        torch.Tensor: Masked time series with same shape as input
    """
    if not 0 <= mask_prob <= 1:
        raise ValueError(f"mask_prob must be in [0, 1], got {mask_prob}")
    
    original_shape = x.shape
    time_dim = x.size(-1)
    
    if mask_mode == 'random':
        # Random masking - each timestep independently
        mask = torch.rand(x.shape, generator=generator, 
                         device=x.device, dtype=x.dtype) < mask_prob
    elif mask_mode == 'contiguous':
        # Contiguous block masking
        if mask_prob == 0:
            mask = torch.zeros_like(x, dtype=torch.bool)
        else:
            # Determine block length based on mask_prob
            block_len = max(1, int(time_dim * mask_prob))
            # Random start position for each sample in batch
            batch_shape = x.shape[:-1]  # All dimensions except time
            starts = torch.randint(0, max(1, time_dim - block_len + 1), 
                                 batch_shape, generator=generator, device=x.device)
            
            # Create mask tensor
            mask = torch.zeros_like(x, dtype=torch.bool)
            
            # Handle different batch shapes efficiently
            if len(batch_shape) == 1:
                # 2D input: (batch, time)
                for i in range(batch_shape[0]):
                    start = starts[i].item()
                    end = min(start + block_len, time_dim)
                    mask[i, start:end] = True
            elif len(batch_shape) == 2:
                # 3D input: (batch, features, time)
                for i in range(batch_shape[0]):
                    for j in range(batch_shape[1]):
                        start = starts[i, j].item()
                        end = min(start + block_len, time_dim)
                        mask[i, j, start:end] = True
            else:
                # General case for higher dimensions (less efficient but works)
                flat_batch_size = 1
                for dim in batch_shape:
                    flat_batch_size *= dim
                
                starts_flat = starts.flatten()
                mask_flat = mask.view(flat_batch_size, time_dim)
                
                for i in range(flat_batch_size):
                    start = starts_flat[i].item()
                    end = min(start + block_len, time_dim)
                    mask_flat[i, start:end] = True
    else:
        raise ValueError(f"mask_mode must be 'random' or 'contiguous', got {mask_mode}")
    
    # Apply mask
    augmented = x.clone()
    augmented[mask] = mask_value
    
    # Log augmentation parameters
    _aug_logger.log_augmentation(
        method='mask',
        params={'mask_prob': mask_prob, 'mask_value': mask_value, 'mask_mode': mask_mode},
        input_shape=original_shape,
        output_shape=augmented.shape
    )
    
    return augmented


def crop_series(x: torch.Tensor, crop_ratio: float = 0.8,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Apply random temporal cropping with resizing.
    
    Randomly crops a portion of the time series and resizes back to original
    length using interpolation. Provides temporal scale augmentation.
    
    Args:
        x (torch.Tensor): Input time series of shape (..., T)
        crop_ratio (float): Fraction of original length to keep (default: 0.8)
        generator (Optional[torch.Generator]): RNG generator for reproducibility
        
    Returns:
        torch.Tensor: Cropped and resized time series with same shape as input
    """
    if not 0 < crop_ratio <= 1:
        raise ValueError(f"crop_ratio must be in (0, 1], got {crop_ratio}")
    
    original_shape = x.shape
    time_dim = x.size(-1)
    
    if crop_ratio == 1.0:
        # No cropping needed
        return x
    
    # Calculate crop window
    crop_len = max(1, int(time_dim * crop_ratio))
    max_start = time_dim - crop_len
    
    if max_start <= 0:
        # Cannot crop, return original
        return x
    
    # Random start position (same for all samples in batch for simplicity)
    start = torch.randint(0, max_start + 1, (1,), generator=generator).item()
    end = start + crop_len
    
    # Extract crop
    cropped = x[..., start:end]  # (..., crop_len)
    
    # Resize back to original length using interpolation
    # Reshape for F.interpolate: (..., T) -> (N, 1, T) where N = prod(batch_dims)
    batch_dims = x.shape[:-1]
    n_samples = 1
    for dim in batch_dims:
        n_samples *= dim
    
    # Flatten batch dimensions and add channel dimension
    cropped_flat = cropped.view(n_samples, 1, crop_len)
    
    # Interpolate to original length
    resized_flat = F.interpolate(cropped_flat, size=time_dim, 
                               mode='linear', align_corners=True)
    
    # Reshape back to original batch shape
    augmented = resized_flat.view(original_shape)
    
    # Log augmentation parameters
    _aug_logger.log_augmentation(
        method='crop',
        params={'crop_ratio': crop_ratio, 'crop_window': (start, end)},
        input_shape=original_shape,
        output_shape=augmented.shape
    )
    
    return augmented


def augment_series(x: torch.Tensor, method: str, 
                  generator: Optional[torch.Generator] = None,
                  **kwargs) -> torch.Tensor:
    """
    Apply temporal data augmentation to time series.
    
    Unified interface for different augmentation methods with consistent
    parameter handling and logging.
    
    Args:
        x (torch.Tensor): Input time series of shape (..., T)
        method (str): Augmentation method ('jitter', 'mask', or 'crop')
        generator (Optional[torch.Generator]): RNG generator for reproducibility
        **kwargs: Method-specific parameters
        
    Returns:
        torch.Tensor: Augmented time series with same shape as input
        
    Raises:
        ValueError: If method is not supported
    """
    if method == 'jitter':
        noise_std = kwargs.get('noise_std', 0.01)
        return jitter_series(x, noise_std=noise_std, generator=generator)
    
    elif method == 'mask':
        mask_prob = kwargs.get('mask_prob', 0.1)
        mask_value = kwargs.get('mask_value', 0.0)
        mask_mode = kwargs.get('mask_mode', 'random')
        return mask_series(x, mask_prob=mask_prob, mask_value=mask_value,
                          mask_mode=mask_mode, generator=generator)
    
    elif method == 'crop':
        crop_ratio = kwargs.get('crop_ratio', 0.8)
        return crop_series(x, crop_ratio=crop_ratio, generator=generator)
    
    else:
        raise ValueError(f"Unsupported augmentation method: {method}. "
                        f"Supported methods: ['jitter', 'mask', 'crop']")


def create_augmented_pair(x: torch.Tensor, 
                         aug1_method: str = 'jitter',
                         aug2_method: str = 'crop',
                         generator: Optional[torch.Generator] = None,
                         **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a pair of differently augmented views for contrastive learning.
    
    Generates two different augmentations of the same input for use in
    contrastive learning objectives.
    
    Args:
        x (torch.Tensor): Input time series of shape (..., T)
        aug1_method (str): First augmentation method (default: 'jitter')
        aug2_method (str): Second augmentation method (default: 'crop')
        generator (Optional[torch.Generator]): RNG generator for reproducibility
        **kwargs: Parameters for augmentation methods (prefixed by method name)
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Two augmented views of the input
    """
    # Extract method-specific parameters
    aug1_params = {k[len(aug1_method)+1:]: v for k, v in kwargs.items() 
                   if k.startswith(f"{aug1_method}_")}
    aug2_params = {k[len(aug2_method)+1:]: v for k, v in kwargs.items() 
                   if k.startswith(f"{aug2_method}_")}
    
    # Create augmented views
    x_aug1 = augment_series(x, aug1_method, generator=generator, **aug1_params)
    x_aug2 = augment_series(x, aug2_method, generator=generator, **aug2_params)
    
    return x_aug1, x_aug2