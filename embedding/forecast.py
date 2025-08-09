"""
Forecasting utilities for time series data.

This module provides utilities for temporal splitting and autoregressive dataset
creation for predictive learning tasks:
- Forecast horizon splitting for future prediction
- Autoregressive dataset creation with sliding windows
- Flexible stride and stacking options
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional, Union


def forecast_horizon_split(x: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits time series into input and horizon target.
    
    Separates the last `horizon` time steps as prediction targets,
    using the remaining steps as input features.
    
    Args:
        x (np.ndarray): Time series of shape (n_geos, T) or (n_geos, n_features, T)
        horizon (int): Number of future steps to predict
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - input: Shape (n_geos, T-horizon) or (n_geos, n_features, T-horizon)
            - target: Shape (n_geos, horizon) or (n_geos, n_features, horizon)
            
    Raises:
        ValueError: If horizon is larger than time dimension
    """
    if x.ndim < 2:
        raise ValueError(f"Input must have at least 2 dimensions, got {x.ndim}")
    
    time_dim = x.shape[-1]  # Last dimension is always time
    
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    
    if horizon >= time_dim:
        raise ValueError(f"horizon ({horizon}) must be less than time dimension ({time_dim})")
    
    split_point = time_dim - horizon
    
    # Split along time dimension (last axis)
    input_seq = x[..., :split_point]
    target_seq = x[..., split_point:]
    
    return input_seq, target_seq


def temporal_split(x: np.ndarray, input_len: int, target_len: int, 
                  overlap: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split time series into input and target sequences of specified lengths.
    
    More flexible alternative to forecast_horizon_split that allows custom
    input and target sequence lengths.
    
    Args:
        x (np.ndarray): Time series of shape (n_geos, T) or (n_geos, n_features, T)
        input_len (int): Length of input sequences
        target_len (int): Length of target sequences  
        overlap (bool): Whether input and target can overlap (default: False)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - input: Shape (n_geos, input_len) or (n_geos, n_features, input_len)
            - target: Shape (n_geos, target_len) or (n_geos, n_features, target_len)
            
    Raises:
        ValueError: If sequences don't fit in time dimension
    """
    if x.ndim < 2:
        raise ValueError(f"Input must have at least 2 dimensions, got {x.ndim}")
    
    time_dim = x.shape[-1]
    
    if input_len <= 0 or target_len <= 0:
        raise ValueError("input_len and target_len must be positive")
    
    if overlap:
        min_required = max(input_len, target_len)
    else:
        min_required = input_len + target_len
    
    if min_required > time_dim:
        raise ValueError(f"Required length ({min_required}) exceeds time dimension ({time_dim})")
    
    if overlap:
        # Input and target can overlap, place target at the end
        input_seq = x[..., :input_len]
        target_seq = x[..., -target_len:]
    else:
        # Non-overlapping: input first, then target
        input_seq = x[..., :input_len] 
        target_seq = x[..., input_len:input_len + target_len]
    
    return input_seq, target_seq


class AutoregressiveDataset(Dataset):
    """
    Dataset for autoregressive learning with sliding windows.
    
    Creates input-target pairs by sliding a window over time series data.
    Supports flexible stride, multiple targets per input, and stacking options.
    
    Args:
        data (np.ndarray): Time series data of shape (n_geos, n_features, T)
        input_len (int): Length of input sequences
        target_len (int): Length of target sequences (default: 1)
        stride (int): Stride between consecutive windows (default: 1)
        stack_targets (bool): Whether to stack targets into single array (default: True)
        
    Example:
        >>> data = np.random.randn(100, 3, 50)  # 100 geos, 3 features, 50 timesteps
        >>> dataset = AutoregressiveDataset(data, input_len=10, target_len=5)
        >>> len(dataset)  # Number of valid windows
        >>> x, y = dataset[0]  # First input-target pair
    """
    
    def __init__(self, data: np.ndarray, input_len: int, target_len: int = 1,
                 stride: int = 1, stack_targets: bool = True):
        
        if data.ndim not in [2, 3]:
            raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")
        
        # Ensure 3D: (n_geos, n_features, T)
        if data.ndim == 2:
            data = data[:, None, :]  # Add feature dimension
        
        self.data = data
        self.n_geos, self.n_features, self.time_len = data.shape
        self.input_len = input_len
        self.target_len = target_len
        self.stride = stride
        self.stack_targets = stack_targets
        
        if input_len <= 0 or target_len <= 0:
            raise ValueError("input_len and target_len must be positive")
        
        if stride <= 0:
            raise ValueError("stride must be positive")
        
        # Calculate number of valid windows per geo
        min_required = input_len + target_len
        if self.time_len < min_required:
            raise ValueError(f"Time length ({self.time_len}) too short for "
                           f"input_len={input_len} + target_len={target_len}")
        
        # Number of valid starting positions per geo
        max_start = self.time_len - min_required
        available_positions = max_start + 1  # max_start is the last valid position, so positions are 0 to max_start
        
        if available_positions <= 0:
            self.windows_per_geo = 0
        elif stride >= available_positions and available_positions > 1:
            # If stride is larger than available positions, can't meaningfully stride
            self.windows_per_geo = 0
        else:
            self.windows_per_geo = (available_positions - 1) // stride + 1
        
        self.total_windows = self.n_geos * self.windows_per_geo
        
        if self.total_windows == 0:
            raise ValueError("No valid windows can be created with given parameters")
    
    def __len__(self) -> int:
        """Return total number of input-target pairs."""
        return self.total_windows
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get input-target pair at given index.
        
        Args:
            idx (int): Index of the window
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - input: Shape (n_features, input_len) 
                - target: Shape (n_features, target_len) if stack_targets=True,
                         or list of tensors of shape (n_features,) if stack_targets=False
        """
        if idx >= self.total_windows:
            raise IndexError(f"Index {idx} out of range [0, {self.total_windows})")
        
        # Determine which geo and which window within that geo
        geo_idx = idx // self.windows_per_geo
        window_idx = idx % self.windows_per_geo
        
        # Calculate start position for this window
        start_pos = window_idx * self.stride
        
        # Extract input and target sequences
        input_start = start_pos
        input_end = start_pos + self.input_len
        target_start = input_end
        target_end = target_start + self.target_len
        
        # Get data for this geo
        geo_data = self.data[geo_idx]  # Shape: (n_features, T)
        
        input_seq = geo_data[:, input_start:input_end]  # (n_features, input_len)
        target_seq = geo_data[:, target_start:target_end]  # (n_features, target_len)
        
        # Convert to tensors
        input_tensor = torch.tensor(input_seq, dtype=torch.float32)
        
        if self.stack_targets:
            target_tensor = torch.tensor(target_seq, dtype=torch.float32)
        else:
            # Return individual timesteps as list
            target_tensor = [torch.tensor(target_seq[:, i], dtype=torch.float32) 
                           for i in range(self.target_len)]
        
        return input_tensor, target_tensor
    
    def get_window_info(self, idx: int) -> dict:
        """
        Get metadata about a specific window.
        
        Args:
            idx (int): Index of the window
            
        Returns:
            dict: Window metadata including geo_idx, window_idx, start_pos, etc.
        """
        if idx >= self.total_windows:
            raise IndexError(f"Index {idx} out of range [0, {self.total_windows})")
        
        geo_idx = idx // self.windows_per_geo
        window_idx = idx % self.windows_per_geo
        start_pos = window_idx * self.stride
        
        return {
            'geo_idx': geo_idx,
            'window_idx': window_idx,
            'start_pos': start_pos,
            'input_range': (start_pos, start_pos + self.input_len),
            'target_range': (start_pos + self.input_len, 
                           start_pos + self.input_len + self.target_len)
        }


def create_sliding_windows(x: np.ndarray, window_len: int, 
                          stride: int = 1) -> np.ndarray:
    """
    Create sliding windows over the time dimension.
    
    Utility function for creating sliding windows that can be used
    for various time series analysis tasks.
    
    Args:
        x (np.ndarray): Input time series of shape (..., T)
        window_len (int): Length of each window
        stride (int): Stride between windows (default: 1)
        
    Returns:
        np.ndarray: Windowed data of shape (..., n_windows, window_len)
        
    Raises:
        ValueError: If window_len is larger than time dimension
    """
    if x.ndim < 1:
        raise ValueError("Input must have at least 1 dimension")
    
    time_dim = x.shape[-1]
    
    if window_len <= 0:
        raise ValueError("window_len must be positive")
    
    if window_len > time_dim:
        raise ValueError(f"window_len ({window_len}) cannot exceed time dimension ({time_dim})")
    
    if stride <= 0:
        raise ValueError("stride must be positive")
    
    # Calculate number of windows
    available_positions = time_dim - window_len + 1  # Number of valid starting positions
    if available_positions <= 0:
        n_windows = 0
    elif stride >= available_positions and available_positions > 1:
        # If stride is larger than or equal to available positions (except for single window case),
        # it doesn't make semantic sense - you can't meaningfully stride
        n_windows = 0
    else:
        n_windows = (available_positions - 1) // stride + 1
    
    if n_windows <= 0:
        raise ValueError("Cannot create any windows with given parameters")
    
    # Create output shape
    batch_shape = x.shape[:-1]
    output_shape = batch_shape + (n_windows, window_len)
    
    # Initialize output array
    windows = np.zeros(output_shape, dtype=x.dtype)
    
    # Fill windows
    for i in range(n_windows):
        start = i * stride
        end = start + window_len
        windows[..., i, :] = x[..., start:end]
    
    return windows