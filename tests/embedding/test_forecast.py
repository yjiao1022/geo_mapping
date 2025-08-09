"""
Tests for forecasting utilities in embedding.forecast module.
"""

import torch
import pytest
import numpy as np
from embedding.forecast import (
    forecast_horizon_split,
    temporal_split,
    AutoregressiveDataset,
    create_sliding_windows
)


class TestForecastHorizonSplit:
    """Test forecast horizon splitting function."""
    
    def test_forecast_split_2d_input(self):
        """Test forecast splitting with 2D input (n_geos, T)."""
        x = np.random.randn(10, 50)
        horizon = 7
        
        input_seq, target_seq = forecast_horizon_split(x, horizon)
        
        assert input_seq.shape == (10, 43)  # 50 - 7
        assert target_seq.shape == (10, 7)
    
    def test_forecast_split_3d_input(self):
        """Test forecast splitting with 3D input (n_geos, n_features, T)."""
        x = np.random.randn(15, 4, 60)
        horizon = 10
        
        input_seq, target_seq = forecast_horizon_split(x, horizon)
        
        assert input_seq.shape == (15, 4, 50)  # 60 - 10
        assert target_seq.shape == (15, 4, 10)
    
    def test_forecast_split_horizon_too_large_error(self):
        """Test error when horizon is too large."""
        x = np.random.randn(5, 20)
        
        with pytest.raises(ValueError, match="horizon .* must be less than time dimension"):
            forecast_horizon_split(x, horizon=25)
    
    def test_forecast_split_negative_horizon_error(self):
        """Test error when horizon is negative."""
        x = np.random.randn(5, 20)
        
        with pytest.raises(ValueError, match="horizon must be positive"):
            forecast_horizon_split(x, horizon=-1)
    
    def test_forecast_split_1d_input_error(self):
        """Test error when input has insufficient dimensions."""
        x = np.random.randn(20)
        
        with pytest.raises(ValueError, match="Input must have at least 2 dimensions"):
            forecast_horizon_split(x, horizon=5)
    
    def test_forecast_split_preserves_data(self):
        """Test that split preserves original data."""
        x = np.random.randn(8, 30)
        horizon = 5
        
        input_seq, target_seq = forecast_horizon_split(x, horizon)
        
        # Reconstructed sequence should match original
        reconstructed = np.concatenate([input_seq, target_seq], axis=-1)
        assert np.allclose(reconstructed, x)


class TestTemporalSplit:
    """Test temporal splitting function."""
    
    def test_temporal_split_non_overlapping(self):
        """Test temporal split without overlap."""
        x = np.random.randn(12, 40)
        input_len, target_len = 15, 10
        
        input_seq, target_seq = temporal_split(x, input_len, target_len, overlap=False)
        
        assert input_seq.shape == (12, 15)
        assert target_seq.shape == (12, 10)
        
        # Should be consecutive sequences
        expected_target = x[:, input_len:input_len + target_len]
        assert np.allclose(target_seq, expected_target)
    
    def test_temporal_split_overlapping(self):
        """Test temporal split with overlap."""
        x = np.random.randn(8, 30)
        input_len, target_len = 20, 15
        
        input_seq, target_seq = temporal_split(x, input_len, target_len, overlap=True)
        
        assert input_seq.shape == (8, 20)
        assert target_seq.shape == (8, 15)
        
        # Input should be first 20 timesteps
        assert np.allclose(input_seq, x[:, :input_len])
        # Target should be last 15 timesteps
        assert np.allclose(target_seq, x[:, -target_len:])
    
    def test_temporal_split_insufficient_length_error(self):
        """Test error when sequences don't fit."""
        x = np.random.randn(5, 20)
        
        # Non-overlapping case: need 15 + 10 = 25 > 20
        with pytest.raises(ValueError, match="Required length .* exceeds time dimension"):
            temporal_split(x, 15, 10, overlap=False)
        
        # Overlapping case: need max(25, 15) = 25 > 20  
        with pytest.raises(ValueError, match="Required length .* exceeds time dimension"):
            temporal_split(x, 25, 15, overlap=True)
    
    def test_temporal_split_invalid_lengths_error(self):
        """Test error for invalid sequence lengths."""
        x = np.random.randn(5, 30)
        
        with pytest.raises(ValueError, match="input_len and target_len must be positive"):
            temporal_split(x, 0, 10)
        
        with pytest.raises(ValueError, match="input_len and target_len must be positive"):
            temporal_split(x, 10, -5)


class TestAutoregressiveDataset:
    """Test autoregressive dataset class."""
    
    def test_dataset_initialization_2d(self):
        """Test dataset initialization with 2D data."""
        data = np.random.randn(20, 50)
        dataset = AutoregressiveDataset(data, input_len=10, target_len=5)
        
        assert dataset.n_geos == 20
        assert dataset.n_features == 1  # Added feature dimension
        assert dataset.time_len == 50
        assert len(dataset) > 0
    
    def test_dataset_initialization_3d(self):
        """Test dataset initialization with 3D data."""
        data = np.random.randn(15, 3, 40)
        dataset = AutoregressiveDataset(data, input_len=8, target_len=3)
        
        assert dataset.n_geos == 15
        assert dataset.n_features == 3
        assert dataset.time_len == 40
    
    def test_dataset_length_calculation(self):
        """Test dataset length calculation with different strides."""
        data = np.random.randn(10, 30)
        
        # Stride 1: every position
        dataset1 = AutoregressiveDataset(data, input_len=5, target_len=2, stride=1)
        expected_windows = (30 - 5 - 2) // 1 + 1  # 24 windows per geo
        assert len(dataset1) == 10 * expected_windows
        
        # Stride 2: every other position
        dataset2 = AutoregressiveDataset(data, input_len=5, target_len=2, stride=2)
        expected_windows = (30 - 5 - 2) // 2 + 1  # 12 windows per geo
        assert len(dataset2) == 10 * expected_windows
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        data = np.random.randn(5, 2, 25)
        dataset = AutoregressiveDataset(data, input_len=6, target_len=3)
        
        input_tensor, target_tensor = dataset[0]
        
        assert isinstance(input_tensor, torch.Tensor)
        assert isinstance(target_tensor, torch.Tensor)
        assert input_tensor.shape == (2, 6)  # (n_features, input_len)
        assert target_tensor.shape == (2, 3)  # (n_features, target_len)
        assert input_tensor.dtype == torch.float32
        assert target_tensor.dtype == torch.float32
    
    def test_dataset_getitem_stacked_targets(self):
        """Test dataset with stacked targets."""
        data = np.random.randn(3, 20)
        dataset = AutoregressiveDataset(data, input_len=5, target_len=4, stack_targets=True)
        
        input_tensor, target_tensor = dataset[0]
        assert target_tensor.shape == (1, 4)  # Stacked as single tensor
    
    def test_dataset_getitem_unstacked_targets(self):
        """Test dataset with unstacked targets."""
        data = np.random.randn(3, 20)
        dataset = AutoregressiveDataset(data, input_len=5, target_len=3, stack_targets=False)
        
        input_tensor, target_list = dataset[0]
        assert isinstance(target_list, list)
        assert len(target_list) == 3
        assert all(t.shape == (1,) for t in target_list)  # Each timestep separately
    
    def test_dataset_index_out_of_bounds_error(self):
        """Test error for out of bounds index."""
        data = np.random.randn(5, 20)
        dataset = AutoregressiveDataset(data, input_len=5, target_len=2)
        
        with pytest.raises(IndexError, match="Index .* out of range"):
            dataset[len(dataset)]
    
    def test_dataset_window_info(self):
        """Test window metadata retrieval."""
        data = np.random.randn(4, 25)
        dataset = AutoregressiveDataset(data, input_len=6, target_len=3, stride=2)
        
        info = dataset.get_window_info(5)  # Some window index
        
        assert 'geo_idx' in info
        assert 'window_idx' in info
        assert 'start_pos' in info
        assert 'input_range' in info
        assert 'target_range' in info
        
        # Validate ranges make sense
        input_range = info['input_range']
        target_range = info['target_range']
        assert input_range[1] - input_range[0] == 6  # input_len
        assert target_range[1] - target_range[0] == 3  # target_len
        assert input_range[1] == target_range[0]  # Sequential
    
    def test_dataset_insufficient_time_error(self):
        """Test error when time dimension is too short."""
        data = np.random.randn(5, 10)
        
        with pytest.raises(ValueError, match="Time length .* too short"):
            AutoregressiveDataset(data, input_len=8, target_len=5)  # Need 13 but have 10
    
    def test_dataset_invalid_parameters_error(self):
        """Test errors for invalid parameters."""
        data = np.random.randn(5, 20)
        
        # Invalid input_len
        with pytest.raises(ValueError, match="input_len and target_len must be positive"):
            AutoregressiveDataset(data, input_len=0, target_len=5)
        
        # Invalid stride
        with pytest.raises(ValueError, match="stride must be positive"):
            AutoregressiveDataset(data, input_len=5, target_len=2, stride=0)
    
    def test_dataset_no_valid_windows_error(self):
        """Test error when no valid windows can be created."""
        data = np.random.randn(3, 15)
        
        # Large stride makes no windows possible
        with pytest.raises(ValueError, match="No valid windows can be created"):
            AutoregressiveDataset(data, input_len=10, target_len=3, stride=20)
    
    def test_dataset_1d_input_error(self):
        """Test error for 1D input."""
        data = np.random.randn(20)
        
        with pytest.raises(ValueError, match="Data must be 2D or 3D"):
            AutoregressiveDataset(data, input_len=5, target_len=2)


class TestCreateSlidingWindows:
    """Test sliding window creation function."""
    
    def test_sliding_windows_basic(self):
        """Test basic sliding window creation."""
        x = np.random.randn(10, 25)
        windows = create_sliding_windows(x, window_len=5, stride=2)
        
        expected_n_windows = (25 - 5) // 2 + 1  # 11 windows
        assert windows.shape == (10, expected_n_windows, 5)
    
    def test_sliding_windows_stride_one(self):
        """Test sliding windows with stride 1."""
        x = np.random.randn(8, 20)
        windows = create_sliding_windows(x, window_len=6, stride=1)
        
        expected_n_windows = 20 - 6 + 1  # 15 windows
        assert windows.shape == (8, expected_n_windows, 6)
        
        # First and second windows should overlap by window_len - stride
        assert np.allclose(windows[0, 0, 1:], windows[0, 1, :-1])
    
    def test_sliding_windows_preserves_data(self):
        """Test that sliding windows preserve original data."""
        x = np.arange(20).reshape(1, 20)
        windows = create_sliding_windows(x, window_len=4, stride=1)
        
        # First window should be [0, 1, 2, 3]
        assert np.array_equal(windows[0, 0, :], [0, 1, 2, 3])
        # Last window should be [16, 17, 18, 19]
        assert np.array_equal(windows[0, -1, :], [16, 17, 18, 19])
    
    def test_sliding_windows_window_too_large_error(self):
        """Test error when window is too large."""
        x = np.random.randn(5, 10)
        
        with pytest.raises(ValueError, match="window_len .* cannot exceed time dimension"):
            create_sliding_windows(x, window_len=15)
    
    def test_sliding_windows_invalid_parameters_error(self):
        """Test errors for invalid parameters."""
        x = np.random.randn(5, 20)
        
        # Invalid window_len
        with pytest.raises(ValueError, match="window_len must be positive"):
            create_sliding_windows(x, window_len=0)
        
        # Invalid stride
        with pytest.raises(ValueError, match="stride must be positive"):
            create_sliding_windows(x, window_len=5, stride=-1)
    
    def test_sliding_windows_no_windows_error(self):
        """Test error when no windows can be created."""
        x = np.random.randn(3, 10)
        
        # Large stride makes no windows possible
        with pytest.raises(ValueError, match="Cannot create any windows"):
            create_sliding_windows(x, window_len=5, stride=20)
    
    def test_sliding_windows_1d_input(self):
        """Test sliding windows with 1D input."""
        x = np.arange(15)
        windows = create_sliding_windows(x, window_len=3, stride=2)
        
        expected_n_windows = (15 - 3) // 2 + 1  # 7 windows
        assert windows.shape == (expected_n_windows, 3)
    
    def test_sliding_windows_3d_input(self):
        """Test sliding windows with 3D input."""
        x = np.random.randn(4, 6, 30)
        windows = create_sliding_windows(x, window_len=8, stride=3)
        
        expected_n_windows = (30 - 8) // 3 + 1  # 8 windows
        assert windows.shape == (4, 6, expected_n_windows, 8)


class TestForecastIntegration:
    """Integration tests for forecasting utilities."""
    
    def test_forecast_pipeline(self):
        """Test complete forecast preprocessing pipeline."""
        # Generate synthetic time series data
        data = np.random.randn(20, 4, 100)  # 20 geos, 4 features, 100 timesteps
        
        # Step 1: Create train/test split using horizon split
        train_data, test_targets = forecast_horizon_split(data, horizon=10)
        assert train_data.shape == (20, 4, 90)
        assert test_targets.shape == (20, 4, 10)
        
        # Step 2: Create autoregressive dataset for training
        ar_dataset = AutoregressiveDataset(train_data, input_len=15, target_len=5)
        assert len(ar_dataset) > 0
        
        # Step 3: Sample from dataset
        input_batch, target_batch = ar_dataset[0]
        assert input_batch.shape == (4, 15)
        assert target_batch.shape == (4, 5)
        
        # Step 4: Create sliding windows for validation
        val_windows = create_sliding_windows(train_data, window_len=20, stride=10)
        assert val_windows.shape[0] == 20  # n_geos preserved
    
    def test_forecast_data_types_preserved(self):
        """Test that forecasting utilities preserve data types."""
        # Test with different dtypes
        for dtype in [np.float32, np.float64]:
            data = np.random.randn(5, 30).astype(dtype)
            
            # Test all functions preserve dtype
            input_seq, target_seq = forecast_horizon_split(data, horizon=5)
            assert input_seq.dtype == dtype
            assert target_seq.dtype == dtype
            
            input_seq2, target_seq2 = temporal_split(data, 10, 8, overlap=False)
            assert input_seq2.dtype == dtype
            assert target_seq2.dtype == dtype
            
            windows = create_sliding_windows(data, window_len=6, stride=2)
            assert windows.dtype == dtype
    
    def test_forecast_edge_cases(self):
        """Test forecasting utilities with edge cases."""
        # Minimal viable input
        min_data = np.random.randn(1, 5)  # 1 geo, 5 timesteps
        
        # Should work with minimal horizons
        input_seq, target_seq = forecast_horizon_split(min_data, horizon=1)
        assert input_seq.shape == (1, 4)
        assert target_seq.shape == (1, 1)
        
        # Minimal AR dataset
        ar_dataset = AutoregressiveDataset(min_data, input_len=2, target_len=1)
        assert len(ar_dataset) >= 1
        
        # Single window
        windows = create_sliding_windows(min_data, window_len=3, stride=1)
        assert windows.shape == (1, 3, 3)  # 1 geo, 3 windows, window_len 3


if __name__ == "__main__":
    pytest.main([__file__])