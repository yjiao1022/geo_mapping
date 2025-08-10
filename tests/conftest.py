"""
pytest configuration with performance testing markers.
"""

import pytest
import numpy as np


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "memory_intensive: marks tests that use significant memory"
    )


@pytest.fixture
def small_synthetic_data():
    """Small synthetic dataset for quick unit tests."""
    np.random.seed(42)
    n_points = 100
    
    import pandas as pd
    df = pd.DataFrame({
        'zip': [f'Z{i:03d}' for i in range(n_points)],
        'e0': np.random.randn(n_points) * 0.1,
        'e1': np.random.randn(n_points) * 0.1,
        'attr1': np.random.randint(1, 100, n_points)
    })
    return df


@pytest.fixture
def large_synthetic_data():
    """Large synthetic dataset for performance tests."""
    np.random.seed(42)
    n_points = 10000
    
    import pandas as pd
    df = pd.DataFrame({
        'zip': [f'Z{i:05d}' for i in range(n_points)],
        'e0': np.random.randn(n_points) * 0.1,
        'e1': np.random.randn(n_points) * 0.1,
        'attr1': np.random.randint(1, 100, n_points)
    })
    return df