import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock
from embedding.utils import (
    extract_temporal_features,
    create_graph_adjacency_matrix,
    extract_model_embeddings
)


class TestExtractTemporalFeatures:
    """Unit tests for extract_temporal_features function."""
    
    def test_extract_temporal_features_basic(self):
        """Test basic functionality with spend and impressions data."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        geo_ids = ['geo_1', 'geo_2', 'geo_3']
        
        data = []
        for date in dates:
            for geo_id in geo_ids:
                data.append({
                    'geo_id': geo_id,
                    'date': date,
                    'spend': np.random.rand() * 1000,
                    'impressions': np.random.randint(1000, 10000)
                })
        
        df = pd.DataFrame(data)
        
        # Extract features
        features = extract_temporal_features(df)
        
        # Assertions
        assert features.shape == (3, 2, 10)  # (n_geos, n_features, time_steps)
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()
        assert features.dtype == np.float64
    
    def test_extract_temporal_features_dynamic_columns(self):
        """Test with different number of feature columns."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        geo_ids = ['geo_1', 'geo_2']
        
        # Test with 3 feature columns
        data = []
        for date in dates:
            for geo_id in geo_ids:
                data.append({
                    'geo_id': geo_id,
                    'date': date,
                    'spend': np.random.rand() * 1000,
                    'impressions': np.random.randint(1000, 10000),
                    'clicks': np.random.randint(10, 100)
                })
        
        df = pd.DataFrame(data)
        features = extract_temporal_features(df)
        
        # Should have 3 feature channels
        assert features.shape == (2, 3, 5)  # (n_geos, n_features, time_steps)
    
    def test_extract_temporal_features_missing_values(self):
        """Test handling of missing values (should be filled with 0)."""
        # Create data with missing values
        data = [
            {'geo_id': 'geo_1', 'date': '2023-01-01', 'spend': 100, 'impressions': 1000},
            {'geo_id': 'geo_2', 'date': '2023-01-01', 'spend': 200, 'impressions': 2000},
            # Missing data for geo_1 on 2023-01-02
            {'geo_id': 'geo_2', 'date': '2023-01-02', 'spend': 250, 'impressions': 2500},
        ]
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        
        features = extract_temporal_features(df)
        
        # Check that missing values are filled with 0
        assert features.shape == (2, 2, 2)
        # geo_1 should have zeros for second time step
        assert features[0, :, 1].sum() == 0
    
    def test_extract_temporal_features_no_feature_columns(self):
        """Test error when no feature columns are present."""
        data = [
            {'geo_id': 'geo_1', 'date': '2023-01-01'},
            {'geo_id': 'geo_2', 'date': '2023-01-01'},
        ]
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        
        with pytest.raises(ValueError, match="No feature columns found"):
            extract_temporal_features(df)


class TestCreateGraphAdjacencyMatrix:
    """Unit tests for create_graph_adjacency_matrix function."""
    
    def test_create_adjacency_matrix_knn(self):
        """Test k-nearest neighbors adjacency matrix creation."""
        # Create sample geographic data
        np.random.seed(42)  # For reproducible results
        n_geos = 20
        geo_data = {
            'geo_id': [f'geo_{i}' for i in range(n_geos)],
            'latitude': np.random.uniform(30, 50, n_geos),
            'longitude': np.random.uniform(-120, -70, n_geos)
        }
        
        df = pd.DataFrame(geo_data)
        
        # Create adjacency matrix
        k = 5
        adj_matrix = create_graph_adjacency_matrix(df, method="knn", k=k)
        
        # Assertions
        assert adj_matrix.shape == (n_geos, n_geos)
        assert adj_matrix.dtype == np.float32
        
        # Matrix should be symmetric
        assert np.allclose(adj_matrix, adj_matrix.T)
        
        # Matrix should be binary
        assert np.all((adj_matrix == 0) | (adj_matrix == 1))
        
        # Diagonal should be zeros (no self-connections)
        assert np.all(np.diag(adj_matrix) == 0)
        # Actually, let's check if each node has approximately k neighbors
        # Note: Due to symmetry, nodes might have more than k connections
        row_sums = adj_matrix.sum(axis=1)
        assert np.all(row_sums >= k)  # At least k connections per node
    
    def test_create_adjacency_matrix_invalid_method(self):
        """Test error for invalid adjacency method."""
        geo_data = {
            'geo_id': ['geo_1', 'geo_2'],
            'latitude': [40.0, 41.0],
            'longitude': [-74.0, -73.0]
        }
        
        df = pd.DataFrame(geo_data)
        
        with pytest.raises(ValueError, match="Unsupported method"):
            create_graph_adjacency_matrix(df, method="invalid_method")
    
    def test_create_adjacency_matrix_small_dataset(self):
        """Test with very small dataset."""
        geo_data = {
            'geo_id': ['geo_1', 'geo_2', 'geo_3'],
            'latitude': [40.0, 41.0, 42.0],
            'longitude': [-74.0, -73.0, -72.0]
        }
        
        df = pd.DataFrame(geo_data)
        
        adj_matrix = create_graph_adjacency_matrix(df, method="knn", k=2)
        
        assert adj_matrix.shape == (3, 3)
        assert adj_matrix.dtype == np.float32


class TestExtractModelEmbeddings:
    """Unit tests for extract_model_embeddings function."""
    
    @patch('embedding.temporal_graph.GMAEmbeddingLightningModule')
    @patch('pytorch_lightning.Trainer')
    @patch('torch.utils.data.DataLoader')
    def test_extract_model_embeddings_basic(self, mock_dataloader, mock_trainer, mock_lightning_module):
        """Test basic embedding extraction functionality."""
        # Setup mocks
        mock_model = MagicMock()
        mock_lightning_module.load_from_checkpoint.return_value = mock_model
        
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        
        # Mock predictions
        embedding_dim = 16
        n_geos = 10
        mock_predictions = [torch.randn(n_geos, embedding_dim)]
        mock_trainer_instance.predict.return_value = mock_predictions
        
        # Test inputs
        feature_array = np.random.randn(n_geos, 2, 50)
        adj_matrix = np.random.randint(0, 2, (n_geos, n_geos)).astype(np.float32)
        checkpoint_path = "test_checkpoint.pt"
        
        # Call function
        embeddings = extract_model_embeddings(
            checkpoint_path, feature_array, adj_matrix, batch_size=32
        )
        
        # Assertions
        assert embeddings.shape == (n_geos, embedding_dim)
        mock_lightning_module.load_from_checkpoint.assert_called_once_with(checkpoint_path)
        mock_trainer_instance.predict.assert_called_once()
    
    @patch('embedding.temporal_graph.GMAEmbeddingLightningModule')
    @patch('pytorch_lightning.Trainer')
    def test_extract_model_embeddings_single_batch_processing(self, mock_trainer, mock_lightning_module):
        """Test embedding extraction processes all nodes in single batch for graph operations."""
        # Setup mocks
        mock_model = MagicMock()
        mock_lightning_module.load_from_checkpoint.return_value = mock_model
        
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        
        # Mock single batch prediction (all nodes processed together for graph operations)
        embedding_dim = 16
        n_geos = 20
        single_batch = torch.randn(n_geos, embedding_dim)
        mock_trainer_instance.predict.return_value = [single_batch]
        
        # Test inputs
        feature_array = np.random.randn(n_geos, 2, 50)
        adj_matrix = np.random.randint(0, 2, (n_geos, n_geos)).astype(np.float32)
        
        # Call function
        embeddings = extract_model_embeddings(
            "test_checkpoint.pt", feature_array, adj_matrix, batch_size=10
        )
        
        # Should return all embeddings in single batch
        assert embeddings.shape == (n_geos, embedding_dim)


class TestBoundaryConditions:
    """Boundary tests for invalid inputs."""
    
    def test_empty_dataframe(self):
        """Test behavior with empty input dataframe."""
        df = pd.DataFrame(columns=['geo_id', 'date', 'spend'])
        
        with pytest.raises((ValueError, IndexError)):
            extract_temporal_features(df)
    
    def test_single_geo_single_timepoint(self):
        """Test with minimal valid input."""
        data = [{
            'geo_id': 'geo_1',
            'date': pd.Timestamp('2023-01-01'),
            'spend': 100.0
        }]
        
        df = pd.DataFrame(data)
        features = extract_temporal_features(df)
        
        assert features.shape == (1, 1, 1)
        assert features[0, 0, 0] == 100.0
    
    def test_zero_k_neighbors(self):
        """Test adjacency matrix with k=0 (should raise error or handle gracefully)."""
        geo_data = {
            'geo_id': ['geo_1', 'geo_2'],
            'latitude': [40.0, 41.0],
            'longitude': [-74.0, -73.0]
        }
        
        df = pd.DataFrame(geo_data)
        
        # k=0 should either raise error or return empty adjacency
        with pytest.raises(ValueError):
            create_graph_adjacency_matrix(df, method="knn", k=0)