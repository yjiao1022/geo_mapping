import pytest
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os

from embedding.temporal_graph import GMAEmbeddingLightningModule
from embedding.utils import (
    extract_temporal_features,
    create_graph_adjacency_matrix,
    extract_model_embeddings
)


class TestOverfitMicroDataset:
    """Integration test for overfitting a small toy dataset."""
    
    def test_overfit_micro_dataset(self):
        """Test that model can overfit a 4-point toy dataset with near-zero loss."""
        # Create 4-point toy dataset
        n_geos = 4
        n_features = 2
        time_steps = 10
        embedding_dim = 8
        
        # Create synthetic time series data that's learnable
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create simple patterns that are easy to learn
        feature_array = np.zeros((n_geos, n_features, time_steps))
        
        # Geo 1: increasing trend
        feature_array[0, 0, :] = np.linspace(0, 1, time_steps)
        feature_array[0, 1, :] = np.linspace(0, 0.5, time_steps)
        
        # Geo 2: decreasing trend  
        feature_array[1, 0, :] = np.linspace(1, 0, time_steps)
        feature_array[1, 1, :] = np.linspace(0.5, 0, time_steps)
        
        # Geo 3: constant values
        feature_array[2, 0, :] = 0.5
        feature_array[2, 1, :] = 0.3
        
        # Geo 4: oscillating
        feature_array[3, 0, :] = 0.5 + 0.3 * np.sin(np.linspace(0, 2*np.pi, time_steps))
        feature_array[3, 1, :] = 0.3 + 0.2 * np.cos(np.linspace(0, 2*np.pi, time_steps))
        
        # Create simple adjacency matrix (fully connected small graph)
        adj_matrix = torch.ones((n_geos, n_geos)) - torch.eye(n_geos)
        
        # Create model with small architecture for overfitting
        model = GMAEmbeddingLightningModule(
            in_channels=n_features,
            hidden_channels=16,
            kernel_size=3,
            embedding_dim=embedding_dim,
            adj_matrix=adj_matrix,
            lr=0.01,  # Higher learning rate for fast overfitting
            loss_type='reconstruction'
        )
        
        # Create dataset
        feature_tensor = torch.tensor(feature_array, dtype=torch.float32)
        dummy_targets = torch.zeros(n_geos)
        dataset = TensorDataset(feature_tensor, dummy_targets)
        dataloader = DataLoader(dataset, batch_size=n_geos, shuffle=False)
        
        # Train with many epochs to ensure overfitting
        trainer = pl.Trainer(
            max_epochs=100,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator='cpu'  # Force CPU to avoid device mismatch in tests
        )
        
        # Train
        trainer.fit(model, dataloader)
        
        # Test final loss
        model.eval()
        with torch.no_grad():
            embeddings = model(feature_tensor)
            final_loss = model._compute_reconstruction_loss(feature_tensor)
        
        # Should achieve very low loss (near zero) on this simple dataset
        assert final_loss < 0.01, f"Model failed to overfit toy dataset, loss: {final_loss}"
        
        # Embeddings should be finite and reasonable
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
        assert embeddings.shape == (n_geos, embedding_dim)


class TestEndToEndPipeline:
    """End-to-end integration test of the complete embedding pipeline."""
    
    def test_complete_embedding_pipeline(self):
        """Test the complete workflow from raw data to embeddings."""
        
        # Step 1: Create synthetic input data (mimicking real spend/mobility data)
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        geo_ids = [f'geo_{i:03d}' for i in range(10)]
        
        # Create realistic time series data
        data = []
        for date in dates:
            for i, geo_id in enumerate(geo_ids):
                # Add some spatial and temporal correlation
                base_spend = 1000 + i * 100  # Spatial variation
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * len(data) / (len(dates) * len(geo_ids)))
                noise = np.random.normal(0, 50)
                
                spend = base_spend * seasonal_factor + noise
                impressions = spend * np.random.uniform(8, 12)  # Correlated with spend
                
                data.append({
                    'geo_id': geo_id,
                    'date': date,
                    'spend': max(0, spend),  # Ensure non-negative
                    'impressions': max(0, impressions)
                })
        
        df = pd.DataFrame(data)
        
        # Create geographic coordinates
        geo_df = pd.DataFrame({
            'geo_id': geo_ids,
            'latitude': np.random.uniform(30, 50, len(geo_ids)),
            'longitude': np.random.uniform(-120, -70, len(geo_ids))
        })
        
        # Step 2: Extract temporal features
        feature_array = extract_temporal_features(df, window=30)
        
        assert feature_array.shape == (len(geo_ids), 2, len(dates))
        assert not np.isnan(feature_array).any()
        
        # Step 3: Build adjacency matrix
        adj_matrix = create_graph_adjacency_matrix(geo_df, method="knn", k=3)
        
        assert adj_matrix.shape == (len(geo_ids), len(geo_ids))
        assert adj_matrix.dtype == np.float32
        
        # Step 4: Train model (short training for integration test)
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
        
        model = GMAEmbeddingLightningModule(
            in_channels=2,
            hidden_channels=16,
            kernel_size=5,
            embedding_dim=12,
            adj_matrix=adj_tensor,
            lr=0.001,
            loss_type='reconstruction'
        )
        
        # Quick training
        feature_tensor = torch.tensor(feature_array, dtype=torch.float32)
        dummy_targets = torch.zeros(len(geo_ids))
        dataset = TensorDataset(feature_tensor, dummy_targets)
        dataloader = DataLoader(dataset, batch_size=len(geo_ids), shuffle=False)
        
        trainer = pl.Trainer(
            max_epochs=5,  # Short training for integration test
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator='cpu'  # Force CPU to avoid device mismatch in tests
        )
        
        trainer.fit(model, dataloader)
        
        # Step 5: Extract embeddings using trained model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            trainer.save_checkpoint(tmp_file.name)
            checkpoint_path = tmp_file.name
        
        try:
            # Test embedding extraction
            embeddings = extract_model_embeddings(
                checkpoint_path, feature_array, adj_matrix, batch_size=5
            )
            
            # Verify embeddings
            assert embeddings.shape == (len(geo_ids), 12)
            assert not np.isnan(embeddings).any()
            assert not np.isinf(embeddings).any()
            
            # Embeddings should have some variation (not all zeros)
            assert np.std(embeddings) > 0.001
            
        finally:
            # Clean up temporary file
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)


class TestDataConsistency:
    """Test data consistency throughout the pipeline."""
    
    def test_feature_extraction_consistency(self):
        """Test that feature extraction maintains geo ordering consistency."""
        # Create data with specific geo ordering
        geo_ids = ['geo_003', 'geo_001', 'geo_002']  # Non-alphabetical order
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        
        data = []
        for date in dates:
            for geo_id in geo_ids:
                data.append({
                    'geo_id': geo_id,
                    'date': date,
                    'spend': float(geo_id.split('_')[1]) * 100  # Unique per geo
                })
        
        df = pd.DataFrame(data)
        features = extract_temporal_features(df)
        
        # Features should be ordered by geo_id (sorted)
        expected_order = ['geo_001', 'geo_002', 'geo_003']
        
        # Check that geo ordering is consistent (sorted)
        # geo_001 should have spend values of 100, geo_002 should have 200, etc.
        assert np.allclose(features[0, 0, :], 100)  # geo_001
        assert np.allclose(features[1, 0, :], 200)  # geo_002  
        assert np.allclose(features[2, 0, :], 300)  # geo_003
    
    def test_adjacency_matrix_geo_consistency(self):
        """Test that adjacency matrix maintains consistent geo ordering."""
        geo_data = {
            'geo_id': ['geo_003', 'geo_001', 'geo_002'],
            'latitude': [40.0, 41.0, 42.0],
            'longitude': [-74.0, -73.0, -72.0]
        }
        
        df = pd.DataFrame(geo_data)
        adj_matrix = create_graph_adjacency_matrix(df, method="knn", k=2)
        
        # Matrix should be square with correct dimensions
        assert adj_matrix.shape == (3, 3)
        
        # Should be symmetric
        assert np.allclose(adj_matrix, adj_matrix.T)