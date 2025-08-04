import pytest
import torch
import numpy as np
from embedding.temporal_graph import TemporalConvNet, GraphSAGEEncoder, GMAEmbeddingLightningModule


class TestTemporalConvNet:
    """Unit tests for TemporalConvNet module."""
    
    def test_temporal_convnet_shape(self):
        """Assert random input (batch, in_channels, seq_len) returns (batch, hidden_channels) without errors."""
        # Setup
        batch_size = 32
        in_channels = 3
        hidden_channels = 64
        kernel_size = 7
        seq_len = 100
        
        # Create model
        model = TemporalConvNet(in_channels, hidden_channels, kernel_size)
        
        # Create random input
        x = torch.randn(batch_size, in_channels, seq_len)
        
        # Forward pass
        output = model(x)
        
        # Assertions
        assert output.shape == (batch_size, hidden_channels)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_temporal_convnet_different_inputs(self):
        """Test with different input dimensions."""
        model = TemporalConvNet(in_channels=2, hidden_channels=32, kernel_size=5)
        
        # Test with different batch sizes and sequence lengths
        test_cases = [
            (1, 2, 50),    # Small batch, short sequence
            (16, 2, 200),  # Medium batch, long sequence
            (64, 2, 30),   # Large batch, short sequence
        ]
        
        for batch_size, in_channels, seq_len in test_cases:
            x = torch.randn(batch_size, in_channels, seq_len)
            output = model(x)
            assert output.shape == (batch_size, 32)


class TestGraphSAGEEncoder:
    """Unit tests for GraphSAGEEncoder module."""
    
    def test_graphsageencoder_shape(self):
        """Verify output (n_nodes, out_channels) given random x & adj."""
        # Setup
        n_nodes = 100
        in_channels = 64
        out_channels = 32
        
        # Create random adjacency matrix (binary)
        adj_matrix = torch.randint(0, 2, (n_nodes, n_nodes)).float()
        # Make symmetric
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        adj_matrix = (adj_matrix > 0.5).float()
        
        # Create model
        model = GraphSAGEEncoder(in_channels, out_channels, adj_matrix)
        
        # Create random input
        x = torch.randn(n_nodes, in_channels)
        
        # Forward pass
        output = model(x)
        
        # Assertions
        assert output.shape == (n_nodes, out_channels)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        # Output should be non-negative due to ReLU
        assert (output >= 0).all()
    
    def test_graphsageencoder_isolated_nodes(self):
        """Test behavior with isolated nodes (no neighbors)."""
        n_nodes = 10
        in_channels = 16
        out_channels = 8
        
        # Create adjacency matrix with isolated nodes
        adj_matrix = torch.zeros(n_nodes, n_nodes)
        # Connect only first 5 nodes in a chain
        for i in range(4):
            adj_matrix[i, i+1] = 1
            adj_matrix[i+1, i] = 1
        
        model = GraphSAGEEncoder(in_channels, out_channels, adj_matrix)
        x = torch.randn(n_nodes, in_channels)
        
        output = model(x)
        assert output.shape == (n_nodes, out_channels)
        # Should still produce valid output for isolated nodes


class TestGMAEmbeddingLightningModule:
    """Unit tests for the complete Lightning module."""
    
    def test_lightning_module_forward(self):
        """Test forward pass through complete module."""
        # Setup
        batch_size = 16
        in_channels = 2
        hidden_channels = 32
        kernel_size = 5
        embedding_dim = 16
        seq_len = 50
        n_nodes = 100
        
        # Create adjacency matrix
        adj_matrix = torch.randint(0, 2, (n_nodes, n_nodes)).float()
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        adj_matrix = (adj_matrix > 0.5).float()
        
        # Create model
        model = GMAEmbeddingLightningModule(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            embedding_dim=embedding_dim,
            adj_matrix=adj_matrix,
            lr=0.001
        )
        
        # Create input (note: batch_size should match n_nodes for graph convolution)
        x = torch.randn(n_nodes, in_channels, seq_len)
        
        # Forward pass
        output = model(x)
        
        # Assertions
        assert output.shape == (n_nodes, embedding_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_lightning_module_training_step(self):
        """Test training step execution."""
        # Setup smaller dimensions for faster test
        n_nodes = 20
        in_channels = 2
        hidden_channels = 16
        embedding_dim = 8
        seq_len = 30
        
        adj_matrix = torch.eye(n_nodes)  # Simple identity matrix
        
        model = GMAEmbeddingLightningModule(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=3,
            embedding_dim=embedding_dim,
            adj_matrix=adj_matrix,
            lr=0.001
        )
        
        # Create batch
        x = torch.randn(n_nodes, in_channels, seq_len)
        dummy_targets = torch.zeros(n_nodes)
        batch = (x, dummy_targets)
        
        # Training step
        loss = model.training_step(batch, 0)
        
        # Assertions
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss >= 0  # MSE loss should be non-negative
    
    def test_lightning_module_invalid_loss_type(self):
        """Test that invalid loss type raises error."""
        adj_matrix = torch.eye(10)
        
        with pytest.raises(ValueError, match="Unsupported loss_type"):
            GMAEmbeddingLightningModule(
                in_channels=2,
                hidden_channels=16,
                kernel_size=3,
                embedding_dim=8,
                adj_matrix=adj_matrix,
                lr=0.001,
                loss_type="invalid_loss"
            )