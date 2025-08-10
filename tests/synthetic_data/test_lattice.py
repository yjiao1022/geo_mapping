"""Unit tests for lattice generator."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
import networkx as nx

from synthetic_data.lattice import lattice_grid, make_embeddings

try:
    from libpysal.weights import lat2W
    HAS_LAT2W = True
except ImportError:
    HAS_LAT2W = False


class TestLatticeGrid:
    """Test lattice grid generation."""
    
    def test_lattice_basic_structure(self):
        """Test basic lattice structure and deterministic IDs."""
        df, w = lattice_grid(width=3, height=2, block_w=1, block_h=1, use_lat2w=False)
        
        # Should have 6 points
        assert len(df) == 6
        assert w.n == 6
        
        # Check column structure
        expected_cols = ['zip', 'row', 'col', 'block_id']
        assert list(df.columns) == expected_cols
        
        # Check deterministic ZIP IDs - keep docs consistent with generated ID format
        expected_zips = ['Z000000', 'Z000001', 'Z000002', 'Z001000', 'Z001001', 'Z001002']
        assert list(df['zip']) == expected_zips
        
        # Check row/col values
        assert df['row'].min() == 0
        assert df['row'].max() == 1
        assert df['col'].min() == 0  
        assert df['col'].max() == 2
    
    def test_lattice_block_ids(self):
        """Test block ID computation."""
        df, w = lattice_grid(width=4, height=4, block_w=2, block_h=2, use_lat2w=False)
        
        # Should create 2x2 = 4 blocks
        expected_blocks = 4
        assert df['block_id'].nunique() == expected_blocks
        
        # Check specific block assignments
        # Top-left 2x2 should be block 0
        top_left = df[(df['row'] < 2) & (df['col'] < 2)]
        assert all(top_left['block_id'] == 0)
    
    def test_lattice_connectivity(self):
        """Test that adjacency creates proper grid connectivity."""
        df, w = lattice_grid(width=3, height=3, block_w=1, block_h=1, use_lat2w=False)
        
        # Convert to NetworkX for easier testing
        G = w.to_networkx()
        
        # Should be connected
        assert nx.is_connected(G), "Grid should be fully connected"
        
        # Check specific connections (center point should have 4 neighbors)
        center_zip = 'Z001001'  # Row 1, Col 1 (center of 3x3)
        center_index = w.id_order.index(center_zip)  # Get integer index for NetworkX
        center_neighbors = list(G.neighbors(center_index))
        assert len(center_neighbors) == 4, "Center point should have 4 neighbors"
    
    @pytest.mark.skipif(not HAS_LAT2W, reason="lat2W not available")
    def test_lat2w_parity(self):
        """lat2W vs manual adjacency produce identical neighbor sets."""
        width, height = 4, 3
        
        df_manual, w_manual = lattice_grid(width, height, 2, 2, use_lat2w=False)
        df_lat2w, w_lat2w = lattice_grid(width, height, 2, 2, use_lat2w=True)
        
        # Should have same structure
        assert w_manual.n == w_lat2w.n
        
        # Every node should have identical neighbor sets
        for zip_id in w_manual.neighbors:
            manual_set = set(w_manual.neighbors[zip_id])
            lat2w_set = set(w_lat2w.neighbors[zip_id])
            assert manual_set == lat2w_set, f"Mismatch for {zip_id}"


class TestMakeEmbeddings:
    """Test embedding generation from lattice."""
    
    def test_embeddings_basic(self):
        """Test basic embedding generation."""
        df = pd.DataFrame({
            'zip': ['A', 'B', 'C'],
            'row': [0, 1, 2],
            'col': [0, 1, 2]
        })
        
        df_with_emb = make_embeddings(df, noise=0.0)
        
        # Should add embedding columns
        assert 'e0' in df_with_emb.columns
        assert 'e1' in df_with_emb.columns
        
        # Original columns should be preserved
        assert 'zip' in df_with_emb.columns
        assert 'row' in df_with_emb.columns
        assert 'col' in df_with_emb.columns
        
        # With no noise, should be perfectly correlated with normalized row/col
        max_row, max_col = 2, 2
        expected_e0 = df['row'] / max_row
        expected_e1 = df['col'] / max_col
        
        np.testing.assert_array_almost_equal(df_with_emb['e0'], expected_e0, decimal=5)
        np.testing.assert_array_almost_equal(df_with_emb['e1'], expected_e1, decimal=5)
    
    def test_embeddings_reproducible(self):
        """Test that embeddings are reproducible."""
        df = pd.DataFrame({
            'zip': ['A', 'B', 'C'],
            'row': [0, 1, 2],
            'col': [0, 1, 2]
        })
        
        # Generate twice with same noise settings
        df_emb1 = make_embeddings(df, noise=0.05, random_state=42)
        df_emb2 = make_embeddings(df, noise=0.05, random_state=42)
        
        # Should be identical (fixed seed)
        np.testing.assert_array_equal(df_emb1['e0'], df_emb2['e0'])
        np.testing.assert_array_equal(df_emb1['e1'], df_emb2['e1'])
    
    def test_rng_local_generator(self):
        """Test that make_embeddings uses local RNG."""
        df = pd.DataFrame({'zip': ['A', 'B'], 'row': [0, 1], 'col': [0, 1]})
        
        # Should not affect global RNG state
        np.random.seed(999)
        global_before = np.random.get_state()
        
        df_emb = make_embeddings(df, noise=0.1, random_state=42)
        
        global_after = np.random.get_state()
        
        # Global state should be unchanged
        assert np.array_equal(global_before[1], global_after[1])
        assert 'e0' in df_emb.columns
        assert 'e1' in df_emb.columns