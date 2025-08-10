"""Unit tests for adjacency builders and reordering."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from libpysal import weights

from clustering.adjacency import mobility_graph, reorder_w_to_zip_order

try:
    import geopandas as gpd
    from shapely.geometry import Polygon
    from clustering.adjacency import contiguity_graph
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


class TestMobilityGraph:
    """Test mobility graph creation."""
    
    def test_basic_mobility_graph(self):
        """Test basic mobility graph construction."""
        # Create test mobility data
        edges_data = {
            'src_zip': ['A', 'A', 'B', 'C'],
            'dst_zip': ['B', 'C', 'C', 'A'],
            'weight': [1.0, 0.5, 2.0, 1.5]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(edges_data)
            df.to_csv(f.name, index=False)
            
            # Build graph
            w = mobility_graph(f.name, threshold=0.0)
            
            # Check basic properties
            assert w.n == 3  # A, B, C
            assert set(w.neighbors.keys()) == {'A', 'B', 'C'}
            
            # Check symmetry (undirected)
            assert 'B' in w.neighbors['A']
            assert 'A' in w.neighbors['B']
            
            # Clean up
            Path(f.name).unlink()
    
    def test_mobility_threshold_filtering(self):
        """Test that threshold properly filters edges."""
        edges_data = {
            'src_zip': ['A', 'A', 'B'],
            'dst_zip': ['B', 'C', 'C'],
            'weight': [1.0, 0.1, 2.0]  # Middle edge below threshold
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(edges_data)
            df.to_csv(f.name, index=False)
            
            # Build graph with threshold
            w = mobility_graph(f.name, threshold=0.5)
            
            # A-C edge should be filtered out
            assert 'C' not in w.neighbors['A']
            assert 'A' not in w.neighbors['C']
            
            # A-B and B-C should remain
            assert 'B' in w.neighbors['A']
            assert 'C' in w.neighbors['B']
            
            # Clean up
            Path(f.name).unlink()


@pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not available")
class TestContiguityGraph:
    """Test contiguity graph creation."""
    
    def test_basic_contiguity_graph(self):
        """Test basic contiguity graph with simple polygons."""
        # Create simple adjacent squares
        polygons = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # Left square
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])   # Right square (adjacent)
        ]
        
        gdf = gpd.GeoDataFrame({
            'zip': ['A', 'B'],
            'geometry': polygons
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            shp_path = Path(temp_dir) / "test_shapes.shp"
            gdf.to_file(shp_path)
            
            w = contiguity_graph(str(shp_path), rule='rook')
            
            # Should have 2 nodes
            assert w.n == 2
            
            # Check that there's at least one neighbor relationship
            total_neighbors = sum(len(neighbors) for neighbors in w.neighbors.values())
            assert total_neighbors >= 2  # Should be symmetric


class TestReorderW:
    """Test W reordering functionality."""
    
    def test_reorder_w_basic(self):
        """Test basic reordering of weights matrix."""
        # Create simple mobility graph
        edges_data = {
            'src_zip': ['C', 'A', 'B'],
            'dst_zip': ['A', 'B', 'C'],
            'weight': [1.0, 1.0, 1.0]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(edges_data)
            df.to_csv(f.name, index=False)
            
            w = mobility_graph(f.name)
            
            # Reorder to specific order
            new_order = ['A', 'B', 'C']
            w_reordered = reorder_w_to_zip_order(w, new_order)
            
            # Check new order
            assert list(w_reordered.id_order) == new_order
            
            # Check that connectivity is preserved
            assert w_reordered.n == w.n
            
            # Clean up
            Path(f.name).unlink()
    
    def test_reorder_w_alignment(self):
        """Test that reordering preserves neighbor relationships."""
        edges_data = {
            'src_zip': ['Z1', 'Z2'],
            'dst_zip': ['Z2', 'Z3'],
            'weight': [1.0, 1.0]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(edges_data)
            df.to_csv(f.name, index=False)
            
            w = mobility_graph(f.name)
            
            # Reorder
            new_order = ['Z3', 'Z1', 'Z2']  # Different from original
            w_reordered = reorder_w_to_zip_order(w, new_order)
            
            # Check that Z1-Z2 and Z2-Z3 relationships are preserved
            assert 'Z2' in w_reordered.neighbors['Z1']
            assert 'Z3' in w_reordered.neighbors['Z2']
            assert 'Z1' in w_reordered.neighbors['Z2']  # Symmetry
            
            # Clean up
            Path(f.name).unlink()


class TestReviewSuggestions:
    """Tests covering all suggestions from the detailed review."""
    
    def test_mobility_threshold_monotonicity(self):
        """Edge count decreases monotonically as threshold increases."""
        edges_data = pd.DataFrame({
            'src_zip': ['A', 'A', 'B', 'B'] * 10,
            'dst_zip': ['B', 'C', 'C', 'D'] * 10, 
            'weight': [0.1, 0.5, 0.9, 0.95] * 10
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            edges_data.to_csv(f.name, index=False)
            
            thresholds = [0.0, 0.5, 0.9]
            edge_counts = []
            
            for thresh in thresholds:
                w = mobility_graph(f.name, threshold=thresh)
                # Count edges (undirected, so divide by 2)
                total_edges = sum(len(neighbors) for neighbors in w.neighbors.values()) // 2
                edge_counts.append(total_edges)
            
            # Edge count should decrease monotonically
            assert edge_counts[0] >= edge_counts[1] >= edge_counts[2]
            
            Path(f.name).unlink()
    
    def test_reorder_invariants(self):
        """Neighbor sets identical after multiple random permutations."""
        # Create baseline graph
        w_base = self._mobility_graph_from_edges([('A', 'B'), ('B', 'C'), ('C', 'A')])
        baseline_neighbors = {k: set(v) for k, v in w_base.neighbors.items()}
        
        # Test multiple random permutations
        zip_list = ['A', 'B', 'C']
        for _ in range(5):
            np.random.shuffle(zip_list)
            w_reordered = reorder_w_to_zip_order(w_base, zip_list.copy())
            
            # Neighbor sets should be identical
            for zip_code in zip_list:
                reordered_neighbors = set(w_reordered.neighbors[zip_code])
                assert reordered_neighbors == baseline_neighbors[zip_code]
    
    def test_reorder_list_copying_fix(self):
        """Test that reorder properly copies lists, not dicts."""
        # Create simple mobility graph
        edges_data = pd.DataFrame({
            'src_zip': ['A', 'B'],
            'dst_zip': ['B', 'C'],
            'weight': [1.0, 1.0]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            edges_data.to_csv(f.name, index=False)
            w = mobility_graph(f.name)
            
            # This should not raise dict(list) error
            w_reordered = reorder_w_to_zip_order(w, ['C', 'A', 'B'])
            
            # Verify structure is preserved
            assert isinstance(w_reordered.neighbors['A'], list)
            assert 'B' in w_reordered.neighbors['A']
            
            Path(f.name).unlink()
    
    def _mobility_graph_from_edges(self, edge_list):
        """Helper to create mobility graph from edge list."""
        df = pd.DataFrame(edge_list, columns=['src_zip', 'dst_zip'])
        df['weight'] = 1.0
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            w = mobility_graph(f.name)
            Path(f.name).unlink()
            return w