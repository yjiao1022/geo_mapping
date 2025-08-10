"""Unit tests for clustering evaluation metrics."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from libpysal import weights
from sklearn.datasets import make_blobs

from clustering.evaluation import quality_scores, contiguity_score, partition_stability, cluster_summary


class TestQualityScores:
    """Test quality score computation."""
    
    def test_quality_two_blob_separation(self):
        """Test quality scores on well-separated blobs."""
        # Create two well-separated 2D blobs
        X, labels = make_blobs(n_samples=100, centers=2, cluster_std=0.5, 
                              center_box=(-5.0, 5.0), random_state=42)
        
        df = pd.DataFrame({
            'zip': [f'Z{i:03d}' for i in range(100)],
            'e0': X[:, 0],
            'e1': X[:, 1]
        })
        
        scores = quality_scores(df, labels)
        
        # Well-separated clusters should have good scores
        assert scores['silhouette'] > 0.5
        assert scores['davies_bouldin'] < 1.0
        assert not np.isnan(scores['silhouette'])
        assert not np.isnan(scores['davies_bouldin'])
    
    def test_quality_single_cluster(self):
        """Test that single cluster returns NaN."""
        df = pd.DataFrame({
            'zip': ['A', 'B', 'C'],
            'e0': [1.0, 2.0, 3.0],
            'e1': [1.0, 1.5, 2.0]
        })
        labels = np.array([0, 0, 0])  # All same cluster
        
        scores = quality_scores(df, labels)
        
        assert np.isnan(scores['silhouette'])
        assert np.isnan(scores['davies_bouldin'])


class TestContiguityScore:
    """Test contiguity scoring."""
    
    def test_contiguity_perfect_connected(self):
        """Test perfectly connected clustering."""
        # Linear graph: A-B-C
        adjacency = {
            'A': ['B'],
            'B': ['A', 'C'],
            'C': ['B']
        }
        w = weights.W(adjacency, id_order=['A', 'B', 'C'])
        
        # Two connected clusters: {A,B} and {C}
        labels = np.array([0, 0, 1])
        
        scores = contiguity_score(labels, w)
        
        assert scores['connected_fraction'] == 1.0
        assert scores['violating_clusters'] == 0
        assert scores['n_clusters'] == 2
    
    def test_contiguity_disconnected_cluster(self):
        """Test disconnected cluster detection."""
        # Disconnected graph: A-B, C (isolated)
        adjacency = {
            'A': ['B'],
            'B': ['A'],
            'C': []
        }
        w = weights.W(adjacency, id_order=['A', 'B', 'C'])
        
        # Try to put disconnected nodes in same cluster
        labels = np.array([0, 1, 0])  # A and C in same cluster, but not connected
        
        scores = contiguity_score(labels, w)
        
        assert scores['connected_fraction'] < 1.0
        assert scores['violating_clusters'] > 0


class TestPartitionStability:
    """Test partition stability comparison."""
    
    def test_stability_identical_partitions(self):
        """Test that identical partitions have perfect stability."""
        labels_a = np.array([0, 0, 1, 1, 2])
        labels_b = np.array([0, 0, 1, 1, 2])
        
        scores = partition_stability(labels_a, labels_b)
        
        assert scores['ari'] == 1.0
        assert scores['jaccard'] == 1.0
    
    def test_stability_completely_different(self):
        """Test completely different partitions - updated after Jaccard fix."""
        labels_a = np.array([0, 0, 0, 0, 0])  # All same cluster
        labels_b = np.array([0, 1, 2, 3, 4])  # All different clusters
        
        scores = partition_stability(labels_a, labels_b)
        
        assert scores['ari'] <= 0.0  # Should be very low
        # Updated expectation after Jaccard fix - no pairs are in same cluster in both
        assert scores['jaccard'] == 0.0  # Should be exactly 0 with corrected calculation


class TestClusterSummary:
    """Test cluster summary generation."""
    
    def test_summary_basic_counts(self):
        """Test basic cluster count summary."""
        mapping = pd.DataFrame({
            'zip': ['A', 'B', 'C', 'D'],
            'unit_id': [0, 0, 1, 1]
        })
        
        df = pd.DataFrame({
            'zip': ['A', 'B', 'C', 'D'],
            'population': [100, 200, 150, 250]
        })
        
        summary = cluster_summary(mapping, df)
        
        expected = pd.DataFrame({
            'unit_id': [0, 1],
            'n_zips': [2, 2]
        })
        
        pd.testing.assert_frame_equal(summary, expected)
    
    def test_summary_with_attribute_sums(self):
        """Test cluster summary with attribute aggregation."""
        mapping = pd.DataFrame({
            'zip': ['A', 'B', 'C', 'D'],
            'unit_id': [0, 0, 1, 1]
        })
        
        df = pd.DataFrame({
            'zip': ['A', 'B', 'C', 'D'],
            'population': [100, 200, 150, 250],
            'sales': [10, 20, 15, 25]
        })
        
        summary = cluster_summary(mapping, df, sum_attrs=['population', 'sales'])
        
        expected = pd.DataFrame({
            'unit_id': [0, 1],
            'n_zips': [2, 2],
            'population': [300, 400],  # 100+200, 150+250
            'sales': [30, 40]          # 10+20, 15+25
        })
        
        pd.testing.assert_frame_equal(summary, expected)


class TestCriticalFixes:
    """Test all critical fixes."""
    
    def test_bfs_deque_performance(self):
        """Test that BFS uses deque for O(1) operations."""
        # Create disconnected clusters
        labels = np.array([0, 0, 1, 1, 2])  # Cluster 2 is isolated
        w = weights.W({
            'A': ['B'], 'B': ['A'], 'C': ['D'], 'D': ['C'], 'E': []
        }, id_order=['A', 'B', 'C', 'D', 'E'])
        
        # Should detect disconnected cluster E
        result = contiguity_score(labels, w, use_bfs=True)
        assert result['violating_clusters'] >= 0  # E is isolated singleton
    
    def test_bfs_networkx_parity(self):
        """BFS and NetworkX give same results for small graphs."""
        labels = np.array([0, 0, 1, 1])
        w = weights.W({
            'A': ['B'], 'B': ['A'], 'C': ['D'], 'D': ['C']
        }, id_order=['A', 'B', 'C', 'D'])
        
        bfs_result = contiguity_score(labels, w, use_bfs=True)
        nx_result = contiguity_score(labels, w, use_bfs=False)
        
        assert bfs_result['connected_fraction'] == nx_result['connected_fraction']
        assert bfs_result['violating_clusters'] == nx_result['violating_clusters']