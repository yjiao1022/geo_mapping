"""
Evaluation metrics for clustering quality, contiguity, and stability.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import logging
from collections import deque
from typing import Dict, List, Optional
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.metrics.cluster import pair_confusion_matrix
from libpysal import weights
import networkx as nx

logger = logging.getLogger(__name__)

# Global cache for NetworkX graphs to avoid repeated conversion
_networkx_cache = {}


def quality_scores(
    df: pd.DataFrame, 
    labels: np.ndarray, 
    emb_prefix: str = "e", 
    max_samples: int = 10000,
    use_sklearn_sample_size: bool = True,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute compactness/separation metrics in embedding space.
    
    Parameters
    ----------
    df : pd.DataFrame
        Contains embedding columns with a common prefix (default 'e').
    labels : np.ndarray
        Cluster assignment for each row of df.
    emb_prefix : str
        Embedding column prefix; default 'e'.
    max_samples : int
        Maximum samples for silhouette computation to avoid memory issues.
    use_sklearn_sample_size : bool
        Use sklearn's built-in sample_size parameter for better performance.
    random_state : int
        Random seed for reproducible subsampling.
        
    Returns
    -------
    dict
        {'silhouette': float | nan, 'davies_bouldin': float | nan}
        
    Notes
    -----
    Performance: Uses sklearn's sample_size param for O(N²) silhouette computation.
    Stability: Uses np.isfinite to catch both NaN and Inf values.
    Reproducibility: Uses local RandomState for subsampling.
    """
    # Find embedding columns
    emb_cols = [col for col in df.columns if col.startswith(emb_prefix) and col[len(emb_prefix):].isdigit()]
    if not emb_cols:
        raise ValueError(f"No embedding columns found with prefix '{emb_prefix}'")
    
    # Sort embedding columns numerically
    emb_cols = sorted(emb_cols, key=lambda x: int(x[len(emb_prefix):]))
    X = df[emb_cols].values
    
    # FIXED: Handle both NaN and Inf values in embeddings
    if not np.isfinite(X).all():
        logger.warning("Non-finite values found in embeddings - returning NaN scores")
        return {'silhouette': np.nan, 'davies_bouldin': np.nan}
    
    # Check for degenerate cases - robustified singleton detection
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1:
        return {'silhouette': np.nan, 'davies_bouldin': np.nan}
    
    # Check for singleton clusters (more robust check)
    min_cluster_size = np.min(label_counts)
    if min_cluster_size == 1:
        logger.warning("Singleton clusters detected - returning NaN scores")
        return {'silhouette': np.nan, 'davies_bouldin': np.nan}
    
    # FIXED: Reproducible subsampling with local RandomState
    rng = np.random.RandomState(random_state)
    
    # Performance optimization: Use sklearn's built-in sample_size for large datasets
    silhouette = np.nan
    if len(X) > 50000 and use_sklearn_sample_size:
        # Use sklearn's efficient sampling for very large datasets
        try:
            sample_size = min(max_samples, len(X))
            logger.info(f"Using sklearn sample_size={sample_size} for silhouette score")
            silhouette = silhouette_score(X, labels, sample_size=sample_size, random_state=random_state)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Silhouette score computation failed: {e}")
            silhouette = np.nan
    else:
        # Manual subsampling for moderate-sized datasets with local RNG
        if len(X) > max_samples:
            logger.info(f"Subsampling {max_samples} points for silhouette score computation")
            indices = rng.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
            labels_sample = labels[indices]
        else:
            X_sample = X
            labels_sample = labels
        
        # Compute silhouette with proper error handling
        try:
            silhouette = silhouette_score(X_sample, labels_sample)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Silhouette score computation failed: {e}")
            silhouette = np.nan
    
    # Davies-Bouldin score (use full data, it's more efficient)
    try:
        davies_bouldin = davies_bouldin_score(X, labels)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Davies-Bouldin score computation failed: {e}")
        davies_bouldin = np.nan
    
    return {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin
    }


def contiguity_score(
    labels: np.ndarray, 
    w: weights.W, 
    use_bfs: bool = True,
    cache_networkx: bool = True
) -> Dict[str, float]:
    """
    Verify each cluster is a single connected component under W.
    
    Parameters
    ----------
    labels : np.ndarray
        Cluster IDs aligned with w.id_order.
    w : weights.W
        Spatial graph.
    use_bfs : bool
        Use BFS instead of NetworkX for large clusters (more memory efficient).
    cache_networkx : bool
        Cache NetworkX conversion for repeated calls.
        
    Returns
    -------
    dict
        {'connected_fraction': float in [0,1], 'violating_clusters': int, 'n_clusters': int}
        
    Notes
    -----
    Performance: Caches w.to_networkx() conversion and reuses for repeated calls.
    Uses optimized BFS with deque for memory efficiency with large clusters.
    """
    if len(labels) != len(w.id_order):
        raise ValueError("Labels length doesn't match weights graph size")
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    violating_clusters = 0
    
    if use_bfs:
        # FIXED: Use deque for O(1) pops instead of list.pop(0) which is O(n)
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_nodes = [w.id_order[i] for i in cluster_indices]
            
            if len(cluster_nodes) <= 1:
                continue  # Single node is trivially connected
            
            # Optimized BFS to check connectivity with deque
            visited = set()
            queue = deque([cluster_nodes[0]])
            visited.add(cluster_nodes[0])
            cluster_node_set = set(cluster_nodes)  # For O(1) lookup
            
            while queue:
                node = queue.popleft()  # O(1) with deque
                # Check neighbors that are in the same cluster
                for neighbor in w.neighbors.get(node, []):
                    if neighbor in cluster_node_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Early break: If disconnected, increment and continue
            if len(visited) < len(cluster_nodes):
                violating_clusters += 1
    else:
        # Use NetworkX with caching for repeated calls
        w_hash = hash((id(w), w.n))  # Simple hash based on object id and size
        
        if cache_networkx and w_hash in _networkx_cache:
            G = _networkx_cache[w_hash]
            logger.debug("Using cached NetworkX graph")
        else:
            G = w.to_networkx()
            if cache_networkx:
                _networkx_cache[w_hash] = G
                # Keep cache size reasonable
                if len(_networkx_cache) > 10:
                    oldest_key = next(iter(_networkx_cache))
                    del _networkx_cache[oldest_key]
        
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_nodes = [w.id_order[i] for i in cluster_indices]
            
            if len(cluster_nodes) <= 1:
                continue  # Single node is trivially connected
            
            # Extract subgraph for this cluster - NetworkX uses integer indices
            subgraph = G.subgraph(cluster_indices)
            
            # Check if it's connected - handle empty subgraph case
            if len(subgraph.nodes) == 0:
                # Empty subgraph - should not happen given our length check
                violating_clusters += 1
            elif not nx.is_connected(subgraph):
                violating_clusters += 1
    
    connected_fraction = (n_clusters - violating_clusters) / n_clusters if n_clusters > 0 else 1.0
    
    return {
        'connected_fraction': connected_fraction,
        'violating_clusters': violating_clusters,
        'n_clusters': n_clusters
    }


def partition_stability(labels_a: np.ndarray, labels_b: np.ndarray) -> Dict[str, float]:
    """
    Compare two partitions of the same items.
    
    Parameters
    ----------
    labels_a : np.ndarray
        First partition labels.
    labels_b : np.ndarray
        Second partition labels (same length and order).
        
    Returns
    -------
    dict
        {'ari': float, 'jaccard': float}
        
    Notes
    -----
    ARI is permutation-invariant; Jaccard computed correctly using TP/(TP+FP+FN).
    Performance: Uses efficient pair confusion matrix computation.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Label arrays must have same length")
    
    # Adjusted Rand Index (already optimized in sklearn)
    ari = adjusted_rand_score(labels_a, labels_b)
    
    # Jaccard index on pairwise co-membership - FIXED calculation
    cm = pair_confusion_matrix(labels_a, labels_b)
    
    # Correct confusion matrix interpretation:
    # cm[0,0] = both in different clusters (TN)
    # cm[0,1] = same cluster in A, different in B (FP) 
    # cm[1,0] = different in A, same cluster in B (FN)
    # cm[1,1] = both in same cluster (TP)
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]
    
    # Jaccard = TP / (TP + FP + FN)
    jaccard = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 1.0
    
    return {
        'ari': ari,
        'jaccard': jaccard
    }


def cluster_summary(
    mapping: pd.DataFrame, 
    df: pd.DataFrame, 
    sum_attrs: Optional[List[str]] = None,
    use_efficient_merge: bool = True,
    handle_nan_attrs: bool = True
) -> pd.DataFrame:
    """
    Aggregate counts and optional attribute sums per unit_id.
    
    Parameters
    ----------
    mapping : pd.DataFrame
        ['zip','unit_id'] assignment.
    df : pd.DataFrame
        Original dataframe that includes 'zip' and attributes.
    sum_attrs : list[str] | None
        Attribute columns to sum per cluster.
    use_efficient_merge : bool
        Use more efficient merge strategy for large datasets.
    handle_nan_attrs : bool
        Use min_count=1 to handle all-NaN attribute groups.
        
    Returns
    -------
    pd.DataFrame
        One row per unit_id with columns ['unit_id','n_zips', <sums...>].
        
    Notes
    -----
    Performance: Uses efficient merge strategies and vectorized operations.
    Option: Adds min_count=1 semantics for NaN-safe attribute aggregation.
    """
    # Validate inputs
    if 'zip' not in mapping.columns or 'unit_id' not in mapping.columns:
        raise ValueError("mapping must contain 'zip' and 'unit_id' columns")
    
    if 'zip' not in df.columns:
        raise ValueError("df must contain 'zip' column")
    
    # Performance optimization: Use efficient merge for large datasets
    if use_efficient_merge and len(mapping) > 10000:
        # Pre-sort both DataFrames on zip for more efficient merge
        mapping_sorted = mapping.sort_values('zip')
        df_sorted = df.sort_values('zip')
        merged = mapping_sorted.merge(df_sorted, on='zip', how='left', indicator=True)
    else:
        # Standard merge with indicator
        merged = mapping.merge(df, on='zip', how='left', indicator=True)
    
    # Check for missing ZIPs using indicator column
    if (merged['_merge'] != 'both').any():
        missing = merged.loc[merged['_merge'] != 'both', 'zip'].unique()
        missing_display = list(missing[:5]) + (['...'] if len(missing) > 5 else [])
        raise ValueError(f"ZIPs in mapping missing from df: {missing_display}")
    
    # Drop indicator column
    merged = merged.drop(columns=['_merge'])
    
    # Start with count aggregation (vectorized)
    summary = merged.groupby('unit_id', sort=False).size().reset_index(name='n_zips')
    
    # Add attribute sums if requested (vectorized)
    if sum_attrs:
        # Validate attribute columns exist
        missing_attrs = [col for col in sum_attrs if col not in df.columns]
        if missing_attrs:
            raise ValueError(f"Missing attribute columns in df: {missing_attrs}")
        
        # Compute sums per unit_id using efficient groupby
        if handle_nan_attrs:
            # Use min_count=1 to handle all-NaN groups properly
            attr_sums = merged.groupby('unit_id', sort=False)[sum_attrs].sum(min_count=1).reset_index()
        else:
            attr_sums = merged.groupby('unit_id', sort=False)[sum_attrs].sum().reset_index()
        
        # Merge with summary
        summary = summary.merge(attr_sums, on='unit_id', how='left')
    
    # Sort by unit_id for consistent output
    return summary.sort_values('unit_id').reset_index(drop=True)


def clear_performance_caches():
    """Clear performance caches to free memory."""
    global _networkx_cache
    _networkx_cache.clear()
    logger.info("Cleared performance caches")