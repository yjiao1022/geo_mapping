"""
Spatial graph builders for contiguity and mobility networks.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import logging
import hashlib
from collections import defaultdict
from libpysal import weights
from libpysal.weights import contiguity

try:
    from scipy.sparse import coo_matrix
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


def contiguity_graph(
    shapefile_path: str, 
    rule: str = "rook", 
    use_direct_shapefile: bool = True,
    zip_column: str = None
) -> weights.W:
    """
    Create a polygon contiguity graph using libpysal.
    
    Parameters
    ----------
    shapefile_path : str
        Path to the ZIP polygon shapefile.
    rule : str
        'rook' or 'queen' contiguity; default 'rook'.
    use_direct_shapefile : bool
        If True, use direct shapefile reading for large files to avoid memory overhead.
    zip_column : str, optional
        Column name containing ZIP codes (e.g., 'ZCTA5CE10', 'zip'). 
        If specified, uses from_dataframe with ids parameter.
        
    Returns
    -------
    weights.W
        Undirected, symmetrized contiguity graph.
        
    Notes
    -----
    Node IDs come from the shapefile and may not match df['zip'] order; align later.
    Uses libpysal.weights.contiguity for version stability.
    Performance: For large shapefiles, uses direct shapefile reading to avoid GeoDataFrame overhead.
    Compatibility: from_shapefile() path handling can be finicky with zipped shapefiles.
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas is required for contiguity_graph")
    
    # If ZIP column specified, use from_dataframe approach for better ID alignment
    if zip_column is not None:
        logger.info(f"Using ZIP column '{zip_column}' for direct ID alignment")
        gdf = gpd.read_file(shapefile_path)
        
        if zip_column not in gdf.columns:
            raise ValueError(f"ZIP column '{zip_column}' not found in shapefile")
        
        if rule.lower() == "rook":
            w = contiguity.Rook.from_dataframe(gdf, ids=gdf[zip_column])
        elif rule.lower() == "queen":
            w = contiguity.Queen.from_dataframe(gdf, ids=gdf[zip_column])
        else:
            raise ValueError(f"Unsupported contiguity rule: {rule}")
    
    # Performance optimization: Use direct shapefile reading for large files
    elif use_direct_shapefile:
        try:
            # Use libpysal's direct shapefile reading for better performance
            if rule.lower() == "rook":
                w = contiguity.Rook.from_shapefile(shapefile_path)
            elif rule.lower() == "queen":
                w = contiguity.Queen.from_shapefile(shapefile_path)
            else:
                raise ValueError(f"Unsupported contiguity rule: {rule}")
        except (AttributeError, TypeError):
            # Fallback to GeoDataFrame method if direct method not available
            logger.info("Direct shapefile reading not available, using GeoDataFrame method")
            use_direct_shapefile = False
    
    if not use_direct_shapefile and zip_column is None:
        # Read shapefile via GeoDataFrame
        gdf = gpd.read_file(shapefile_path)
        
        # Build contiguity weights using libpysal.weights.contiguity for stability
        if rule.lower() == "rook":
            w = contiguity.Rook.from_dataframe(gdf)
        elif rule.lower() == "queen":
            w = contiguity.Queen.from_dataframe(gdf)
        else:
            raise ValueError(f"Unsupported contiguity rule: {rule}")
    
    # Symmetrize before applying transform (ensures undirected graph)
    w.symmetrize()
    w.transform = 'b'  # binary weights
    
    # Downgraded to debug - this is expected behavior with shapefiles
    if zip_column is None and list(w.id_order) != list(range(w.n)):
        logger.debug("Contiguity graph id_order uses shapefile indices (expected)")
    
    return w


def mobility_graph(
    edges_csv: str, 
    threshold: float = 0.0, 
    keep_numeric_weights: bool = False,
    use_sparse_matrix: bool = True,
    chunk_size: int = 50000,
    deduplicate_edges: bool = True
) -> weights.W:
    """
    Build an undirected graph from mobility edges.
    
    Parameters
    ----------
    edges_csv : str
        CSV with columns [src_zip, dst_zip, weight].
    threshold : float
        Drop edges with weight < threshold (default 0.0).
    keep_numeric_weights : bool
        If True, preserve numeric weights; if False, use binary transform.
    use_sparse_matrix : bool
        Use sparse matrix for large graphs (requires scipy).
    chunk_size : int
        Process CSV in chunks to reduce memory usage.
    deduplicate_edges : bool
        If True, handle CSV files that already contain both directions.
        
    Returns
    -------
    weights.W
        Symmetrized graph keyed by ZIP strings.
        
    Notes
    -----
    Performance: Uses compressed sparse matrix (CSR) for very large graphs.
    Logic: Deduplicates edges if CSV contains both (A,B) and (B,A) entries.
    """
    logger.info(f"Loading mobility edges from {edges_csv} with threshold={threshold}")
    
    # Performance optimization: Read large CSV in chunks
    edge_chunks = []
    chunk_count = 0
    
    for chunk in pd.read_csv(edges_csv, chunksize=chunk_size):
        chunk_count += 1
        if chunk_count % 10 == 0:
            logger.info(f"Processed {chunk_count * chunk_size:,} rows...")
        
        # Validate required columns
        required_cols = ['src_zip', 'dst_zip', 'weight']
        missing_cols = [col for col in required_cols if col not in chunk.columns]
        if missing_cols and chunk_count == 1:  # Only check first chunk
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter chunk
        chunk = chunk.dropna(subset=['src_zip', 'dst_zip'])
        chunk = chunk[chunk['weight'] >= threshold]
        chunk = chunk[chunk['src_zip'] != chunk['dst_zip']]  # Remove self-loops
        
        # Convert to string
        chunk['src_zip'] = chunk['src_zip'].astype(str)
        chunk['dst_zip'] = chunk['dst_zip'].astype(str)
        
        edge_chunks.append(chunk)
    
    # Combine all chunks
    df = pd.concat(edge_chunks, ignore_index=True) if edge_chunks else pd.DataFrame()
    
    if len(df) == 0:
        logger.warning("No valid edges found after filtering")
        return weights.W({})
    
    # Deduplicate edges if CSV contains both directions
    if deduplicate_edges:
        # Keep only one direction to avoid double-counting
        df['edge_id'] = df.apply(lambda row: tuple(sorted([row['src_zip'], row['dst_zip']])), axis=1)
        df = df.drop_duplicates(subset=['edge_id']).drop(columns=['edge_id'])
        logger.info(f"After deduplication: {len(df):,} unique edges")
    
    logger.info(f"Loaded {len(df):,} mobility edges after filtering")
    
    # Performance optimization: Use sparse matrix for large graphs
    if use_sparse_matrix and HAS_SCIPY and len(df) > 10000:
        return _build_sparse_mobility_graph(df, keep_numeric_weights)
    else:
        return _build_dict_mobility_graph(df, keep_numeric_weights)


def _build_sparse_mobility_graph(df: pd.DataFrame, keep_numeric_weights: bool) -> weights.W:
    """Build mobility graph using sparse matrix for performance."""
    logger.info("Using sparse matrix construction for large mobility graph")
    
    # Create node mapping
    all_nodes = pd.concat([df['src_zip'], df['dst_zip']]).unique()
    sorted_nodes = sorted(all_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    n_nodes = len(sorted_nodes)
    
    # Create edge lists with indices
    src_indices = [node_to_idx[src] for src in df['src_zip']]
    dst_indices = [node_to_idx[dst] for dst in df['dst_zip']]
    weights_list = df['weight'].tolist()
    
    # Add reverse edges for undirected graph (A + A.T approach)
    all_src = src_indices + dst_indices
    all_dst = dst_indices + src_indices  
    all_weights = weights_list + weights_list
    
    # Create sparse matrix - tocsr() handles duplicates via summation
    csr_matrix = coo_matrix(
        (all_weights, (all_src, all_dst)), 
        shape=(n_nodes, n_nodes)
    ).tocsr()
    
    # Convert to libpysal weights - optimized node mapping
    neighbors = {}
    weights_dict = {} if keep_numeric_weights else None
    
    # Micro-perf: Direct array indexing instead of dict lookup
    idx_to_node = np.array(sorted_nodes)
    
    for i in range(n_nodes):
        node = idx_to_node[i]
        row = csr_matrix.getrow(i)
        neighbor_indices = row.indices
        neighbor_weights = row.data
        
        neighbors[node] = [idx_to_node[j] for j in neighbor_indices]
        if weights_dict is not None:
            weights_dict[node] = neighbor_weights.tolist()
    
    # Create weights object
    w = weights.W(neighbors, weights=weights_dict, id_order=sorted_nodes)
    w.symmetrize()
    
    if not keep_numeric_weights:
        w.transform = 'b'
    
    return w


def _build_dict_mobility_graph(df: pd.DataFrame, keep_numeric_weights: bool) -> weights.W:
    """Build mobility graph using dictionary approach for smaller graphs."""
    # Original dictionary-based approach (from previous version)
    neighbors = defaultdict(list)
    weights_dict = defaultdict(list) if keep_numeric_weights else None
    
    for src, dst, wgt in df[['src_zip', 'dst_zip', 'weight']].itertuples(index=False):
        if src == dst:
            continue  # Extra safety against self-loops
        
        # Add both directions for undirected graph
        for a, b in ((src, dst), (dst, src)):
            neighbors[a].append(b)
            if keep_numeric_weights:
                weights_dict[a].append(float(wgt))
    
    # Sort IDs for consistent ordering
    ids = sorted(neighbors.keys())
    
    # Create weights.W object with both neighbors and weights
    w = weights.W(neighbors, weights=weights_dict, id_order=ids)
    
    # Symmetrize to ensure proper undirected structure
    w.symmetrize()
    
    # Apply transform conditionally
    if not keep_numeric_weights:
        w.transform = 'b'  # binary weights
    
    return w


def reorder_w_to_zip_order(w: weights.W, zip_order: list[str], avoid_copy: bool = True) -> weights.W:
    """
    Align W to a specific ZIP order so labels[i] ↔ zip_order[i].
    
    Parameters
    ----------
    w : weights.W
        Input graph whose keys include all ZIPs in zip_order.
    zip_order : list[str]
        Exact row order used for clustering/evaluation.
    avoid_copy : bool
        If True, avoid copying dicts when weights are unchanged.
        
    Returns
    -------
    weights.W
        Copy with id_order == zip_order.
        
    Notes
    -----
    Performance: Avoids full dict copying when weights unchanged; rebuilds id_order in place when possible.
    Safety: Creates deep copies of neighbor/weight lists to prevent aliasing.
    """
    # Check for ZIPs in zip_order that aren't in w
    w_ids = set(w.neighbors.keys())
    zip_set = set(zip_order)
    
    missing_zips = zip_set - w_ids
    if missing_zips:
        raise ValueError(f"ZIPs not found in weights graph: {list(missing_zips)}")
    
    # Check for extra IDs in w that aren't in zip_order (may be acceptable)
    extra_ids = w_ids - zip_set
    if extra_ids:
        logger.warning(f"Weights graph contains {len(extra_ids)} IDs not in zip_order")
    
    # Performance optimization: Check if reordering is actually needed
    if list(w.id_order) == zip_order:
        logger.debug("No reordering needed - id_order already matches")
        return w
    
    # Performance optimization: Avoid copying if possible
    if avoid_copy and set(w.id_order) == zip_set:
        # Only id_order needs to change, not the actual neighbor/weight data
        try:
            # Try to create a new W with same data but different id_order
            w_reordered = weights.W(w.neighbors, weights=w.weights, id_order=zip_order)
            w_reordered.transform = w.transform
            return w_reordered
        except Exception:
            # Fall back to full copy if optimization fails
            logger.debug("Fast reordering failed, using full copy")
    
    # FIXED: Full copy approach - properly handle lists, not dicts
    new_neighbors = {}
    new_weights = {} if w.weights else None
    
    for zip_code in zip_order:
        # CRITICAL FIX: w.neighbors[zip_code] is a list, not dict
        new_neighbors[zip_code] = list(w.neighbors.get(zip_code, []))
        
        # CRITICAL FIX: w.weights[zip_code] is also a list, not dict
        if new_weights is not None and w.weights and zip_code in w.weights:
            new_weights[zip_code] = list(w.weights.get(zip_code, []))
    
    # Create new weights object with specified id_order
    w_reordered = weights.W(new_neighbors, weights=new_weights, id_order=zip_order)
    w_reordered.transform = w.transform
    
    return w_reordered