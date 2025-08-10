"""
Deterministic lattice generator for tests and end-to-end validation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import math
import logging
from typing import Tuple, Optional
from libpysal import weights

try:
    from libpysal.weights import lat2W
    HAS_LAT2W = True
except ImportError:
    HAS_LAT2W = False

logger = logging.getLogger(__name__)


def lattice_grid(
    width: int, 
    height: int, 
    block_w: int, 
    block_h: int, 
    use_vectorized: bool = True,
    use_lat2w: bool = True
) -> Tuple[pd.DataFrame, weights.W]:
    """
    Build a rook-adjacent grid graph and base ZIP list without embeddings.
    
    Parameters
    ----------
    width : int
        Number of columns.
    height : int
        Number of rows.
    block_w : int
        Number of columns per coarse block (for ground truth labels).
    block_h : int
        Number of rows per coarse block.
    use_vectorized : bool
        Use vectorized NumPy operations for large grids.
    use_lat2w : bool
        Use libpysal's lat2W for optimal grid construction when available.
        
    Returns
    -------
    df : pd.DataFrame
        Columns ['zip','row','col','block_id'].
    w : weights.W
        Rook contiguity over the grid with id_order matching df['zip'].
        
    Notes
    -----
    Deterministic IDs follow format Z{row:03d}{col:03d} (e.g., Z000001, Z001002).
    Performance: Uses lat2W for optimal grid construction when available.
    Vectorized block_id computation using NumPy broadcasting for large grids.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    
    if block_w <= 0 or block_h <= 0:
        raise ValueError("block_w and block_h must be positive")
    
    # Calculate number of blocks for validation
    n_block_rows = math.ceil(height / block_h)
    n_block_cols = math.ceil(width / block_w)
    
    logger.info(f"Creating {width}x{height} lattice with {n_block_cols}x{n_block_rows} blocks")
    
    # Use lat2W optimization for grid construction if available
    if use_lat2w and HAS_LAT2W:
        return _lattice_grid_lat2w(width, height, block_w, block_h, n_block_cols)
    elif use_vectorized and width * height > 1000:
        # Performance optimization: Vectorized computation for large grids
        return _lattice_grid_vectorized(width, height, block_w, block_h, n_block_cols)
    else:
        # Original loop-based approach for small grids
        return _lattice_grid_loops(width, height, block_w, block_h, n_block_cols)


def _lattice_grid_lat2w(width: int, height: int, block_w: int, block_h: int, n_block_cols: int) -> Tuple[pd.DataFrame, weights.W]:
    """Optimal lattice grid generation using libpysal's lat2W."""
    logger.debug("Using lat2W for optimal grid construction")
    
    # Create coordinate arrays using broadcasting
    rows, cols = np.indices((height, width))
    
    # Vectorized block_id computation using NumPy broadcasting
    block_ids = (rows // block_h) * n_block_cols + (cols // block_w)
    
    # Flatten arrays
    rows_flat = rows.flatten()
    cols_flat = cols.flatten()
    block_ids_flat = block_ids.flatten()
    
    # Create ZIP IDs - format Z{row:03d}{col:03d}
    zip_ids = np.array([f"Z{row:03d}{col:03d}" for row, col in zip(rows_flat, cols_flat)])
    
    # Create DataFrame
    df = pd.DataFrame({
        'zip': zip_ids,
        'row': rows_flat,
        'col': cols_flat,
        'block_id': block_ids_flat
    })
    
    # Use lat2W for optimal grid contiguity construction
    w_grid = lat2W(nrows=height, ncols=width, rook=True)
    
    # Remap neighbors from integer indices to ZIP string IDs
    zip_neighbors = {}
    zip_weights = {}
    for i, zip_id in enumerate(zip_ids):
        if i in w_grid.neighbors:
            # Map neighbor indices to ZIP IDs
            zip_neighbors[zip_id] = [zip_ids[neighbor_idx] for neighbor_idx in w_grid.neighbors[i]]
            zip_weights[zip_id] = w_grid.weights[i]
        else:
            zip_neighbors[zip_id] = []
            zip_weights[zip_id] = []
    
    # Create weights object with ZIP string IDs
    w = weights.W(zip_neighbors, weights=zip_weights, id_order=zip_ids.tolist())
    w.transform = 'b'  # binary weights
    
    return df, w


def _lattice_grid_vectorized(width: int, height: int, block_w: int, block_h: int, n_block_cols: int) -> Tuple[pd.DataFrame, weights.W]:
    """Vectorized lattice grid generation for performance."""
    logger.debug("Using vectorized lattice generation")
    
    # Create coordinate arrays using broadcasting
    rows, cols = np.indices((height, width))
    
    # Vectorized block_id computation using NumPy broadcasting
    block_ids = (rows // block_h) * n_block_cols + (cols // block_w)
    
    # Flatten arrays
    rows_flat = rows.flatten()
    cols_flat = cols.flatten()
    block_ids_flat = block_ids.flatten()
    
    # Create ZIP IDs vectorized
    zip_ids = np.array([f"Z{row:03d}{col:03d}" for row, col in zip(rows_flat, cols_flat)])
    
    # Create DataFrame
    df = pd.DataFrame({
        'zip': zip_ids,
        'row': rows_flat,
        'col': cols_flat,
        'block_id': block_ids_flat
    })
    
    # Build adjacency more efficiently for large grids using vectorized operations
    # Create index mapping for neighbor lookups
    coord_to_idx = {(r, c): i for i, (r, c) in enumerate(zip(rows_flat, cols_flat))}
    
    adjacency = {}
    for i, (row, col) in enumerate(zip(rows_flat, cols_flat)):
        zip_id = zip_ids[i]
        neighbors = {}
        
        # Check 4-connected neighbors (rook contiguity) - vectorized bounds checking
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in deltas:
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width:
                neighbor_zip = f"Z{nr:03d}{nc:03d}"
                neighbors[neighbor_zip] = 1.0
        
        adjacency[zip_id] = neighbors
    
    # Create weights object
    zip_order = df['zip'].tolist()
    w = weights.W(adjacency, id_order=zip_order)
    w.transform = 'b'  # binary weights
    
    return df, w


def _lattice_grid_loops(width: int, height: int, block_w: int, block_h: int, n_block_cols: int) -> Tuple[pd.DataFrame, weights.W]:
    """Original loop-based lattice generation for small grids."""
    logger.debug("Using loop-based lattice generation")
    
    # Generate grid points using loops (original approach)
    rows = []
    for row in range(height):
        for col in range(width):
            # Create deterministic ZIP ID - documented format Z{row:03d}{col:03d}
            zip_id = f"Z{row:03d}{col:03d}"
            
            # Compute block ID using math.ceil for proper boundary handling
            block_row = math.ceil((row + 1) / block_h) - 1  # 0-indexed
            block_col = math.ceil((col + 1) / block_w) - 1  # 0-indexed
            block_id = block_row * n_block_cols + block_col
            
            rows.append({
                'zip': zip_id,
                'row': row,
                'col': col,
                'block_id': block_id
            })
    
    df = pd.DataFrame(rows)
    
    # Build adjacency dictionary for rook contiguity
    adjacency = {}
    
    for _, point in df.iterrows():
        zip_id = point['zip']
        row, col = point['row'], point['col']
        
        neighbors = {}
        
        # Check 4-connected neighbors (rook contiguity)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_row = row + dr
            neighbor_col = col + dc
            
            # Check bounds
            if 0 <= neighbor_row < height and 0 <= neighbor_col < width:
                neighbor_zip = f"Z{neighbor_row:03d}{neighbor_col:03d}"
                neighbors[neighbor_zip] = 1.0  # weight of 1 for adjacent
        
        adjacency[zip_id] = neighbors
    
    # Create weights object with consistent ordering
    zip_order = df['zip'].tolist()
    w = weights.W(adjacency, id_order=zip_order)
    w.transform = 'b'  # binary weights
    
    return df, w


def make_embeddings(
    df: pd.DataFrame, 
    noise: float = 0.05, 
    emb_prefix: str = "e", 
    random_state: Optional[int] = 42,
    use_float32: bool = True
) -> pd.DataFrame:
    """
    Create simple 2-D embeddings from (row,col) with Gaussian noise for testing.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'row' and 'col'.
    noise : float
        Standard deviation of noise added to each dimension.
    emb_prefix : str
        Prefix for embedding columns; default 'e'.
    random_state : int | None
        Random seed for reproducible noise; default 42.
    use_float32 : bool
        Use float32 for memory efficiency.
        
    Returns
    -------
    pd.DataFrame
        Original df plus columns e0, e1.
        
    Notes
    -----
    Performance: Uses float32 for memory efficiency by default.
    RNG: Uses local Generator to avoid global RNG side effects.
    Guards against division by zero when max_row or max_col is 0.
    """
    if 'row' not in df.columns or 'col' not in df.columns:
        raise ValueError("df must contain 'row' and 'col' columns")
    
    if noise < 0:
        raise ValueError("noise must be non-negative")
    
    df = df.copy()
    
    # Guard against division by zero - use max(1, max_value) to avoid div0
    max_row = max(1, df['row'].max())
    max_col = max(1, df['col'].max())
    
    # FIXED: Use local Generator to avoid global RNG side effects
    rng = np.random.default_rng(random_state)
    
    # Create embeddings based on normalized row/col coordinates with noise
    # Performance optimization: Vectorized computation
    normalized_rows = df['row'].values / max_row
    normalized_cols = df['col'].values / max_col
    
    noise_e0 = rng.normal(0, noise, len(df)) if noise > 0 else 0.0
    noise_e1 = rng.normal(0, noise, len(df)) if noise > 0 else 0.0
    
    e0 = normalized_rows + noise_e0
    e1 = normalized_cols + noise_e1
    
    # Performance optimization: Use float32 by default
    if use_float32:
        e0 = e0.astype(np.float32)
        e1 = e1.astype(np.float32)
    
    df[f'{emb_prefix}0'] = e0
    df[f'{emb_prefix}1'] = e1
    
    logger.info(f"Generated {len(df)} embeddings with noise={noise}, random_state={random_state}, dtype={e0.dtype}")
    
    return df