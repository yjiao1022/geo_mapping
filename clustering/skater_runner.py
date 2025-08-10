"""
SKATER regionalization wrapper with configuration support.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, TypedDict
from libpysal import weights

logger = logging.getLogger(__name__)


class SkaterConfig(TypedDict):
    """Configuration for SKATER algorithm."""
    n_clusters: int
    use_attributes_in_skater: bool
    random_state: Optional[int]


def run_skater(
    df: pd.DataFrame, 
    w: weights.W, 
    cfg: SkaterConfig, 
    attr_cols: Optional[List[str]] = None, 
    emb_prefix: str = "e",
    use_float32: bool = True,
    subsample_quality_debug: bool = False
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Run SKATER and return a mapping DataFrame and labels aligned to df order.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'zip' and embedding columns (e0..e{d-1}); may include attributes.
    w : weights.W
        Spatial graph aligned to df['zip'] via reorder_w_to_zip_order.
    cfg : SkaterConfig
        Algorithm parameters.
    attr_cols : list[str] | None
        Attribute columns from df to include in features.
    emb_prefix : str
        Prefix for embedding columns; default 'e'.
    use_float32 : bool
        Cast feature matrix to float32 to reduce memory usage.
    subsample_quality_debug : bool
        Subsample embedding vectors for development/debug quality metrics.
        
    Returns
    -------
    mapping : pd.DataFrame
        Columns ['zip','unit_id'] in the same order as df.
    labels : np.ndarray
        Cluster IDs (0..K-1) aligned to df rows.
        
    Raises
    ------
    ValueError
        If no embedding columns are present or n_clusters is invalid.
        
    Notes
    -----
    Uses GDF + attrs_name API for spopt.skater compatibility.
    Performance: Casts to float32 for memory efficiency, optional quality subsampling.
    """
    try:
        from spopt.region import Skater
    except ImportError:
        raise ImportError("spopt is required for SKATER clustering")
    
    try:
        import geopandas as gpd
    except ImportError:
        # Create a minimal GeoDataFrame-like object if geopandas not available
        logger.warning("geopandas not available - using DataFrame directly")
        gpd = None
    
    # Validate inputs
    if cfg['n_clusters'] < 1:
        raise ValueError("n_clusters must be >= 1")
    
    if len(df) < cfg['n_clusters']:
        raise ValueError("n_clusters cannot exceed number of observations")
    
    # Check for duplicate ZIPs
    if df['zip'].duplicated().any():
        duplicated_zips = df.loc[df['zip'].duplicated(), 'zip'].unique()
        raise ValueError(f"Duplicate ZIPs found in DataFrame: {duplicated_zips}")
    
    # Find embedding columns
    emb_cols = [col for col in df.columns if col.startswith(emb_prefix) and col[len(emb_prefix):].isdigit()]
    if not emb_cols:
        raise ValueError(f"No embedding columns found with prefix '{emb_prefix}'")
    
    # Sort embedding columns numerically
    emb_cols = sorted(emb_cols, key=lambda x: int(x[len(emb_prefix):]))
    logger.info(f"Using {len(emb_cols)} embedding columns")
    
    # FIXED: Validate ALL embedding columns are numeric and finite
    embedding_data = df[emb_cols]
    if not all(np.issubdtype(dt, np.number) for dt in embedding_data.dtypes):
        raise ValueError("All embedding columns must be numeric")
    
    if not np.isfinite(embedding_data.values).all():
        raise ValueError("Embedding columns must be finite (no NaN/Inf values)")
    
    # Build feature matrix
    feature_cols = emb_cols.copy()
    if cfg['use_attributes_in_skater'] and attr_cols:
        # Validate attribute columns exist
        missing_attrs = [col for col in attr_cols if col not in df.columns]
        if missing_attrs:
            raise ValueError(f"Missing attribute columns: {missing_attrs}")
        
        # Validate attributes are numeric and finite
        attr_data = df[attr_cols]
        for col in attr_cols:
            if not np.issubdtype(attr_data[col].dtype, np.number):
                raise ValueError(f"Attribute column '{col}' must be numeric")
        
        if not np.isfinite(attr_data.values).all():
            raise ValueError("Attribute columns must be finite (no NaN/Inf values)")
        
        feature_cols.extend(attr_cols)
        logger.info(f"Using {len(attr_cols)} additional attribute columns")
    
    # Validate W alignment
    if len(w.id_order) != len(df):
        raise ValueError("Weights graph size doesn't match DataFrame")
    
    if list(w.id_order) != list(df['zip'].astype(str)):
        raise ValueError("Weights id_order doesn't match df['zip'] order")
    
    # FIXED: Performance optimization with correct memory calculation
    original_features = df[feature_cols].to_numpy()
    original_nbytes = original_features.nbytes
    
    if use_float32:
        feature_matrix = original_features.astype(np.float32)
        saved_bytes = original_nbytes - feature_matrix.nbytes
        if saved_bytes > 1024**2:  # Only log if > 1MB saved
            logger.info(f"Cast features to float32, saved ~{saved_bytes/1024**2:.1f} MB memory")
    else:
        feature_matrix = original_features
    
    # Prepare data for SKATER - use GDF + attrs_name API for spopt compatibility
    if gpd is not None:
        # Create GeoDataFrame with feature matrix (without actual geometry for transparency)
        feature_df = pd.DataFrame(feature_matrix, columns=feature_cols)
        gdf = gpd.GeoDataFrame(feature_df)
        logger.debug("Using GeoDataFrame without geometry for spopt compatibility")
    else:
        gdf = pd.DataFrame(feature_matrix, columns=feature_cols)
    
    # Performance debug: Subsample for quality metrics if requested
    if subsample_quality_debug and len(gdf) > 5000:
        logger.info("Subsampling enabled for development - not recommended for production")
        sample_size = min(5000, len(gdf))
        sample_indices = np.random.choice(len(gdf), sample_size, replace=False)
        gdf_sample = gdf.iloc[sample_indices].copy()
        
        # Need to adjust weights matrix for subsampling (complex - skip for now)
        logger.warning("Quality subsampling not implemented for weights matrix")
    
    # Run SKATER with attrs_name parameter for better spopt compatibility
    skater_kwargs = {
        'n_clusters': cfg['n_clusters']
    }
    
    logger.info(f"Running SKATER clustering with {cfg['n_clusters']} clusters...")
    
    # Add random_state if supported and specified
    if cfg['random_state'] is not None:
        try:
            skater = Skater(gdf, w, attrs_name=feature_cols, random_state=cfg['random_state'], **skater_kwargs)
        except TypeError:
            # Fallback if random_state not supported in this version
            logger.warning("random_state not supported in this spopt version")
            skater = Skater(gdf, w, attrs_name=feature_cols, **skater_kwargs)
    else:
        skater = Skater(gdf, w, attrs_name=feature_cols, **skater_kwargs)
    
    skater.solve()
    labels = skater.labels_
    
    # Optionally relabel clusters to contiguous integers starting from 0
    unique_labels_before = np.unique(labels)
    if not np.array_equal(unique_labels_before, np.arange(len(unique_labels_before))):
        logger.info("Relabeling clusters to contiguous integers")
        label_map = {old: new for new, old in enumerate(unique_labels_before)}
        labels = np.array([label_map[label] for label in labels])
    
    # FIXED: Log final cluster count after relabeling
    final_cluster_count = len(np.unique(labels))
    logger.info(f"SKATER completed, created {final_cluster_count} clusters")
    
    # Create mapping DataFrame
    mapping = pd.DataFrame({
        'zip': df['zip'].astype(str),
        'unit_id': labels
    })
    
    return mapping, labels