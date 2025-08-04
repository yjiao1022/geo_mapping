import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from typing import Dict, Any, Tuple, List


def extract_temporal_features(df: pd.DataFrame, window: int = 30) -> np.ndarray:
    """
    Extract temporal features from timeseries data.
    
    Args:
        df (pd.DataFrame): Input dataframe with columns ['geo_id', 'date'] + feature columns
        window (int): Rolling window size for feature extraction (currently unused, for future extensions)
        
    Returns:
        np.ndarray: Feature array of shape (n_geos, n_features, time_steps)
    """
    # Identify feature columns (all columns except geo_id and date)
    feature_cols = [col for col in df.columns if col not in ['geo_id', 'date']]
    n_features = len(feature_cols)
    
    if n_features == 0:
        raise ValueError("No feature columns found. Expected columns other than 'geo_id' and 'date'")
    
    # Check for empty dataframe
    if len(df) == 0:
        raise ValueError("Empty dataframe provided. Cannot extract features from empty data.")
    
    # Get unique geos and sort for consistent ordering
    geo_ids = sorted(df['geo_id'].unique())
    n_geos = len(geo_ids)
    
    # Get time steps
    dates = sorted(df['date'].unique())
    time_steps = len(dates)
    
    # Initialize feature array
    features = np.zeros((n_geos, n_features, time_steps))
    
    # Fill feature array
    for feat_idx, feature_col in enumerate(feature_cols):
        # Pivot to get time series for each geo for this feature
        feature_pivot = df.pivot(index='date', columns='geo_id', values=feature_col).fillna(0)
        
        for geo_idx, geo_id in enumerate(geo_ids):
            if geo_id in feature_pivot.columns:
                features[geo_idx, feat_idx, :] = feature_pivot[geo_id].values
    
    return features


def create_graph_adjacency_matrix(
    geo_df: pd.DataFrame, 
    method: str = "knn", 
    k: int = 8
) -> np.ndarray:
    """
    Create spatial adjacency matrix for geographic units.
    
    Args:
        geo_df (pd.DataFrame): DataFrame with columns ['geo_id', 'latitude', 'longitude']
        method (str): Method for creating adjacency ('knn' or 'radius')
        k (int): Number of nearest neighbors for knn method
        
    Returns:
        np.ndarray: Binary adjacency matrix of shape (n_geos, n_geos)
    """
    if method not in ["knn", "radius"]:
        raise ValueError(f"Unsupported method: {method}. Use 'knn' or 'radius'")
    
    if method == "knn" and k <= 0:
        raise ValueError(f"k must be positive for knn method, got k={k}")
    
    # Extract coordinates
    coords = geo_df[['latitude', 'longitude']].values
    n_geos = len(coords)
    
    if method == "knn":
        # Use k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Create adjacency matrix
        adj_matrix = np.zeros((n_geos, n_geos))
        for i in range(n_geos):
            # Skip first neighbor (self) and connect to k nearest neighbors
            for j in indices[i][1:]:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  # Make symmetric
    
    return adj_matrix.astype(np.float32)


def extract_model_embeddings(
    checkpoint_path: str,
    feature_array: np.ndarray,
    adj_matrix: np.ndarray,
    batch_size: int = 128
) -> np.ndarray:
    """
    Extract embeddings using PyTorch Lightning trainer.predict() method.
    
    Note: For graph operations, we process all nodes at once since adjacency
    matrix operations require the full graph structure.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        feature_array (np.ndarray): Input features of shape (n_geos, in_channels, time_steps)
        adj_matrix (np.ndarray): Adjacency matrix of shape (n_geos, n_geos)
        batch_size (int): Batch size for prediction (ignored for graph operations)
        
    Returns:
        np.ndarray: Embeddings of shape (n_geos, embedding_dim)
    """
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset
    from embedding.temporal_graph import GMAEmbeddingLightningModule
    
    # Load model from checkpoint
    model = GMAEmbeddingLightningModule.load_from_checkpoint(checkpoint_path)
    
    # For graph operations, we must process all nodes together
    # Create single-batch dataset with all nodes
    feature_tensor = torch.tensor(feature_array, dtype=torch.float32)
    dummy_targets = torch.zeros(feature_array.shape[0])
    
    dataset = TensorDataset(feature_tensor, dummy_targets)
    dataloader = DataLoader(dataset, batch_size=feature_array.shape[0], shuffle=False)
    
    # Use Lightning trainer for prediction, force CPU to match training
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, accelerator='cpu')
    predictions = trainer.predict(model, dataloader)
    
    # Should be a single batch containing all embeddings
    embeddings = predictions[0]
    
    return embeddings.numpy()