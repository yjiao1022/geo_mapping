"""
GMA Embedding Module

This module contains the core components for learning geographic embeddings
from spend and mobility time series data.
"""

from .temporal_graph import (
    TemporalConvNet,
    GraphSAGEEncoder, 
    GMAEmbeddingLightningModule
)

from .utils import (
    extract_temporal_features,
    create_graph_adjacency_matrix,
    extract_model_embeddings
)

from .losses import (
    reconstruction_loss,
    contrastive_loss,
    predictive_loss,
    MemoryBankContrastiveLoss
)

from .augment import (
    jitter_series,
    mask_series,
    crop_series,
    augment_series,
    create_augmented_pair,
    get_augmentation_logger,
    AugmentationLogger
)

from .forecast import (
    forecast_horizon_split,
    temporal_split,
    AutoregressiveDataset,
    create_sliding_windows
)

__all__ = [
    # Core models
    'TemporalConvNet',
    'GraphSAGEEncoder',
    'GMAEmbeddingLightningModule',
    
    # Utilities
    'extract_temporal_features',
    'create_graph_adjacency_matrix',
    'extract_model_embeddings',
    
    # Loss functions
    'reconstruction_loss',
    'contrastive_loss',
    'predictive_loss',
    'MemoryBankContrastiveLoss',
    
    # Augmentation
    'jitter_series',
    'mask_series',
    'crop_series',
    'augment_series',
    'create_augmented_pair',
    'get_augmentation_logger',
    'AugmentationLogger',
    
    # Forecasting
    'forecast_horizon_split',
    'temporal_split',
    'AutoregressiveDataset',
    'create_sliding_windows'
]