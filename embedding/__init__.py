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

__all__ = [
    'TemporalConvNet',
    'GraphSAGEEncoder',
    'GMAEmbeddingLightningModule',
    'extract_temporal_features',
    'create_graph_adjacency_matrix',
    'extract_model_embeddings'
]