"""
Synthetic data generators for testing and development.

This module provides deterministic generators for creating test datasets
with known ground truth clustering patterns.
"""

from .lattice import lattice_grid, make_embeddings

__all__ = ['lattice_grid', 'make_embeddings']