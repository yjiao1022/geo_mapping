"""
Loss functions for GMA embedding learning.

This module implements various loss functions for geographic embedding learning:
- NT-Xent contrastive loss with temperature scaling
- Predictive loss for time series forecasting
- Memory bank enhanced contrastive loss with detached negatives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def reconstruction_loss(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    """
    Computes mean squared error between original and reconstructed input.
    
    Args:
        x (torch.Tensor): Original input of shape (B, C, T) or (B, D)
        x_recon (torch.Tensor): Reconstructed input of same shape
        
    Returns:
        torch.Tensor: Scalar loss
    """
    if x.shape != x_recon.shape:
        raise ValueError(f"Input shapes must match: {x.shape} vs {x_recon.shape}")
    
    return F.mse_loss(x_recon, x)


def contrastive_loss(z_i: torch.Tensor, z_j: torch.Tensor, 
                    temperature: float = 0.1) -> torch.Tensor:
    """
    NT-Xent contrastive loss between two augmented views of embeddings.
    
    Based on SimCLR implementation. Creates positive pairs from corresponding 
    indices in z_i and z_j, with all other samples as negatives.
    
    Args:
        z_i (torch.Tensor): View 1, shape (B, D)
        z_j (torch.Tensor): View 2, shape (B, D)
        temperature (float): Scaling factor for logits (default: 0.1)
        
    Returns:
        torch.Tensor: Scalar loss
    """
    batch_size = z_i.size(0)
    
    if z_i.shape != z_j.shape:
        raise ValueError(f"Input shapes must match: {z_i.shape} vs {z_j.shape}")
    
    # Normalize embeddings
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # Concatenate both views
    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.T) / temperature  # (2B, 2B)
    
    # Create positive pair mask - diagonal blocks
    # Positive pairs: (i, i+B) and (i+B, i) 
    pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, 
                          dtype=torch.bool, device=z.device)
    pos_mask[torch.arange(batch_size), torch.arange(batch_size, 2 * batch_size)] = True
    pos_mask[torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)] = True
    
    # Exclude self-similarities (diagonal)
    neg_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    
    # Compute InfoNCE loss
    pos_sim = sim_matrix[pos_mask]  # Positive similarities
    neg_sim = sim_matrix[neg_mask].view(2 * batch_size, -1)  # Negative similarities
    
    # For each sample, compute log-likelihood
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (2B, 1 + 2B-1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
    
    return F.cross_entropy(logits, labels)


def predictive_loss(past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    """
    MSE loss between embedding from past sequence and future target.
    
    Used for self-supervised learning where the model predicts future 
    representations from past observations.
    
    Args:
        past (torch.Tensor): Encoded features from input sequence, shape (B, D)
        future (torch.Tensor): Encoded target to predict, shape (B, D)
        
    Returns:
        torch.Tensor: Scalar loss
    """
    if past.shape != future.shape:
        raise ValueError(f"Input shapes must match: {past.shape} vs {future.shape}")
    
    return F.mse_loss(past, future)


class MemoryBankContrastiveLoss(nn.Module):
    """
    Contrastive loss with memory bank for larger negative sampling.
    
    Maintains a memory bank of past embeddings to provide more negatives
    without requiring larger batch sizes.
    
    Args:
        embedding_dim (int): Dimension of embeddings
        memory_size (int): Size of memory bank (default: 4096)
        temperature (float): Temperature for InfoNCE loss (default: 0.1)
    """
    
    def __init__(self, embedding_dim: int, memory_size: int = 4096, 
                 temperature: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.temperature = temperature
        
        # Initialize memory bank
        self.register_buffer('memory_bank', 
                           torch.randn(memory_size, embedding_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # Normalize memory bank
        self.memory_bank = F.normalize(self.memory_bank, dim=1)
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss with memory bank negatives.
        
        Args:
            z_i (torch.Tensor): View 1, shape (B, D)
            z_j (torch.Tensor): View 2, shape (B, D)
            
        Returns:
            torch.Tensor: Scalar loss
        """
        batch_size = z_i.size(0)
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Positive similarities (between corresponding pairs)
        pos_sim = torch.sum(z_i * z_j, dim=1) / self.temperature  # (B,)
        
        # Get current memory bank state (detached to prevent gradient issues)
        memory_bank_snapshot = self.memory_bank.detach().clone()
        
        # Negative similarities with memory bank
        neg_sim_i = torch.mm(z_i, memory_bank_snapshot.T) / self.temperature  # (B, M)
        neg_sim_j = torch.mm(z_j, memory_bank_snapshot.T) / self.temperature  # (B, M)
        
        # Compute InfoNCE loss for both views
        logits_i = torch.cat([pos_sim.unsqueeze(1), neg_sim_i], dim=1)  # (B, 1+M)
        logits_j = torch.cat([pos_sim.unsqueeze(1), neg_sim_j], dim=1)  # (B, 1+M)
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=z_i.device)
        
        loss_i = F.cross_entropy(logits_i, labels)
        loss_j = F.cross_entropy(logits_j, labels)
        
        # Update memory bank with current batch (detached and after loss computation)
        self._update_memory_bank(z_i.detach())
        self._update_memory_bank(z_j.detach())
        
        return (loss_i + loss_j) / 2
    
    def _update_memory_bank(self, embeddings: torch.Tensor) -> None:
        """
        Update memory bank with new embeddings using circular buffer.
        
        Args:
            embeddings (torch.Tensor): New embeddings to add, shape (B, D)
        """
        batch_size = embeddings.size(0)
        ptr = int(self.memory_ptr)
        
        # Use no_grad to avoid autograd issues with in-place operations
        with torch.no_grad():
            if ptr + batch_size <= self.memory_size:
                # Enough space, update contiguously
                self.memory_bank[ptr:ptr + batch_size].copy_(embeddings)
                self.memory_ptr[0] = (ptr + batch_size) % self.memory_size
            else:
                # Wrap around
                remaining = self.memory_size - ptr
                if remaining > 0:
                    self.memory_bank[ptr:].copy_(embeddings[:remaining])
                if batch_size > remaining:
                    self.memory_bank[:batch_size - remaining].copy_(embeddings[remaining:])
                self.memory_ptr[0] = batch_size - remaining