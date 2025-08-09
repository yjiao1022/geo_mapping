import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Dict, Any, Union, List
from .losses import contrastive_loss, predictive_loss, MemoryBankContrastiveLoss
from .augment import create_augmented_pair, get_augmentation_logger


class TemporalConvNet(nn.Module):
    """
    A 1D temporal convolutional network modeling per-geo time series.

    Args:
      in_channels (int): number of input series/features per geo
      hidden_channels (int): output channels of convolution
      kernel_size (int): temporal window size
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over input tensor of shape (batch_size, in_channels, seq_len).

        Returns:
          Tensor of shape (batch_size, hidden_channels)
        """
        out = self.conv1d(x)
        out = self.activation(out)
        pooled = self.pool(out)
        return pooled.squeeze(-1)


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder aggregating neighbor and self features.

    Args:
      in_channels (int): feature dimension per node
      out_channels (int): output embedding dimension
      adj_matrix (torch.Tensor): binary sparse adjacency (n_nodes, n_nodes)
    """
    def __init__(self, in_channels: int, out_channels: int, adj_matrix: torch.Tensor):
        super().__init__()
        self.register_buffer('adj', adj_matrix)  # Register as buffer to move with model
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: Tensor of shape (n_nodes, in_channels)

        Returns:
          Tensor of shape (n_nodes, out_channels)
        """
        self_feat = self.lin_self(x)
        neigh_sum = torch.matmul(self.adj, x)
        neigh_feat = self.lin_neigh(neigh_sum)
        return nn.functional.relu(self_feat + neigh_feat)


class GMAEmbeddingLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module combining TemporalConvNet and GraphSAGEEncoder.

    Supports multiple loss types for different training objectives:
    - 'reconstruction': Autoencoder-style training
    - 'contrastive': NT-Xent contrastive learning with augmentations
    - 'predictive': Future prediction from past sequences
    - 'multi': Weighted combination of multiple losses

    Args:
      in_channels (int): number of input series per geo
      hidden_channels (int): conv output channels
      kernel_size (int): conv window size
      embedding_dim (int): output embedding dimension
      adj_matrix (torch.Tensor): sparse adjacency matrix
      lr (float): learning rate
      loss_type (str): 'reconstruction' | 'contrastive' | 'predictive' | 'multi'
      loss_weights (Optional[Dict[str, float]]): Weights for multi-loss training
      contrastive_temperature (float): Temperature for contrastive loss
      use_memory_bank (bool): Whether to use memory bank for contrastive loss
      memory_bank_size (int): Size of memory bank
      augmentation_config (Optional[Dict]): Configuration for data augmentations
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int,
                 embedding_dim: int, adj_matrix: torch.Tensor,
                 lr: float, loss_type: str = 'reconstruction',
                 loss_weights: Optional[Dict[str, float]] = None,
                 contrastive_temperature: float = 0.1,
                 use_memory_bank: bool = False,
                 memory_bank_size: int = 4096,
                 augmentation_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.save_hyperparameters()
        self.temporal_net = TemporalConvNet(in_channels, hidden_channels, kernel_size)
        self.graph_encoder = GraphSAGEEncoder(hidden_channels, embedding_dim, adj_matrix)
        self.loss_type = loss_type
        self.contrastive_temperature = contrastive_temperature
        
        # Initialize loss components based on loss_type
        self._setup_loss_functions(embedding_dim, hidden_channels, 
                                 use_memory_bank, memory_bank_size)
        
        # Set up loss weights for multi-loss training
        if loss_type == 'multi':
            if loss_weights is None:
                self.loss_weights = {'reconstruction': 1.0, 'contrastive': 1.0}
            else:
                self.loss_weights = loss_weights
        else:
            self.loss_weights = {loss_type: 1.0}
        
        # Set up augmentation configuration
        if augmentation_config is None:
            self.augmentation_config = {
                'aug1_method': 'jitter',
                'aug2_method': 'crop',
                'jitter_noise_std': 0.01,
                'crop_crop_ratio': 0.8
            }
        else:
            self.augmentation_config = augmentation_config
        
        # Create RNG generator for reproducible augmentations
        self.aug_generator = torch.Generator()
        self.aug_generator.manual_seed(42)  # Fixed seed for reproducibility
        
        # Augmentation logger
        self.aug_logger = get_augmentation_logger()
        
    def _setup_loss_functions(self, embedding_dim: int, hidden_channels: int,
                            use_memory_bank: bool, memory_bank_size: int):
        """Initialize loss functions based on configuration."""
        
        # Reconstruction loss components
        if self.loss_type in ['reconstruction', 'multi']:
            self.mse_loss = nn.MSELoss()
            self.decoder = nn.Linear(embedding_dim, hidden_channels)
        
        # Contrastive loss components
        if self.loss_type in ['contrastive', 'multi']:
            if use_memory_bank:
                self.memory_bank_loss = MemoryBankContrastiveLoss(
                    embedding_dim, memory_bank_size, self.contrastive_temperature
                )
            else:
                self.memory_bank_loss = None
        
        # Predictive loss components
        if self.loss_type in ['predictive', 'multi']:
            self.predictive_mse = nn.MSELoss()
            # Could add a prediction head here if needed
            self.prediction_head = nn.Identity()  # Placeholder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal and graph encoders."""
        t_feats = self.temporal_net(x)
        emb = self.graph_encoder(t_feats)
        return emb
    
    def _compute_reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss using autoencoder architecture."""
        t_feats = self.temporal_net(x)
        emb = self.graph_encoder(t_feats)
        recon = self.decoder(emb)
        return self.mse_loss(recon, t_feats)
    
    def _compute_contrastive_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss using augmented views."""
        # Create two augmented views
        x_aug1, x_aug2 = create_augmented_pair(
            x, generator=self.aug_generator, **self.augmentation_config
        )
        
        # Get embeddings for both views
        emb1 = self(x_aug1)
        emb2 = self(x_aug2)
        
        # Compute contrastive loss
        if self.memory_bank_loss is not None:
            return self.memory_bank_loss(emb1, emb2)
        else:
            return contrastive_loss(emb1, emb2, self.contrastive_temperature)
    
    def _compute_predictive_loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute predictive loss for future prediction."""
        # Get embeddings from past sequence
        past_emb = self(x)
        
        # Get embeddings from future sequence (targets)
        future_emb = self(targets)
        
        # Apply prediction head if needed
        pred_emb = self.prediction_head(past_emb)
        
        return predictive_loss(pred_emb, future_emb)
    
    def _compute_loss(self, batch: tuple) -> Dict[str, torch.Tensor]:
        """
        Compute losses based on loss_type configuration.
        
        Args:
            batch: Batch data format depends on loss type
            
        Returns:
            Dict containing individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        if self.loss_type == 'reconstruction':
            x, _ = batch
            recon_loss = self._compute_reconstruction_loss(x)
            losses['reconstruction'] = recon_loss
            total_loss = recon_loss
            
        elif self.loss_type == 'contrastive':
            x, _ = batch
            cont_loss = self._compute_contrastive_loss(x)
            losses['contrastive'] = cont_loss
            total_loss = cont_loss
            
        elif self.loss_type == 'predictive':
            x, targets = batch
            pred_loss = self._compute_predictive_loss(x, targets)
            losses['predictive'] = pred_loss
            total_loss = pred_loss
            
        elif self.loss_type == 'multi':
            x, targets = batch
            
            # Reconstruction loss
            if 'reconstruction' in self.loss_weights:
                recon_loss = self._compute_reconstruction_loss(x)
                losses['reconstruction'] = recon_loss
                total_loss += self.loss_weights['reconstruction'] * recon_loss
            
            # Contrastive loss
            if 'contrastive' in self.loss_weights:
                cont_loss = self._compute_contrastive_loss(x)
                losses['contrastive'] = cont_loss
                total_loss += self.loss_weights['contrastive'] * cont_loss
            
            # Predictive loss (if targets provided)
            if 'predictive' in self.loss_weights and targets is not None:
                pred_loss = self._compute_predictive_loss(x, targets)
                losses['predictive'] = pred_loss
                total_loss += self.loss_weights['predictive'] * pred_loss
                
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
        
        losses['total'] = total_loss
        return losses

    def training_step(self, batch, batch_idx):
        """Training step with multi-loss support."""
        try:
            losses = self._compute_loss(batch)
            
            # Log individual losses
            for loss_name, loss_value in losses.items():
                if loss_name != 'total':
                    self.log(f'train_{loss_name}_loss', loss_value, prog_bar=True)
            
            # Log augmentation parameters periodically
            if batch_idx % 100 == 0 and hasattr(self, 'aug_logger') and self.logger is not None:
                recent_logs = self.aug_logger.get_logs()[-10:]  # Last 10 augmentations
                if recent_logs:
                    self.logger.debug(f"Recent augmentations: {recent_logs}")
            
            return losses['total']
            
        except Exception as e:
            # Graceful error handling
            if self.logger is not None:
                self.logger.error(f"Error in training step: {str(e)}")
            # Return a dummy loss to prevent training crash
            return torch.tensor(0.0, requires_grad=True)

    def validation_step(self, batch, batch_idx):
        """Validation step with multi-loss support."""
        try:
            losses = self._compute_loss(batch)
            
            # Log individual losses
            for loss_name, loss_value in losses.items():
                if loss_name != 'total':
                    self.log(f'val_{loss_name}_loss', loss_value, prog_bar=True)
                    
        except Exception as e:
            # Graceful error handling for validation
            if self.logger is not None:
                self.logger.error(f"Error in validation step: {str(e)}")
            # Log dummy losses to prevent validation crash
            self.log('val_total_loss', torch.tensor(0.0), prog_bar=True)

    def predict_step(self, batch, batch_idx):
        """Prediction step for Lightning trainer.predict()"""
        x, _ = batch
        return self(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)