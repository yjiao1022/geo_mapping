import torch
import torch.nn as nn
import pytorch_lightning as pl


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

    Args:
      in_channels (int): number of input series per geo
      hidden_channels (int): conv output channels
      kernel_size (int): conv window size
      embedding_dim (int): output embedding dimension
      adj_matrix (torch.Tensor): sparse adjacency matrix
      lr (float): learning rate
      loss_type (str): 'reconstruction' | 'contrastive' | 'forecast'
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int,
                 embedding_dim: int, adj_matrix: torch.Tensor,
                 lr: float, loss_type: str = 'reconstruction'):
        super().__init__()
        self.save_hyperparameters()
        self.temporal_net = TemporalConvNet(in_channels, hidden_channels, kernel_size)
        self.graph_encoder = GraphSAGEEncoder(hidden_channels, embedding_dim, adj_matrix)
        self.loss_type = loss_type
        if loss_type == 'reconstruction':
            self.loss_fn = nn.MSELoss()
            # Add decoder for reconstruction
            self.decoder = nn.Linear(embedding_dim, hidden_channels)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_feats = self.temporal_net(x)
        emb = self.graph_encoder(t_feats)
        return emb
    
    def _compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Helper method to compute loss for both training and validation."""
        t_feats = self.temporal_net(x)
        emb = self.graph_encoder(t_feats)
        if self.loss_type == 'reconstruction':
            recon = self.decoder(emb)
            return self.loss_fn(recon, t_feats)
        else:
            return self.loss_fn(emb, emb)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self._compute_loss(x)
        self.log('train_recon_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        loss = self._compute_loss(x)
        self.log('val_recon_loss', loss, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        """Prediction step for Lightning trainer.predict()"""
        x, _ = batch
        return self(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)