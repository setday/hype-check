"""
TEMPLATE MODEL
==============
Minimal working example: MLP classifier.

TODO: Replace this with your model architecture.
Steps:
  1. Subclass AbstractModel.
  2. Implement calculate_loss(batch) -> torch.Tensor (required for training).
  3. Implement forward(**batch) -> dict (used for validation metrics).
  4. The batch dict keys come from your collate_fn — ensure consistency.

Example model types you might implement:
  - Transformer (NLP, sequence tasks)     → use nn.Transformer, HuggingFace, etc.
  - CNN (vision)                          → use nn.Conv2d blocks
  - GNN (graph)                           → use torch_geometric
  - Diffusion model (generative)          → noise schedule + UNet
  - Any PyTorch nn.Module subclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.abstract_model import AbstractModel


class TemplateMLP(AbstractModel):
    """
    A simple MLP for synthetic classification data.

    Architecture:
      input_dim → Linear(128) → ReLU → Linear(num_classes) → logits
    """

    def __init__(self, input_dim=20, hidden_dim=64, num_classes=5, **kwargs):
        # AbstractModel expects config= dict — pass kwargs as config
        super().__init__(config=kwargs)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        # TODO: Replace with your model's layers
        # Example for a Transformer:
        #   self.encoder = nn.TransformerEncoder(...)
        #   self.projection = nn.Linear(d_model, num_classes)

    def forward(self, features, **batch):
        """
        Forward pass for validation / inference.

        Args:
            features: FloatTensor of shape (B, input_dim)
            **batch: Additional batch keys (ignored by this model)

        Returns:
            dict with "logits": FloatTensor of shape (B, num_classes)
        """
        logits = self.net(features)
        return {"logits": logits}

    def calculate_loss(self, batch):
        """
        Compute training loss for a batch.

        Args:
            batch (dict): Must contain "features" and "label"
                          (keys come from collate_fn + dataset __getitem__)

        Returns:
            torch.Tensor: scalar loss
        """
        features = batch.get("features")
        labels = batch.get("label")
        logits = self.net(features)

        # TODO: Replace with your loss function
        loss = F.cross_entropy(logits, labels)
        return loss
