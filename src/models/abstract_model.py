"""
TEMPLATE MODEL — Abstract interface
====================================
All models must inherit from AbstractModel and implement calculate_loss.

TEMPLATE ADAPTATION:
  Subclass AbstractModel and implement:
    - calculate_loss(batch) -> torch.Tensor  (REQUIRED — used in training)
    - forward(**batch) -> dict               (RECOMMENDED — used in validation)
"""

from typing import Tuple

import torch
import torch.nn as nn


class AbstractModel(nn.Module):
    def __init__(self, config: dict):
        super(AbstractModel, self).__init__()
        self.config = config

    def calculate_loss(self, batch: dict) -> torch.Tensor:
        """
        Calculate the loss for the given batch.

        This is the primary method called during training_step.
        It receives the full batch dict (post-collation).

        Args:
            batch (dict): A dictionary containing batched input data and labels.
                         Keys match those returned by your collate_fn.

        Returns:
            torch.Tensor: The scalar loss for the batch.
        """
        raise NotImplementedError("calculate_loss method must be implemented.")

    def forward(self, **batch) -> dict:
        """
        Forward pass for validation / inference.

        Returns a dict of predictions (e.g., {"logits": ..., "probs": ...})
        so that metrics can extract what they need.

        Args:
            **batch: Unpacked batch dict keys.

        Returns:
            dict: Model outputs (e.g., logits, embeddings, etc.).
        """
        raise NotImplementedError("forward method must be implemented.")

    # ── Optional: seq2seq methods ────────────────────────────────────────
    # Uncomment and implement if your task requires encoder-decoder inference
    # (e.g., beam search, autoregressive generation).

    # def encode(self, batch: dict) -> torch.Tensor:
    #     """
    #     Encode input (context / history) and return a fixed representation.
    #     Used to precompute encoder outputs before decoding.
    #     """
    #     raise NotImplementedError("encode method must be implemented.")

    # def decode(
    #     self,
    #     batch: dict,
    #     digit=None,
    #     past_key_values=None,
    #     use_cache=False,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Decode one step given encoder hidden states and optional cache.
    #     """
    #     raise NotImplementedError("decode method must be implemented.")
