"""
BASE TRAINING PIPELINE
======================
Generic LightningModule that wires together model, optimizer, scheduler,
and metrics.

This is the core training loop for the template. It delegates to:
  - model.calculate_loss(batch)   → training step
  - model.forward(**batch)        → validation step (predictions)

TODO: Customize for your task:
  - Add teacher forcing or scheduled sampling
  - Add multi-task losses
  - Add gradient accumulation custom logic
  - Override on_before_optimizer_step for gradient clipping strategies
"""

from typing import Dict, List

import torch
import lightning as L

from src.models.abstract_model import AbstractModel
from src.metrics.base_metric import BaseMetric


class BasePipeline(L.LightningModule):
    def __init__(
        self,
        model: AbstractModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        metrics: Dict[str, List[BaseMetric]] = None,
        **config,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        # Store metrics as ModuleDict so Lightning moves them to device
        metrics = metrics or {}
        self.train_metrics = torch.nn.ModuleDict(
            {m.name: m for m in metrics.get("train", [])}
        )
        self.evaluation_metrics = torch.nn.ModuleDict(
            {m.name: m for m in metrics.get("inference", [])}
        )

    # ── Training ─────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        loss = self.model.calculate_loss(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        # Update training metrics (optional)
        if self.train_metrics:
            with torch.no_grad():
                preds = self.model(**batch)
                for metric in self.train_metrics.values():
                    metric.update(preds=preds, **batch)

        return loss

    def on_train_epoch_end(self):
        for key, metric in self.train_metrics.items():
            value = metric.compute()
            self.log(f"train_{key}", value, prog_bar=True, sync_dist=True)
            metric.reset()

    # ── Validation ───────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        # TODO: Customize inference logic here (e.g., beam search, sampling)
        preds = self.model(**batch)

        for metric in self.evaluation_metrics.values():
            metric.update(preds=preds, **batch)

    def on_validation_epoch_end(self):
        for key, metric in self.evaluation_metrics.items():
            value = metric.compute()
            self.log(f"val_{key}", value, prog_bar=True, sync_dist=True)
            metric.reset()

    # ── Optimizer / Scheduler ────────────────────────────────────────────

    def configure_optimizers(self):
        if self.scheduler is not None:
            return [self.optimizer], [self.scheduler]
        return self.optimizer

    # ── Batch Transforms (optional) ──────────────────────────────────────

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        """
        Apply batch-level transforms (e.g., augmentation, device transfer).
        Override in subclass if you need custom logic.
        """
        if hasattr(self, "batch_transforms") and self.batch_transforms is not None:
            for transform_type, transforms in self.batch_transforms.items():
                if transform_type in batch:
                    for transform_name, transform in transforms.items():
                        batch[transform_type] = transform(batch[transform_type])
        return batch
