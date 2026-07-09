"""
Uplift Training Pipeline
========================

LightningModule for training and evaluating uplift models.
"""

import logging
from typing import Dict, Any, Optional

import pytorch_lightning as L
import torch
import torch.nn as nn

from src.pipelines.base_pipeline import BasePipeline

logger = logging.getLogger(__name__)


class UpliftPipeline(BasePipeline):
    """
    Training pipeline for uplift models.

    Extends BasePipeline to handle:
    - Separate treatment and control group evaluation
    - Ranking metrics (Qini, AUUC, uplift@k)
    - PEHE computation (if ground-truth CATE available)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        metrics: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            **kwargs,
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step: compute loss on treatment effect estimation.
        """
        loss = self.model.calculate_loss(batch)

        # Log training loss
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Validation step: compute CATE predictions and ranking metrics.
        """
        # Forward pass
        output = self.model.forward(**batch)
        cate_pred = output.get("cate_pred", torch.zeros(batch["features"].size(0)))

        # Update metrics
        if "val" in self.metrics:
            for metric in self.metrics["val"]:
                try:
                    metric.update(
                        cate_pred=cate_pred,
                        outcome=batch.get("outcome"),
                        treatment=batch.get("treatment"),
                        cate_true=batch.get("cate_true"),
                    )
                except Exception as e:
                    logger.warning(f"Metric {metric.name} failed: {e}")

    def on_validation_epoch_end(self) -> None:
        """
        Compute and log validation metrics.
        """
        if "val" not in self.metrics:
            return

        for metric in self.metrics["val"]:
            try:
                value = metric.compute()
                metric_name = f"val_{metric.name}"

                if isinstance(value, dict):
                    # Multi-value metric (e.g., ranking_correlation)
                    for k, v in value.items():
                        self.log(
                            f"{metric_name}_{k}",
                            v,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=True,
                            logger=True,
                        )
                else:
                    self.log(
                        metric_name,
                        value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                    )

                # Reset metric for next epoch
                metric.reset()
            except Exception as e:
                logger.warning(f"Metric {metric.name} computation failed: {e}")


class InferencePipeline(L.LightningModule):
    """
    Inference-only pipeline for frozen models (e.g., CausalPFN).

    No training loop; only validation and inference.
    """

    def __init__(
        self,
        model: nn.Module,
        metrics: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.metrics = metrics or {}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Run inference and compute metrics (no training).
        """
        output = self.model.forward(**batch)
        cate_pred = output.get("cate_pred", torch.zeros(batch["features"].size(0)))

        if "inference" in self.metrics:
            for metric in self.metrics["inference"]:
                try:
                    metric.update(
                        cate_pred=cate_pred,
                        outcome=batch.get("outcome"),
                        treatment=batch.get("treatment"),
                        cate_true=batch.get("cate_true"),
                    )
                except Exception as e:
                    logger.warning(f"Metric {metric.name} failed: {e}")

    def on_validation_epoch_end(self) -> None:
        """
        Compute and log inference metrics.
        """
        if "inference" not in self.metrics:
            return

        for metric in self.metrics["inference"]:
            try:
                value = metric.compute()
                metric_name = f"val_{metric.name}"

                if isinstance(value, dict):
                    for k, v in value.items():
                        self.log(
                            f"{metric_name}_{k}",
                            v,
                            on_step=False,
                            on_epoch=True,
                            prog_bar=True,
                            logger=True,
                        )
                else:
                    self.log(
                        metric_name,
                        value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                    )

                metric.reset()
            except Exception as e:
                logger.warning(f"Metric {metric.name} computation failed: {e}")

    def forward(self, **batch) -> Dict[str, torch.Tensor]:
        """
        Forward pass (inference).
        """
        return self.model.forward(**batch)
