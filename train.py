"""
TEMPLATE TRAINING SCRIPT
========================
Single consolidated entry point for training.

This script demonstrates the config-driven training pattern:
  1. Hydra loads config from config/train.yaml (composed via defaults list)
  2. Datasets, model, metrics, optimizer, scheduler are instantiated via Hydra
  3. A LightningModule pipeline wraps everything
  4. Lightning Trainer handles the training loop

HOW TO CUSTOMIZE FOR YOUR TASK:
  - Create your dataset  → see src/datasets/template_dataset.py
  - Create your model    → see src/models/template_model.py
  - Create your metrics  → see src/metrics/template_metrics.py
  - Update configs       → see config/datasets/, config/model/, config/metrics/
  - Add preprocessing    → TODO: add any data preprocessing / tokenizer training here
  - Add post-training    → TODO: add model export, analysis, or evaluation here
"""

import warnings

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_datasets, get_dataloaders, move_batch_transforms_to_device
from src.utils.init_utils import setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config):
    """
    Main training entry point.
    All components (datasets, model, metrics, optimizer, scheduler, pipeline,
    trainer) are configured via Hydra YAML files.

    Args:
        config (DictConfig): Hydra experiment config.
    """
    # ── Reproducibility ───────────────────────────────────────────────
    L.seed_everything(config.global_setings.seed)

    # Save full config for reproducibility
    project_config = OmegaConf.to_container(config)

    # ── Experiment setup (save dir, logging) ──────────────────────────
    setup_saving_and_logging(config)

    # ── Datasets ──────────────────────────────────────────────────────
    # TODO: Replace TemplateDataset with your own dataset implementation
    datasets = get_datasets(config)

    # ── DataLoaders ───────────────────────────────────────────────────
    # TODO: Add custom collation or preprocessing here
    dataloaders = get_dataloaders(config, datasets)

    # ── Model ─────────────────────────────────────────────────────────
    # TODO: Pass any additional model parameters (vocab_size, etc.)
    model = instantiate(config.model)

    # Apply model transforms if configured (e.g., weight initialization)
    for transform_config in config.global_setings.get("model_transforms", []):
        instantiate(transform_config, model)

    # ── Metrics ───────────────────────────────────────────────────────
    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            metrics[metric_type].append(instantiate(metric_config))

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    # Read from global_setings (not training_pipeline) — keeps config flat
    # TODO: Adjust optimizer params for your model
    optimizer = instantiate(config.global_setings.optimizer, model.parameters())
    scheduler = instantiate(config.global_setings.scheduler, optimizer)

    # ── Training Pipeline (LightningModule) ───────────────────────────
    training_pipeline = instantiate(
        config.training_pipeline,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
    )

    # Apply batch transforms (e.g., GPU augmentation)
    if config.get("transforms", {}).get("batch_transforms", None) is not None:
        batch_transforms = instantiate(config.transforms.batch_transforms)
        move_batch_transforms_to_device(
            batch_transforms,
            config.trainer.get("accelerator", "auto"),
        )
        training_pipeline.batch_transforms = batch_transforms

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = instantiate(config.trainer)

    for logger in trainer.loggers:
        logger.log_hyperparams(project_config)

    # ── Train ─────────────────────────────────────────────────────────
    # TODO: Add validation before training, resume from checkpoint, etc.
    trainer.fit(
        model=training_pipeline,
        train_dataloaders=dataloaders.get("train"),
        val_dataloaders=dataloaders.get("val"),
    )

    # ── Post-training ─────────────────────────────────────────────────
    # TODO: Add model export, evaluation on test set, visualization, etc.


if __name__ == "__main__":
    main()
