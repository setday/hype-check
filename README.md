# ML Project Template

A config-driven ML project template built with **Hydra** + **PyTorch Lightning** + **torchmetrics**.

Supports rapid prototyping and experimentation through modular component design: swap datasets, models, metrics, and pipelines by editing YAML configs — no code changes needed.

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the template example (synthetic data + MLP classifier)
python train.py
```

**Override settings from CLI** (Hydra):
```bash
python train.py trainer.max_epochs=3 trainer.fast_dev_run=true
python train.py model=my_model datasets=my_dataset
```

## Architecture

```
train.py              → Single training entrypoint (Hydra + Lightning)
config/
  train.yaml           → Root config (defaults list composes sub-configs)
  model/               → Model hyperparameters
  datasets/            → Dataset configurations
  metrics/             → Metric configurations
  dataloaders/         → DataLoader configurations
  training_pipeline/   → LightningModule pipeline config
src/
  datasets/            → BaseDataset + template implementations
  models/              → AbstractModel + template implementations
  pipelines/           → BasePipeline (LightningModule training loop)
  metrics/             → BaseMetric + template implementations
  logger/              → Logging setup
  utils/               → Experiment init, I/O utilities
```

## How to Adapt for Your Task

The template ships with a working synthetic-data example (`TemplateDataset` → `TemplateMLP` → `AccuracyMetric`). Replace each component step by step:

### 1. Create your dataset

See `src/datasets/template_dataset.py` for a complete annotated example:
- Subclass `BaseDataset`
- Build `self._index` as a `list[dict]` in `__init__`
- Each dict's keys must match your `collate_fn` and model

### 2. Create your model

See `src/models/template_model.py`:
- Subclass `AbstractModel`
- Implement `calculate_loss(batch) → Tensor`
- Implement `forward(**batch) → dict`

### 3. Define your metrics

See `src/metrics/template_metrics.py`:
- Subclass `BaseMetric` (which extends `torchmetrics.Metric`)
- Implement `update(**batch)` and `compute()`

### 4. Wire it up in config

Add your configs to `config/model/`, `config/datasets/`, `config/metrics/`, then update the `defaults` list in `config/train.yaml`.

## Key Features

- **Hydra config composition** — modular YAML files, CLI overrides, nested configs
- **Lightning training** — `training_step`/`validation_step`/`configure_optimizers` in `BasePipeline`
- **Abstract interfaces** — `BaseDataset`, `AbstractModel`, `BaseMetric` for extensibility
- **Reproducible experiments** — seed setting, config dumps, checkpointing
- **Batch transforms** — GPU-side data augmentation via `on_after_batch_transfer`
- **W&B / TensorBoard** — swappable loggers via config

## Requirements

- Python 3.10+
- CUDA/cuDNN for GPU training (optional, auto-detected)

## Repository Structure

```
├── train.py                   # Single training entry point
├── config/
│   ├── train.yaml             # Root Hydra config
│   ├── model/                 # Model configs
│   ├── datasets/              # Dataset configs
│   ├── metrics/               # Metric configs
│   ├── dataloaders/           # DataLoader configs
│   └── training_pipeline/     # Pipeline configs
└── src/
    ├── datasets/              # BaseDataset + custom datasets
    ├── models/                # AbstractModel + custom models
    ├── pipelines/             # BasePipeline (LightningModule)
    ├── metrics/               # BaseMetric + custom metrics
    ├── logger/                # Logging infrastructure
    └── utils/                 # Experiment setup, I/O utilities
```

## Contributing

PRs, issues, and experiments are welcome. For substantial changes, open an issue first to discuss design.

## License

MIT — see `LICENSE`.
