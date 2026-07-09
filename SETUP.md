# Research Setup Guide: Hype-Check

## Project Overview

This repository evaluates whether **untuned causal foundation models** (CausalPFN) match tuned uplift learners on ranking metrics and develops training-free adaptations. The work spans three phases:

- **Check**: Compare untuned CausalPFN vs. tuned baselines on ranking quality (Qini, AUUC)
- **Break**: Map failure regimes (small control groups, low conversion, out-of-context samples)
- **Fix**: Develop CausalPFN-Rank with context selection and output re-ranking

## Infrastructure Setup

### 1. Environment & Dependencies

Install dependencies:
```bash
pip install -r requirements.txt
```

**Key libraries**:
- `torch`, `lightning`: Model training & validation
- `hydra-core`, `omegaconf`: Config-driven experiments
- `torchmetrics`: Distributed metric tracking
- `causalml`, `econml`: Causal inference methods
- `pandas`, `scipy`, `numpy`: Data processing
- `matplotlib`, `seaborn`, `plotly`: Visualization

### 2. Data Structure

Datasets are cached locally in `data/raw/{dataset_name}/`:
```
data/
├── raw/
│   ├── hillstrom/
│   │   └── cache/
│   │       └── {hash}.pkl    # Cached dataset split
│   ├── criteo/
│   ├── x5/
│   ├── ihdp/
│   └── acic/
```

**Hybrid caching strategy**:
- First call: Fetch from original source (or generate synthetic if not available)
- Subsequent calls: Load from cache (`.pkl` files)
- No git tracking (added to `.gitignore`)

### 3. Config-Driven Experiments

All experiments are defined via Hydra YAML configs:
```
config/
├── train.yaml                 # Root config (defaults composition)
├── model/                     # Model architecture
│   ├── s_learner.yaml
│   ├── t_learner.yaml
│   └── causalpfn.yaml
├── datasets/                  # Dataset loading
│   ├── hillstrom.yaml
│   ├── criteo.yaml
│   └── x5.yaml
├── metrics/                   # Evaluation metrics
│   └── uplift.yaml
├── dataloaders/               # DataLoader config
│   └── template.yaml
├── training_pipeline/         # Pipeline selection
│   └── base.yaml
└── experiments/               # Experiment templates
    ├── w1_check.yaml          # Untuned vs. tuned comparison
    ├── w2_break.yaml          # Regime mapping
    └── w3_rank.yaml           # CausalPFN-Rank ablations
```

## Data Infrastructure

### Dataset Classes

**UpliftDataset** (base class):
```python
from src.datasets.uplift_dataset import UpliftDataset

index = [
    {"features": array, "treatment": 0/1, "outcome": float, "cate_true": float},
    ...
]
dataset = UpliftDataset(index)
stats = dataset.get_statistics()
```

**Supported datasets**:
- `hillstrom`: Email marketing RCT (5K–64K rows, 10–20 features)
- `criteo`: Uplift modeling challenge (100K+ rows, 20+ features)
- `x5`: Retail dataset (50K rows, 15+ features)
- `ihdp`: Semi-synthetic with ground truth CATE (7K rows, 25 features)
- `acic`: Semi-synthetic causal benchmark (10K rows, 30 features)

### Data Utils

```python
from src.datasets.uplift_data_utils import (
    load_hillstrom_dataset,
    load_criteo_dataset,
    bootstrap_resample,
)

# Load dataset with caching
index = load_hillstrom_dataset(
    limit=10000,
    control_ratio=0.5,  # Downsample control (for regime testing)
    target_rows=5000,   # Resample to exact size
)

# Bootstrap for confidence intervals
resamples = bootstrap_resample(index, n_resamples=100, seed=42)
```

## Model Infrastructure

### Model Classes

**UpliftModel**: Base class for all methods
- `.forward(features, treatment)` → `{"cate_pred": tensor}`
- `.calculate_loss(batch)` → scalar (for training)

**FrozenFoundationModel**: Inference-only wrapper
- No training loop
- Used for pretrained models (CausalPFN, etc.)

### Implemented Baselines

1. **S-Learner** (`SLearnerWrapper`):
   - Single model with treatment as feature
   - `CATE = f(X, T=1) - f(X, T=0)`

2. **T-Learner** (`TLearnerWrapper`):
   - Separate models for treatment and control groups
   - `CATE = f_1(X) - f_0(X)`

3. **CausalPFN** (`CausalPFNModel`):
   - Frozen foundation model
   - Pretrained for causal inference (HF Model Hub)

### Usage Example

```python
from src.models.uplift_model import SLearnerWrapper
from src.models.causalpfn_model import CausalPFNModel

# S-Learner: train on data
model = SLearnerWrapper({
    "params": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
    }
})
model.fit(X_train, T_train, Y_train)
output = model.forward(X_test)  # {"cate_pred": tensor}

# CausalPFN: inference only
model = CausalPFNModel({
    "model_name": "vdblm/causalpfn-v1",
    "device": "cuda",
    "batch_size": 32,
})
output = model.forward(X_test)
```

## Metrics & Evaluation

### Ranking Metrics

```python
from src.metrics.uplift_metrics import (
    QiniMetric,
    AUUCMetric,
    UpliftAtKMetric,
    PEHEMetric,
    RankingCorrelationMetric,
)

# Qini coefficient (primary ranking metric)
qini_metric = QiniMetric()
qini_metric.update(cate_pred=predictions, outcome=outcomes, treatment=treatments)
qini = qini_metric.compute()

# AUUC (cumulative gains)
auuc_metric = AUUCMetric()
auuc_metric.update(...)
auuc = auuc_metric.compute()

# Uplift at k (targeting deciles)
uplift_10_metric = UpliftAtKMetric(k=0.1)
uplift_10 = uplift_10_metric.compute()

# PEHE (prediction error, for semi-synthetic)
pehe_metric = PEHEMetric()
pehe_metric.update(cate_pred=predictions, cate_true=ground_truth)
pehe = pehe_metric.compute()
```

### Metric Configs

Metrics are configured via `config/metrics/uplift.yaml`:
```yaml
train:
  _target_: src.metrics.uplift_metrics.QiniMetric

val:
  - _target_: src.metrics.uplift_metrics.QiniMetric
  - _target_: src.metrics.uplift_metrics.AUUCMetric
  - _target_: src.metrics.uplift_metrics.UpliftAtKMetric
    k: 0.3
```

## Pipeline & Training

### UpliftPipeline (LightningModule)

Coordinates training and evaluation:
- `training_step()`: Compute loss
- `validation_step()`: Compute CATE + metrics
- `on_validation_epoch_end()`: Log metrics

```python
from src.pipelines.uplift_pipeline import UpliftPipeline

pipeline = UpliftPipeline(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    metrics={"val": [qini_metric, auuc_metric]},
)

trainer = L.Trainer(max_epochs=10, accelerator="cuda")
trainer.fit(pipeline, train_dataloaders, val_dataloaders)
```

### InferencePipeline (Frozen Models)

For models that don't train (e.g., CausalPFN):
```python
from src.pipelines.uplift_pipeline import InferencePipeline

pipeline = InferencePipeline(
    model=model,
    metrics={"inference": [qini_metric]},
)

trainer = L.Trainer(limit_val_batches=1.0)
trainer.validate(pipeline, val_dataloaders)
```

## Quick Start

### 1. Smoke Test

Verify the pipeline works end-to-end:
```bash
python scripts/smoke_test.py --dataset hillstrom --models s_learner,causalpfn
```

Output:
```
Loading hillstrom (train split)...
Testing S-Learner...
S-Learner Qini: 0.1234
Testing CausalPFN...
CausalPFN Qini: 0.1890
✓ Smoke test passed!
```

### 2. Run Single Experiment

Override config from CLI:
```bash
python train.py \
  model=s_learner \
  datasets=hillstrom \
  trainer.max_epochs=5 \
  trainer.accelerator=cuda
```

Results saved to: `saved/experiment/{timestamp}/`

### 3. Batch Experiments (via cluster launcher)

```bash
python scripts/launch_grid.py \
  --experiment w1_check \
  --models s_learner,t_learner,causalpfn \
  --datasets hillstrom,criteo \
  --gpus_per_job 1 \
  --time_limit 60
```

This submits SLURM/SGE jobs for all model × dataset combinations.

### 4. Results & Visualization

Aggregate results:
```bash
python scripts/generate_report.py \
  --results_dir results/ \
  --output report.html
```

Generates:
- Summary metrics table (Qini × model × dataset)
- Qini curves (top 30% of population)
- Regime breakdowns (control ratio, conversion rate)
- Ranking vs. accuracy correlation (H2 test)

## Project Structure

```
hype-check/
├── src/
│   ├── datasets/
│   │   ├── uplift_dataset.py           # Base class for uplift data
│   │   ├── uplift_data_utils.py        # Load/cache/resample functions
│   │   ├── base_dataset.py             # Template base class
│   │   └── collate.py
│   ├── models/
│   │   ├── uplift_model.py             # UpliftModel, FrozenFoundationModel
│   │   ├── causalpfn_model.py          # CausalPFN wrapper
│   │   ├── abstract_model.py           # Template abstract class
│   │   └── template_model.py
│   ├── metrics/
│   │   ├── uplift_metrics.py           # Qini, AUUC, PEHE, etc.
│   │   ├── base_metric.py              # Template base class
│   │   └── template_metrics.py
│   ├── pipelines/
│   │   ├── uplift_pipeline.py          # UpliftPipeline, InferencePipeline
│   │   └── base_pipeline.py            # Template LightningModule
│   ├── logger/
│   ├── utils/
│   └── experiments/                    # TBD: experiment runner harness
├── config/
│   ├── train.yaml                      # Root config (Hydra)
│   ├── model/                          # Model configs
│   │   ├── s_learner.yaml
│   │   ├── t_learner.yaml
│   │   └── causalpfn.yaml
│   ├── datasets/                       # Dataset configs
│   │   ├── hillstrom.yaml
│   │   ├── criteo.yaml
│   │   └── x5.yaml
│   ├── metrics/
│   │   └── uplift.yaml
│   ├── dataloaders/
│   ├── training_pipeline/
│   └── experiments/                    # TBD: experiment specs
├── scripts/
│   ├── smoke_test.py                   # End-to-end validation
│   ├── launch_grid.py                  # TBD: SLURM/SGE job launcher
│   ├── plot_results.py                 # TBD: visualization
│   └── generate_report.py              # TBD: result aggregation
├── train.py                            # Main entry point (Hydra + Lightning)
├── requirements.txt
├── README.md
└── TASK.md                             # Project goals and hypotheses
```

## Key Hypotheses

1. **H1**: Untuned CausalPFN matches tuned learners on semi-synthetic accuracy (PEHE) but not on real ranking (Qini)
2. **H2**: Ranking by accuracy correlates weakly with ranking by targeting value
3. **H3**: Advantage breaks in three regimes: out-of-context data, small control group, low conversion
4. **H4**: Engineered context (treatment-balanced selection + multi-context averaging) recovers targeting value
5. **H5**: Re-ranking outputs improves targeting even when accuracy is unchanged

## Next Steps

### Phase 5: Smoke Test & Validation (In Progress)
- [ ] Run smoke test on Hillstrom
- [ ] Verify Qini scores match benchmarks
- [ ] Create experiment templates (W1, W2, W3)

### Phase 6: Cluster Orchestration
- [ ] Implement `scripts/launch_grid.py` for SLURM/SGE
- [ ] Create `scripts/plot_results.py` for visualization
- [ ] Implement `scripts/generate_report.py` for result aggregation

### Phase 7: CausalPFN Integration
- [ ] Install CausalPFN from GitHub (as submodule or dependency)
- [ ] Implement actual CausalPFN inference (currently placeholder)
- [ ] Validate on Hillstrom RCT

### Phase 8: Comparative Experiments
- [ ] Run W1: untuned vs. tuned on real + synthetic data
- [ ] Run W2: regime mapping (control ratio, conversion, context size)
- [ ] Implement & ablate CausalPFN-Rank

## References

- CausalPFN paper: https://arxiv.org/abs/2506.07918
- CausalPFN repo: https://github.com/vdblm/CausalPFN
- Uplift modeling overview: https://arxiv.org/abs/2410.07021v1
- Hillstrom RCT: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
