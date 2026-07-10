# Integration Guide: How to Extend the Research Repo

This document shows how to integrate new methods, datasets, or metrics into the established infrastructure.

## Adding a New Dataset

### Example: Custom RCT Dataset

1. **Implement loader** in `src/datasets/uplift_data_utils.py`:
   ```python
   def load_mydataset_dataset(
       limit: Optional[int] = None,
       control_ratio: float = 1.0,
       target_rows: Optional[int] = None,
   ) -> List[Dict[str, Any]]:
       cache_path = get_cache_path("mydataset", limit=limit, control_ratio=control_ratio)
       if cache_path.exists():
           with open(cache_path, "rb") as f:
               return pickle.load(f)
       
       # Load data (from CSV, API, etc.)
       df = pd.read_csv("path/to/mydataset.csv")
       
       # Convert to index format
       index = []
       for _, row in df.iterrows():
           index.append({
               "features": row[feature_cols].values.astype(np.float32),
               "treatment": int(row["treatment"]),
               "outcome": float(row["outcome"]),
               # Optional: "cate_true": float(row["cate_true"]),
           })
       
       # Apply subsampling if needed
       if target_rows and len(index) > target_rows:
           indices = np.random.choice(len(index), target_rows, replace=False)
           index = [index[i] for i in indices]
       
       # Cache and return
       with open(cache_path, "wb") as f:
           pickle.dump(index, f)
       return index
   ```

2. **Create config** `config/datasets/mydataset.yaml`:
   ```yaml
   train:
     _target_: src.datasets.uplift_dataset.UpliftDataset
     index: ${oc.env:MYDATASET_TRAIN_INDEX}
     shuffle_index: true
   
   val:
     _target_: src.datasets.uplift_dataset.UpliftDataset
     index: ${oc.env:MYDATASET_VAL_INDEX}
     shuffle_index: false
   ```

3. **Use in train.py**:
   ```bash
   python train.py datasets=mydataset model=s_learner
   ```

## Adding a New Model / Baseline

### Example: X-Learner

1. **Implement wrapper** in `src/models/uplift_model.py` (or new file):
   ```python
   class XLearnerWrapper(UpliftModel):
       """X-Learner (Kuenzel et al. 2019)"""
       
       def __init__(self, config: dict):
           super().__init__(config)
           self.model_1 = None
           self.model_0 = None
           self._is_fitted = False
       
       def fit(self, X: torch.Tensor, T: torch.Tensor, Y: torch.Tensor) -> None:
           # 1. Fit f_0 on control group
           # 2. Fit f_1 on treatment group
           # 3. Compute pseudo-outcomes: X_0 = Y - f_0(X), X_1 = Y - f_1(X)
           # 4. Fit g_1 on (X, X_1) for treatment group
           # 5. Fit g_0 on (X, X_0) for control group
           pass
       
       def forward(self, features: torch.Tensor, **batch) -> Dict[str, torch.Tensor]:
           # CATE = T(X) * g_1(X) + (1 - T(X)) * g_0(X)
           pass
   ```

2. **Create config** `config/model/x_learner.yaml`:
   ```yaml
   _target_: src.models.uplift_model.XLearnerWrapper
   params:
     n_estimators: 100
     max_depth: 5
   ```

3. **Use**:
   ```bash
   python train.py model=x_learner
   ```

## Adding a New Metric

### Example: Custom Metric

1. **Implement in** `src/metrics/uplift_metrics.py`:
   ```python
   class MyCustomMetric(Metric):
       def __init__(self, name: str = "my_metric", **kwargs):
           super().__init__(**kwargs)
           self.name = name
           self.add_state("pred", default=[], dist_reduce_fx="cat")
           self.add_state("true", default=[], dist_reduce_fx="cat")
       
       def update(self, cate_pred: torch.Tensor, cate_true: torch.Tensor, **kwargs):
           self.pred.append(cate_pred.detach().cpu())
           self.true.append(cate_true.detach().cpu())
       
       def compute(self) -> torch.Tensor:
           if not self.pred:
               return torch.tensor(float("nan"))
           pred = torch.cat(self.pred).numpy()
           true = torch.cat(self.true).numpy()
           # Compute metric
           value = my_metric_function(pred, true)
           return torch.tensor(value)
   ```

2. **Add to config** `config/metrics/uplift.yaml`:
   ```yaml
   val:
     - _target_: src.metrics.uplift_metrics.MyCustomMetric
       name: my_metric
   ```

## Running Comparative Experiments

### W1: Untuned vs. Tuned Comparison

```bash
# S-Learner on Hillstrom
python train.py model=s_learner datasets=hillstrom trainer.max_epochs=10

# T-Learner on Hillstrom
python train.py model=t_learner datasets=hillstrom trainer.max_epochs=10

# CausalPFN (inference only, no training)
python train.py model=causalpfn datasets=hillstrom trainer.limit_val_batches=1.0
```

### W2: Regime Mapping

Test how performance varies with control group size:
```python
# In scripts/regime_test.py (TBD)
for control_ratio in [0.1, 0.3, 0.5, 1.0]:
    index = load_hillstrom_dataset(control_ratio=control_ratio)
    # Train & evaluate
    # Log results
```

### W3: CausalPFN-Rank Ablations

Once CausalPFN-Rank is implemented:
```bash
# Baseline CausalPFN
python train.py model=causalpfn ...

# CausalPFN + context selection
python train.py model=causalpfn_rank config/ablate/context_selection.yaml

# CausalPFN + re-ranking
python train.py model=causalpfn_rank config/ablate/reranking.yaml

# Full CausalPFN-Rank
python train.py model=causalpfn_rank
```

## Cluster Integration (H200 Cluster)

### Batch Submission

Once `scripts/launch_grid.py` is implemented:

```bash
python scripts/launch_grid.py \
  --experiment w1_check \
  --models s_learner,t_learner,causalpfn \
  --datasets hillstrom,criteo,x5 \
  --gpus_per_job 1 \
  --time_limit 60 \
  --partition gpu_h200
```

This generates a job script and submits to SLURM/SGE.

### Result Aggregation

```bash
# Generate report
python scripts/generate_report.py \
  --results_dir results/ \
  --output report.html

# Create visualizations
python scripts/plot_results.py \
  --results_dir results/ \
  --plot_dir plots/
```

## Reproducibility

### Save Experiment Config

Configs are automatically saved:
```bash
python train.py model=s_learner datasets=hillstrom

# Check saved config:
ls saved/experiment/*/config.yaml
```

### Reproduce Experiment

```bash
# Same results: use same config + seed
python train.py --config-path saved/experiment/{id}/config.yaml
```

## Testing Your Extensions

1. **Unit test your code** (pytest, if added to repo)
2. **Run smoke test** with your new component:
   ```bash
   python scripts/smoke_test.py --models your_new_model
   ```
3. **Check metrics logging** (wandb, tensorboard)
4. **Verify reproducibility** with multiple runs

## Performance Tips

- **GPU acceleration**: Set `trainer.accelerator=cuda` (default)
- **Multi-GPU**: Set `trainer.strategy=ddp` for distributed training
- **Batch size tuning**: Edit `config/dataloaders/template.yaml`
- **Data caching**: First run is slow (caching), subsequent runs are fast
- **Metric computation**: Only computed on validation, not training (efficient)

---

**All extensions follow the same pattern**: data → model → metric → pipeline → config → experiment.
