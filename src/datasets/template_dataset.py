"""
TEMPLATE DATASET
================
Minimal working example: synthetic classification data.

TODO: Replace this file with your actual dataset implementation.
Steps:
  1. Subclass BaseDataset (or torch.utils.data.Dataset directly).
  2. In __init__, build `self._index` as a list[dict], where each dict
     is one training example. The keys in each dict must match what
     your collate_fn and model.calculate_loss / forward expect.
  3. If using files on disk, override the loading logic below.
  4. Ensure __getitem__ returns a dict with the same keys.

Typical index entry keys:
  - "features": input tensor / list
  - "label": ground truth (int, float, or tensor)
  - "input_ids": token IDs (for NLP)
  - "attention_mask": binary mask (for NLP)
"""

import torch
from src.datasets.base_dataset import BaseDataset


class TemplateDataset(BaseDataset):
    """
    A synthetic classification dataset for demonstration.

    Generates random feature vectors and integer labels.
    This serves as a drop-in replacement that lets you run
    and verify the training pipeline end-to-end before
    implementing your real dataset.
    """

    def __init__(
        self,
        num_samples=1000,
        input_dim=20,
        num_classes=5,
        shuffle_index=False,
        limit=None,
        instance_transforms=None,
        **kwargs,
    ):
        # ── Build index (list[dict]) ──────────────────────────────────────
        # TODO: Replace with your data loading logic (CSV, parquet, images, etc.)
        rng = torch.Generator().manual_seed(42)
        index = []
        for i in range(num_samples):
            features = torch.randn(input_dim, generator=rng).tolist()
            label = int(torch.randint(0, num_classes, (1,), generator=rng).item())
            index.append({"features": features, "label": label})

        # Let BaseDataset handle shuffle, limit, sorting
        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def preprocess_data(self, instance_data):
        """
        Convert raw index entries into tensors for collation.
        Override BaseDataset's no-op preprocess if you need transforms.
        """
        instance_data["features"] = torch.tensor(instance_data["features"])
        instance_data["label"] = torch.tensor(instance_data["label"], dtype=torch.long)
        return instance_data
