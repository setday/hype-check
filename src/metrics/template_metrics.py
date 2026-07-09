"""
TEMPLATE METRICS
================
Minimal working example: top-K accuracy.

TODO: Replace or extend with your task-specific metrics.
  - For regression: MAE, MSE, RMSE
  - For ranking: Recall@K, NDCG@K
  - For NLP: BLEU, ROUGE, perplexity
  - For generation: FID, IS, CLIP score
"""

import torch
from src.metrics.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    """
    Top-K accuracy metric.

    Computes the fraction of samples where the true label is
    among the top-K predicted logits.

    Inherits from BaseMetric → torchmetrics.Metric, so it supports
    distributed sync via add_state(..., dist_reduce_fx="sum").
    """

    def __init__(self, name="accuracy", top_k=1, **kwargs):
        super().__init__(name=name, **kwargs)
        self.top_k = top_k

        # Distributed-safe accumulators
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits=None, label=None, preds=None, labels=None, **batch):
        """
        Update metric state with a batch of predictions.

        Accepts both 'logits'/'label' and 'preds'/'labels' for flexibility.
        """
        logits = logits if logits is not None else preds
        label = label if label is not None else labels

        if logits is None or label is None:
            return  # skip if missing (e.g., empty metric group)

        pred = torch.topk(logits, self.top_k, dim=-1).indices         # (B, top_k)
        self.correct += (pred == label.unsqueeze(-1)).any(dim=-1).sum()
        self.total += label.size(0)

    def compute(self):
        """:return: accuracy as a float tensor"""
        return self.correct.float() / self.total
