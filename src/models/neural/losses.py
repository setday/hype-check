"""Shared loss helpers for neural uplift models."""

import torch


def factual_outcome_loss(y0, y1, y, t, loss_fn=torch.nn.BCELoss):
    """Binary outcome loss on the observed potential outcome."""
    pred = t * y1 + (1.0 - t) * y0
    return loss_fn()(pred, y)


def propensity_loss(logit, t):
    return torch.nn.functional.binary_cross_entropy_with_logits(logit, t)


def dragonnet_targeted_regularization(y0, y1, propensity, y, t, epsilon):
    """Targeted regularization from Shi et al. (2019)."""
    pred = t * y1 + (1.0 - t) * y0
    e = torch.clamp(propensity, 1e-6, 1.0 - 1e-6)
    h = t / e - (1.0 - t) / (1.0 - e)
    y_pert = torch.clamp(pred + epsilon * h, 1e-6, 1.0 - 1e-6)
    return torch.nn.functional.binary_cross_entropy(y_pert, y)


def linear_mmd(rep, t):
    """Linear MMD between treated and control representations."""
    mask1 = t.view(-1) > 0.5
    mask0 = ~mask1
    if mask1.sum() == 0 or mask0.sum() == 0:
        return rep.new_zeros(())
    return (rep[mask1].mean(dim=0) - rep[mask0].mean(dim=0)).pow(2).mean()
