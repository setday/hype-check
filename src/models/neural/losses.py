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


# Continuous treatment losses


def continuous_outcome_loss(y_pred, y, t=None, weights=None):
    """
    MSE outcome loss for continuous treatment.

    Args:
        y_pred: (batch_size, 1) predicted outcomes
        y: (batch_size, 1) observed outcomes
        t: (batch_size, 1) optional treatment (for weighting)
        weights: (batch_size,) optional IPW weights

    Returns:
        Scalar loss
    """
    se = (y_pred.view(-1) - y.view(-1)).pow(2)
    if weights is not None:
        se = se * weights.view(-1)
    return se.mean()


def continuous_propensity_loss(t, mu, log_sigma):
    """
    Gaussian NLL for continuous treatment propensity.

    Assumes T ~ N(μ(X), σ²(X))

    Args:
        t: (batch_size, 1) observed treatment
        mu: (batch_size, 1) predicted mean
        log_sigma: (batch_size, 1) predicted log std dev

    Returns:
        Scalar loss (negative log likelihood)
    """
    sigma = torch.exp(log_sigma)
    sigma = torch.clamp(sigma, min=1e-3)  # Numerical stability
    se = (t.view(-1) - mu.view(-1)).pow(2) / (2 * sigma.view(-1).pow(2))
    nll = se + log_sigma.view(-1)
    return nll.mean()


def continuous_representation_loss(rep_t, t, kernel="rbf", bandwidth=1.0):
    """
    Kernel-based representation balance for continuous treatment.

    Uses maximum mean discrepancy (MMD) in RKHS to balance representations
    across treatment spectrum.

    Args:
        rep_t: (batch_size, rep_dim) learned representations
        t: (batch_size, 1) continuous treatment
        kernel: 'rbf' or 'linear'
        bandwidth: kernel bandwidth (for RBF)

    Returns:
        Scalar loss
    """
    if kernel == "rbf":
        # RBF kernel: exp(-||x - y||² / bandwidth)
        t_centered = t.view(-1, 1) - t.view(1, -1)  # (batch, batch)
        K = torch.exp(-t_centered.pow(2) / (2 * bandwidth**2))
    elif kernel == "linear":
        K = torch.mm(t, t.t())  # (batch, batch)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # MMD: E[K(rep_i, rep_j)] for i, j ~ T vs. independent T
    rep_diff = rep_t.unsqueeze(0) - rep_t.unsqueeze(1)  # (batch, batch, rep_dim)
    rep_dist = (rep_diff.pow(2).sum(dim=2) + 1e-8).sqrt()  # (batch, batch)

    # Weighted kernel in representation space
    rep_kernel = torch.exp(-rep_dist / bandwidth)

    mmd = (K * rep_kernel).mean()
    return mmd

