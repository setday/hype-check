import abc
import numpy as np
from typing import Callable, List, Dict, Any, Optional, Union
import random


class BiasInducer(abc.ABC):
    """
    Abstract base class for bias induction methods.

    Subclasses must implement the `fit` and `apply` methods.
    The `fit` method can compute any statistics from the original dataset
    (e.g., marginal treatment probabilities) needed for the sampling procedure.
    The `apply` method performs the subsampling and returns a new dataset.

    Attributes:
        random_seed (int): seed for reproducibility (default 42).
        confounder_func (Callable): a function that takes a feature vector
            (as a 1D numpy array) and returns the probability of treatment=1
            given those covariates. Used to bias the treatment assignment.
    """

    def __init__(self, confounder_func: Callable[[np.ndarray], float], random_seed: int = 42):
        """
        Args:
            confounder_func: callable that maps a feature vector (1D np.ndarray)
                to the probability P(T=1 | C=c). Must return a float in (0,1).
            random_seed: seed for random number generation.
        """
        self.confounder_func = confounder_func
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)

    @abc.abstractmethod
    def fit(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Compute any necessary statistics from the original RCT dataset.

        Args:
            dataset: list of dictionaries, each with 'features', 'treatment', 'outcome'.
        """
        pass

    @abc.abstractmethod
    def apply(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply the bias induction to produce a confounded observational dataset.

        Args:
            dataset: original RCT dataset (list of dicts).

        Returns:
            A new list of dictionaries (subsampled) that exhibits confounding.
        """
        pass


class OSRCTBiasInducer(BiasInducer):
    """
    Observational Sampling from RCT (OSRCT) as described in Gentzel et al. (2021).

    For each unit, sample a treatment `t_s` from Bernoulli(confounder_func(C)).
    If `t_s` equals the unit's actual treatment, the unit is kept in the biased dataset;
    otherwise it is discarded.

    This induces dependence between the covariates C and the observed treatment T,
    creating confounding. The resulting dataset has, in expectation, half the size
    of the original when treatment assignment is balanced.
    """

    def fit(self, dataset: List[Dict[str, Any]]) -> None:
        """
        No fitting is required for OSRCT; we only store the confounder function.
        """
        pass

    def apply(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply the OSRCT subsampling.

        For each unit, draw t_s ~ Bernoulli(confounder_func(features)).
        Keep the unit if t_s == treatment.

        Returns:
            Biased dataset (list of dicts) with only the retained units.
        """
        rng = self._rng
        biased = []
        for unit in dataset:
            features = unit['features']
            t_actual = unit['treatment']
            # Compute P(T=1 | C)
            p1 = self.confounder_func(features)
            # Clamp to avoid numerical issues
            p1 = max(0.0, min(1.0, p1))
            # Sample treatment indicator
            t_s = 1 if rng.random() < p1 else 0
            if t_s == t_actual:
                biased.append(unit)
        return biased


class RCTRejectionBiasInducer(BiasInducer):
    """
    RCT Rejection Sampling as described in Keith et al. (2023).

    This method uses rejection sampling to generate a confounded dataset while
    preserving the marginal distribution of covariates and the conditional outcome
    distribution, ensuring that the true ATE is still identified from the biased data
    via backdoor adjustment.

    The user specifies the target conditional distribution P*(T | C) via the
    `confounder_func`. The algorithm computes:
        M = max_{i} P*(T=t_i | C_i) / P(T=t_i)
    and then accepts each unit with probability:
        (1/M) * (P*(T=t_i | C_i) / P(T=t_i))

    This yields a sample from P*(C,T,Y) = P(C) * P*(T|C) * P(Y|T,C),
    where the covariate and outcome distributions remain unchanged.
    """

    def __init__(self, confounder_func: Callable[[np.ndarray], float], random_seed: int = 42):
        super().__init__(confounder_func, random_seed)
        self._p_t1: Optional[float] = None  # marginal P(T=1)
        self._p_t0: Optional[float] = None  # marginal P(T=0)
        self._M: Optional[float] = None      # upper bound constant

    def fit(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Compute the marginal treatment probabilities and the bound M
        from the original RCT dataset.

        Args:
            dataset: list of dictionaries with 'features', 'treatment'.
        """
        treatments = np.array([unit['treatment'] for unit in dataset])
        n = len(treatments)
        self._p_t1 = np.mean(treatments)
        self._p_t0 = 1.0 - self._p_t1

        # Compute the maximum likelihood ratio over the dataset
        max_ratio = 0.0
        for unit in dataset:
            t = unit['treatment']
            p_t_given_c = self.confounder_func(unit['features'])
            # Clamp to avoid 0 or 1 (positivity)
            p_t_given_c = max(1e-10, min(1.0 - 1e-10, p_t_given_c))
            if t == 1:
                p_t = self._p_t1
                prob = p_t_given_c
            else:
                p_t = self._p_t0
                prob = 1.0 - p_t_given_c
            ratio = prob / p_t
            if ratio > max_ratio:
                max_ratio = ratio

        self._M = max(1.0, max_ratio)

    def apply(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply RCT rejection sampling to create a confounded dataset.

        Returns:
            Biased dataset (list of dicts) after rejection sampling.
        """
        if self._M is None or self._p_t1 is None:
            raise RuntimeError("Must call fit() before apply().")

        rng = self._rng
        biased = []
        for unit in dataset:
            t = unit['treatment']
            p_t_given_c = self.confounder_func(unit['features'])
            p_t_given_c = max(1e-10, min(1.0 - 1e-10, p_t_given_c))
            if t == 1:
                prob = p_t_given_c
                p_t = self._p_t1
            else:
                prob = 1.0 - p_t_given_c
                p_t = self._p_t0

            accept_prob = (1.0 / self._M) * (prob / p_t)
            # Clamp to [0,1] for safety
            accept_prob = max(0.0, min(1.0, accept_prob))
            if rng.random() < accept_prob:
                biased.append(unit)

        return biased
