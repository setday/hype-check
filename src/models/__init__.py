from src.models.template_model import TemplateMLP
from src.models.uplift_model import (
    UpliftModel,
    FrozenFoundationModel,
    SLearnerWrapper,
    TLearnerWrapper,
    XLearnerWrapper,
    DRLearnerWrapper,
)
from src.models.causalpfn_model import CausalPFNModel

# Registry used by the evaluation harness (scripts/run_baselines.py).
BASELINES = {
    "s_learner": SLearnerWrapper,
    "t_learner": TLearnerWrapper,
    "x_learner": XLearnerWrapper,
    "dr_learner": DRLearnerWrapper,
}

__all__ = [
    "TemplateMLP",
    "UpliftModel",
    "FrozenFoundationModel",
    "SLearnerWrapper",
    "TLearnerWrapper",
    "XLearnerWrapper",
    "DRLearnerWrapper",
    "CausalPFNModel",
    "BASELINES",
]
