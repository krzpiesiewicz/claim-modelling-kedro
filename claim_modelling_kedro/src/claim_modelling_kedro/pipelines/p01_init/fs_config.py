import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

logger = logging.getLogger(__name__)


class FeatureSelectionMethod(Enum):
    MODEL: str = "model"
    PYGLMNET: str = "pyglmnet"
    LASSO: str = "lasso"
    RFE: str = "rfe"


@dataclass
class FeatureSelectionConfig:
    enabled: bool
    method: str
    model_class: str
    max_n_features: int
    min_importance: float
    max_iter: int
    params: Dict[str, Any]

    def __init__(self, params: Dict):
        self.enabled = params["enabled"]
        self.method = FeatureSelectionMethod(params["method"])
        self.max_n_features = params["max_n_features"]
        self.min_importance = params["min_importance"]
        self.max_iter = params["max_iter"]
        if (params["params"] is not None
                and params["method"] in params["params"]
                and params["params"][params["method"]] is not None):
            self.params = params["params"][params["method"]]
        else:
            self.params = {}
        match self.method:
            case FeatureSelectionMethod.MODEL:
                self.model_class = "claim_modelling_kedro.pipelines.p07_data_science.selectors.ModelWrapperFeatureSelector"
            case FeatureSelectionMethod.PYGLMNET:
                self.model_class = "claim_modelling_kedro.pipelines.p07_data_science.selectors.PyGLMNetFeatureSelector"
            case FeatureSelectionMethod.LASSO:
                self.model_class = "claim_modelling_kedro.pipelines.p07_data_science.selectors.LassoFeatureSelector"
            case FeatureSelectionMethod.RFE:
                self.model_class = "claim_modelling_kedro.pipelines.p07_data_science.selectors.RecursiveFeatureEliminationSelector"
            case _:
                raise ValueError(f"Feature selection method \"{self.method}\" not supported.")
