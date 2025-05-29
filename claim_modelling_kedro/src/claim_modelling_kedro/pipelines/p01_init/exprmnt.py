from dataclasses import dataclass
from enum import Enum
from typing import Dict


class ModelEnum(Enum):
    DUMMY_MEAN_REGRESSOR: str = "DummyMeanRegressor"
    STATSMODELS_POISSON_GLM: str = "StatsmodelsPoissonGLM"
    STATSMODELS_GAMMA_GLM: str = "StatsmodelsGammaGLM"
    STATSMODELS_TWEEDIE_GLM: str = "StatsmodelsTweedieGLM"
    STATSMODELS_GAUSSIAN_GLM: str = "StatsmodelsGaussianGLM"
    SKLEARN_POISSON_GLM: str = "SklearnPoissonGLM"
    SKLEARN_GAMMA_GLM: str = "SklearnGammaGLM"
    SKLEARN_TWEEDIE_GLM: str = "SklearnTweedieGLM"
    RGLMNET_POISSON_GLM: str = "RGLMNetPoissonGLM"
    PYGLMNET_POISSON_GLM: str = "PyGLMNetPoissonGLM"
    PYGLMNET_GAMMA_GLM: str = "PyGLMNetGammaGLM"


class Target(Enum):
    CLAIMS_NUMBER: str = "claims_number"
    FREQUENCY: str = "frequency"
    TOTAL_AMOUNT: str = "total_claims_amount"
    AVG_CLAIM_AMOUNT: str = "avg_claim_amount"
    PURE_PREMIUM: str = "pure_premium"


@dataclass
class ExperimentInfo:
    name: str
    author: str
    target: Target
    description: str

    def __init__(self, parameters: Dict):
        params = parameters["experiment"]
        self.name = params["name"]
        self.author = params["author"]
        self.target = Target(params["target"])
        self.description = params["full_description"]
