from dataclasses import dataclass
from enum import Enum
from typing import Dict


class ModelEnum(Enum):
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
class ModelInfo:
    name: str
    author: str
    type: str
    model: ModelEnum
    target: Target
    short_description: str
    full_description: str

    def __init__(self, parameters: Dict):
        params = parameters["model"]
        self.name = params["name"]
        self.author = params["author"]
        self.type = params["type"]
        self.model = ModelEnum(params["model"])
        self.target = Target(params["target"])
        self.short_description = params["short_description"]
        self.full_description = params["full_description"]
