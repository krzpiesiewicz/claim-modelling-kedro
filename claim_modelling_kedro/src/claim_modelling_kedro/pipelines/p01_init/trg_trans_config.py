import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TargetTransformerEnum(Enum):
    LOG_TARGET_TRANSFORMER: str = "LogTargetTransformer"
    POWER_TARGET_TRANSFORMER: str = "PowerTargetTransformer"

@dataclass
class TargetTransformerConfig:
    enabled: bool
    model_class: str
    model_params: Dict[str, Any]

    def __init__(self, trans_params: Dict[str, Any]):
        self.enabled = trans_params["enabled"]
        if self.enabled:
            model = TargetTransformerEnum(trans_params["model"])
            match model:
                case TargetTransformerEnum.LOG_TARGET_TRANSFORMER:
                    self.model_class = "claim_modelling_kedro.pipelines.p07_data_science.target_transformers.LogTargetTransformer"
                case TargetTransformerEnum.POWER_TARGET_TRANSFORMER:
                    self.model_class = "claim_modelling_kedro.pipelines.p07_data_science.target_transformers.PowerTargetTransformer"
        else:
            self.model_class = None
        # Check if target transformer parameters are provided
        if (self.enabled
                and trans_params["params"] is not None
                and trans_params["model"] in trans_params["params"]
                and trans_params["params"][trans_params["model"]] is not None):
            self.model_params = trans_params["params"][trans_params["model"]]
        else:
            self.model_params = {}
