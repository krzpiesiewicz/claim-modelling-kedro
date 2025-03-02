from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

from claim_modelling_kedro.pipelines.p01_init.mdl_task_config import ModelTask


class CalibrationMethod(Enum):
    ISOTONIC_REGRESSION: str = "IsotonicRegression"
    CENTERED_ISOTONIC_REGRESSION: str = "CenteredIsotonicRegression"
    STATS_MODELS_POISSON_GLM: str = "StatsmodelsPoissonGLM"
    STATS_MODELS_GAMMA_GLM: str = "StatsmodelsGammaGLM"
    STATS_MODELS_TWEEDIE_GLM: str = "StatsmodelsTweedieGLM"
    SKLEARN_POISSON_GLM: str = "SklearnPoissonGLM"
    SKLEARN_GAMMA_GLM: str = "SklearnGammaGLM"
    SKLEARN_TWEEDIE_GLM: str = "SklearnTweedieGLM"


@dataclass
class CalibrationConfig:
    mlflow_run_id: str
    enabled: bool
    method: CalibrationMethod
    model_class: str
    model_const_hparams: Dict[str, Any]
    lower_bound: float
    upper_bound: float
    pure_prediction_col: str
    calibrated_prediction_col: str

    def __init__(self, parameters: Dict, mdl_task: ModelTask):
        params = parameters["calibration"]
        self.mlflow_run_id = params["mlflow_run_id"]
        self.enabled = params["enabled"]
        self.method = CalibrationMethod(params["method"])
        self.lower_bound = params["lower_bound"]
        self.upper_bound = params["upper_bound"]
        self.pure_prediction_col = "pure_" + mdl_task.prediction_col
        self.calibrated_prediction_col = "calibrated_" + mdl_task.prediction_col
        match self.method:
            case CalibrationMethod.ISOTONIC_REGRESSION:
                self.model_class = "SklearnIsotonicRegression"
            case CalibrationMethod.CENTERED_ISOTONIC_REGRESSION:
                self.model_class = "CIRModelCenteredIsotonicRegression"
            case CalibrationMethod.STATS_MODELS_POISSON_GLM:
                self.model_class = "StatsmodelsGLMCalibration"
            case CalibrationMethod.STATS_MODELS_GAMMA_GLM:
                self.model_class = "StatsmodelsGLMCalibration"
            case CalibrationMethod.SKLEARN_POISSON_GLM:
                self.model_class = "SklearnPoissonGLM"
            case CalibrationMethod.SKLEARN_GAMMA_GLM:
                self.model_class = "SklearnGammaGLM"
            case CalibrationMethod.STATS_MODELS_TWEEDIE_GLM:
                self.model_class = "StatsmodelsTweedieGLM"
            case CalibrationMethod.SKLEARN_TWEEDIE_GLM:
                self.model_class = "SklearnTweedieGLM"
            case _:
                raise ValueError(f"Calibration method \"{self.method}\" not supported.")
        self.model_class = f"claim_modelling_kedro.pipelines.p08_model_calibration.calibration_models.{self.model_class}"
        if (params["const_hparams"] is not None
                and self.method.value in params["const_hparams"]
                and params["const_hparams"][self.method.value] is not None):
            self.model_const_hparams = {k: v for k, v in params["const_hparams"][self.method.value].items() if v is not None}
        else:
            self.model_const_hparams = {}
