from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

from claim_modelling_kedro.pipelines.p01_init.data_config import DataConfig
from claim_modelling_kedro.pipelines.p01_init.mdl_task_config import ModelTask
from claim_modelling_kedro.pipelines.p01_init.outliers_config import OutliersConfig


class CalibrationMethod(Enum):
    PURE: str = "Pure"
    ISOTONIC_REGRESSION: str = "IsotonicRegression"
    CENTERED_ISOTONIC_REGRESSION: str = "CenteredIsotonicRegression"
    STATS_MODELS_POISSON_GLM: str = "StatsmodelsPoissonGLM"
    STATS_MODELS_GAMMA_GLM: str = "StatsmodelsGammaGLM"
    STATS_MODELS_TWEEDIE_GLM: str = "StatsmodelsTweedieGLM"
    SKLEARN_POISSON_GLM: str = "SklearnPoissonGLM"
    SKLEARN_GAMMA_GLM: str = "SklearnGammaGLM"
    SKLEARN_TWEEDIE_GLM: str = "SklearnTweedieGLM"
    EQUAL_BINS_MEANS: str = "EqualBinsMeans"
    LOCAL_POLYNOMIAL_REGRESSION: str = "LocalPolynomialRegression"
    LOCAL_STATS_MODELS_GLM: str = "LocalStatsmodelsGLM"


class RebalanceInTotalsMethod(Enum):
    SCALE: str = "scale"
    SHIFT: str = "shift"


@dataclass
class CalibrationConfig:
    mlflow_run_id: str
    enabled: bool
    method: CalibrationMethod
    model_class: str
    model_const_hparams: Dict[str, Any]
    outliers: OutliersConfig
    pure_prediction_col: str
    calibrated_prediction_col: str
    post_calibration_rebalancing_enabled: bool
    post_clb_rebalance_method: RebalanceInTotalsMethod

    def __init__(self, parameters: Dict, mdl_task: ModelTask, data: DataConfig):
        params = parameters["calibration"]
        self.mlflow_run_id = params["mlflow_run_id"]
        self.enabled = params["enabled"]
        if self.enabled:
            if not data.calib_set_enabled:
                raise ValueError("Calibration is enabled, but calibration set is not enabled in data config.")
            self.pure_prediction_col = "pure_" + mdl_task.prediction_col
            self.calibrated_prediction_col = "calibrated_" + mdl_task.prediction_col
            self.method = CalibrationMethod(params["method"])
            match self.method:
                case CalibrationMethod.PURE:
                    self.model_class = "PureCalibration"
                case CalibrationMethod.ISOTONIC_REGRESSION:
                    self.model_class = "SklearnIsotonicRegression"
                case CalibrationMethod.CENTERED_ISOTONIC_REGRESSION:
                    self.model_class = "CIRModelCenteredIsotonicRegression"
                case CalibrationMethod.STATS_MODELS_POISSON_GLM:
                    self.model_class = "StatsmodelsGLMCalibration"
                case CalibrationMethod.STATS_MODELS_GAMMA_GLM:
                    self.model_class = "StatsmodelsGammaGLMCalibration"
                case CalibrationMethod.SKLEARN_POISSON_GLM:
                    self.model_class = "SklearnPoissonGLMCalibration"
                case CalibrationMethod.SKLEARN_GAMMA_GLM:
                    self.model_class = "SklearnGammaGLMCalibration"
                case CalibrationMethod.STATS_MODELS_TWEEDIE_GLM:
                    self.model_class = "StatsmodelsGLMCalibration"
                case CalibrationMethod.SKLEARN_TWEEDIE_GLM:
                    self.model_class = "SklearnTweedieGLMCalibration"
                case CalibrationMethod.EQUAL_BINS_MEANS:
                    self.model_class = "EqualBinsMeansCalibration"
                case CalibrationMethod.LOCAL_POLYNOMIAL_REGRESSION:
                    self.model_class = "LocalPolynomialRegressionCalibration"
                case CalibrationMethod.LOCAL_STATS_MODELS_GLM:
                    self.model_class = "LocalStatsmodelsGLMCalibration"
                case _:
                    raise ValueError(f"Calibration method \"{self.method}\" not supported.")
            self.model_class = f"claim_modelling_kedro.pipelines.p08_model_calibration.calibration_models.{self.model_class}"
            if (params["const_hparams"] is not None
                    and self.method.value in params["const_hparams"]
                    and params["const_hparams"][self.method.value] is not None):
                self.model_const_hparams = {k: v for k, v in params["const_hparams"][self.method.value].items() if v is not None}
            else:
                self.model_const_hparams = {}
            self.outliers = OutliersConfig(params["outliers"])
        else:
            self.method = None
            self.model_class = None
            self.model_const_hparams = None
            self.outliers = None
            self.pure_prediction_col = None
            self.calibrated_prediction_col = None
        post_clb_params = params["post_calibration_rebalancing"]
        self.post_calibration_rebalancing_enabled = post_clb_params["enabled"]
        if self.post_calibration_rebalancing_enabled:
            self.post_clb_rebalance_method = RebalanceInTotalsMethod(post_clb_params["method"])
        else:
            self.post_clb_rebalance_method = None
