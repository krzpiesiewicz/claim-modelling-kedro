import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

from claim_modelling_kedro.pipelines.p01_init.exprmnt import ExperimentInfo
from claim_modelling_kedro.pipelines.p01_init.fs_config import FeatureSelectionConfig
from claim_modelling_kedro.pipelines.p01_init.hp_config import HypertuneConfig
from claim_modelling_kedro.pipelines.p01_init.smpl_config import SamplingConfig

logger = logging.getLogger(__name__)

class TargetTransformerEnum(Enum):
    LOG_TARGET_TRANSFORMER: str = "LogTargetTransformer"

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
    LIGHTGBM_POISSON_REGRESSOR: str = "LightGBMPoissonRegressor"
    LIGHTGBM_GAMMA_REGRESSOR: str = "LightGBMGammaRegressor"
    LIGHTGBM_TWEEDIE_REGRESSOR: str = "LightGBMTweedieRegressor"


@dataclass
class DataScienceConfig:
    mlflow_run_id: str
    model: ModelEnum
    model_class: str
    model_const_hparams: Dict[str, Any]
    target_transformer_enabled: bool
    target_transformer_class: str
    fs: FeatureSelectionConfig
    hp: HypertuneConfig

    def __init__(self, parameters: Dict, smpl: SamplingConfig):
        params = parameters["data_science"]
        self.mlflow_run_id = params["mlflow_run_id"]

        model_params = params["model"]
        self.model = ModelEnum(model_params["model"])
        if model_params["model_class"] is not None:
            self.model_class = model_params["model_class"]
        else:
            match self.model:
                case ModelEnum.DUMMY_MEAN_REGRESSOR:
                    model_class_name = "DummyMeanRegressor"
                case ModelEnum.STATSMODELS_POISSON_GLM:
                    model_class_name = "StatsmodelsPoissonGLM"
                case ModelEnum.STATSMODELS_GAMMA_GLM:
                    model_class_name = "StatsmodelsGammaGLM"
                case ModelEnum.STATSMODELS_TWEEDIE_GLM:
                    model_class_name = "StatsmodelsTweedieGLM"
                case ModelEnum.STATSMODELS_GAUSSIAN_GLM:
                    model_class_name = "StatsmodelsGaussianGLM"
                case ModelEnum.SKLEARN_POISSON_GLM:
                    model_class_name = "SklearnPoissonGLM"
                case ModelEnum.SKLEARN_GAMMA_GLM:
                    model_class_name = "SklearnGammaGLM"
                case ModelEnum.SKLEARN_TWEEDIE_GLM:
                    model_class_name = "SklearnTweedieGLM"
                case ModelEnum.PYGLMNET_POISSON_GLM:
                    model_class_name = "PyGLMNetGLM"
                case ModelEnum.PYGLMNET_GAMMA_GLM:
                    model_class_name = "PyGLMNetGLM"
                case ModelEnum.LIGHTGBM_POISSON_REGRESSOR:
                    model_class_name = "LightGBMPoissonRegressor"
                case ModelEnum.LIGHTGBM_GAMMA_REGRESSOR:
                    model_class_name = "LightGBMGammaRegressor"
                case ModelEnum.LIGHTGBM_TWEEDIE_REGRESSOR:
                    model_class_name = "LightGBMTweedieRegressor"
                case _:
                    raise ValueError(f"Model {self.model} not supported.")
            self.model_class = f"claim_modelling_kedro.pipelines.p07_data_science.models.{model_class_name}"
        logger.debug(f'model_params["const_hparams"]: {model_params["const_hparams"]}')
        logger.debug(f'mdl_info.model.value: {self.model.value}')
        if self.model.value in model_params["const_hparams"]:
            logger.debug(
                f'model_params["const_hparams"][mdl_info.model.value]: {model_params["const_hparams"][self.model.value]}')
        if (model_params["const_hparams"] is not None
                and self.model.value in model_params["const_hparams"]
                and model_params["const_hparams"][self.model.value] is not None):
            self.model_const_hparams = model_params["const_hparams"][self.model.value]
        else:
            self.model_const_hparams = {}
        logger.debug(f"model_const_hparams: {self.model_const_hparams}")

        self.target_transformer_enabled = params["target_transformer"]["enabled"]
        if self.target_transformer_enabled:
            model = TargetTransformerEnum(params["target_transformer"]["model"])
            match model:
                case TargetTransformerEnum.LOG_TARGET_TRANSFORMER:
                    self.target_transformer_class = "claim_modelling_kedro.pipelines.p07_data_science.target_transformers.LogTargetTransformer"
                case _:
                    raise ValueError(f"Target transformer model \"{model}\" not supported.")
        else:
            self.target_transformer_class = None

        self.fs = FeatureSelectionConfig(params["feature_selection"])
        self.hp = HypertuneConfig(params["hyperopt"], smpl=smpl, model_name=self.model.value)
