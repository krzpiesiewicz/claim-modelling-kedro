import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List

from claim_modelling_kedro.pipelines.p01_init.mdl_info_config import ModelInfo, Target, ModelEnum
from claim_modelling_kedro.pipelines.p01_init.metric_config import MetricEnum


logger = logging.getLogger(__name__)


class FeatureSelectionMethod(Enum):
    LASSO: str = "lasso"
    RFE: str = "rfe"

class HyperoptAlgoEnum(Enum):
    TPE: str = "tpe"
    RANDOM: str = "random"

@dataclass
class DataScienceConfig:
    mlflow_run_id: str
    split_random_seed: int
    split_val_size: float
    fs_enabled: bool
    fs_method: str
    fs_model_class: str
    fs_max_n_features: int
    fs_min_importance: float
    fs_max_iter: int
    fs_random_seed: int
    fs_params: Dict[str, Any]
    hopt_enabled: bool
    hopt_metric: MetricEnum
    hopt_max_evals: int
    hopt_algo: HyperoptAlgoEnum
    hopt_show_progressbar: bool
    hopt_trial_verbose: bool
    hopt_fmin_random_seed: int
    hopt_split_random_seed: int
    hopt_split_val_size: float
    hopt_excluded_params: List[str]
    model_class: str
    model_random_seed: int
    model_const_hparams: Dict[str, Any]

    def __init__(self, parameters: Dict, mdl_info: ModelInfo):
        params = parameters["data_science"]
        self.mlflow_run_id = params["mlflow_run_id"]
        self.split_random_seed = params["split_random_seed"]
        self.split_val_size = params["split_val_size"]
        fs_params = params["feature_selection"]
        self.fs_enabled = fs_params["enabled"]
        self.fs_method = FeatureSelectionMethod(fs_params["method"])
        self.fs_max_n_features = fs_params["max_n_features"]
        self.fs_min_importance = fs_params["min_importance"]
        self.fs_max_iter = fs_params["max_iter"]
        self.fs_random_seed = fs_params["random_seed"]
        if (fs_params["params"] is not None
                and mdl_info.model.value in fs_params["params"]
                and fs_params["params"][mdl_info.model.value] is not None):
            self.fs_params = fs_params["params"][mdl_info.model.value]
        else:
            self.fs_params = {}
        self.fs_params = fs_params["params"]
        match self.fs_method:
            case FeatureSelectionMethod.LASSO:
                self.fs_model_class = "claim_modelling_kedro.pipelines.p07_data_science.selectors.LassoFeatureSelector"
            case FeatureSelectionMethod.RFE:
                self.fs_model_class = "claim_modelling_kedro.pipelines.p07_data_science.selectors.RecursiveFeatureEliminationSelector"
            case _:
                raise ValueError(f"Feature selection method \"{self.fs_method}\" not supported.")

        hopt_params = params["hyperopt"]
        self.hopt_enabled = hopt_params["enabled"]
        if hopt_params["metric"] is None or hopt_params["metric"] != "auto":
            self.hopt_metric = MetricEnum(hopt_params["metric"])
        else:
            match mdl_info.target:
                case Target.CLAIMS_NUMBER:
                    self.hopt_metric = MetricEnum.POISSON_DEV
                case Target.FREQUENCY:
                    self.hopt_metric = MetricEnum.POISSON_DEV
                case Target.TOTAL_AMOUNT:
                    self.hopt_metric = None     # each model class has a method metric_function(self)
                                                # e.g. TweedieRegressor has a method tweedie_deviance where the parameter p
                                                # is extracted from hyper parameters
                case Target.POSITIVE_AMOUNT:
                    self.hopt_metric = MetricEnum.GAMMA_DEV
                case Target.AVG_CLAIM_AMOUNT:
                    self.hopt_metric = None
                case Target.POSITIVE_SEVERITY:
                    self.hopt_metric = MetricEnum.GAMMA_DEV
                case _:
                    raise ValueError(f"Metric for target {mdl_info.target} is not defined -> see p01_init/config.py.")
        self.hopt_max_evals = hopt_params["max_evals"]
        self.hopt_algo = HyperoptAlgoEnum(hopt_params["algo"])
        self.hopt_show_progressbar = hopt_params["show_progressbar"]
        self.hopt_trial_verbose = hopt_params["trial_verbose"]
        self.hopt_fmin_random_seed = hopt_params["random_seed"]
        self.hopt_split_random_seed = hopt_params["split_random_seed"]
        self.hopt_cv_enabled = hopt_params["cross_validation"]["enabled"]
        if self.hopt_cv_enabled:
            self.hopt_cv_folds = hopt_params["cross_validation"]["folds"]
            if self.hopt_cv_folds <= 1:
                raise ValueError("Number of folds should be greater than 1.")
            self.hopt_split_val_size = None
        else:
            self.hopt_cv_folds = None
            self.hopt_split_val_size = hopt_params["split_val_size"]
        if (hopt_params["excluded_params"] is not None
                and mdl_info.model.value in hopt_params["excluded_params"]
                and hopt_params["excluded_params"][mdl_info.model.value] is not None):
            self.hopt_excluded_params = hopt_params["excluded_params"][mdl_info.model.value]
        else:
            self.hopt_excluded_params = []
        model_params = params["model"]
        if model_params["model_class"] is not None:
            self.model_class = model_params["model_class"]
        else:
            match mdl_info.model:
                case ModelEnum.STATSMODELS_POISSON_GLM:
                    model_class_name = "StatsmodelsGLM"
                case ModelEnum.STATSMODELS_GAMMA_GLM:
                    model_class_name = "StatsmodelsGLM"
                case ModelEnum.STATSMODELS_TWEEDIE_GLM:
                    model_class_name = "StatsmodelsGLM"
                case ModelEnum.STATSMODELS_GAUSSIAN_GLM:
                    model_class_name = "StatsmodelsGLM"
                case ModelEnum.SKLEARN_POISSON_GLM:
                    model_class_name = "SklearnPoissonGLM"
                case ModelEnum.SKLEARN_GAMMA_GLM:
                    model_class_name = "SklearnGammaGLM"
                case ModelEnum.SKLEARN_TWEEDIE_GLM:
                    model_class_name = "SklearnTweedieGLM"
                case _:
                    raise ValueError(f"Model {mdl_info.model} not supported.")
            self.model_class = f"claim_modelling_kedro.pipelines.p07_data_science.models.{model_class_name}"
        self.model_random_seed = model_params["random_seed"]
        logger.debug(f'model_params["const_hparams"]: {model_params["const_hparams"]}')
        logger.debug(f'mdl_info.model.value: {mdl_info.model.value}')
        if mdl_info.model.value in model_params["const_hparams"]:
            logger.debug(f'model_params["const_hparams"][mdl_info.model.value]: {model_params["const_hparams"][mdl_info.model.value]}')
        if (model_params["const_hparams"] is not None
                and mdl_info.model.value in model_params["const_hparams"]
                and model_params["const_hparams"][mdl_info.model.value] is not None):
            self.model_const_hparams = model_params["const_hparams"][mdl_info.model.value]
        else:
            self.model_const_hparams = {}
        logger.debug(f"model_const_hparams: {self.model_const_hparams}")
