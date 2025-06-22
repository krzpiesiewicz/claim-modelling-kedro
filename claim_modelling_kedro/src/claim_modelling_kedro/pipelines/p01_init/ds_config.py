import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional

from claim_modelling_kedro.pipelines.p01_init.exprmnt import ExperimentInfo, Target
from claim_modelling_kedro.pipelines.p01_init.metric_config import MetricEnum
from claim_modelling_kedro.pipelines.p01_init.smpl_config import SamplingConfig, SampleValidationSet

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

class FeatureSelectionMethod(Enum):
    MODEL: str = "model"
    PYGLMNET: str = "pyglmnet"
    LASSO: str = "lasso"
    RFE: str = "rfe"

class HyperoptAlgoEnum(Enum):
    TPE: str = "tpe"
    RANDOM: str = "random"

class HypertuneValidationEnum(Enum):
    SAMPLE_VAL_SET: str = "sample_val_set"
    CROSS_VALIDATION: str = "cross_validation"
    REPEATED_SPLIT: str = "repeated_split"

@dataclass
class DataScienceConfig:
    mlflow_run_id: str
    model: ModelEnum
    model_class: str
    model_const_hparams: Dict[str, Any]
    target_transformer_enabled: bool
    target_transformer_class: str
    fs_enabled: bool
    fs_method: str
    fs_model_class: str
    fs_max_n_features: int
    fs_min_importance: float
    fs_max_iter: int
    fs_params: Dict[str, Any]
    hopt_enabled: bool
    hopt_metric: MetricEnum
    hopt_max_evals: int
    hopt_overfit_penalty: float
    hopt_early_stop_enabled: bool
    hopt_early_iteration_stop_count: int
    hopt_early_stop_percent_increase: float
    hopt_algo: HyperoptAlgoEnum
    hopt_show_progressbar: bool
    hopt_trial_verbose: bool
    hopt_fmin_random_seed: int
    hopt_validation_method: HypertuneValidationEnum
    hopt_cv_folds: int
    hopt_cv_random_seed: int
    hopt_repeated_split_val_size: float
    hopt_repeated_split_n_repeats: int
    hopt_repeated_split_random_seed: int
    hopt_excluded_params: List[str]

    def __init__(self, parameters: Dict, exprmnt: ExperimentInfo, smpl: SamplingConfig):
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

        fs_params = params["feature_selection"]
        self.fs_enabled = fs_params["enabled"]
        self.fs_method = FeatureSelectionMethod(fs_params["method"])
        self.fs_max_n_features = fs_params["max_n_features"]
        self.fs_min_importance = fs_params["min_importance"]
        self.fs_max_iter = fs_params["max_iter"]
        if (fs_params["params"] is not None
                and fs_params["method"] in fs_params["params"]
                and fs_params["params"][fs_params["method"]] is not None):
            self.fs_params = fs_params["params"][fs_params["method"]]
        else:
            self.fs_params = {}
        match self.fs_method:
            case FeatureSelectionMethod.MODEL:
                self.fs_model_class = "claim_modelling_kedro.pipelines.p07_data_science.selectors.ModelWrapperFeatureSelector"
            case FeatureSelectionMethod.PYGLMNET:
                self.fs_model_class = "claim_modelling_kedro.pipelines.p07_data_science.selectors.PyGLMNetFeatureSelector"
            case FeatureSelectionMethod.LASSO:
                self.fs_model_class = "claim_modelling_kedro.pipelines.p07_data_science.selectors.LassoFeatureSelector"
            case FeatureSelectionMethod.RFE:
                self.fs_model_class = "claim_modelling_kedro.pipelines.p07_data_science.selectors.RecursiveFeatureEliminationSelector"
            case _:
                raise ValueError(f"Feature selection method \"{self.fs_method}\" not supported.")

        hopt_params = params["hyperopt"]
        self.hopt_enabled = hopt_params["enabled"]
        if self.hopt_enabled:
            if hopt_params["metric"] is None or hopt_params["metric"] != "auto":
                self.hopt_metric = MetricEnum(hopt_params["metric"])
            else:
                match exprmnt.target:
                    case Target.CLAIMS_NUMBER:
                        self.hopt_metric = MetricEnum.POISSON_DEV
                    case Target.FREQUENCY:
                        self.hopt_metric = MetricEnum.POISSON_DEV
                    case Target.TOTAL_AMOUNT:
                        self.hopt_metric = None     # each model class has a method metric_function(self)
                                                    # e.g. TweedieRegressor has a method tweedie_deviance where the parameter p
                                                    # is extracted from hyperparameters
                    case Target.POSITIVE_AMOUNT:
                        self.hopt_metric = MetricEnum.GAMMA_DEV
                    case Target.AVG_CLAIM_AMOUNT:
                        self.hopt_metric = None
                    case Target.POSITIVE_SEVERITY:
                        self.hopt_metric = MetricEnum.GAMMA_DEV
                    case _:
                        raise ValueError(f"Metric for target {exprmnt.target} is not defined -> see p01_init/config.py.")
            self.hopt_overfit_penalty = hopt_params["overfit_penalty"]
            self.hopt_max_evals = hopt_params["max_evals"]
            self.hopt_early_stop_enabled = hopt_params["early_stopping"]["enabled"]
            if self.hopt_early_stop_enabled:
                self.hopt_early_iteration_stop_count= hopt_params["early_stopping"]["iteration_stop_count"]
                self.hopt_early_stop_percent_increase = hopt_params["early_stopping"]["percent_increase"]
            self.hopt_algo = HyperoptAlgoEnum(hopt_params["algo"])
            self.hopt_show_progressbar = hopt_params["show_progressbar"]
            self.hopt_trial_verbose = hopt_params["trial_verbose"]
            self.hopt_fmin_random_seed = hopt_params["random_seed"]

            hopt_val_params = hopt_params["validation"]
            self.hopt_validation_method = HypertuneValidationEnum(hopt_val_params["method"])
            self.hopt_cv_folds = hopt_val_params["cross_validation"]["folds"]
            self.hopt_cv_random_seed = hopt_val_params["cross_validation"]["random_seed"]
            self.hopt_repeated_split_val_size = hopt_val_params["repeated_split"]["val_size"]
            self.hopt_repeated_split_n_repeats = hopt_val_params["repeated_split"]["n_repeats"]
            self.hopt_repeated_split_random_seed = hopt_val_params["repeated_split"]["random_seed"]
            match self.hopt_validation_method:
                case HypertuneValidationEnum.CROSS_VALIDATION:
                    if self.hopt_cv_folds <= 1:
                        raise ValueError("Number of folds should be greater than 1.")
                case HypertuneValidationEnum.SAMPLE_VAL_SET:
                    if smpl.validation_set == SampleValidationSet.NONE:
                        raise ValueError("Sample validation set hypertuning method is not supported when no validation set is created in sampling pipeline.\n"
                                         "Please set `sampling.validation.val_set` to `calib_set` or `split_train_val`.\n"
                                         "Alternatively, set `data_science.hyperopt.validation.method` to `cross_validation` or `repeated_split`.")

            if (hopt_params["excluded_params"] is not None
                    and self.model.value in hopt_params["excluded_params"]
                    and hopt_params["excluded_params"][self.model.value] is not None):
                self.hopt_excluded_params = hopt_params["excluded_params"][self.model.value]
            else:
                self.hopt_excluded_params = []
        else:
            self.hopt_metric = None
            self.hopt_overfit_penalty = None
            self.hopt_max_evals = None
            self.hopt_early_stop_enabled = False
            self.hopt_early_iteration_stop_count = None
            self.hopt_early_stop_percent_increase = None
            self.hopt_algo = None
            self.hopt_show_progressbar = False
            self.hopt_trial_verbose = False
            self.hopt_fmin_random_seed = None
            self.hopt_validation_method = None
            self.hopt_cv_folds = None
            self.hopt_cv_random_seed = None
            self.hopt_repeated_split_val_size = None
            self.hopt_repeated_split_n_repeats = None
            self.hopt_repeated_split_random_seed = None
            self.hopt_excluded_params = None
