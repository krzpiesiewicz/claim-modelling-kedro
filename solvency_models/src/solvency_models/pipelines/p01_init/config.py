import ast
import logging
import os
import tempfile
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Any

import mlflow
import yaml

logger = logging.getLogger(__name__)


class ModelEnum(Enum):
    STATSMODELS_POISSON_GLM: str = "StatsmodelsPoissonGLM"
    STATSMODELS_GAMMA_GLM: str = "StatsmodelsGammaGLM"
    STATSMODELS_TWEEDIE_GLM: str = "StatsmodelsTweedieGLM"
    STATSMODELS_GAUSSIAN_GLM: str = "StatsmodelsGaussianGLM"
    SKLEARN_POISSON_GLM: str = "SklearnPoissonGLM"


class Target(Enum):
    CLAIMS_NUMBER: str = "claims_number"
    FREQUENCY: str = "frequency"
    TOTAL_AMOUNT: str = "total_claims_amount"
    POSITIVE_AMOUNT: str = "positive_claims_amount"
    SEVERITY: str = "severity"
    POSITIVE_SEVERITY: str = "positive_severity"


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


@dataclass
class DataConfig:
    raw_policy_id_col: str
    raw_claims_number_col: str
    raw_claims_amount_col: str
    raw_exposure_col: str
    claims_number_target_col: str
    claims_freq_target_col: str
    claims_total_amount_target_col: str
    claims_severity_target_col: str
    targets_cols: List[str]
    claims_number_pred_col: str
    claims_freq_pred_col: str
    claims_total_amount_pred_col: str
    claims_severity_pred_col: str
    policy_exposure_col: str
    policy_id_col: str
    split_random_seed: int
    cv_enabled: bool
    cv_folds: int
    cv_parts_names: List[str]
    cv_mlflow_runs_ids: Dict[str, str]
    one_part_key = "0"
    calib_size: float
    test_size: float

    def __init__(self, parameters: Dict):
        params = parameters["data"]
        self.raw_policy_id_col = params["raw_files"]["policy_id_col"]
        self.raw_claims_number_col = params["raw_files"]["claims_number_col"]
        self.raw_claims_amount_col = params["raw_files"]["claims_amount_col"]
        self.raw_exposure_col = params["raw_files"]["exposure_col"]
        self.claims_number_target_col = params["claims_number_target_col"]
        self.claims_freq_target_col = params["claims_freq_target_col"]
        self.claims_total_amount_target_col = params["claims_total_amount_target_col"]
        self.claims_severity_target_col = params["claims_severity_target_col"]
        self.claims_number_pred_col = params["claims_number_pred_col"]
        self.claims_freq_pred_col = params["claims_freq_pred_col"]
        self.claims_total_amount_pred_col = params["claims_total_amount_pred_col"]
        self.claims_severity_pred_col = params["claims_severity_pred_col"]
        self.policy_exposure_col = params["policy_exposure_col"]
        self.policy_id_col = params["policy_id_col"]
        self.targets_cols = [self.claims_number_target_col, self.claims_freq_target_col,
                             self.claims_severity_target_col, self.claims_total_amount_target_col,
                             self.policy_exposure_col]
        self.split_random_seed = params["split_random_seed"]

        self.calib_size = params["calib_size"]
        if self.calib_size is None:
            self.calib_size = 0
        if self.calib_size < 0:
            raise ValueError("calib_size should be greater or equal to 0.")

        self.cv_enabled = params["cross_validation"]["enabled"]
        self.cv_mlflow_runs_ids = None
        if self.cv_enabled:
            self.cv_folds = params["cross_validation"]["folds"]
            if self.cv_folds <= 1:
                raise ValueError("Number of folds should be greater than 1.")
            self.cv_parts_names = list(map(str, range(self.cv_folds)))
            self.test_size = 1 / self.cv_folds
            self.one_part_key = None
        else:
            self.cv_folds = None
            self.cv_parts_names = None
            self.test_size = params["test_size"]
            if self.test_size + self.calib_size >= 1:
                raise ValueError("Sum of calib_size and test_size should be less than 1.")
            if self.test_size <= 0:
                raise ValueError("test_size should be greater than 0.")
            self.one_part_key = "0"


class MetricEnum(Enum):
    POISSON_DEV: str = "poisson_deviance"
    WEIGHTED_POISSON_DEV: str = "weighted_poisson_deviance"
    GAMMA_DEV: str = "gamma_deviance"
    WEIGHTED_GAMMA_DEV: str = "weighted_gamma_deviance"
    RMSE: str = "rmse"
    WEIGHTED_RMSE: str = "weighted_rmse"
    R2: str = "r2"
    WEIGHTED_R2: str = "weighted_r2"
    SPEARMAN: str = "spearman_correlation"
    SPEARMAN_RANK: str = "spearman_rank_correlation"


@dataclass
class ModelTask:
    target_col: str
    prediction_col: str
    evaluation_metrics: List[MetricEnum]

    def __init__(self, mdl_info: ModelInfo, data: DataConfig):
        match mdl_info.target:
            case Target.CLAIMS_NUMBER:
                self.target_col = data.claims_number_target_col
                self.prediction_col = data.claims_number_pred_col
                self.evaluation_metrics = [MetricEnum.POISSON_DEV, MetricEnum.WEIGHTED_POISSON_DEV, MetricEnum.RMSE,
                                           MetricEnum.WEIGHTED_RMSE, MetricEnum.R2, MetricEnum.WEIGHTED_R2,
                                           MetricEnum.SPEARMAN, MetricEnum.SPEARMAN_RANK]
            case Target.FREQUENCY:
                self.target_col = data.claims_freq_target_col
                self.prediction_col = data.claims_freq_pred_col
                self.evaluation_metrics = [MetricEnum.POISSON_DEV, MetricEnum.WEIGHTED_POISSON_DEV, MetricEnum.RMSE,
                                           MetricEnum.WEIGHTED_RMSE, MetricEnum.R2, MetricEnum.WEIGHTED_R2,
                                           MetricEnum.SPEARMAN, MetricEnum.SPEARMAN_RANK]
            case Target.TOTAL_AMOUNT:
                self.target_col = data.claims_total_amount_target_col
                self.prediction_col = data.claims_total_amount_pred_col
                self.evaluation_metrics = [MetricEnum.RMSE, MetricEnum.WEIGHTED_RMSE, MetricEnum.R2,
                                           MetricEnum.SPEARMAN, MetricEnum.SPEARMAN_RANK]
            case Target.POSITIVE_AMOUNT:
                self.target_col = data.claims_total_amount_target_col
                self.prediction_col = data.claims_total_amount_pred_col
                self.evaluation_metrics = [MetricEnum.GAMMA_DEV, MetricEnum.WEIGHTED_GAMMA_DEV, MetricEnum.RMSE,
                                           MetricEnum.WEIGHTED_RMSE, MetricEnum.R2, MetricEnum.WEIGHTED_R2,
                                           MetricEnum.SPEARMAN, MetricEnum.SPEARMAN_RANK]
            case Target.SEVERITY:
                self.target_col = data.claims_severity_target_col
                self.prediction_col = data.claims_severity_pred_col
                self.evaluation_metrics = [MetricEnum.RMSE, MetricEnum.WEIGHTED_RMSE, MetricEnum.R2,
                                           MetricEnum.SPEARMAN, MetricEnum.WEIGHTED_R2, MetricEnum.SPEARMAN_RANK]
            case Target.POSITIVE_SEVERITY:
                self.target_col = data.claims_severity_target_col
                self.prediction_col = data.claims_severity_pred_col
                self.evaluation_metrics = [MetricEnum.GAMMA_DEV, MetricEnum.WEIGHTED_GAMMA_DEV, MetricEnum.RMSE,
                                           MetricEnum.WEIGHTED_RMSE, MetricEnum.R2, MetricEnum.WEIGHTED_R2,
                                           MetricEnum.SPEARMAN, MetricEnum.SPEARMAN_RANK]


@dataclass
class SamplingConfig:
    use_calib_data: bool
    random_seed: int
    n_obs: int
    target_ratio: float
    allow_lower_ratio: bool
    include_zeros: bool
    lower_bound: float
    upper_bound: float

    def __init__(self, parameters: Dict):
        params = parameters["sampling"]
        self.use_calib_data = params["use_calib_data"]
        self.random_seed = params["random_seed"]
        self.n_obs = params["n_obs"]
        self.target_ratio = params["target_ratio"]
        self.allow_lower_ratio = params["allow_lower_ratio"]
        self.include_zeros = params["include_zeros"]
        self.lower_bound = params["lower_bound"]
        self.upper_bound = params["upper_bound"]


@dataclass
class CustomFeatureConfig:
    name: str
    type: str
    single_column: bool
    description: str
    model_class: str

    def __init__(self, feature_params: Dict):
        self.name = feature_params["name"]
        self.type = feature_params["type"]
        self.single_column = feature_params["single_column"]
        self.description = feature_params["description"]
        self.model_class = feature_params["model_class"]


@dataclass
class DataEngineeringConfig:
    mlflow_run_id: str
    is_custom_features_creator_enabled: bool
    custom_features: List[CustomFeatureConfig]
    is_cat_reducer_enabled: bool
    reducer_max_categories: int
    reducer_join_exceeding_max_categories: str
    reducer_min_frequency: int
    reducer_join_infrequent: str
    is_cat_imputer_enabled: bool
    cat_imputer_strategy: str
    cat_imputer_fill_value: str
    is_num_imputer_enabled: bool
    num_imputer_strategy: str
    num_imputer_fill_value: float
    is_ohe_enabled: bool
    ohe_drop: str
    ohe_max_categories: int
    ohe_min_frequency: int
    is_scaler_enabled: bool
    join_exceeding_and_join_infrequent_values_warning: str = None

    def __init__(self, parameters: Dict):
        params = parameters["data_engineering"]
        self.mlflow_run_id = params["mlflow_run_id"]
        self.is_custom_features_creator_enabled = params["custom_features"]["enabled"]
        self.custom_features = [CustomFeatureConfig(feature_params) for feature_params in
                                params["custom_features"]["features"]]
        self.is_cat_reducer_enabled = params["reduce_categories"]["enabled"]
        self.reducer_max_categories = params["reduce_categories"]["max_categories"]
        self.reducer_min_frequency = params["reduce_categories"]["min_frequency"]
        self.reducer_join_exceeding_max_categories = params["reduce_categories"]["join_exceeding_max_categories"]
        self.reducer_join_infrequent = params["reduce_categories"]["join_infrequent"]
        if self.is_cat_reducer_enabled and all(val is not None for val in
                                               [self.reducer_max_categories, self.reducer_min_frequency,
                                                self.reducer_join_infrequent,
                                                self.reducer_join_exceeding_max_categories]):
            self.join_exceeding_and_join_infrequent_values_warning = (
                "Both parameters join_exceeding_max_categories and join_infrequent are provided "
                "and have different values, "
                f"{self.join_exceeding_max_categories} and {self.join_infrequent} respectively.\n"
                f"Thus, {self.join_exceeding_max_categories} will be used for both infrequent and exceeding categories.")
            logger.warning(self.join_exceeding_and_join_infrequent_values_warning)

        self.is_cat_imputer_enabled = params["cat_ftrs_imputer"]["enabled"]
        self.cat_imputer_strategy = params["cat_ftrs_imputer"]["strategy"]
        self.cat_imputer_fill_value = params["cat_ftrs_imputer"]["fill_value"]
        self.is_num_imputer_enabled = params["num_ftrs_imputer"]["enabled"]
        self.num_imputer_strategy = params["num_ftrs_imputer"]["strategy"]
        self.num_imputer_fill_value = params["num_ftrs_imputer"]["fill_value"]
        self.is_ohe_enabled = params["ohe"]["enabled"]
        self.ohe_drop = params["ohe"]["drop"]
        self.ohe_max_categories = params["ohe"]["max_categories"]
        self.ohe_min_frequency = params["ohe"]["min_frequency"]
        self.is_scaler_enabled = params["scaler"]["enabled"]


class FeatureSelectionMethod(Enum):
    LASSO: str = "lasso"
    RFE: str = "rfe"


class LossEnum(Enum):
    POISSON_DEV: str = "poisson_deviance"
    WEIGHTED_POISSON_DEV: str = "weighted_poisson_deviance"
    GAMMA_DEV: str = "gamma_deviance"
    WEIGHTED_GAMMA_DEV: str = "weighted_gamma_deviance"
    MSE: str = "mse"
    WEIGHTED_MSE: str = "weighted_mse"
    TWEEDIE_DEV: str = "tweedie_deviance"
    WEIGHTED_TWEEDIE_DEV: str = "weighted_tweedie_deviance"


@dataclass
class DataScienceConfig:
    mlflow_run_id: str
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
    hopt_algo: str
    hopt_show_progressbar: bool
    hopt_trial_verbose: bool
    hopt_fmin_random_seed: int
    hopt_split_random_seed: int
    hopt_split_test_size: float
    hopt_excluded_params: List[str]
    model_class: str
    model_random_seed: int
    model_const_hparams: Dict[str, Any]

    def __init__(self, parameters: Dict, mdl_info: ModelInfo):
        params = parameters["data_science"]
        self.mlflow_run_id = params["mlflow_run_id"]
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
                self.fs_model_class = "solvency_models.pipelines.p07_data_science.selectors.LassoFeatureSelector"
            case FeatureSelectionMethod.RFE:
                self.fs_model_class = "solvency_models.pipelines.p07_data_science.selectors.RecursiveFeatureEliminationSelector"
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
                    self.hopt_metric = MetricEnum.RMSE
                case Target.POSITIVE_AMOUNT:
                    self.hopt_metric = MetricEnum.GAMMA_DEV
                case Target.SEVERITY:
                    self.hopt_metric = MetricEnum.RMSE
                case Target.POSITIVE_SEVERITY:
                    self.hopt_metric = MetricEnum.GAMMA_DEV
                case _:
                    raise ValueError(f"Metric for target {mdl_info.target} is not defined -> see p01_init/config.py.")
        self.hopt_max_evals = hopt_params["max_evals"]
        self.hopt_algo = hopt_params["algo"]
        self.hopt_show_progressbar = hopt_params["show_progressbar"]
        self.hopt_trial_verbose = hopt_params["trial_verbose"]
        self.hopt_fmin_random_seed = hopt_params["random_seed"]
        self.hopt_split_random_seed = hopt_params["split_random_seed"]
        self.hopt_split_test_size = hopt_params["split_test_size"]
        if (hopt_params["excluded_params"] is not None
                and mdl_info.model.value in hopt_params["excluded_params"]
                and hopt_params["excluded_params"][mdl_info.model.value] is not None):
            self.hopt_excluded_params = hopt_params["excluded_params"][mdl_info.model.value]
        else:
            self.hopt_excluded_params = []
        model_params = params["model"]
        match mdl_info.model:
            case ModelEnum.STATSMODELS_POISSON_GLM:
                self.model_class = "StatsmodelsGLM"
            case ModelEnum.STATSMODELS_GAMMA_GLM:
                self.model_class = "StatsmodelsGLM"
            case ModelEnum.STATSMODELS_TWEEDIE_GLM:
                self.model_class = "StatsmodelsGLM"
            case ModelEnum.STATSMODELS_GAUSSIAN_GLM:
                self.model_class = "StatsmodelsGLM"
            case ModelEnum.SKLEARN_POISSON_GLM:
                self.model_class = "SklearnPoissonGLM"
            case _:
                raise ValueError(f"Model {mdl_info.model} not supported.")
        self.model_class = f"solvency_models.pipelines.p07_data_science.models.{self.model_class}"
        self.model_random_seed = model_params["random_seed"]
        if (model_params["const_hparams"] is not None
                and mdl_info.model.value in model_params["const_hparams"]
                and model_params["const_hparams"][mdl_info.model.value] is not None):
            self.model_const_hparams = model_params["const_hparams"][mdl_info.model.value]
        else:
            self.model_const_hparams = {}


class CalibrationMethod(Enum):
    ISOTONIC_REGRESSION: str = "IsotonicRegression"
    CENTERED_ISOTONIC_REGRESSION: str = "CenteredIsotonicRegression"
    SKLEARN_POISSON_GLM: str = "SklearnPoissonGLM"


@dataclass
class CalibrationConfig:
    mlflow_run_id: str
    enabled: bool
    method: CalibrationMethod
    model_class: str
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
            case CalibrationMethod.SKLEARN_POISSON_GLM:
                self.model_class = "SklearnPoissonGLM"
            case _:
                raise ValueError(f"Calibration method \"{self.method}\" not supported.")
        self.model_class = f"solvency_models.pipelines.p08_model_calibration.calibration_models.{self.model_class}"


@dataclass
class TestConfig:
    lower_bound: float
    upper_bound: float

    def __init__(self, parameters: Dict):
        params = parameters["test"]
        self.lower_bound = params["lower_bound"]
        self.upper_bound = params["upper_bound"]


@dataclass
class Config:
    mdl_info: ModelInfo
    data: DataConfig
    mdl_task: ModelTask
    smpl: SamplingConfig
    de: DataEngineeringConfig
    ds: DataScienceConfig
    clb: CalibrationConfig
    test: TestConfig

    def __init__(self, parameters: Dict):
        self.mdl_info = ModelInfo(parameters)
        self.data = DataConfig(parameters)
        self.mdl_task = ModelTask(mdl_info=self.mdl_info, data=self.data)
        self.smpl = SamplingConfig(parameters)
        self.de = DataEngineeringConfig(parameters)
        self.ds = DataScienceConfig(parameters, mdl_info=self.mdl_info)
        self.clb = CalibrationConfig(parameters, mdl_task=self.mdl_task)
        self.test = TestConfig(parameters)


def log_config_to_mlflow(config: Config):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "config.yml")
        with open(file_path, "w") as file:
            yaml.dump(asdict(config), file)
        mlflow.log_artifact(file_path)


def add_sub_runs_ids_to_config(config: Config) -> Config:
    run = mlflow.active_run()
    logger.info(f"Searching for sub runs in MLFlow run ID: {run.info.run_id}.")
    if "cv_subruns" in run.data.params:
        subruns_dct = get_subruns_dct(run)
        config.data.cv_mlflow_runs_ids = subruns_dct
        logger.info(f"Loaded cross validation sub runs: \n" +
                    "\n".join([f" - {part}: {run_id}" for part, run_id in subruns_dct.items()]))
    else:
        logger.info("No sub runs found.")
        logger.info("Creating sub runs...")
        subruns_dct = {}
        for part in config.data.cv_parts_names:
            with mlflow.start_run(run_name=f"CV_{part}", nested=True) as subrun:
                subrun_id = subrun.info.run_id
                subruns_dct[part] = subrun_id
                logger.info(f"Created sub run for cross validation split {part} with ID: {subrun_id}")
        config.data.cv_mlflow_runs_ids = subruns_dct
        mlflow.log_param(f"cv_subruns", subruns_dct)
        logger.info("Created all sub runs and logged ids to MLflow.")
    return config


def get_subruns_dct(run) -> Dict[str, str]:
    logger.info(f"type(run): {type(run)}")
    if "cv_subruns" in run.data.params:
        return ast.literal_eval(run.data.params["cv_subruns"])
    else:
        return None
