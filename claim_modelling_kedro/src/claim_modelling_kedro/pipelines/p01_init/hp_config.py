import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any

from claim_modelling_kedro.pipelines.p01_init.hp_space_config import HyperOptSpaceConfig
from claim_modelling_kedro.pipelines.p01_init.metric_config import MetricEnum
from claim_modelling_kedro.pipelines.p01_init.smpl_config import SamplingConfig, SampleValidationSet

logger = logging.getLogger(__name__)

class HyperoptAlgoEnum(Enum):
    TPE: str = "tpe"
    RANDOM: str = "random"

class HypertuneValidationEnum(Enum):
    SAMPLE_VAL_SET: str = "sample_val_set"
    CROSS_VALIDATION: str = "cross_validation"
    REPEATED_SPLIT: str = "repeated_split"

@dataclass
class HypertuneConfig:
    enabled: bool
    metric: MetricEnum
    max_evals: int
    overfit_penalty: float
    early_stop_enabled: bool
    early_iteration_stop_count: int
    early_stop_percent_increase: float
    algo: HyperoptAlgoEnum
    show_progressbar: bool
    trial_verbose: bool
    fmin_random_seed: int
    validation_method: HypertuneValidationEnum
    cv_folds: int
    cv_random_seed: int
    repeated_split_val_size: float
    repeated_split_n_repeats: int
    repeated_split_random_seed: int
    excluded_params: List[str]
    space: HyperOptSpaceConfig

    def __init__(self, params: Dict, smpl: SamplingConfig, model_name: str, model_const_hparams: Dict[str, Any]):
        self.enabled = params["enabled"]
        if self.enabled:
            self.metric = MetricEnum(params["metric"]) if params["metric"] is None or params["metric"] != "auto" else None
            self.overfit_penalty = params["overfit_penalty"]
            self.max_evals = params["max_evals"]
            self.early_stop_enabled = params["early_stopping"]["enabled"]
            if self.early_stop_enabled:
                self.early_iteration_stop_count= params["early_stopping"]["iteration_stop_count"]
                self.early_stop_percent_increase = params["early_stopping"]["percent_increase"]
            self.algo = HyperoptAlgoEnum(params["algo"])
            self.show_progressbar = params["show_progressbar"]
            self.trial_verbose = params["trial_verbose"]
            self.fmin_random_seed = params["random_seed"]
            # Validation method
            val_params = params["validation"]
            self.validation_method = HypertuneValidationEnum(val_params["method"])
            self.cv_folds = val_params["cross_validation"]["folds"]
            self.cv_random_seed = val_params["cross_validation"]["random_seed"]
            self.repeated_split_val_size = val_params["repeated_split"]["val_size"]
            self.repeated_split_n_repeats = val_params["repeated_split"]["n_repeats"]
            self.repeated_split_random_seed = val_params["repeated_split"]["random_seed"]
            match self.validation_method:
                case HypertuneValidationEnum.CROSS_VALIDATION:
                    if self.cv_folds <= 1:
                        raise ValueError("Number of folds should be greater than 1.")
                case HypertuneValidationEnum.SAMPLE_VAL_SET:
                    if smpl.validation_set == SampleValidationSet.NONE:
                        raise ValueError("Sample validation set hypertuning method is not supported when no validation set is created in sampling pipeline.\n"
                                         "Please set `sampling.validation.val_set` to `calib_set` or `split_train_val`.\n"
                                         "Alternatively, set `data_science.hyperopt.validation.method` to `cross_validation` or `repeated_split`.")
            # Excluded params
            if (params["excluded_params"] is not None
                    and model_name in params["excluded_params"]
                    and params["excluded_params"][model_name] is not None):
                self.excluded_params = params["excluded_params"][model_name]
            else:
                self.excluded_params = []
            # Params space
            if (params["params_space"] is not None
                    and model_name in params["params_space"]
                    and params["params_space"][model_name] is not None):
                self.space = HyperOptSpaceConfig(params["params_space"][model_name], excluded_params=self.excluded_params,
                                                 const_params=model_const_hparams)
            else:
                self.space = None
        else:
            self.metric = None
            self.overfit_penalty = None
            self.max_evals = None
            self.early_stop_enabled = False
            self.early_iteration_stop_count = None
            self.early_stop_percent_increase = None
            self.algo = None
            self.show_progressbar = False
            self.trial_verbose = False
            self.fmin_random_seed = None
            self.validation_method = None
            self.cv_folds = None
            self.cv_random_seed = None
            self.repeated_split_val_size = None
            self.repeated_split_n_repeats = None
            self.repeated_split_random_seed = None
            self.excluded_params = None
            self.space = None
