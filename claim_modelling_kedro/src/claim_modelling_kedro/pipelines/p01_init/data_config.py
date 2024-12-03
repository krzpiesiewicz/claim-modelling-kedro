from dataclasses import dataclass
from typing import List, Dict


@dataclass
class DataConfig:
    raw_policy_id_col: str
    raw_claims_number_col: str
    raw_claims_amount_col: str
    raw_exposure_col: str
    claims_number_target_col: str
    claims_freq_target_col: str
    claims_total_amount_target_col: str
    claims_avg_amount_target_col: str
    targets_cols: List[str]
    claims_number_pred_col: str
    claims_freq_pred_col: str
    claims_total_amount_pred_col: str
    claims_avg_amount_pred_col: str
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
        self.claims_avg_amount_target_col = params["claims_avg_amount_target_col"]
        self.claims_pure_premium_target_col = params["claims_pure_premium_target_col"]
        self.claims_number_pred_col = params["claims_number_pred_col"]
        self.claims_freq_pred_col = params["claims_freq_pred_col"]
        self.claims_total_amount_pred_col = params["claims_total_amount_pred_col"]
        self.claims_avg_amount_pred_col = params["claims_avg_amount_pred_col"]
        self.claims_pure_premium_pred_col = params["claims_pure_premium_pred_col"]
        self.policy_exposure_col = params["policy_exposure_col"]
        self.policy_id_col = params["policy_id_col"]
        self.targets_cols = [self.claims_number_target_col, self.claims_freq_target_col,
                             self.claims_avg_amount_target_col, self.claims_total_amount_target_col,
                             self.claims_pure_premium_target_col, self.policy_exposure_col]
        self.split_random_seed = params["split_random_seed"]

        self.cv_enabled = params["cross_validation"]["enabled"]
        self.cv_mlflow_runs_ids = None
        if self.cv_enabled:
            self.cv_folds = params["cross_validation"]["folds"]
            if self.cv_folds <= 1:
                raise ValueError("Number of folds should be greater than 1.")
            self.cv_parts_names = list(map(str, range(self.cv_folds)))
            self.test_size = 1 / self.cv_folds
            self.calib_size = 1 / self.cv_folds
            self.one_part_key = None
        else:
            self.cv_folds = None
            self.cv_parts_names = None
            self.calib_size = params["calib_size"]
            if self.calib_size is None:
                self.calib_size = 0
            if self.calib_size < 0:
                raise ValueError("calib_size should be greater or equal to 0.")
            self.test_size = params["test_size"]
            if self.test_size + self.calib_size >= 1:
                raise ValueError("Sum of calib_size and test_size should be less than 1.")
            if self.test_size <= 0:
                raise ValueError("test_size should be greater than 0.")
            self.one_part_key = "0"
