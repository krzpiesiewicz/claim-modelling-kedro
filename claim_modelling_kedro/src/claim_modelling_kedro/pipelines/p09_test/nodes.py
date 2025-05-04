import logging
from typing import Dict, Tuple

import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p09_test.utils import run_eval_pipeline

logger = logging.getLogger(__name__)


def eval_model_on_test_and_compute_stats(config: Config, calibrated_calib_predictions_df: Dict[str, pd.DataFrame],
                                         features_df: Dict[str, pd.DataFrame], target_df: Dict[str, pd.DataFrame],
                                         test_keys: Dict[str, pd.Index]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    calibrated_calib_predictions_df is only used to force the pipeline order (p09 after p08)
    """
    return run_eval_pipeline(config, features_df, target_df, test_keys, "test")


def eval_model_on_train_and_compute_stats(config: Config, calibrated_calib_predictions_df: Dict[str, pd.DataFrame],
                                          features_df: Dict[str, pd.DataFrame], target_df: Dict[str, pd.DataFrame],
                                          train_keys: Dict[str, pd.Index]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    calibrated_calib_predictions_df is only used to force the pipeline order (p09 after p08)
    """
    return run_eval_pipeline(config, features_df, target_df, train_keys, "train")
