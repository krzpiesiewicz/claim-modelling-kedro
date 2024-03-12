import logging
from typing import Tuple

import pandas as pd

from solvency_models.pipelines.p01_init.config import Config, Target

logger = logging.getLogger(__name__)

def create_modeling_task(
    config: Config,
    freqs: pd.DataFrame,
    severities: pd.DataFrame,
    features: pd.DataFrame,
    train_policies: pd.Series,
    calib_policies: pd.Series,
    backtest_policies: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Index, pd.Index, pd.Index]:
    logger.info(f"Defining modelling task for target {config.mdl_info.target}...")
    target_df = None
    match config.mdl_info.target:
        case Target.FREQUENCY:
            target_df = freqs
        case Target.SEVERITY:
            target_df = severities

    features_df = features

    train_idx = train_policies.index
    calib_idx = calib_policies.index
    backtest_idx = backtest_policies.index

    logger.info(f"""...defined task's parameters:
        - target_col: {config.mdl_task.target_col}""")

    return target_df, features_df, train_idx, calib_idx, backtest_idx
