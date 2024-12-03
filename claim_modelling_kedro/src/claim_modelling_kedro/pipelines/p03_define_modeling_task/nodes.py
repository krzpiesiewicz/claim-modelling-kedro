import logging
from typing import Tuple, Callable

import pandas as pd
from pandera.typing import Series

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p01_init.mdl_info_config import Target
from claim_modelling_kedro.pipelines.p05_sampling.utils import is_target_gt_zero, return_none

logger = logging.getLogger(__name__)


def create_modeling_task(
        config: Config,
        targets_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Callable[[Series], Series[bool]]]:
    logger.info(f"Defining modelling task for target {config.mdl_info.target}...")
    match config.mdl_info.target:
        case Target.CLAIMS_NUMBER:
            target_df = targets_df
            is_event = is_target_gt_zero
        case Target.FREQUENCY:
            target_df = targets_df
            is_event = is_target_gt_zero
        case Target.TOTAL_AMOUNT:
            target_df = targets_df
            is_event = is_target_gt_zero
        case Target.AVG_CLAIM_AMOUNT:
            target_df = targets_df[targets_df[config.mdl_task.target_col] > 0]
            is_event = return_none
        case Target.PURE_PREMIUM:
            target_df = targets_df
            is_event = is_target_gt_zero
    logger.info("Defined the task.")
    return target_df, is_event
