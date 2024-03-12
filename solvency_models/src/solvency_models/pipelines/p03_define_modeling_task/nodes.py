import logging
from typing import Tuple, Callable

import pandas as pd
from pandera.typing import Series

from solvency_models.pipelines.p01_init.config import Config, Target
from solvency_models.pipelines.p05_sampling.utils import is_target_gt_zero, return_none

logger = logging.getLogger(__name__)


def create_modeling_task(
        config: Config,
        targets_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Callable[[Series], Series[bool]]]:
    logger.info(f"Defining modelling task for target {config.mdl_info.target}...")
    match config.mdl_info.target:
        case Target.CLAIMS_NUMBER:
            target_df = targets_df
            is_event = is_target_gt_zero if config.smpl.target_ratio is not None else return_none
        case Target.FREQUENCY:
            target_df = targets_df
            is_event = is_target_gt_zero if config.smpl.target_ratio is not None else return_none
        case Target.TOTAL_AMOUNT:
            target_df = targets_df
            is_event = is_target_gt_zero if config.smpl.target_ratio is not None else return_none
        case Target.POSITIVE_AMOUNT:
            target_df = targets_df[targets_df[config.mdl_task.target_col] > 0]
            is_event = return_none
        case Target.SEVERITY:
            target_df = targets_df
            is_event = is_target_gt_zero if config.smpl.target_ratio is not None else return_none
        case Target.POSITIVE_SEVERITY:
            target_df = targets_df[targets_df[config.mdl_task.target_col] > 0]
            is_event = return_none
    logger.info("Defined the task.")
    return target_df, is_event
