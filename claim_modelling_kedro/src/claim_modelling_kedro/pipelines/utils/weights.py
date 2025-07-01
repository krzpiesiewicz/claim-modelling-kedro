from typing import Union

import numpy as np
import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config


import logging

logger = logging.getLogger(__name__)


def get_sample_weight(config: Config, target_df: Union[pd.DataFrame, np.ndarray]) -> Union[pd.Series, None]:
    sample_weight = None
    if config.mdl_task.exposure_weighted and config.mdl_task.claim_nb_weighted:
        raise ValueError("Only one of exposure_weighted and claim_nb_weighted can be True.")
    if type(target_df) is pd.DataFrame:
        if config.mdl_task.exposure_weighted:
            sample_weight = target_df[config.data.policy_exposure_col]
            logger.debug("Using policy exposure as sample weight.")
        if config.mdl_task.claim_nb_weighted:
            sample_weight = np.fmax(target_df[config.data.claims_number_target_col], 1).astype(int)
            logger.debug("Using claims number as sample weight.")
    return sample_weight
