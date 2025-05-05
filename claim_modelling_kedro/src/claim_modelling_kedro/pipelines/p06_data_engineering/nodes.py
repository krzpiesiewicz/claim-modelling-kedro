from typing import Dict, List

import pandas as pd

from claim_modelling_kedro.pipelines.p01_init.config import Config
from claim_modelling_kedro.pipelines.p06_data_engineering.utils import fit_transform_features


def fit_features_transformer(
    config: Config,
    sample_features_df: Dict[str, pd.DataFrame],
    features_blacklist_text: str
) -> Dict[str, pd.DataFrame]:
    return fit_transform_features(config, sample_features_df, features_blacklist_text)
