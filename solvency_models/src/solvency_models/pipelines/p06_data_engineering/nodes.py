from typing import Dict

import pandas as pd

from solvency_models.pipelines.p01_init.config import Config
from solvency_models.pipelines.p06_data_engineering.utils import fit_transform_features


def fit_features_transformer(config: Config, sample_features_df: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return fit_transform_features(config, sample_features_df)
