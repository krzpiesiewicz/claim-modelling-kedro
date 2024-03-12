import pandas as pd

from solvency_models.pipelines.p01_init.config import Config


def hypertune_transform(config: Config, transformed_sample_features_df: pd.DataFrame,
                        sample_target_df: pd.DataFrame) -> pd.DataFrame:
    pass
