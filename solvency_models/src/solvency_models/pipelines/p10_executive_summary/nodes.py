from typing import Dict

import pandas as pd

from solvency_models.pipelines.p01_init.config import Config


def load_test_predictions(config: Config,
                          calibrated_test_predictions_df: pd.Dict[str, pd.DataFrame],
                          target_df: Dict[str, pd.DataFrame],
                          test_keys: Dict[str, pd.Index]) -> Dict[str, pd.DataFrame]:
    pass

def create_stats_tables(config: Config,
                        loaded_test_predictions_df: Dict[str, pd.DataFrame],
                        target_df: Dict[str, pd.DataFrame],
                        test_keys: Dict[str, pd.Index]) -> pd.DataFrame:
    pass
