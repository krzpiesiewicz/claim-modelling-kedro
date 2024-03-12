from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_features_and_targets, split_train_calib_backtest


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_features_and_targets,
                inputs=["config", "raw_features_and_claims_numbers", "raw_severities"],
                outputs=["features", "freqs", "severities"],
                name="split_features_and_targets",
            ),
            node(
                func=split_train_calib_backtest,
                inputs=["config", "freqs"],
                outputs=["train_policies", "calib_policies", "backtest_policies"],
                name="split_train_calib_backtest",
            ),
        ]
    )
