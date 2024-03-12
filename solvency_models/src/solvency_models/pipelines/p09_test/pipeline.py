from kedro.pipeline import Pipeline, pipeline, node

from solvency_models.pipelines.p09_test.nodes import test_model_and_compute_stats


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=test_model_and_compute_stats,
                inputs=["config", "calibrated_calib_predictions_df", "features_df", "target_df", "test_keys"],
                outputs="calibrated_test_predictions_df",
                name="test_model_and_compute_stats",
            ),
        ]
    )
