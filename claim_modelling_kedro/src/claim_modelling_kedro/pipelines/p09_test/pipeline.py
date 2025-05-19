from kedro.pipeline import Pipeline, pipeline, node

from claim_modelling_kedro.pipelines.p09_test.nodes import eval_model_on_test_and_compute_stats, eval_model_on_train_and_compute_stats


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=eval_model_on_test_and_compute_stats,
                inputs=["config", "calibrated_calib_predictions_df", "features_df", "target_df", "test_keys"],
                outputs="dummy_test_1_df",
                name="test_model_and_compute_stats",
            ),
            node(
                func=eval_model_on_train_and_compute_stats,
                inputs=["config", "calibrated_calib_predictions_df", "features_df", "target_df", "train_keys"],
                outputs="dummy_test_2_df",
                name="eval_model_and_compute_stats",
            ),
        ]
    )
