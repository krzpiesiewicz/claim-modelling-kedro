from kedro.pipeline import Pipeline, pipeline, node




def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=test_model_and_compute_stats,
                inputs=["config", calibrated_test_predictions_df],
                outputs="calibrated_test_predictions_df",
                name="test_model_and_compute_stats",
            ),
        ]
    )
