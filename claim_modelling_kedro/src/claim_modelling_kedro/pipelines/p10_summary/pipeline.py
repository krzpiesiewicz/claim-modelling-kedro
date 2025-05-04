from kedro.pipeline import Pipeline, pipeline, node

from claim_modelling_kedro.pipelines.p10_summary.nodes import load_test_predictions, create_stats_tables



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_test_predictions,
                inputs=["config", "calibrated_test_predictions_df", "target_df", "test_keys"],
                outputs="loaded_test_predictions_df",
                name="load_test_predictions",
            ),
            node(
                func=create_stats_tables,
                inputs=["config", "loaded_test_predictions_df", "target_df", "test_keys"],
                outputs="percentiles_scores_df",
                name="create_stats_tables",
            ),
        ]
    )
