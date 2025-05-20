from kedro.pipeline import Pipeline, pipeline, node

from claim_modelling_kedro.pipelines.p10_summary.nodes import load_target_and_predictions, create_concentration_curves, \
    create_prediction_groups_stats_tables



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_target_and_predictions,
                inputs=["dummy_test_1_df", "dummy_test_2_df"],
                outputs=["calib_predictions_df", "calib_target_df",
                         "train_predictions_df", "train_target_df",
                         "test_predictions_df", "test_target_df"],
                name="load_test_predictions",
            ),
            node(
                func=create_concentration_curves,
                inputs=["config", "calib_predictions_df", "calib_target_df", "train_predictions_df", "train_target_df",
                        "test_predictions_df", "test_target_df"],
                outputs="dummy_summary_1_df",
                name="create_concentration_curves",
            ),
            node(
                func=create_prediction_groups_stats_tables,
                inputs=["config", "calib_predictions_df", "calib_target_df", "train_predictions_df", "train_target_df",
                        "test_predictions_df", "test_target_df"],
                outputs="dummy_summary_2_df",
                name="create_prediction_groups_stats_tables",
            ),
        ]
    )
