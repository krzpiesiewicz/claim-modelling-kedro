from kedro.pipeline import Pipeline, pipeline, node

from claim_modelling_kedro.pipelines.p10_summary.nodes import load_target_and_predictions, create_curves_plots, \
    create_lift_charts



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_target_and_predictions,
                inputs=["dummy_test_1_df", "dummy_test_2_df"],
                outputs=["sample_train_predictions_df", "sample_train_target_df",
                         "sample_valid_predictions_df", "sample_valid_target_df",
                         "calib_predictions_df", "calib_target_df",
                         "train_predictions_df", "train_target_df",
                         "test_predictions_df", "test_target_df"],
                name="load_test_predictions",
            ),
            node(
                func=create_curves_plots,
                inputs=["config", "sample_train_predictions_df", "sample_train_target_df",
                        "sample_valid_predictions_df", "sample_valid_target_df",
                        "calib_predictions_df", "calib_target_df", "train_predictions_df", "train_target_df",
                        "test_predictions_df", "test_target_df"],
                outputs="dummy_summary_1_df",
                name="create_curves_plots",
            ),
            node(
                func=create_lift_charts,
                inputs=["config", "sample_train_predictions_df", "sample_train_target_df",
                        "sample_valid_predictions_df", "sample_valid_target_df",
                        "calib_predictions_df", "calib_target_df", "train_predictions_df", "train_target_df",
                        "test_predictions_df", "test_target_df"],
                outputs="dummy_summary_2_df",
                name="create_lift_charts",
            ),
        ]
    )
