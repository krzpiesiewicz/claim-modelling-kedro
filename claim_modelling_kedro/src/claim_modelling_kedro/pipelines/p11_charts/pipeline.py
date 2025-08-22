from kedro.pipeline import Pipeline, pipeline, node

from claim_modelling_kedro.pipelines.p11_charts.nodes import create_curves_plots, create_lift_charts



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_curves_plots,
                inputs=["dummy_load_pred_and_trg_df", "config", "sample_train_predictions_df", "sample_train_target_df",
                        "sample_valid_predictions_df", "sample_valid_target_df",
                        "calib_predictions_df", "calib_target_df", "train_predictions_df", "train_target_df",
                        "test_predictions_df", "test_target_df"],
                outputs="dummy_charts_1_df",
                name="create_curves_plots",
            ),
            node(
                func=create_lift_charts,
                inputs=["dummy_load_pred_and_trg_df", "config", "sample_train_predictions_df", "sample_train_target_df",
                        "sample_valid_predictions_df", "sample_valid_target_df",
                        "calib_predictions_df", "calib_target_df", "train_predictions_df", "train_target_df",
                        "test_predictions_df", "test_target_df"],
                outputs="dummy_charts_2_df",
                name="create_lift_charts",
            ),
        ]
    )
