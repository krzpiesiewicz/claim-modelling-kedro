from kedro.pipeline import Pipeline, pipeline, node

from claim_modelling_kedro.pipelines.p10_load_pred_and_trg.nodes import load_target_and_predictions


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_target_and_predictions,
                inputs=["dummy_test_1_df", "dummy_test_2_df"],
                outputs=["dummy_load_pred_and_trg_df",
                         "sample_train_predictions_df", "sample_train_target_df",
                         "sample_valid_predictions_df", "sample_valid_target_df",
                         "calib_predictions_df", "calib_target_df",
                         "train_predictions_df", "train_target_df",
                         "test_predictions_df", "test_target_df"],
                name="load_target_and_predictions",
            ),
        ]
    )
