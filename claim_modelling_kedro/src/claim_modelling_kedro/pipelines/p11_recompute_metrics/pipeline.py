from kedro.pipeline import Pipeline, pipeline, node

from claim_modelling_kedro.pipelines.p11_recompute_metrics.nodes import recompute_metrics



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=recompute_metrics,
                inputs=["dummy_load_pred_and_trg_df", "config", "sample_train_predictions_df", "sample_train_target_df",
                        "sample_valid_predictions_df", "sample_valid_target_df",
                        "calib_predictions_df", "calib_target_df", "train_predictions_df", "train_target_df",
                        "test_predictions_df", "test_target_df"],
                outputs="dummy_recomp_metrics_df",
                name="recompute_metrics",
            ),
        ]
    )
