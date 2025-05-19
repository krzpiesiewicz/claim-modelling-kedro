from kedro.pipeline import Pipeline, pipeline, node

from claim_modelling_kedro.pipelines.p10_summary.nodes import load_train_and_test_predictions, create_concentration_curves



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_train_and_test_predictions,
                inputs=["dummy_test_1_df", "dummy_test_2_df"],
                outputs=["train_predictions_df", "train_target_df", "test_predictions_df", "test_target_df"],
                name="load_test_predictions",
            ),
            node(
                func=create_concentration_curves,
                inputs=["config", "train_predictions_df", "train_target_df", "test_predictions_df", "test_target_df"],
                outputs="dummy_summary_1_df",
                name="create_concentration_curves",
            ),
        ]
    )
