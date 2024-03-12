from kedro.pipeline import Pipeline, pipeline, node

from solvency_models.pipelines.p07_data_science.nodes import select_features, fit_predictive_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=select_features,
                inputs=["config", "transformed_sample_features_df", "sample_target_df"],
                outputs="selected_sample_features_df",
                name="select_features",
            ),
            node(
                func=fit_predictive_model,
                inputs=["config", "selected_sample_features_df", "sample_target_df"],
                outputs="pure_sample_predictions_df",
                name="fit_predictive_model",
            ),
        ]
    )
