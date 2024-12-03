from kedro.pipeline import Pipeline, pipeline, node

from claim_modelling_kedro.pipelines.p07_data_science.nodes import fit_features_selector, tune_hyper_parameters, \
    fit_predictive_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=fit_features_selector,
                inputs=["config", "transformed_sample_features_df", "sample_target_df"],
                outputs="selected_sample_features_df",
                name="fit_features_selector",
            ),
            node(
                func=tune_hyper_parameters,
                inputs=["config", "selected_sample_features_df", "sample_target_df"],
                outputs="best_hparams",
                name="tune_hyper_parameters",
            ),
            node(
                func=fit_predictive_model,
                inputs=["config", "selected_sample_features_df", "sample_target_df", "best_hparams"],
                outputs="pure_sample_predictions_df",
                name="fit_predictive_model",
            ),
        ]
    )
