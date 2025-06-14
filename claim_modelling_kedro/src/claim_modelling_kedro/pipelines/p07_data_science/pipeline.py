from kedro.pipeline import Pipeline, pipeline, node

from claim_modelling_kedro.pipelines.p07_data_science.nodes import (fit_target_transformer, fit_features_selector,
                                                                    tune_hyper_parameters, fit_predictive_model)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=fit_target_transformer,
                inputs=["config", "sample_target_df"],
                outputs="transformed_sample_target_df",
                name="fit_target_transformer",
            ),
            node(
                func=fit_features_selector,
                inputs=["config", "transformed_sample_features_df", "transformed_sample_target_df", "sample_train_keys",
                        "sample_val_keys"],
                outputs="selected_sample_features_df",
                name="fit_features_selector",
            ),
            node(
                func=tune_hyper_parameters,
                inputs=["config", "selected_sample_features_df", "transformed_sample_target_df", "sample_train_keys",
                        "sample_val_keys"],
                outputs=["best_hparams", "best_models"],
                name="tune_hyper_parameters",
            ),
            node(
                func=fit_predictive_model,
                inputs=["config", "selected_sample_features_df", "transformed_sample_target_df", "sample_target_df",
                        "sample_train_keys", "sample_val_keys", "best_hparams", "best_models"],
                outputs="pure_sample_predictions_df",
                name="fit_predictive_model",
            ),
        ]
    )
