from kedro.pipeline import Pipeline, pipeline, node

from solvency_models.pipelines.p08_model_calibration.nodes import fit_calibration_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=fit_calibration_model,
                inputs=["config", "pure_sample_predictions_df", "features_df", "target_df", "calib_keys"],
                outputs="calibrated_calib_predictions_df",
                name="fit_calibration_model",
            ),
        ]
    )
