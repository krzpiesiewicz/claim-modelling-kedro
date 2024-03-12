from kedro.pipeline import Pipeline, node, pipeline

from solvency_models.pipelines.p03_define_modeling_task.nodes import create_modeling_task

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_modeling_task,
                inputs=["config", "freqs", "severities", "features", "train_policies", "calib_policies", "backtest_policies"],
                outputs=["target_df", "features_df", "train_idx", "calib_idx", "backtest_idx"],
                name="create_modeling_task",
            ),
        ]
    )
