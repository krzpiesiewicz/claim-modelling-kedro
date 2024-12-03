from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_train_calib_test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_train_calib_test,
                inputs=["config", "target_df"],
                outputs=["train_policies", "calib_policies", "test_policies", "train_keys", "calib_keys",
                         "test_keys"],
                # outputs = None,
                # confirms=["train_policies", "calib_policies", "test_policies"],
                name="split_train_calib_test",
            ),
        ]
    )
