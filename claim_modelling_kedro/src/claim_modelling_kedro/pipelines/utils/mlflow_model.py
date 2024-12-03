import inspect
import logging

import mlflow
import mlflow.pyfunc
import pandas as pd


def _has_argument(method):
    sig = inspect.signature(method)
    params = sig.parameters
    return len(params) > 1  # Exclude 'self' for instance methods


def _assert_model_has_transform_method(model):
    if not hasattr(model, "transform"):  # or not _has_argument(model.transform):
        raise ValueError("The model in MLFlowModelLogger must have a \"transform\" method "
                         "which have at least one argument.")


class MLFlowModelLogger(mlflow.pyfunc.PythonModel):
    model_name: str
    verbose: bool

    def __init__(self, model, model_name: str = "model", logger=None):
        _assert_model_has_transform_method(model)
        self.model = model
        self.model_name = model_name
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def predict(self, *args, **kwargs):
        """
        This function is only for compatibility with the MLflow API.
        It is required by mlflow.pyfunc.PythonModel.
        """
        return self.model

    def log_model(self, path: str):
        logger = self.logger
        logger.info(f"Saving the {self.model_name} of type {type(self.model)} to MLFlow path {path}...")
        mlflow.pyfunc.log_model(artifact_path=path, python_model=self)
        logger.info(f"Successfully saved the {self.model_name} in {path}.")


class MLFlowModelLoader():
    model_name: str
    verbose: bool

    def __init__(self, model_name: str = "model", logger=None):
        self.model_name = model_name
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def load_model(self, path: str, run_id: str = None):
        logger = self.logger
        # If no MLflow run ID is provided, use the ID of the currently active run
        if run_id is None:
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Using the active MLFlow run ID: {run_id}.")
        else:
            logger.info(f"Using the provided MLFlow run ID: {run_id}.")
        # Define the URI for the MLflow model
        model_uri = f"runs:/{run_id}/{path}"
        logger.info(f"Loading the {self.model_name} from MLFlow path {path}...")
        # Load the model wrapper from the MLflow model registry and extract the model
        wrapper = mlflow.pyfunc.load_model(model_uri)
        model = wrapper.predict(pd.DataFrame([]))
        logger.info(f"Successfully loaded the {self.model_name} of type {type(model)}.")
        # Assert that the model has a transform method
        _assert_model_has_transform_method(model)
        return model
