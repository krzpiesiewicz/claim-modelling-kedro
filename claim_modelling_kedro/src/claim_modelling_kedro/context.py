import fcntl
import logging
import os
import tempfile
from pathlib import Path
from typing import Union, Dict, Any
import subprocess

import mlflow
import pandas as pd
from IPython import get_ipython
from kedro.config import AbstractConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.session import KedroSession
from kedro_mlflow.config.kedro_mlflow_config import KedroMlflowConfig
from pluggy import PluginManager

from claim_modelling_kedro.experiments import get_run_mlflow_id, set_run_mlflow_id

logger = logging.getLogger(__name__)


def get_git_commit_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        return sha
    except Exception:
        return "unknown"


class ProjectContext(KedroContext):
    kedro_runs_df_filename = "kedro_runs.csv"

    def __init__(
            self,
            package_name: str,
            project_path: Union[Path, str],
            config_loader: AbstractConfigLoader,
            hook_manager: PluginManager,
            env: str = None,
            extra_params: Dict[str, Any] = None,
    ):
        if extra_params is None:
            raise Exception("Probably you are running kedro by %load_ext kedro.ipython. Now, run\n"
                        "%reload_kedro --params ipython_notebook=True\n"
                        "in your jupyter notebook, please. Alternatively, you can run\n"
                        "%reload_kedro --params ipython_notebook=True,mlflow_run_id=\"<mlflow-run-id>\"")

        self.mlflow_run_id = None
        self.experiment_name, self.experiment_run_name = extra_params.get("experiment_and_run_names") or (None, None)
        if self.experiment_name is not None:
            experiment_mlflow_run_id = get_run_mlflow_id(self.experiment_name, self.experiment_run_name)
            self.mlflow_run_id = extra_params["mlflow_run_id"] or experiment_mlflow_run_id

        self.ipython_notebook = extra_params.get("ipython_notebook", False)
        if self.ipython_notebook:
            extra_params.pop("ipython_notebook")
            self.mlflow_run_id = extra_params.get("mlflow_run_id") or self.mlflow_run_id
            if "mlflow_run_id" in extra_params:
                extra_params.pop("mlflow_run_id")
        else:
            self.kedro_no = int(extra_params["kedro"]["run_no"])
            self.kedro_log_dir = extra_params["kedro"]["log_dir"]
            self.kedro_pipeline = extra_params["kedro"]["pipeline"]
            self.kedro_log_to_mlflow = extra_params["kedro"]["log_to_mlflow"]
            if self.kedro_log_to_mlflow:
                self.kedro_html_file = extra_params["kedro"]["log_html"]
                self.kedro_execution_time = extra_params["kedro"]["elapsed_time"]
                self.kedro_run_status = extra_params["kedro"]["run_status"]
                self.mlflow_run_id = self.read_mlflow_run_id()
            else:
                self.kedro_runtime = pd.Timestamp(extra_params["kedro"]["runtime"])
                self.logs_index_file = extra_params["kedro"]["logs_index_file"]
                self.mlflow_run_id = extra_params["mlflow_run_id"] or self.mlflow_run_id
            for k in ["kedro", "mlflow_run_id"]:
                extra_params.pop(k)


        config_loader.config_patterns.update(
            {"mlflow": ["mlflow*", "mlflow*/**", "**/mlflow*"],
             "logging": ["logging*", "logging*/**", "**/logging*"]}
        )
        super().__init__(package_name=package_name, project_path=project_path, config_loader=config_loader,
                         hook_manager=hook_manager, env=env,
                         extra_params=extra_params)

        conf_mlflow_yml = config_loader["mlflow"]
        mlflow_config = KedroMlflowConfig.parse_obj({**conf_mlflow_yml})
        mlflow_tracking_uri = mlflow_config.server.mlflow_tracking_uri
        self.experiment_name = self.experiment_name or mlflow_config.tracking.experiment.name
        logger.info(f"mlflow_config.server.mlflow_tracking_uri: {mlflow_tracking_uri}")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        self.mlflow_run_id = self.mlflow_run_id or mlflow_config.tracking.run.id
        if self.mlflow_run_id is not None:
            if mlflow.active_run() is not None:
                mlflow.end_run()
            mlflow.start_run(self.mlflow_run_id, run_name=self.experiment_run_name)
        else:
            if mlflow.active_run() is not None:
                mlflow.end_run()
            mlflow.start_run(run_name=self.experiment_run_name)
            self.mlflow_run_id = mlflow.active_run().info.run_id

        logger.info(f"mlflow_run_id: {self.mlflow_run_id}")
        logger.debug(f"self.params: {self.params}")

        if self.experiment_run_name is not None:
            experiment_mlflow_run_id = get_run_mlflow_id(self.experiment_name, self.experiment_run_name)
            if experiment_mlflow_run_id is None or experiment_mlflow_run_id != self.mlflow_run_id:
                set_run_mlflow_id(self.experiment_name, self.experiment_run_name, self.mlflow_run_id)

        if not self.ipython_notebook:
            if self.kedro_log_to_mlflow:
                kedro_execution_time_td = pd.to_timedelta(self.kedro_execution_time, unit='s')
                logger.info(f"Kedro execution time: {kedro_execution_time_td}")
                mlflow.log_artifact(self.kedro_html_file, artifact_path="kedro_runs")
                # Add the execution time to kedro_runs_df and log it to mlflow
                kedro_runs_df = self.get_kedro_runs_df_from_mlflow()
                last_row_idx = len(kedro_runs_df) - 1
                logger.debug(
                    f'kedro_runs_df.loc[last_row_idx, "run_no"]: {kedro_runs_df.loc[last_row_idx, "run_no"]} ({type(kedro_runs_df.loc[last_row_idx, "run_no"])})')
                logger.debug(f'self.kedro_no: {self.kedro_no} ({type(self.kedro_no)})')
                assert kedro_runs_df.loc[last_row_idx, "run_no"] == self.kedro_no
                if "execution_time" not in kedro_runs_df.columns:
                    kedro_runs_df["execution_time"] = None
                kedro_runs_df.loc[last_row_idx, "execution_time"] = kedro_execution_time_td
                kedro_run_status = "SUCCESS" if self.kedro_run_status == 0 else "FAILURE"
                if "status" not in kedro_runs_df.columns:
                    kedro_runs_df["status"] = None
                kedro_runs_df.loc[last_row_idx, "status"] = kedro_run_status
                self.log_kedro_runs_df_to_mlflow(kedro_runs_df)
            else:
                # Save mlflow_run_id to a file and append to index
                with open(os.path.join(self.kedro_log_dir, 'mlflow_run_id.txt'), 'w') as file:
                    file.write(self.mlflow_run_id)
                self.append_mlflow_run_id()

                kedro_runs_df = self.get_kedro_runs_df_from_mlflow()
                logger.debug(f"kedro_runs_df: {kedro_runs_df}")

                # Add a new row to the DataFrame
                new_row = pd.DataFrame(dict(
                    run_no=[self.kedro_no],
                    pipeline=[self.kedro_pipeline],
                    runtime=[self.kedro_runtime],
                    execution_time=None,
                    status=None,
                    logdir=[self.kedro_log_dir],
                    commit_sha=[get_git_commit_sha()]
                ))
                logger.debug(f"new_row: {new_row}")
                kedro_runs_df = pd.concat([kedro_runs_df, new_row],
                                          ignore_index=True) if kedro_runs_df is not None else new_row
                self.log_kedro_runs_df_to_mlflow(kedro_runs_df)


    def append_mlflow_run_id(self):
        with open(self.logs_index_file, "r+") as file:
            fcntl.flock(file, fcntl.LOCK_EX)  # acquire an exclusive lock
            lines = file.readlines()

            for i, line in enumerate(lines):
                if line.startswith(f"run: {self.kedro_no}"):
                    lines[i] = line.strip() + f", mlflow_run_id: {self.mlflow_run_id}\n"

            file.seek(0)  # move file pointer to the beginning
            file.writelines(lines)
            fcntl.flock(file, fcntl.LOCK_UN)  # release the lock

    def read_mlflow_run_id(self) -> str:
        mlflow_run_id_file = os.path.join(self.kedro_log_dir, "mlflow_run_id.txt")
        try:
            with open(mlflow_run_id_file, "r") as file:
                mlflow_run_id = file.read().strip()
            return mlflow_run_id
        except FileNotFoundError:
            logger.error(f"File {mlflow_run_id_file} not found.")
            return None

    def get_kedro_runs_df_from_mlflow(self) -> pd.DataFrame:
        artifacts_info_files = mlflow.artifacts.list_artifacts(run_id=self.mlflow_run_id)
        logger.debug(f"artifacts_info_files: {artifacts_info_files}")
        for file_info in artifacts_info_files:
            if file_info.path == self.kedro_runs_df_filename:
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        local_path = mlflow.artifacts.download_artifacts(
                            run_id=self.mlflow_run_id,
                            artifact_path=self.kedro_runs_df_filename,
                            dst_path=os.path.join(temp_dir, self.kedro_runs_df_filename))
                        logger.info("Artifact 'kedro_runs.csv' exists in the current mlflow session.")

                        # Read the artifact as a pandas DataFrame
                        return pd.read_csv(local_path)
                except Exception as e:
                    logger.warning(
                        f"Artifact 'kedro_runs.csv' does not exist in the current mlflow session. Error: {str(e)}")
        logger.debug(f"Artifact 'kedro_runs.csv' does not exist in the current mlflow session.")
        return None

    def log_kedro_runs_df_to_mlflow(self, kedro_runs_df: pd.DataFrame):
        # Save the DataFrame to a CSV file in a temporary directory and log it as an artifact
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, self.kedro_runs_df_filename)
            kedro_runs_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)
