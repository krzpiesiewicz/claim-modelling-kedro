import ast
import logging
import os
import tempfile
from dataclasses import dataclass, asdict
from typing import Dict

import mlflow
import yaml

from claim_modelling_kedro.pipelines.p01_init.clb_config import CalibrationConfig
from claim_modelling_kedro.pipelines.p01_init.data_config import DataConfig
from claim_modelling_kedro.pipelines.p01_init.de_config import DataEngineeringConfig
from claim_modelling_kedro.pipelines.p01_init.ds_config import DataScienceConfig
from claim_modelling_kedro.pipelines.p01_init.exprmnt import ExperimentInfo
from claim_modelling_kedro.pipelines.p01_init.mdl_task_config import ModelTask
from claim_modelling_kedro.pipelines.p01_init.smpl_config import SamplingConfig
from claim_modelling_kedro.pipelines.p01_init.summary_config import SummaryConfig
from claim_modelling_kedro.pipelines.p01_init.test_config import TestConfig

logger = logging.getLogger(__name__)


@dataclass
class Config:
    exprmnt: ExperimentInfo
    data: DataConfig
    mdl_task: ModelTask
    smpl: SamplingConfig
    de: DataEngineeringConfig
    ds: DataScienceConfig
    clb: CalibrationConfig
    test: TestConfig
    summary: SummaryConfig

    def __init__(self, parameters: Dict):
        self.exprmnt = ExperimentInfo(parameters)
        self.data = DataConfig(parameters)
        self.mdl_task = ModelTask(exprmnt=self.exprmnt, data=self.data)
        self.smpl = SamplingConfig(parameters)
        self.de = DataEngineeringConfig(parameters)
        self.ds = DataScienceConfig(parameters, exprmnt=self.exprmnt, smpl=self.smpl)
        self.clb = CalibrationConfig(parameters, mdl_task=self.mdl_task)
        self.test = TestConfig(parameters)
        self.summary = SummaryConfig(parameters)


def log_config_to_mlflow(config: Config):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "config.yml")
        with open(file_path, "w") as file:
            yaml.dump(asdict(config), file)
        mlflow.log_artifact(file_path)


def add_sub_runs_ids_to_config(config: Config) -> Config:
    run = mlflow.active_run()
    root_run_id = run.info.run_id
    root_run_name = run.info.run_name
    logger.info(f"Searching for sub runs in MLFlow run ID: {run.info.run_id}.")
    if "cv_subruns" in run.data.params:
        subruns_dct = get_subruns_dct(run)
        config.data.cv_mlflow_runs_ids = subruns_dct
        logger.info(f"Loaded cross validation sub runs: \n" +
                    "\n".join([f" - {part}: {run_id}" for part, run_id in subruns_dct.items()]))
    else:
        logger.info("No sub runs found.")
        logger.info("Creating sub runs...")
        subruns_dct = {}
        for part in config.data.cv_parts_names:
            with mlflow.start_run(run_name=f"CV_{part}", nested=True) as subrun:
                subrun_id = subrun.info.run_id
                subruns_dct[part] = subrun_id
                mlflow.log_param("cv_part", part)
                mlflow.log_param("parent_run_id", root_run_id)
                mlflow.log_param(f"parent_run_name", root_run_name)
                mlflow.log_param(f"root_run_id", root_run_id)
                mlflow.log_param(f"root_run_name", root_run_name)
                mlflow.log_param(f"is_root", False)
                logger.info(f"Created sub run for cross validation split {part} with ID: {subrun_id}")
        config.data.cv_mlflow_runs_ids = subruns_dct
        mlflow.log_param(f"cv_subruns", subruns_dct)
        mlflow.log_param(f"root_run_id", root_run_id)
        mlflow.log_param(f"root_run_name", root_run_name)
        mlflow.log_param(f"is_root", True)
        logger.info("Created all sub runs and logged ids to MLflow.")
    return config


def get_subruns_dct(run) -> Dict[str, str]:
    logger.info(f"type(run): {type(run)}")
    if "cv_subruns" in run.data.params:
        return ast.literal_eval(run.data.params["cv_subruns"])
    else:
        return None
