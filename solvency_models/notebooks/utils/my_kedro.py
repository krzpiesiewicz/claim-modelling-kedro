import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Union, Iterable

import mlflow
from IPython import get_ipython
from kedro.framework.session import KedroSession
from kedro.runner import AbstractRunner

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


@contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


@contextmanager
def suppress_stdout_stderr():
    with suppress_stdout():
        with suppress_stderr():
            yield


def _str_to_logging_level(name: str) -> int:
    return logging._nameToLevel[name.upper()]


def get_all_loggers_and_levels() -> Dict[str, int]:
    logger_dict = logging.Logger.manager.loggerDict
    return {name: logger.level for name, logger in logger_dict.items() if isinstance(logger, logging.Logger)}


def set_all_loggers_to_level(level: Union[int, str], only_if_higher: bool = True) -> None:
    if type(level) is str:
        level = _str_to_logging_level(level)
    logger_dict = logging.Logger.manager.loggerDict
    for logger in logger_dict.values():
        if isinstance(logger, logging.Logger):
            if level > logger.getEffectiveLevel() or not only_if_higher:
                logger.setLevel(level)


def restore_loggers(previous_levels: Dict[str, int]) -> None:
    logger_dict = logging.Logger.manager.loggerDict
    for name, level in previous_levels.items():
        if name in logger_dict and isinstance(logger_dict[name], logging.Logger):
            logger_dict[name].setLevel(level)


@contextmanager
def suppress_all_loggers(level: Union[int, str], only_if_higher: bool = True):
    previous_levels = get_all_loggers_and_levels()
    try:
        # Set all loggers to ERROR
        set_all_loggers_to_level(level, only_if_higher)
        yield
    finally:
        # Restore previous levels
        restore_loggers(previous_levels)


class MyKedro():

    def __init__(self, mlflow_run_id: str = None, project_path: str = None, level: Union[int, str] = "ERROR",
                 only_if_higher: bool = True):
        ipython = get_ipython()

        if ipython is None:
            raise EnvironmentError("This class can be created only in an IPython environment.")

        self.mlflow_run_id = mlflow_run_id
        if project_path is not None:
            self.project_path = project_path
        else:
            self.project_path = "/home/krzysiek/Development/solvency-modelling/solvency_models"

        def _load():
            self._reset_path()
            try:
                ipython.run_line_magic("load_ext", "kedro.ipython")
            except:
                pass

        if level is not None:
            with suppress_all_loggers(level, only_if_higher):
                _load()
        else:
            _load()
        self.reload(mlflow_run_id, level, only_if_higher)

    def reload(self, mlflow_run_id: str = None, level: Union[int, str] = "ERROR", only_if_higher: bool = True) -> None:
        self._reset_path()
        if mlflow_run_id is None:
            mlflow_run_id = self.mlflow_run_id
        params = "--params ipython_notebook=True"
        if mlflow_run_id is not None:
            params += f",mlflow_run_id={mlflow_run_id}"
        if level is not None:
            with suppress_all_loggers(level, only_if_higher):
                get_ipython().run_line_magic("reload_kedro", params)
        else:
            get_ipython().run_line_magic("reload_kedro", params)
        self.mlflow_run_id = mlflow.active_run().info.run_id
        logger.info(f"Reloaded kedro. Active mlflow_run_id: {self.mlflow_run_id}")

    def run(self, pipeline_name: str = "__default__", level: Union[int, str] = None, only_if_higher: bool = True,
            tags: Iterable[str] = None,
            runner: AbstractRunner = None,
            node_names: Iterable[str] = None,
            from_nodes: Iterable[str] = None,
            to_nodes: Iterable[str] = None,
            from_inputs: Iterable[str] = None,
            to_outputs: Iterable[str] = None,
            load_versions: Dict[str, str] = None,
            namespace: str = None,
            ) -> Any:
        session = self._create_session()
        if level is not None:
            with suppress_all_loggers(level, only_if_higher):
                res = session.run(pipeline_name=pipeline_name,
                                  tags=tags,
                                  runner=runner,
                                  node_names=node_names,
                                  from_nodes=from_nodes,
                                  to_nodes=to_nodes,
                                  from_inputs=from_inputs,
                                  to_outputs=to_outputs,
                                  load_versions=load_versions,
                                  namespace=namespace)
        else:
            res = session.run(pipeline_name=pipeline_name,
                              tags=tags,
                              runner=runner,
                              node_names=node_names,
                              from_nodes=from_nodes,
                              to_nodes=to_nodes,
                              from_inputs=from_inputs,
                              to_outputs=to_outputs,
                              load_versions=load_versions,
                              namespace=namespace)
        return res

    def _reset_path(self) -> None:
        with suppress_stdout_stderr():
            with suppress_all_loggers("ERROR"):
                get_ipython().run_line_magic("cd", self.project_path)

    def _create_session(self) -> KedroSession:
        extra_params = dict(ipython_notebook=True, mlflow_run_id=self.mlflow_run_id)
        self._reset_path()
        return KedroSession.create(self.project_path, env="local", extra_params=extra_params)
