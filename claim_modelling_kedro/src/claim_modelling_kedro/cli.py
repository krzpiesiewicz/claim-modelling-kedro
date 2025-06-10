"""Command line tools for manipulating a Kedro project.
Intended to be invoked via `kedro`."""
import logging
from typing import Dict, Tuple

import click
from kedro.framework.cli.project import (
    ASYNC_ARG_HELP,
    CONFIG_FILE_HELP,
    CONF_SOURCE_HELP,
    FROM_INPUTS_HELP,
    FROM_NODES_HELP,
    LOAD_VERSION_HELP,
    NODE_ARG_HELP,
    PARAMS_ARG_HELP,
    PIPELINE_ARG_HELP,
    RUNNER_ARG_HELP,
    TAG_ARG_HELP,
    TO_NODES_HELP,
    TO_OUTPUTS_HELP,
    project_group,
)
from kedro.framework.cli.utils import (
    CONTEXT_SETTINGS,
    _config_file_callback,
    _split_params,
    _split_load_versions,
    env_option,
    split_string,
    split_node_names,
)
from kedro.framework.session import KedroSession
from kedro.utils import load_obj

logger = logging.getLogger(__name__)

@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""

CONTINUE_MLFLOW_RUN_HELP = ("Continues mlflow run with the given mlflow run_id. If provided 'latest' as mlflow run_id, "
                            "continues the last run, i.e.,\n"
                            "--mlflow-run-id latest")
KEDRO_RUN_NO_HELP = "TODO: KEDRO_RUN_NO_HELP"
KEDRO_RUNTIME_HELP = "TODO: KEDRO_RUNTIME_HELP"
KEDRO_LOGDIR_PATH_HELP = "TODO: KEDRO_LOGDIR_PATH_HELP"
KEDRO_LOG_HTML_FILE_HELP = "TODO: KEDRO_LOG_HTML_FILE_HELP"
KEDRO_LOGS_INDEX_FILE_HELP = "TODO: KEDRO_LOGS_INDEX_FILE_HELP"
KEDRO_LOG_TO_MLFLOW_HELP = "TOOD: KEDRO_LOG_TO_MLFLOW_HELP"
KEDRO_ELAPSED_TIME_HELP = "TODO: KEDRO_ELAPSED_TIME_HELP"
KEDRO_RUN_STATUS_HELP = "TODO: KEDRO_STATUS_HELP"


@project_group.command()
@click.option(
    "--from-inputs", type=str, default="", help=FROM_INPUTS_HELP, callback=split_string
)
@click.option(
    "--to-outputs", type=str, default="", help=TO_OUTPUTS_HELP, callback=split_string
)
@click.option(
    "--from-nodes", type=str, default="", help=FROM_NODES_HELP, callback=split_node_names
)
@click.option(
    "--to-nodes", type=str, default="", help=TO_NODES_HELP, callback=split_node_names
)
@click.option("--nodes", "-n", "node_names", type=str, multiple=True, help=NODE_ARG_HELP)
@click.option(
    "--runner", "-r", type=str, default=None, multiple=False, help=RUNNER_ARG_HELP
)
@click.option("--async", "is_async", is_flag=True, multiple=False, help=ASYNC_ARG_HELP)
@env_option
@click.option("--tags", "-t", type=str, multiple=True, help=TAG_ARG_HELP)
@click.option(
    "--load-versions",
    "-lv",
    type=str,
    multiple=True,
    help=LOAD_VERSION_HELP,
    callback=_split_load_versions,
)
@click.option("--pipeline", "-p", type=str, default="__default__", help=PIPELINE_ARG_HELP)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help=CONFIG_FILE_HELP,
    callback=_config_file_callback,
)
@click.option(
    "--conf-source",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help=CONF_SOURCE_HELP,
)
@click.option(
    "--params",
    type=click.UNPROCESSED,
    default="",
    help=PARAMS_ARG_HELP,
    callback=_split_params
)
@click.option(
    "--mlflow-run-id",
    type=str,
    default=None,
    help=CONTINUE_MLFLOW_RUN_HELP,
)
@click.option(
    "--kedro-run-no",
    type=str,
    default=None,
    help=KEDRO_RUN_NO_HELP,
)
@click.option(
    "--kedro-runtime",
    type=str,
    default=None,
    help=KEDRO_RUNTIME_HELP,
)
@click.option(
    "--kedro-logdir-path",
    type=str,
    default=None,
    help=KEDRO_LOGDIR_PATH_HELP,
)
@click.option(
    "--kedro-log-html-file",
    type=str,
    default=None,
    help=KEDRO_LOG_HTML_FILE_HELP,
)
@click.option(
    "--kedro-logs-index-file",
    type=str,
    default=None,
    help=KEDRO_LOGS_INDEX_FILE_HELP,
)
@click.option(
    "--kedro-log-to-mlflow",
    is_flag=True,
    default=False,
    help=KEDRO_LOG_TO_MLFLOW_HELP,
)
@click.option(
    "--kedro-elapsed-time",
    type=int,
    default=None,
    help=KEDRO_ELAPSED_TIME_HELP,
)
@click.option(
    "--kedro-run-status",
    type=int,
    default=None,
    help=KEDRO_RUN_STATUS_HELP,
)
@click.option(
    "--experiment-and-run-names",
    nargs=2,  # Expect two arguments
    type=(str, str),  # Both arguments are of type string
    default=None,
    help="Specify experiment name and run name as two separate arguments."
)
def run(
    tags,
    env,
    runner,
    is_async,
    node_names,
    to_nodes,
    from_nodes,
    from_inputs,
    to_outputs,
    load_versions,
    pipeline,
    config,
    conf_source,
    params: Dict,
    mlflow_run_id: str,
    kedro_run_no: str,
    kedro_runtime: str,
    kedro_logdir_path: str,
    kedro_log_html_file: str,
    kedro_logs_index_file: str,
    kedro_log_to_mlflow: bool,
    kedro_elapsed_time: int,
    kedro_run_status: int,
    experiment_and_run_names: Tuple[str, str],
):
    """Run the pipeline."""

    runner = load_obj(runner or "SequentialRunner", "kedro.runner")
    tags = tuple(tags)
    node_names = tuple(node_names)

    # logger.info(f"params: {params}")
    # if "kedro_run" not in params:
    #     params["kedro_params"] = dict()
    # kedro_params = params["kedro_params"]
    # kedro_params["kedro_run_no"] = kedro_run_no
    # kedro_params["kedro_runtime"] = kedro_runtime
    # kedro_params["kedro_logdir_path"] = kedro_logdir_path
    extra_params = dict(
        kedro=dict(
            run_no=kedro_run_no,
            runtime=kedro_runtime,
            log_dir=kedro_logdir_path,
            log_html=kedro_log_html_file,
            logs_index_file=kedro_logs_index_file,
            log_to_mlflow=kedro_log_to_mlflow,
            elapsed_time=kedro_elapsed_time,
            pipeline=pipeline,
            run_status=kedro_run_status
        ),
        mlflow_run_id=mlflow_run_id,
        experiment_and_run_names=experiment_and_run_names
    )

    with KedroSession.create(
        env=env, conf_source=conf_source, extra_params=extra_params
    ) as session:

        def run_session():
            session.run(
                tags=tags,
                runner=runner(is_async=is_async),
                node_names=node_names,
                from_nodes=from_nodes,
                to_nodes=to_nodes,
                from_inputs=from_inputs,
                to_outputs=to_outputs,
                load_versions=load_versions,
                pipeline_name=pipeline,
            )

        run_session()
        # from mlflow.tracking import MlflowClient
        # client = MlflowClient()
        # active_runs = client.search_runs(["test-model", "Default"])
        # for run_info in active_runs:
        #     client.set_terminated(run_info.run_id)

        # from kedro.config import OmegaConfigLoader
        # import mlflow
        # from kedro_mlflow.config.kedro_mlflow_config import KedroMlflowConfig

        # project_path = Path.cwd()
        # config_loader = OmegaConfigLoader(str(project_path / settings.CONF_SOURCE))
        # config_loader.config_patterns.update(
        #     {"mlflow": ["mlflow*", "mlflow*/**", "**/mlflow*"]}
        # )
        # logger.info(f"config_loader: {config_loader}")
        # conf_mlflow_yml = config_loader["mlflow"]
        # mlflow_config = KedroMlflowConfig.parse_obj({**conf_mlflow_yml})
        # mlflow_tracking_uri = mlflow_config.server.mlflow_tracking_uri
        # logger.info(f"mlflow_config.server.mlflow_tracking_uri: {mlflow_tracking_uri}")
        # mlflow.set_tracking_uri(mlflow_tracking_uri)
        # mlflow.set_experiment(mlflow_config.tracking.experiment.name)
        #
        # logger.info(f"mlflow_run_id: {mlflow_run_id}")
        # if mlflow_run_id is not None:
        #     if mlflow_run_id == "latest":
        #         pass
        #     with mlflow.start_run(run_id=mlflow_run_id):
        #         logger.info(f"Continuing mlflow_run_id '{mlflow.active_run().info.run_id}'")
        #         run_session()
        # else:
        #     run_session()
