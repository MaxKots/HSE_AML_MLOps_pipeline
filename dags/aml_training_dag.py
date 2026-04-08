from __future__ import annotations

from datetime import datetime, timedelta
from pprint import pformat

from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import get_current_context

from src.pipelines.training_pipeline import run_training_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


default_args = {
    "owner": "aml-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="aml_training_dag",
    description="Обучение AML-модели на базовом датасете",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["aml", "training", "mlops"],
)
def aml_training_dag():
    start = EmptyOperator(task_id="start")
    finish = EmptyOperator(task_id="finish")

    @task(task_id="read_runtime_config")
    def read_runtime_config_task() -> dict:
        context = get_current_context()
        dag_run = context.get("dag_run")

        conf = dag_run.conf if dag_run and dag_run.conf else {}
        dataset_name = conf.get("dataset_name", "base")

        runtime_config = {
            "dataset_name": dataset_name,
        }

        logger.info(f"Runtime-конфигурация прочитана: {runtime_config}")
        return runtime_config

    @task(task_id="run_training")
    def run_training_task(runtime_config: dict) -> dict:
        dataset_name = runtime_config["dataset_name"]

        logger.info(f"Запуск задачи обучения для датасета '{dataset_name}'")

        summary = run_training_pipeline(dataset_name=dataset_name)

        logger.info("Задача обучения завершена")
        logger.info(pformat(summary))

        return summary

    runtime_config = read_runtime_config_task()
    training_summary = run_training_task(runtime_config)

    start >> runtime_config >> training_summary >> finish


aml_training_dag()
