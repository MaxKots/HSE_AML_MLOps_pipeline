from __future__ import annotations

from datetime import datetime, timedelta
from pprint import pformat

from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import get_current_context
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from src.pipelines.drift_pipeline import run_drift_pipeline
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
    dag_id="aml_drift_dag",
    description="Проверка дрейфа данных с выбранным источником и запуск переобучения при необходимости",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["aml", "drift", "mlops"],
    params={
        "reference_dataset_name": "base",
        "current_dataset_name": "variant_1",
        "source": "csv",
    },
)
def aml_drift_dag():
    start = EmptyOperator(task_id="start")
    no_drift = EmptyOperator(task_id="no_drift_detected")
    finish = EmptyOperator(task_id="finish")

    @task(task_id="read_runtime_config")
    def read_runtime_config_task() -> dict:
        context = get_current_context()
        dag_run = context.get("dag_run")
        params = context.get("params") or {}
        conf = dag_run.conf if dag_run and dag_run.conf else {}

        runtime_config = {
            "reference_dataset_name": conf.get("reference_dataset_name") or params.get("reference_dataset_name") or "base",
            "current_dataset_name": conf.get("current_dataset_name") or params.get("current_dataset_name") or "variant_1",
            "source": conf.get("source") or params.get("source") or "csv",
            "reference_source": conf.get("reference_source"),
            "current_source": conf.get("current_source"),
        }
        logger.info(f"Runtime-конфигурация прочитана: {runtime_config}")
        return runtime_config

    @task(task_id="run_drift_check")
    def run_drift_check_task(runtime_config: dict) -> dict:
        logger.info(f"Запуск drift-check: {runtime_config}")

        summary = run_drift_pipeline(
            reference_dataset_name=runtime_config["reference_dataset_name"],
            current_dataset_name=runtime_config["current_dataset_name"],
            source=runtime_config.get("source"),
            reference_source=runtime_config.get("reference_source"),
            current_source=runtime_config.get("current_source"),
        )

        logger.info("Drift-check завершён")
        logger.info(pformat(summary))
        return summary

    @task.branch(task_id="branch_on_drift")
    def branch_on_drift_task(drift_summary: dict) -> str:
        if bool(drift_summary["drift_detected"]):
            logger.info("Обнаружен дрейф, переход в ветку retraining")
            return "trigger_retraining"
        logger.info("Дрейф не обнаружен, завершение без переобучения")
        return "no_drift_detected"

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="aml_training_dag",
        conf={
            "dataset_name": "{{ ti.xcom_pull(task_ids='read_runtime_config')['reference_dataset_name'] }}",
            "source": "{{ ti.xcom_pull(task_ids='read_runtime_config')['source'] }}",
        },
        wait_for_completion=False,
        reset_dag_run=True,
    )

    runtime_config = read_runtime_config_task()
    drift_summary = run_drift_check_task(runtime_config)
    branch_decision = branch_on_drift_task(drift_summary)

    start >> runtime_config >> drift_summary >> branch_decision
    branch_decision >> trigger_retraining >> finish
    branch_decision >> no_drift >> finish


aml_drift_dag()
