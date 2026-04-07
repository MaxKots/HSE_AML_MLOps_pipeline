from future import annotations

from datetime import datetime, timedelta
from pprint import pformat

from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from src.pipelines.drift_pipeline import run_drift_pipeline
from src.utils.logger import get_logger

logger = get_logger(name)

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
description="Проверка дрейфа данных и запуск переобучения при необходимости",
default_args=default_args,
start_date=datetime(2025, 1, 1),
schedule="@daily",
catchup=False,
tags=["aml", "drift", "mlops"],
)
def aml_drift_dag():
start = EmptyOperator(task_id="start")
no_drift = EmptyOperator(task_id="no_drift_detected")
finish = EmptyOperator(task_id="finish")



@task(task_id="run_drift_check")
def run_drift_check_task(current_dataset_name: str = "variant_1") -> dict:
    logger.info(f"Запуск drift-check для датасета '{current_dataset_name}'")

    summary = run_drift_pipeline(
        reference_dataset_name="base",
        current_dataset_name=current_dataset_name,
    )

    logger.info("Drift-check завершён")
    logger.info(pformat(summary))

    return summary

@task.branch(task_id="branch_on_drift")
def branch_on_drift_task(drift_summary: dict) -> str:
    drift_detected = bool(drift_summary["drift_detected"])

    if drift_detected:
        logger.info("Обнаружен дрейф, переход в ветку retraining")
        return "trigger_retraining"

    logger.info("Дрейф не обнаружен, завершение без переобучения")
    return "no_drift_detected"

trigger_retraining = TriggerDagRunOperator(
    task_id="trigger_retraining",
    trigger_dag_id="aml_training_dag",
    conf={"dataset_name": "base"},
    wait_for_completion=False,
    reset_dag_run=True,
)

drift_summary = run_drift_check_task()
branch_decision = branch_on_drift_task(drift_summary)

start >> drift_summary >> branch_decision
branch_decision >> trigger_retraining >> finish
branch_decision >> no_drift >> finish

aml_drift_dag()
