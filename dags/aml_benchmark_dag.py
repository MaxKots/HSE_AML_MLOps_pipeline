from __future__ import annotations

import os
import shutil
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator

from config.settings import settings
from src.data import DataLoader, DataTransformer, DataValidator
from src.data.synthaml import prepare_synthaml_dataset_from_frames
from src.features import FeatureEngineer
from src.models.evaluate import calculate_classification_metrics
from src.models.predict import AMLPredictor
from src.models.train import AMLModelTrainer
from src.utils.logger import get_logger
from src.utils.paths import ensure_directories


logger = get_logger(__name__)

PROJECT_ROOT = Path("/opt/project")
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
BENCHMARK_MODELS_DIR = ARTIFACTS_DIR / "models" / "benchmark"

TARGET_COLUMN = "fraud_bool"
DEFAULT_THRESHOLD = 0.5

THRESHOLD_MODE_DEFAULT = "default_0.5"
THRESHOLD_MODE_BEST_F1 = "best_f1_valid"
THRESHOLD_MODE_ALERT_20 = "best_recall_alert_20_valid"
THRESHOLD_MODE_ALERT_30 = "best_recall_alert_30_valid"
THRESHOLD_MODE_FPR_05 = "best_recall_fpr_05_valid"

BENCHMARK_MLFLOW_TRACKING_URI = os.getenv(
    "BENCHMARK_MLFLOW_TRACKING_URI",
    "file:/opt/project/artifacts/mlruns",
)


BENCHMARK_JOBS: list[dict[str, Any]] = [
    {
        "job_id": "base_lightgbm_raw",
        "scenario": "Base",
        "dataset": "Base",
        "train_dataset_name": "base",
        "eval_dataset_name": "base",
        "evaluation_mode": "held_out_test",
        "model_type": "lightgbm",
        "feature_set": "raw",
        "threshold": DEFAULT_THRESHOLD,
        "output_file": "base_lightgbm_raw.csv",
    },
    {
        "job_id": "base_xgboost_raw",
        "scenario": "Base",
        "dataset": "Base",
        "train_dataset_name": "base",
        "eval_dataset_name": "base",
        "evaluation_mode": "held_out_test",
        "model_type": "xgboost",
        "feature_set": "raw",
        "threshold": DEFAULT_THRESHOLD,
        "output_file": "base_xgboost_raw.csv",
    },
    {
        "job_id": "base_lightgbm_engineered",
        "scenario": "Base",
        "dataset": "Base",
        "train_dataset_name": "base",
        "eval_dataset_name": "base",
        "evaluation_mode": "held_out_test",
        "model_type": "lightgbm",
        "feature_set": "engineered",
        "threshold": DEFAULT_THRESHOLD,
        "output_file": "base_lightgbm_engineered.csv",
    },
    {
        "job_id": "base_xgboost_engineered",
        "scenario": "Base",
        "dataset": "Base",
        "train_dataset_name": "base",
        "eval_dataset_name": "base",
        "evaluation_mode": "held_out_test",
        "model_type": "xgboost",
        "feature_set": "engineered",
        "threshold": DEFAULT_THRESHOLD,
        "output_file": "base_xgboost_engineered.csv",
    },
    {
        "job_id": "synthaml_lightgbm_prepared",
        "scenario": "SynthAML",
        "dataset": "SynthAML",
        "train_dataset_name": "synthaml",
        "eval_dataset_name": "synthaml",
        "evaluation_mode": "held_out_test",
        "model_type": "lightgbm",
        "feature_set": "prepared",
        "threshold": DEFAULT_THRESHOLD,
        "output_file": "synthaml_lightgbm_prepared.csv",
    },
    {
        "job_id": "synthaml_xgboost_prepared",
        "scenario": "SynthAML",
        "dataset": "SynthAML",
        "train_dataset_name": "synthaml",
        "eval_dataset_name": "synthaml",
        "evaluation_mode": "held_out_test",
        "model_type": "xgboost",
        "feature_set": "prepared",
        "threshold": DEFAULT_THRESHOLD,
        "output_file": "synthaml_xgboost_prepared.csv",
    },
    {
        "job_id": "variant_1_lightgbm_engineered",
        "scenario": "Variant I",
        "dataset": "Variant I",
        "train_dataset_name": "base",
        "eval_dataset_name": "variant_1",
        "evaluation_mode": "external_dataset",
        "model_type": "lightgbm",
        "feature_set": "engineered",
        "threshold": DEFAULT_THRESHOLD,
        "output_file": "variant_1_lightgbm_engineered.csv",
    },
    {
        "job_id": "variant_1_xgboost_engineered",
        "scenario": "Variant I",
        "dataset": "Variant I",
        "train_dataset_name": "base",
        "eval_dataset_name": "variant_1",
        "evaluation_mode": "external_dataset",
        "model_type": "xgboost",
        "feature_set": "engineered",
        "threshold": DEFAULT_THRESHOLD,
        "output_file": "variant_1_xgboost_engineered.csv",
    },
    {
        "job_id": "variant_2_lightgbm_engineered",
        "scenario": "Variant II",
        "dataset": "Variant II",
        "train_dataset_name": "base",
        "eval_dataset_name": "variant_2",
        "evaluation_mode": "external_dataset",
        "model_type": "lightgbm",
        "feature_set": "engineered",
        "threshold": DEFAULT_THRESHOLD,
        "output_file": "variant_2_lightgbm_engineered.csv",
    },
    {
        "job_id": "variant_2_xgboost_engineered",
        "scenario": "Variant II",
        "dataset": "Variant II",
        "train_dataset_name": "base",
        "eval_dataset_name": "variant_2",
        "evaluation_mode": "external_dataset",
        "model_type": "xgboost",
        "feature_set": "engineered",
        "threshold": DEFAULT_THRESHOLD,
        "output_file": "variant_2_xgboost_engineered.csv",
    },
]


NUMERIC_COLUMNS = [
    "threshold",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1_score",
    "false_positive_rate",
    "alert_rate",
    "positive_share",
    "precision_at_100",
    "recall_at_100",
    "lift_at_100",
    "precision_at_500",
    "recall_at_500",
    "lift_at_500",
    "tp",
    "fp",
    "fn",
    "tn",
    "train_rows",
    "valid_rows",
    "test_rows",
    "train_time_sec",
    "inference_time_sec",
]


def configure_benchmark_mlflow() -> None:
    """
    Настраивает локальное MLflow-хранилище для benchmark-задач.
    """
    import mlflow

    tracking_uri = BENCHMARK_MLFLOW_TRACKING_URI

    if tracking_uri.startswith("file:"):
        mlruns_dir = Path(tracking_uri.replace("file:", "", 1))
        mlruns_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    settings.mlflow_tracking_uri = tracking_uri

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("aml_benchmark")


def split_counts(total_rows: int) -> tuple[int, int, int]:
    """
    Возвращает размеры train / valid / test так же,
    как это делает AMLModelTrainer.split_dataset().
    """
    train_rows = int(total_rows * 0.7)
    valid_rows = int(total_rows * 0.15)
    test_rows = total_rows - train_rows - valid_rows

    return train_rows, valid_rows, test_rows


def calculate_business_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    top_k_values: tuple[int, ...] = (100, 500),
) -> dict[str, float | int]:
    """
    Считает эксплуатационные метрики:
    FPR, alert rate, confusion matrix, Precision@k, Recall@k и Lift@k.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    y_pred = (y_proba >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    alert_rate = float(y_pred.mean()) if len(y_pred) else 0.0
    false_positive_rate = float(fp / (fp + tn)) if (fp + tn) else 0.0
    positive_share = float(y_true.mean()) if len(y_true) else 0.0

    result: dict[str, float | int] = {
        "false_positive_rate": false_positive_rate,
        "alert_rate": alert_rate,
        "positive_share": positive_share,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

    sorted_idx = np.argsort(y_proba)[::-1]
    positives_total = int(y_true.sum())

    for k in top_k_values:
        k_eff = min(k, len(y_true))
        top_idx = sorted_idx[:k_eff]

        if k_eff == 0:
            precision_at_k = 0.0
            recall_at_k = 0.0
        else:
            top_positives = int(y_true[top_idx].sum())
            precision_at_k = float(top_positives / k_eff)
            recall_at_k = float(top_positives / positives_total) if positives_total else 0.0

        lift_at_k = float(precision_at_k / positive_share) if positive_share else 0.0

        result[f"precision_at_{k}"] = precision_at_k
        result[f"recall_at_{k}"] = recall_at_k
        result[f"lift_at_{k}"] = lift_at_k

    return result


def get_base_categorical_columns(df: pd.DataFrame) -> list[str]:
    """
    Возвращает категориальные признаки основного Fraud-датасета,
    которые реально присутствуют в текущем датафрейме.
    """
    categorical_candidates = [
        "payment_type",
        "employment_status",
        "housing_status",
        "source",
        "device_os",
    ]

    return [column for column in categorical_candidates if column in df.columns]


def get_raw_numerical_columns(
    df: pd.DataFrame,
    categorical_columns: list[str],
) -> list[str]:
    """
    Возвращает числовые признаки для raw-сценария.
    """
    excluded_columns = set(categorical_columns + [TARGET_COLUMN, "event_time"])

    return [
        column
        for column in df.columns
        if column not in excluded_columns
    ]


def load_and_prepare_main_dataset(
    dataset_name: str,
    feature_set: str,
    cache: dict[str, tuple[pd.DataFrame, list[str], list[str]]],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    cache_key = f"{dataset_name}:{feature_set}"

    if cache_key in cache:
        return cache[cache_key]

    loader = DataLoader()
    validator = DataValidator()
    transformer = DataTransformer()
    feature_engineer = FeatureEngineer()

    raw_df = loader.load_dataset(dataset_name)
    validator.run_full_validation(raw_df)

    transformed_df = transformer.transform(raw_df)

    if feature_set == "engineered":
        feature_result = feature_engineer.build_features(transformed_df)
        prepared = (
            feature_result.dataframe,
            feature_result.categorical_columns,
            feature_result.numerical_columns,
        )

    elif feature_set == "raw":
        categorical_columns = get_base_categorical_columns(transformed_df)
        numerical_columns = get_raw_numerical_columns(transformed_df, categorical_columns)
        prepared = (transformed_df, categorical_columns, numerical_columns)

    else:
        raise ValueError(f"Для набора {dataset_name} неизвестный feature_set='{feature_set}'")

    cache[cache_key] = prepared
    return prepared


def load_and_prepare_synthaml_dataset(
    cache: dict[str, tuple[pd.DataFrame, list[str], list[str]]],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    cache_key = "synthaml:prepared"

    if cache_key in cache:
        return cache[cache_key]

    loader = DataLoader()
    alerts_df = loader.load_dataset("synthaml_alerts")
    transactions_df = loader.load_dataset("synthaml_transactions")

    synthaml_df = prepare_synthaml_dataset_from_frames(alerts_df, transactions_df)

    categorical_columns: list[str] = []
    numerical_columns = [
        col
        for col in synthaml_df.columns
        if col not in categorical_columns + [TARGET_COLUMN]
    ]

    prepared = (synthaml_df, categorical_columns, numerical_columns)
    cache[cache_key] = prepared

    return prepared


def load_and_prepare_dataset(
    dataset_name: str,
    feature_set: str,
    cache: dict[str, tuple[pd.DataFrame, list[str], list[str]]],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if dataset_name == "synthaml":
        return load_and_prepare_synthaml_dataset(cache)

    return load_and_prepare_main_dataset(dataset_name, feature_set, cache)


def get_evaluation_dataframe(
    job: dict[str, Any],
    cache: dict[str, tuple[pd.DataFrame, list[str], list[str]]],
) -> pd.DataFrame:
    eval_dataset_name = str(job["eval_dataset_name"])
    feature_set = str(job["feature_set"])
    evaluation_mode = str(job["evaluation_mode"])

    eval_df, _, _ = load_and_prepare_dataset(
        dataset_name=eval_dataset_name,
        feature_set=feature_set,
        cache=cache,
    )

    if evaluation_mode == "held_out_test":
        train_rows, valid_rows, _ = split_counts(len(eval_df))
        return eval_df.iloc[train_rows + valid_rows:].copy()

    if evaluation_mode == "external_dataset":
        return eval_df.copy()

    raise ValueError(f"Неизвестный evaluation_mode='{evaluation_mode}'")


def get_validation_dataframe_from_train_df(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает validation-часть обучающего набора.
    Порог подбирается только на validation, а не на test/external dataset.
    """
    train_rows, valid_rows, _ = split_counts(len(train_df))
    return train_df.iloc[train_rows:train_rows + valid_rows].copy()


def calculate_threshold_modes(
    y_valid: np.ndarray,
    y_valid_proba: np.ndarray,
) -> dict[str, float]:
    """
    Подбирает несколько threshold-режимов на validation-выборке.

    Эти режимы влияют на precision/recall/F1/FPR/alert rate,
    но не меняют ROC-AUC, PR-AUC и top-k метрики.
    """
    rows: list[dict[str, float]] = []

    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        quality_metrics = calculate_classification_metrics(
            y_true=y_valid,
            y_proba=y_valid_proba,
            threshold=float(threshold),
        )

        business_metrics = calculate_business_metrics(
            y_true=y_valid,
            y_proba=y_valid_proba,
            threshold=float(threshold),
            top_k_values=(100, 500),
        )

        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(quality_metrics["precision"]),
                "recall": float(quality_metrics["recall"]),
                "f1_score": float(quality_metrics["f1_score"]),
                "false_positive_rate": float(business_metrics["false_positive_rate"]),
                "alert_rate": float(business_metrics["alert_rate"]),
            }
        )

    modes: dict[str, float] = {
        THRESHOLD_MODE_DEFAULT: DEFAULT_THRESHOLD,
    }

    best_f1 = sorted(
        rows,
        key=lambda row: (row["f1_score"], row["recall"], row["precision"]),
        reverse=True,
    )[0]
    modes[THRESHOLD_MODE_BEST_F1] = float(best_f1["threshold"])

    alert_20_candidates = [
        row for row in rows
        if row["alert_rate"] <= 0.20
    ]
    if alert_20_candidates:
        best_alert_20 = sorted(
            alert_20_candidates,
            key=lambda row: (row["recall"], row["precision"], row["f1_score"]),
            reverse=True,
        )[0]
        modes[THRESHOLD_MODE_ALERT_20] = float(best_alert_20["threshold"])

    alert_30_candidates = [
        row for row in rows
        if row["alert_rate"] <= 0.30
    ]
    if alert_30_candidates:
        best_alert_30 = sorted(
            alert_30_candidates,
            key=lambda row: (row["recall"], row["precision"], row["f1_score"]),
            reverse=True,
        )[0]
        modes[THRESHOLD_MODE_ALERT_30] = float(best_alert_30["threshold"])

    fpr_05_candidates = [
        row for row in rows
        if row["false_positive_rate"] <= 0.05
    ]
    if fpr_05_candidates:
        best_fpr_05 = sorted(
            fpr_05_candidates,
            key=lambda row: (row["recall"], row["precision"], row["f1_score"]),
            reverse=True,
        )[0]
        modes[THRESHOLD_MODE_FPR_05] = float(best_fpr_05["threshold"])

    return modes


def write_single_row_csv(row: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output_path, index=False)


def write_rows_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def sort_benchmark_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    scenario_order = {
        "Base": 1,
        "SynthAML": 2,
        "Variant I": 3,
        "Variant II": 4,
    }

    sorted_df = df.copy()
    sorted_df["_scenario_order"] = sorted_df["scenario"].map(scenario_order).fillna(99)

    sort_columns = ["_scenario_order", "pr_auc", "roc_auc"]
    ascending = [True, False, False]

    sorted_df = sorted_df.sort_values(
        sort_columns,
        ascending=ascending,
    )

    sorted_df = sorted_df.drop(columns=["_scenario_order"])
    return sorted_df


def threshold_modes_file_for_job(job: dict[str, Any]) -> Path:
    output_file = Path(str(job["output_file"]))
    return METRICS_DIR / f"{output_file.stem}_threshold_modes.csv"


@task(task_id="run_benchmark_job")
def run_benchmark_job_task(job: dict[str, Any]) -> dict[str, Any]:
    ensure_directories()
    configure_benchmark_mlflow()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Запуск benchmark job: {job['job_id']}")

    cache: dict[str, tuple[pd.DataFrame, list[str], list[str]]] = {}

    train_df, categorical_columns, numerical_columns = load_and_prepare_dataset(
        dataset_name=str(job["train_dataset_name"]),
        feature_set=str(job["feature_set"]),
        cache=cache,
    )

    train_rows, valid_rows, _ = split_counts(len(train_df))

    trainer = AMLModelTrainer()

    start_train = time.perf_counter()
    training_result = trainer.train(
        df=train_df,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        model_type=str(job["model_type"]),
    )
    train_time_sec = time.perf_counter() - start_train

    unique_bundle_path = (
        BENCHMARK_MODELS_DIR
        / f"{job['job_id']}_{job['model_type']}_{job['feature_set']}_bundle.joblib"
    )
    shutil.copy2(training_result.bundle_path, unique_bundle_path)

    predictor = AMLPredictor(str(unique_bundle_path))

    # Threshold-режимы выбираются на validation-части train dataset.
    valid_df = get_validation_dataframe_from_train_df(train_df)

    valid_prediction_df = predictor.predict_proba(valid_df)
    y_valid = valid_df[TARGET_COLUMN].astype(int).to_numpy()
    y_valid_proba = valid_prediction_df["prediction_score"].astype(float).to_numpy()

    threshold_modes = calculate_threshold_modes(
        y_valid=y_valid,
        y_valid_proba=y_valid_proba,
    )

    # Итоговая оценка выполняется на held-out test или external dataset.
    eval_df = get_evaluation_dataframe(job, cache)

    start_inference = time.perf_counter()
    prediction_df = predictor.predict_proba(eval_df)
    inference_time_sec = time.perf_counter() - start_inference

    y_true = eval_df[TARGET_COLUMN].astype(int).to_numpy()
    y_proba = prediction_df["prediction_score"].astype(float).to_numpy()

    # Основной результат сохраняется как раньше: одна строка при фиксированном threshold=0.5.
    default_threshold = float(job.get("threshold", DEFAULT_THRESHOLD))

    default_quality_metrics = calculate_classification_metrics(
        y_true=y_true,
        y_proba=y_proba,
        threshold=default_threshold,
    )

    default_business_metrics = calculate_business_metrics(
        y_true=y_true,
        y_proba=y_proba,
        threshold=default_threshold,
        top_k_values=(100, 500),
    )

    result_row = {
        "job_id": job["job_id"],
        "scenario": job["scenario"],
        "dataset": job["dataset"],
        "train_dataset_name": job["train_dataset_name"],
        "eval_dataset_name": job["eval_dataset_name"],
        "evaluation_mode": job["evaluation_mode"],
        "model_type": job["model_type"],
        "feature_set": job["feature_set"],
        "threshold_mode": THRESHOLD_MODE_DEFAULT,
        "threshold": default_threshold,
        "threshold_selected_on": "fixed",
        "roc_auc": default_quality_metrics["roc_auc"],
        "pr_auc": default_quality_metrics["pr_auc"],
        "precision": default_quality_metrics["precision"],
        "recall": default_quality_metrics["recall"],
        "f1_score": default_quality_metrics["f1_score"],
        "false_positive_rate": default_business_metrics["false_positive_rate"],
        "alert_rate": default_business_metrics["alert_rate"],
        "positive_share": default_business_metrics["positive_share"],
        "precision_at_100": default_business_metrics["precision_at_100"],
        "recall_at_100": default_business_metrics["recall_at_100"],
        "lift_at_100": default_business_metrics["lift_at_100"],
        "precision_at_500": default_business_metrics["precision_at_500"],
        "recall_at_500": default_business_metrics["recall_at_500"],
        "lift_at_500": default_business_metrics["lift_at_500"],
        "tp": default_business_metrics["tp"],
        "fp": default_business_metrics["fp"],
        "fn": default_business_metrics["fn"],
        "tn": default_business_metrics["tn"],
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "test_rows": len(eval_df),
        "train_time_sec": train_time_sec,
        "inference_time_sec": inference_time_sec,
        "bundle_path": str(unique_bundle_path),
    }

    output_path = METRICS_DIR / str(job["output_file"])
    write_single_row_csv(result_row, output_path)

    # Дополнительный файл: все threshold-режимы для нового анализа.
    threshold_rows: list[dict[str, Any]] = []

    for threshold_mode, threshold in threshold_modes.items():
        threshold = float(threshold)

        quality_metrics = calculate_classification_metrics(
            y_true=y_true,
            y_proba=y_proba,
            threshold=threshold,
        )

        business_metrics = calculate_business_metrics(
            y_true=y_true,
            y_proba=y_proba,
            threshold=threshold,
            top_k_values=(100, 500),
        )

        threshold_rows.append(
            {
                "job_id": job["job_id"],
                "scenario": job["scenario"],
                "dataset": job["dataset"],
                "train_dataset_name": job["train_dataset_name"],
                "eval_dataset_name": job["eval_dataset_name"],
                "evaluation_mode": job["evaluation_mode"],
                "model_type": job["model_type"],
                "feature_set": job["feature_set"],
                "threshold_mode": threshold_mode,
                "threshold": threshold,
                "threshold_selected_on": "validation"
                if threshold_mode != THRESHOLD_MODE_DEFAULT
                else "fixed",
                "roc_auc": quality_metrics["roc_auc"],
                "pr_auc": quality_metrics["pr_auc"],
                "precision": quality_metrics["precision"],
                "recall": quality_metrics["recall"],
                "f1_score": quality_metrics["f1_score"],
                "false_positive_rate": business_metrics["false_positive_rate"],
                "alert_rate": business_metrics["alert_rate"],
                "positive_share": business_metrics["positive_share"],
                "precision_at_100": business_metrics["precision_at_100"],
                "recall_at_100": business_metrics["recall_at_100"],
                "lift_at_100": business_metrics["lift_at_100"],
                "precision_at_500": business_metrics["precision_at_500"],
                "recall_at_500": business_metrics["recall_at_500"],
                "lift_at_500": business_metrics["lift_at_500"],
                "tp": business_metrics["tp"],
                "fp": business_metrics["fp"],
                "fn": business_metrics["fn"],
                "tn": business_metrics["tn"],
                "train_rows": train_rows,
                "valid_rows": valid_rows,
                "test_rows": len(eval_df),
                "train_time_sec": train_time_sec,
                "inference_time_sec": inference_time_sec,
                "bundle_path": str(unique_bundle_path),
            }
        )

    threshold_modes_path = threshold_modes_file_for_job(job)
    write_rows_csv(threshold_rows, threshold_modes_path)

    best_f1_row = max(threshold_rows, key=lambda row: row["f1_score"])

    logger.info(
        f"Benchmark job завершён: {job['job_id']}; "
        f"roc_auc={result_row['roc_auc']:.4f}; "
        f"pr_auc={result_row['pr_auc']:.4f}; "
        f"precision_at_100={result_row['precision_at_100']:.4f}; "
        f"lift_at_100={result_row['lift_at_100']:.2f}; "
        f"best_threshold_mode={best_f1_row['threshold_mode']}; "
        f"best_f1={best_f1_row['f1_score']:.4f}; "
        f"train_time_sec={train_time_sec:.2f}; "
        f"inference_time_sec={inference_time_sec:.2f}"
    )

    return {
        "job_id": job["job_id"],
        "output_path": str(output_path),
        "threshold_modes_path": str(threshold_modes_path),
        "bundle_path": str(unique_bundle_path),
    }


@task(task_id="build_benchmark_tables")
def build_benchmark_tables_task() -> dict[str, Any]:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[pd.DataFrame] = []

    for job in BENCHMARK_JOBS:
        path = METRICS_DIR / str(job["output_file"])

        if not path.exists():
            raise FileNotFoundError(f"Не найден результат benchmark job: {path}")

        df = pd.read_csv(path)
        df["source_file"] = path.name
        rows.append(df)

    all_df = pd.concat(rows, ignore_index=True)

    for column in NUMERIC_COLUMNS:
        if column in all_df.columns:
            all_df[column] = pd.to_numeric(all_df[column], errors="coerce")

    all_df = sort_benchmark_dataframe(all_df)

    output_paths: dict[str, str] = {}

    all_path = METRICS_DIR / "benchmark_results_all.csv"
    all_df.to_csv(all_path, index=False)
    output_paths["benchmark_results_all"] = str(all_path)

    dag_path = METRICS_DIR / "benchmark_results_dag.csv"
    all_df.to_csv(dag_path, index=False)
    output_paths["benchmark_results_dag"] = str(dag_path)

    scenario_file_map = {
        "Base": "benchmark_base.csv",
        "SynthAML": "benchmark_synthaml.csv",
        "Variant I": "benchmark_variant_1.csv",
        "Variant II": "benchmark_variant_2.csv",
    }

    for scenario, file_name in scenario_file_map.items():
        scenario_df = all_df[all_df["scenario"] == scenario].copy()
        scenario_path = METRICS_DIR / file_name
        scenario_df.to_csv(scenario_path, index=False)
        output_paths[file_name.replace(".csv", "")] = str(scenario_path)

    best_idx = all_df.groupby("scenario")["pr_auc"].idxmax()
    best_df = sort_benchmark_dataframe(all_df.loc[best_idx].copy())

    best_path = METRICS_DIR / "benchmark_best_by_pr_auc.csv"
    best_df.to_csv(best_path, index=False)
    output_paths["benchmark_best_by_pr_auc"] = str(best_path)

    chapter_columns = [
        "scenario",
        "model_type",
        "feature_set",
        "threshold_mode",
        "threshold",
        "threshold_selected_on",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
        "alert_rate",
        "positive_share",
        "precision_at_100",
        "recall_at_100",
        "lift_at_100",
        "precision_at_500",
        "recall_at_500",
        "lift_at_500",
        "tp",
        "fp",
        "fn",
        "tn",
        "source_file",
    ]

    chapter_df = all_df[[col for col in chapter_columns if col in all_df.columns]].copy()

    chapter_path = METRICS_DIR / "benchmark_for_chapter_3_2.csv"
    chapter_df.to_csv(chapter_path, index=False)
    output_paths["benchmark_for_chapter_3_2"] = str(chapter_path)

    lift_columns = [
        "scenario",
        "model_type",
        "feature_set",
        "positive_share",
        "precision_at_100",
        "recall_at_100",
        "lift_at_100",
        "precision_at_500",
        "recall_at_500",
        "lift_at_500",
        "source_file",
    ]

    lift_df = all_df[[col for col in lift_columns if col in all_df.columns]].copy()

    lift_path = METRICS_DIR / "benchmark_lift_for_chapter_3_2.csv"
    lift_df.to_csv(lift_path, index=False)
    output_paths["benchmark_lift_for_chapter_3_2"] = str(lift_path)

    threshold_rows: list[pd.DataFrame] = []

    for job in BENCHMARK_JOBS:
        threshold_path = threshold_modes_file_for_job(job)

        if threshold_path.exists():
            threshold_df = pd.read_csv(threshold_path)
            threshold_df["source_file"] = threshold_path.name
            threshold_rows.append(threshold_df)

    if threshold_rows:
        threshold_all_df = pd.concat(threshold_rows, ignore_index=True)

        for column in NUMERIC_COLUMNS:
            if column in threshold_all_df.columns:
                threshold_all_df[column] = pd.to_numeric(
                    threshold_all_df[column],
                    errors="coerce",
                )

        threshold_all_df = sort_benchmark_dataframe(threshold_all_df)

        threshold_all_path = METRICS_DIR / "benchmark_threshold_modes.csv"
        threshold_all_df.to_csv(threshold_all_path, index=False)
        output_paths["benchmark_threshold_modes"] = str(threshold_all_path)

        best_f1_idx = threshold_all_df.groupby("scenario")["f1_score"].idxmax()
        best_f1_df = sort_benchmark_dataframe(threshold_all_df.loc[best_f1_idx].copy())

        best_f1_path = METRICS_DIR / "benchmark_best_by_f1_threshold_modes.csv"
        best_f1_df.to_csv(best_f1_path, index=False)
        output_paths["benchmark_best_by_f1_threshold_modes"] = str(best_f1_path)

        threshold_chapter_df = threshold_all_df[
            [col for col in chapter_columns if col in threshold_all_df.columns]
        ].copy()

        threshold_chapter_path = METRICS_DIR / "benchmark_threshold_modes_for_chapter_3_2.csv"
        threshold_chapter_df.to_csv(threshold_chapter_path, index=False)
        output_paths["benchmark_threshold_modes_for_chapter_3_2"] = str(threshold_chapter_path)

    logger.info(f"Итоговые benchmark-таблицы сохранены в {METRICS_DIR}")

    return {
        "rows": len(all_df),
        "outputs": output_paths,
    }


@task(task_id="measure_inference_time")
def measure_inference_time_task(repeats: int = 3) -> dict[str, Any]:
    input_path = METRICS_DIR / "benchmark_results_all.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Не найден файл с итогами benchmark: {input_path}")

    benchmark_df = pd.read_csv(input_path)

    cache: dict[str, tuple[pd.DataFrame, list[str], list[str]]] = {}
    timing_rows: list[dict[str, Any]] = []

    for _, row in benchmark_df.iterrows():
        job = {
            "eval_dataset_name": row["eval_dataset_name"],
            "feature_set": row["feature_set"],
            "evaluation_mode": row["evaluation_mode"],
        }

        logger.info(f"Замер времени получения предсказаний: {row['job_id']}")

        bundle_path = Path(str(row["bundle_path"]))

        if not bundle_path.exists():
            raise FileNotFoundError(f"Не найден bundle модели: {bundle_path}")

        eval_df = get_evaluation_dataframe(job, cache)
        predictor = AMLPredictor(str(bundle_path))

        warmup_size = min(1000, len(eval_df))

        if warmup_size > 0:
            predictor.predict_proba(eval_df.head(warmup_size))

        durations = []

        for _ in range(repeats):
            start = time.perf_counter()
            prediction_df = predictor.predict_proba(eval_df)
            duration = time.perf_counter() - start

            if "prediction_score" not in prediction_df.columns:
                raise RuntimeError("prediction_score не найден в результате инференса")

            durations.append(duration)

        median_time = statistics.median(durations)
        mean_time = statistics.mean(durations)

        timing_rows.append(
            {
                "job_id": row["job_id"],
                "scenario": row["scenario"],
                "dataset": row["dataset"],
                "eval_dataset_name": row["eval_dataset_name"],
                "evaluation_mode": row["evaluation_mode"],
                "model_type": row["model_type"],
                "feature_set": row["feature_set"],
                "batch_size": len(eval_df),
                "inference_repeats": repeats,
                "inference_time_sec_median": median_time,
                "inference_time_sec_mean": mean_time,
                "inference_time_per_1000_sec": (
                    median_time / len(eval_df) * 1000 if len(eval_df) else np.nan
                ),
                "records_per_second": (
                    len(eval_df) / median_time if median_time > 0 else np.nan
                ),
            }
        )

    timing_df = pd.DataFrame(timing_rows)

    timing_path = METRICS_DIR / "inference_timing.csv"
    timing_df.to_csv(timing_path, index=False)

    merged_df = benchmark_df.merge(
        timing_df[
            [
                "job_id",
                "batch_size",
                "inference_repeats",
                "inference_time_sec_median",
                "inference_time_sec_mean",
                "inference_time_per_1000_sec",
                "records_per_second",
            ]
        ],
        on="job_id",
        how="left",
    )

    merged_path = METRICS_DIR / "benchmark_results_with_inference.csv"
    merged_df.to_csv(merged_path, index=False)

    performance_columns = [
        "scenario",
        "model_type",
        "feature_set",
        "train_time_sec",
        "batch_size",
        "inference_time_sec_median",
        "inference_time_per_1000_sec",
        "records_per_second",
    ]

    performance_df = merged_df[
        [col for col in performance_columns if col in merged_df.columns]
    ].copy()

    performance_path = METRICS_DIR / "performance_for_chapter_3_3.csv"
    performance_df.to_csv(performance_path, index=False)

    logger.info(f"Замеры инференса сохранены: {timing_path}")
    logger.info(f"Объединённая таблица сохранена: {merged_path}")
    logger.info(f"Таблица для раздела 3.3 сохранена: {performance_path}")

    return {
        "rows": len(timing_df),
        "timing_path": str(timing_path),
        "merged_path": str(merged_path),
        "performance_path": str(performance_path),
    }


default_args = {
    "owner": "aml-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="aml_benchmark_dag",
    description="Формирование итоговых benchmark-таблиц и замер времени получения предсказаний",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["aml", "benchmark", "mlops"],
)
def aml_benchmark_dag():
    start = EmptyOperator(task_id="start")
    finish = EmptyOperator(task_id="finish")

    previous_task = start

    for index, job in enumerate(BENCHMARK_JOBS, start=1):
        benchmark_task = run_benchmark_job_task.override(
            task_id=f"benchmark_{index}_{job['job_id']}"
        )(job)

        previous_task >> benchmark_task
        previous_task = benchmark_task

    build_tables = build_benchmark_tables_task()
    measure_inference = measure_inference_time_task()

    previous_task >> build_tables >> measure_inference >> finish


aml_benchmark_dag()
