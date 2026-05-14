from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data import DataLoader, DataTransformer, DataValidator
from src.data.synthaml import prepare_synthaml_dataset_from_frames
from src.features import FeatureEngineer
from src.models.evaluate import calculate_classification_metrics
from src.models.predict import AMLPredictor
from src.models.train import AMLModelTrainer
from src.utils.io import copy_file, load_dataframe, save_dataframe, save_json
from src.utils.logger import get_logger
from src.utils.paths import ensure_directories, get_artifacts_dir


logger = get_logger(__name__)

TARGET_COLUMN = "fraud_bool"

BENCHMARK_JOBS: list[dict[str, Any]] = [
    {
        "job_id": "base_lightgbm_raw",
        "scenario": "Base",
        "dataset_label": "Base",
        "train_dataset_name": "base",
        "eval_dataset_name": None,
        "model_type": "lightgbm",
        "feature_set": "raw",
        "output_file": "base_lightgbm_raw.csv",
    },
    {
        "job_id": "base_xgboost_raw",
        "scenario": "Base",
        "dataset_label": "Base",
        "train_dataset_name": "base",
        "eval_dataset_name": None,
        "model_type": "xgboost",
        "feature_set": "raw",
        "output_file": "base_xgboost_raw.csv",
    },
    {
        "job_id": "base_lightgbm_engineered",
        "scenario": "Base",
        "dataset_label": "Base",
        "train_dataset_name": "base",
        "eval_dataset_name": None,
        "model_type": "lightgbm",
        "feature_set": "engineered",
        "output_file": "base_lightgbm_engineered.csv",
    },
    {
        "job_id": "base_xgboost_engineered",
        "scenario": "Base",
        "dataset_label": "Base",
        "train_dataset_name": "base",
        "eval_dataset_name": None,
        "model_type": "xgboost",
        "feature_set": "engineered",
        "output_file": "base_xgboost_engineered.csv",
    },
    {
        "job_id": "variant_1_lightgbm_engineered",
        "scenario": "Variant I",
        "dataset_label": "Variant I",
        "train_dataset_name": "base",
        "eval_dataset_name": "variant_1",
        "model_type": "lightgbm",
        "feature_set": "engineered",
        "output_file": "variant_1_lightgbm_engineered.csv",
    },
    {
        "job_id": "variant_1_xgboost_engineered",
        "scenario": "Variant I",
        "dataset_label": "Variant I",
        "train_dataset_name": "base",
        "eval_dataset_name": "variant_1",
        "model_type": "xgboost",
        "feature_set": "engineered",
        "output_file": "variant_1_xgboost_engineered.csv",
    },
    {
        "job_id": "variant_2_lightgbm_engineered",
        "scenario": "Variant II",
        "dataset_label": "Variant II",
        "train_dataset_name": "base",
        "eval_dataset_name": "variant_2",
        "model_type": "lightgbm",
        "feature_set": "engineered",
        "output_file": "variant_2_lightgbm_engineered.csv",
    },
    {
        "job_id": "variant_2_xgboost_engineered",
        "scenario": "Variant II",
        "dataset_label": "Variant II",
        "train_dataset_name": "base",
        "eval_dataset_name": "variant_2",
        "model_type": "xgboost",
        "feature_set": "engineered",
        "output_file": "variant_2_xgboost_engineered.csv",
    },
    {
        "job_id": "synthaml_lightgbm_prepared",
        "scenario": "SynthAML",
        "dataset_label": "SynthAML",
        "train_dataset_name": "synthaml",
        "eval_dataset_name": None,
        "model_type": "lightgbm",
        "feature_set": "prepared",
        "output_file": "synthaml_lightgbm_prepared.csv",
    },
    {
        "job_id": "synthaml_xgboost_prepared",
        "scenario": "SynthAML",
        "dataset_label": "SynthAML",
        "train_dataset_name": "synthaml",
        "eval_dataset_name": None,
        "model_type": "xgboost",
        "feature_set": "prepared",
        "output_file": "synthaml_xgboost_prepared.csv",
    },
]


def _safe_divide(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _build_raw_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_columns = [
        col
        for col in [
            "payment_type",
            "employment_status",
            "housing_status",
            "source",
            "device_os",
        ]
        if col in df.columns
    ]

    numerical_columns = [
        col
        for col in df.columns
        if col not in categorical_columns + [TARGET_COLUMN, "event_time"]
    ]

    return categorical_columns, numerical_columns


def _prepare_regular_dataset(
    dataset_name: str,
    feature_set: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    loader = DataLoader()
    validator = DataValidator()
    transformer = DataTransformer()
    feature_engineer = FeatureEngineer()

    raw_df = loader.load_dataset(dataset_name)
    validator.run_full_validation(raw_df)
    transformed_df = transformer.transform(raw_df)

    if feature_set == "engineered":
        feature_result = feature_engineer.build_features(transformed_df)
        return (
            feature_result.dataframe,
            feature_result.categorical_columns,
            feature_result.numerical_columns,
        )

    if feature_set == "raw":
        categorical_columns, numerical_columns = _build_raw_feature_lists(transformed_df)
        return transformed_df, categorical_columns, numerical_columns

    raise ValueError(
        f"Для regular dataset feature_set должен быть 'raw' или 'engineered', получено: {feature_set}"
    )


def _prepare_synthaml_dataset() -> tuple[pd.DataFrame, list[str], list[str]]:
    loader = DataLoader()

    alerts_df = loader.load_dataset("synthaml_alerts")
    transactions_df = loader.load_dataset("synthaml_transactions")
    synthaml_df = prepare_synthaml_dataset_from_frames(alerts_df, transactions_df)

    categorical_columns: list[str] = []
    numerical_columns = [col for col in synthaml_df.columns if col != TARGET_COLUMN]

    return synthaml_df, categorical_columns, numerical_columns


def prepare_dataset(
    dataset_name: str,
    feature_set: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if dataset_name == "synthaml":
        if feature_set != "prepared":
            raise ValueError("Для SynthAML ожидается feature_set='prepared'")
        return _prepare_synthaml_dataset()

    return _prepare_regular_dataset(dataset_name=dataset_name, feature_set=feature_set)


def calculate_business_metrics(
    y_true,
    y_proba,
    threshold: float,
    top_k_values: tuple[int, ...] = (100, 500),
) -> dict[str, float]:
    y_true = pd.Series(y_true).astype(int)
    y_proba = pd.Series(y_proba).astype(float)

    y_pred = (y_proba >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    result: dict[str, float] = {
        "alert_rate": float(y_pred.mean()),
        "false_positive_rate": _safe_divide(fp, fp + tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

    ranking = (
        pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
        .sort_values("y_proba", ascending=False)
        .reset_index(drop=True)
    )
    positives_total = int(ranking["y_true"].sum())

    for k in top_k_values:
        top_k = ranking.head(min(k, len(ranking)))
        result[f"precision_at_{k}"] = float(top_k["y_true"].mean()) if len(top_k) else 0.0
        result[f"recall_at_{k}"] = _safe_divide(int(top_k["y_true"].sum()), positives_total)

    return result


def _stable_bundle_path(job: dict[str, Any]) -> Path:
    bundle_dir = get_artifacts_dir() / "models" / "benchmark"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    return bundle_dir / f"{job['job_id']}_{job['model_type']}_{job['feature_set']}_bundle.joblib"


def _resolve_output_path(job: dict[str, Any]) -> Path:
    output_value = job.get("output_file")
    if output_value is None:
        raise ValueError("В job отсутствует output_file")

    output_path = Path(output_value)
    if not output_path.is_absolute():
        output_path = get_artifacts_dir() / "metrics" / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def run_benchmark_job(job: dict[str, Any]) -> dict[str, Any]:
    ensure_directories()

    trainer = AMLModelTrainer()

    train_df, categorical_columns, numerical_columns = prepare_dataset(
        dataset_name=job["train_dataset_name"],
        feature_set=job["feature_set"],
    )

    split_result = trainer.split_dataset(train_df)

    logger.info(
        "Запуск benchmark job: "
        f"job_id={job['job_id']}, "
        f"scenario={job['scenario']}, "
        f"model={job['model_type']}, "
        f"feature_set={job['feature_set']}"
    )

    training_result = trainer.train(
        df=train_df,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        model_type=job["model_type"],
    )

    bundle_path = _stable_bundle_path(job)
    copy_file(training_result.bundle_path, bundle_path)

    predictor = AMLPredictor(str(bundle_path))

    if job.get("eval_dataset_name"):
        eval_df, _, _ = prepare_dataset(
            dataset_name=job["eval_dataset_name"],
            feature_set=job["feature_set"],
        )
        evaluation_mode = "external_dataset"
        test_rows = len(eval_df)
    else:
        eval_df = split_result.test_df.copy()
        evaluation_mode = "held_out_test"
        test_rows = len(split_result.test_df)

    prediction_df = predictor.predict_proba(eval_df)
    y_true = eval_df[TARGET_COLUMN].values
    y_proba = prediction_df["prediction_score"].values
    threshold = float(predictor.threshold)

    cls_metrics = calculate_classification_metrics(
        y_true=y_true,
        y_proba=y_proba,
        threshold=threshold,
    )
    biz_metrics = calculate_business_metrics(
        y_true=y_true,
        y_proba=y_proba,
        threshold=threshold,
        top_k_values=(100, 500),
    )

    result: dict[str, Any] = {
        "job_id": job["job_id"],
        "scenario": job["scenario"],
        "dataset": job["dataset_label"],
        "train_dataset_name": job["train_dataset_name"],
        "eval_dataset_name": job.get("eval_dataset_name") or job["train_dataset_name"],
        "evaluation_mode": evaluation_mode,
        "model_type": job["model_type"],
        "feature_set": job["feature_set"],
        "threshold": threshold,
        "roc_auc": cls_metrics["roc_auc"],
        "pr_auc": cls_metrics["pr_auc"],
        "precision": cls_metrics["precision"],
        "recall": cls_metrics["recall"],
        "f1_score": cls_metrics["f1_score"],
        "false_positive_rate": biz_metrics["false_positive_rate"],
        "alert_rate": biz_metrics["alert_rate"],
        "precision_at_100": biz_metrics["precision_at_100"],
        "recall_at_100": biz_metrics["recall_at_100"],
        "precision_at_500": biz_metrics["precision_at_500"],
        "recall_at_500": biz_metrics["recall_at_500"],
        "tp": biz_metrics["tp"],
        "fp": biz_metrics["fp"],
        "fn": biz_metrics["fn"],
        "tn": biz_metrics["tn"],
        "train_rows": len(split_result.train_df),
        "valid_rows": len(split_result.valid_df),
        "test_rows": test_rows,
        "train_time_sec": training_result.metrics_test.get("threshold", 0.0),  # replaced below
        "inference_time_sec": 0.0,  # can be added later if needed
        "bundle_path": str(bundle_path),
    }

    # trainer.train does not expose raw timing, so keep explicit placeholders stable for the tables.
    # They can be filled later if timing becomes necessary for the thesis.
    result["train_time_sec"] = None
    result["inference_time_sec"] = None

    output_path = _resolve_output_path(job)
    save_dataframe(pd.DataFrame([result]), output_path, index=False)

    logger.info(f"Benchmark job завершён, результат сохранён в {output_path}")
    return result


def run_single_benchmark_cli(
    *,
    job_id: str,
    scenario: str,
    dataset_label: str,
    train_dataset_name: str,
    eval_dataset_name: str | None,
    model_type: str,
    feature_set: str,
    output_file: str,
) -> dict[str, Any]:
    job = {
        "job_id": job_id,
        "scenario": scenario,
        "dataset_label": dataset_label,
        "train_dataset_name": train_dataset_name,
        "eval_dataset_name": eval_dataset_name,
        "model_type": model_type,
        "feature_set": feature_set,
        "output_file": output_file,
    }
    return run_benchmark_job(job)


def build_benchmark_tables(
    metrics_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    ensure_directories()

    metrics_dir = Path(metrics_dir) if metrics_dir is not None else get_artifacts_dir() / "metrics"
    output_dir = Path(output_dir) if output_dir is not None else metrics_dir

    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    collected_frames: list[pd.DataFrame] = []
    expected_paths = [metrics_dir / job["output_file"] for job in BENCHMARK_JOBS]

    for path in expected_paths:
        if not path.exists():
            logger.warning(f"Файл benchmark не найден и будет пропущен: {path}")
            continue

        df = load_dataframe(path)
        if df.empty:
            continue

        df = df.copy()
        df["source_file"] = path.name
        collected_frames.append(df)

    if not collected_frames:
        raise FileNotFoundError(
            f"В директории {metrics_dir} не найдено ни одного job-result файла для benchmark"
        )

    all_results = pd.concat(collected_frames, ignore_index=True)
    all_results = all_results.sort_values(
        ["scenario", "pr_auc", "f1_score", "precision_at_100"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    all_csv = output_dir / "benchmark_results_all.csv"
    all_json = output_dir / "benchmark_results_all.json"
    save_dataframe(all_results, all_csv, index=False)
    save_json(all_results.to_dict(orient="records"), all_json)

    scenario_to_filename = {
        "Base": "benchmark_base.csv",
        "Variant I": "benchmark_variant_1.csv",
        "Variant II": "benchmark_variant_2.csv",
        "SynthAML": "benchmark_synthaml.csv",
    }

    for scenario, filename in scenario_to_filename.items():
        scenario_df = (
            all_results[all_results["scenario"] == scenario]
            .sort_values(["pr_auc", "f1_score", "precision_at_100"], ascending=[False, False, False])
            .reset_index(drop=True)
        )
        if not scenario_df.empty:
            save_dataframe(scenario_df, output_dir / filename, index=False)

    best_rows: list[dict[str, Any]] = []
    for scenario in sorted(all_results["scenario"].dropna().unique().tolist()):
        scenario_df = (
            all_results[all_results["scenario"] == scenario]
            .sort_values(["pr_auc", "f1_score", "precision_at_100"], ascending=[False, False, False])
            .reset_index(drop=True)
        )
        if not scenario_df.empty:
            best_rows.append(scenario_df.iloc[0].to_dict())

    best_df = pd.DataFrame(best_rows)
    best_csv = output_dir / "benchmark_best_by_pr_auc.csv"
    best_json = output_dir / "benchmark_best_by_pr_auc.json"
    save_dataframe(best_df, best_csv, index=False)
    save_json(best_df.to_dict(orient="records"), best_json)

    thesis_columns = [
        "scenario",
        "model_type",
        "feature_set",
        "threshold",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
        "alert_rate",
        "precision_at_100",
        "recall_at_100",
        "precision_at_500",
        "recall_at_500",
        "tp",
        "fp",
        "fn",
        "tn",
        "source_file",
    ]
    thesis_df = all_results[[col for col in thesis_columns if col in all_results.columns]].copy()
    thesis_csv = output_dir / "benchmark_for_chapter_3_2.csv"
    save_dataframe(thesis_df, thesis_csv, index=False)

    summary = {
        "rows_total": len(all_results),
        "all_results_csv": str(all_csv),
        "all_results_json": str(all_json),
        "best_results_csv": str(best_csv),
        "best_results_json": str(best_json),
        "thesis_table_csv": str(thesis_csv),
        "scenario_tables": {
            scenario: str(output_dir / filename)
            for scenario, filename in scenario_to_filename.items()
            if (output_dir / filename).exists()
        },
    }

    logger.info(f"Сводные benchmark-таблицы сохранены в {output_dir}")
    return summary
