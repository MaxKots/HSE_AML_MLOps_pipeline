from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data import DataLoader, DataTransformer, DataValidator
from src.features import FeatureEngineer
from src.models.predict import AMLPredictor
from src.utils.paths import ensure_directories, get_artifacts_dir


DATASET_NAME = "variant_2"
TARGET_COLUMN = "fraud_bool"

MODEL_TYPES = [
    model_type.strip()
    for model_type in os.getenv("MODEL_TYPES", "lightgbm,xgboost").split(",")
    if model_type.strip()
]

TOP_N_FOR_SHAP = int(os.getenv("TOP_N_FOR_SHAP", "500"))
TOP_FEATURES = int(os.getenv("TOP_FEATURES", "15"))


def resolve_bundle_path(model_type: str) -> Path:
    artifacts_dir = get_artifacts_dir()
    benchmark_dir = artifacts_dir / "models" / "benchmark"

    exact_candidates = [
        benchmark_dir / f"variant_2_{model_type}_engineered_{model_type}_engineered_bundle.joblib",
        artifacts_dir / "models" / f"{model_type}_bundle.joblib",
        artifacts_dir / "models" / "production_bundle.joblib",
    ]

    for path in exact_candidates:
        if path.exists():
            return path

    if benchmark_dir.exists():
        matches = sorted(
            benchmark_dir.glob(f"*variant_2*{model_type}*bundle*.joblib")
        )

        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Не найден bundle для Variant II и модели {model_type}. "
        "Сначала запусти aml_benchmark_dag или проверь artifacts/models/benchmark."
    )


def prepare_variant_2_dataframe() -> pd.DataFrame:
    loader = DataLoader()
    validator = DataValidator()
    transformer = DataTransformer()
    feature_engineer = FeatureEngineer()

    raw_df = loader.load_dataset(DATASET_NAME)
    validator.run_full_validation(raw_df)

    transformed_df = transformer.transform(raw_df)
    feature_result = feature_engineer.build_features(transformed_df)

    if TARGET_COLUMN not in feature_result.dataframe.columns:
        raise KeyError(
            f"В подготовленном Variant II dataframe нет target-колонки {TARGET_COLUMN}"
        )

    return feature_result.dataframe


def to_dense_array(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()

    return np.asarray(matrix)


def normalize_shap_values(shap_values) -> np.ndarray:
    """
    Приводит SHAP values к матрице shape = [n_rows, n_features].
    Нужно для совместимости LightGBM / XGBoost в binary classification.
    """
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values


def get_bundle_feature_columns(bundle: dict) -> list[str]:
    feature_columns = (
        bundle.get("feature_columns_before_preprocessing")
        or bundle.get("feature_columns")
        or bundle.get("input_columns")
    )

    if feature_columns is None:
        raise KeyError(
            "В bundle отсутствует список исходных признаков: "
            "feature_columns_before_preprocessing / feature_columns / input_columns"
        )

    return list(feature_columns)


def get_bundle_feature_names(bundle: dict, n_features: int) -> list[str]:
    feature_names = (
        bundle.get("feature_names_after_preprocessing")
        or bundle.get("feature_names")
    )

    if feature_names is None or len(feature_names) != n_features:
        return [f"feature_{i}" for i in range(n_features)]

    return list(feature_names)


def build_shap_for_model(
    model_type: str,
    df: pd.DataFrame,
) -> dict[str, str]:
    bundle_path = resolve_bundle_path(model_type)
    print(f"\n=== {model_type.upper()} ===")
    print(f"Используется bundle: {bundle_path}")

    predictor = AMLPredictor(str(bundle_path))
    prediction_df = predictor.predict_proba(df)

    scores = pd.Series(
        prediction_df["prediction_score"].astype(float).to_numpy(),
        index=df.index,
        name="prediction_score",
    )

    top_n = min(TOP_N_FOR_SHAP, len(df))
    top_index = scores.sort_values(ascending=False).head(top_n).index
    shap_df = df.loc[top_index].copy()

    print(f"Variant II rows: {len(df)}")
    print(f"SHAP sample: top-{len(shap_df)} по prediction_score")
    print(f"Средний score в top-{top_n}: {scores.loc[top_index].mean():.6f}")

    bundle = joblib.load(bundle_path)

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    feature_columns = get_bundle_feature_columns(bundle)

    X_raw = shap_df.drop(columns=[TARGET_COLUMN, "event_time"], errors="ignore")
    X_raw = X_raw.reindex(columns=feature_columns)

    X_prepared = preprocessor.transform(X_raw)
    X_prepared = to_dense_array(X_prepared)

    feature_names = get_bundle_feature_names(bundle, X_prepared.shape[1])

    X_prepared_df = pd.DataFrame(
        X_prepared,
        columns=feature_names,
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_prepared)
    shap_values = normalize_shap_values(shap_values)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(TOP_FEATURES)
    )

    output_dir = get_artifacts_dir() / "shap"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"variant_2_{model_type}_top{top_n}_shap_importance.csv"
    summary_dot_path = output_dir / f"variant_2_{model_type}_top{top_n}_shap_summary_dot.png"
    summary_bar_path = output_dir / f"variant_2_{model_type}_top{top_n}_shap_summary_bar.png"

    importance_df.to_csv(csv_path, index=False)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_prepared_df,
        max_display=TOP_FEATURES,
        show=False,
    )
    plt.title(
        f"SHAP summary: {model_type.upper()} на top-{top_n} операций Variant II"
    )
    plt.tight_layout()
    plt.savefig(summary_dot_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_prepared_df,
        max_display=TOP_FEATURES,
        plot_type="bar",
        show=False,
    )
    plt.title(
        f"SHAP importance: {model_type.upper()} на top-{top_n} операций Variant II"
    )
    plt.tight_layout()
    plt.savefig(summary_bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"CSV сохранён: {csv_path.resolve()}")
    print(f"SHAP summary plot сохранён: {summary_dot_path.resolve()}")
    print(f"SHAP bar plot сохранён: {summary_bar_path.resolve()}")

    return {
        "model_type": model_type,
        "bundle_path": str(bundle_path),
        "csv_path": str(csv_path),
        "summary_dot_path": str(summary_dot_path),
        "summary_bar_path": str(summary_bar_path),
    }


def main() -> None:
    ensure_directories()

    df = prepare_variant_2_dataframe()

    results = []

    for model_type in MODEL_TYPES:
        result = build_shap_for_model(
            model_type=model_type,
            df=df,
        )
        results.append(result)

    report_path = get_artifacts_dir() / "shap" / "variant_2_shap_report.csv"
    pd.DataFrame(results).to_csv(report_path, index=False)

    print(f"\nИтоговый отчёт сохранён: {report_path.resolve()}")


if __name__ == "__main__":
    main()
