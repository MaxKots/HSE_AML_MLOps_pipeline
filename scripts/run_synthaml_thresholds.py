from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import DataLoader
from src.data.synthaml import prepare_synthaml_dataset_from_frames
from src.models.evaluate import calculate_classification_metrics
from src.models.predict import AMLPredictor
from src.models.train import AMLModelTrainer
from src.utils.logger import get_logger
from src.utils.paths import get_artifacts_dir

logger = get_logger(__name__)


def _safe_divide(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def calculate_business_metrics_at_threshold(
    y_true,
    y_proba,
    threshold: float,
) -> dict[str, float]:
    y_true = pd.Series(y_true).astype(int)
    y_proba = pd.Series(y_proba).astype(float)

    y_pred = (y_proba >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    alert_rate = float(y_pred.mean())
    false_positive_rate = _safe_divide(fp, fp + tn)

    return {
        "alert_rate": alert_rate,
        "false_positive_rate": false_positive_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def calculate_precision_recall_at_k(
    y_true,
    y_proba,
    k: int,
) -> dict[str, float]:
    df = pd.DataFrame(
        {
            "y_true": pd.Series(y_true).astype(int),
            "y_proba": pd.Series(y_proba).astype(float),
        }
    ).sort_values("y_proba", ascending=False)

    top_k = df.head(k)
    positives_total = int(df["y_true"].sum())

    precision_at_k = float(top_k["y_true"].mean()) if len(top_k) else 0.0
    recall_at_k = _safe_divide(int(top_k["y_true"].sum()), positives_total)

    return {
        f"precision_at_{k}": precision_at_k,
        f"recall_at_{k}": recall_at_k,
    }


def main() -> None:
    loader = DataLoader()

    alerts_df = loader.load_dataset("synthaml_alerts")
    transactions_df = loader.load_dataset("synthaml_transactions")
    synthaml_df = prepare_synthaml_dataset_from_frames(alerts_df, transactions_df)

    target_column = "fraud_bool"
    categorical_columns: list[str] = []

    train_df, test_df = train_test_split(
        synthaml_df,
        test_size=0.15,
        random_state=42,
        stratify=synthaml_df[target_column],
    )

    logger.info(
        f"SynthAML split выполнен: train_rows={len(train_df)}, test_rows={len(test_df)}"
    )

    trainer = AMLModelTrainer()

    results: list[dict] = []
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]

    for model_type in ["lightgbm", "xgboost"]:
        numerical_columns = [
            col for col in train_df.columns
            if col not in categorical_columns + [target_column]
        ]

        logger.info(f"Обучение модели для threshold sweep: {model_type}")

        training_result = trainer.train(
            df=train_df,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            model_type=model_type,
        )

        predictor = AMLPredictor(training_result.bundle_path)
        prediction_df = predictor.predict_proba(test_df)

        y_true = test_df[target_column].values
        y_proba = prediction_df["prediction_score"].values

        base_metrics = calculate_classification_metrics(
            y_true=y_true,
            y_proba=y_proba,
            threshold=0.5,
        )

        p_at_100 = calculate_precision_recall_at_k(y_true, y_proba, 100)
        p_at_500 = calculate_precision_recall_at_k(y_true, y_proba, 500)

        for threshold in thresholds:
            cls_metrics = calculate_classification_metrics(
                y_true=y_true,
                y_proba=y_proba,
                threshold=threshold,
            )
            biz_metrics = calculate_business_metrics_at_threshold(
                y_true=y_true,
                y_proba=y_proba,
                threshold=threshold,
            )

            row = {
                "model_type": model_type,
                "threshold": threshold,
                "roc_auc": base_metrics["roc_auc"],
                "pr_auc": base_metrics["pr_auc"],
                "precision": cls_metrics["precision"],
                "recall": cls_metrics["recall"],
                "f1_score": cls_metrics["f1_score"],
                "false_positive_rate": biz_metrics["false_positive_rate"],
                "alert_rate": biz_metrics["alert_rate"],
                "tp": biz_metrics["tp"],
                "fp": biz_metrics["fp"],
                "fn": biz_metrics["fn"],
                "tn": biz_metrics["tn"],
                **p_at_100,
                **p_at_500,
            }
            results.append(row)

    results_df = pd.DataFrame(results).sort_values(
        ["model_type", "f1_score", "recall"],
        ascending=[True, False, False],
    )

    output_dir = get_artifacts_dir() / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "synthaml_thresholds.csv"
    results_df.to_csv(output_path, index=False)

    logger.info(f"Threshold sweep завершён. Результаты сохранены в {output_path}")

    print("\n=== TOP thresholds by F1 ===")
    for model_type in ["lightgbm", "xgboost"]:
        best = (
            results_df[results_df["model_type"] == model_type]
            .sort_values(["f1_score", "recall"], ascending=[False, False])
            .head(3)
        )
        print(f"\nModel: {model_type}")
        print(best.to_string(index=False))


if __name__ == "__main__":
    main()
