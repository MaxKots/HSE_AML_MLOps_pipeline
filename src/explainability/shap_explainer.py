from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils.io import load_object, save_yaml
from src.utils.logger import get_logger
from src.utils.paths import get_artifacts_dir

logger = get_logger(__name__)


@dataclass
class ShapExplanation:
    row_index: int
    prediction_score: float
    prediction_label: int
    top_positive_factors: list[dict[str, Any]]
    top_negative_factors: list[dict[str, Any]]
    raw_top_features: list[dict[str, Any]]


class AMLShapExplainer:
    def __init__(self, bundle_path: str) -> None:
        self.bundle = load_object(bundle_path)

        self.model = self.bundle["model"]
        self.preprocessor = self.bundle["preprocessor"]
        self.feature_columns_before_preprocessing = self.bundle["feature_columns_before_preprocessing"]
        self.feature_names_after_preprocessing = self.bundle["feature_names_after_preprocessing"]
        self.threshold = self.bundle["threshold"]

        logger.info(f"Model bundle для SHAP загружен из {bundle_path}")

    def _prepare_input(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        X = df.copy()

        if "fraud_bool" in X.columns:
            X = X.drop(columns=["fraud_bool"])

        if "event_time" in X.columns:
            X = X.drop(columns=["event_time"])

        X = X[self.feature_columns_before_preprocessing].copy()
        X_prepared = self.preprocessor.transform(X)

        return X, X_prepared

    def _build_explainer(self, X_background_prepared: np.ndarray):
        logger.info("Создание SHAP explainer")
        explainer = shap.Explainer(self.model, X_background_prepared)
        return explainer

    def _group_feature_name(self, feature_name: str) -> str:
        prefixes_to_group = [
            "payment_type_",
            "employment_status_",
            "housing_status_",
            "source_",
            "device_os_",
        ]

        for prefix in prefixes_to_group:
            if feature_name.startswith(prefix):
                return prefix[:-1]

        return feature_name

    def _aggregate_shap_values(
        self,
        shap_values_row: np.ndarray,
        feature_names: list[str],
    ) -> list[dict[str, Any]]:
        grouped_values: dict[str, float] = {}

        for feature_name, shap_value in zip(feature_names, shap_values_row):
            grouped_name = self._group_feature_name(feature_name)
            grouped_values[grouped_name] = grouped_values.get(grouped_name, 0.0) + float(shap_value)

        sorted_items = sorted(
            grouped_values.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        return [
            {
                "feature": feature,
                "shap_value": value,
                "abs_shap_value": abs(value),
                "direction": "increase_risk" if value > 0 else "decrease_risk",
            }
            for feature, value in sorted_items
        ]

    def explain_rows(
        self,
        df: pd.DataFrame,
        row_indices: list[int] | None = None,
        top_k: int = 5,
        background_size: int = 200,
    ) -> list[ShapExplanation]:
        if row_indices is None:
            row_indices = [0]

        _, X_prepared = self._prepare_input(df)

        background_size = min(background_size, len(X_prepared))
        background = X_prepared[:background_size]

        explainer = self._build_explainer(background)
        shap_values = explainer(X_prepared, check_additivity=False)

        prediction_scores = self.model.predict_proba(X_prepared)[:, 1]
        prediction_labels = (prediction_scores >= self.threshold).astype(int)

        explanations: list[ShapExplanation] = []

        for row_index in row_indices:
            row_shap_values = shap_values.values[row_index]
            aggregated_features = self._aggregate_shap_values(
                shap_values_row=row_shap_values,
                feature_names=self.feature_names_after_preprocessing,
            )

            positive_factors = [
                item for item in aggregated_features if item["shap_value"] > 0
            ][:top_k]

            negative_factors = [
                item for item in aggregated_features if item["shap_value"] < 0
            ][:top_k]

            raw_top_features = aggregated_features[:top_k]

            explanation = ShapExplanation(
                row_index=row_index,
                prediction_score=float(prediction_scores[row_index]),
                prediction_label=int(prediction_labels[row_index]),
                top_positive_factors=positive_factors,
                top_negative_factors=negative_factors,
                raw_top_features=raw_top_features,
            )
            explanations.append(explanation)

        logger.info(f"SHAP-объяснения сформированы для {len(explanations)} строк")
        return explanations

    def build_human_readable_reasons(
        self,
        explanation: ShapExplanation,
    ) -> list[str]:
        reasons: list[str] = []

        for factor in explanation.top_positive_factors:
            feature = factor["feature"]
            shap_value = factor["shap_value"]

            if feature == "credit_risk_score":
                reasons.append(f"Высокий кредитный риск увеличил итоговый риск операции (SHAP={shap_value:.4f})")
            elif feature == "velocity_24h_to_6h_ratio":
                reasons.append(f"Нестандартная динамика активности за 24 часа повысила риск (SHAP={shap_value:.4f})")
            elif feature == "risk_low_similarity_free_email":
                reasons.append(f"Низкая схожесть имени и e-mail при бесплатной почте повысила риск (SHAP={shap_value:.4f})")
            elif feature == "risk_foreign_and_device_fraud":
                reasons.append(f"Иностранный запрос в сочетании с историей fraud-устройства повысил риск (SHAP={shap_value:.4f})")
            elif feature == "device_fraud_count":
                reasons.append(f"История мошеннической активности устройства повысила риск (SHAP={shap_value:.4f})")
            elif feature == "proposed_credit_limit":
                reasons.append(f"Запрошенный кредитный лимит увеличил риск (SHAP={shap_value:.4f})")
            elif feature == "payment_type":
                reasons.append(f"Тип платежа оказал влияние на повышение риска (SHAP={shap_value:.4f})")
            elif feature == "device_os":
                reasons.append(f"Характеристика операционной системы устройства повлияла на риск (SHAP={shap_value:.4f})")
            else:
                reasons.append(f"Признак '{feature}' повысил риск (SHAP={shap_value:.4f})")

        return reasons

    def export_explanations(
        self,
        explanations: list[ShapExplanation],
        output_path: str | Path,
    ) -> None:
        serializable = []
        for explanation in explanations:
            serializable.append(
                {
                    "row_index": explanation.row_index,
                    "prediction_score": explanation.prediction_score,
                    "prediction_label": explanation.prediction_label,
                    "top_positive_factors": explanation.top_positive_factors,
                    "top_negative_factors": explanation.top_negative_factors,
                    "raw_top_features": explanation.raw_top_features,
                    "human_readable_reasons": self.build_human_readable_reasons(explanation),
                }
            )

        save_yaml({"explanations": serializable}, output_path)
        logger.info(f"SHAP-объяснения сохранены в {output_path}")

    def export_summary_plot(
        self,
        df: pd.DataFrame,
        output_path: str | Path | None = None,
        sample_size: int = 1000,
    ) -> str:
        _, X_prepared = self._prepare_input(df)

        if len(X_prepared) > sample_size:
            X_prepared = X_prepared[:sample_size]

        explainer = self._build_explainer(X_prepared[: min(200, len(X_prepared))])
        shap_values = explainer(X_prepared, check_additivity=False)

        if output_path is None:
            output_path = get_artifacts_dir() / "shap" / "summary_plot.png"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values.values,
            features=X_prepared,
            feature_names=self.feature_names_after_preprocessing,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

        logger.info(f"Глобальный SHAP summary plot сохранён в {output_path}")
        return str(output_path)
