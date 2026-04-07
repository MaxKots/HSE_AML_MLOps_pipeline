from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import Any
from uuid import uuid4

import pandas as pd

from src.data import DataTransformer
from src.explainability.shap_explainer import AMLShapExplainer
from src.features import FeatureEngineer
from src.models.predict import AMLPredictor
from src.utils.io import save_json
from src.utils.logger import get_logger
from src.utils.paths import get_artifacts_dir

logger = get_logger(__name__)


class AMLInferenceService:
    def __init__(self, bundle_path: str) -> None:
        self.bundle_path = bundle_path

        self.transformer = DataTransformer()
        self.feature_engineer = FeatureEngineer()
        self.predictor = AMLPredictor(bundle_path=bundle_path)
        self.explainer = AMLShapExplainer(bundle_path=bundle_path)

        self.model_type = self.predictor.bundle["model_type"]

        logger.info(f"Inference service инициализирован с моделью '{self.model_type}'")

    def _build_recommendation(self, prediction_score: float) -> str:
        if prediction_score >= 0.80:
            return "red"
        if prediction_score >= 0.50:
            return "yellow"
        return "green"

    def _prepare_dataframe(self, transactions: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(transactions).copy()

        if "fraud_bool" not in df.columns:
            df["fraud_bool"] = 0

        transformed_df = self.transformer.transform(df)
        feature_result = self.feature_engineer.build_features(transformed_df)

        return feature_result.dataframe

    def _save_prediction_log(
        self,
        payload: dict[str, Any],
        result: dict[str, Any] | list[dict[str, Any]],
        mode: str,
        latency_ms: float,
    ) -> None:
        log_dir = get_artifacts_dir() / "predictions"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_record = {
            "request_id": str(uuid4()),
            "timestamp_utc": datetime.utcnow().isoformat(),
            "mode": mode,
            "model_type": self.model_type,
            "latency_ms": latency_ms,
            "payload": payload,
            "result": result,
        }

        output_path = log_dir / f"{log_record['timestamp_utc'].replace(':', '-')}_{mode}.json"
        save_json(log_record, output_path)

    def predict_one(self, transaction: dict[str, Any]) -> dict[str, Any]:
        start_time = perf_counter()

        feature_df = self._prepare_dataframe([transaction])

        prediction_df = self.predictor.predict_proba(feature_df)
        explanation = self.explainer.explain_rows(
            feature_df,
            row_indices=[0],
            top_k=5,
            background_size=1,
        )[0]

        prediction_score = float(prediction_df.iloc[0]["prediction_score"])
        prediction_label = int(prediction_df.iloc[0]["prediction_label"])
        recommendation = self._build_recommendation(prediction_score)

        result = {
            "prediction_score": prediction_score,
            "prediction_label": prediction_label,
            "recommendation": recommendation,
            "top_positive_factors": explanation.top_positive_factors,
            "top_negative_factors": explanation.top_negative_factors,
            "human_readable_reasons": self.explainer.build_human_readable_reasons(explanation),
        }

        latency_ms = (perf_counter() - start_time) * 1000
        self._save_prediction_log(
            payload=transaction,
            result=result,
            mode="single",
            latency_ms=latency_ms,
        )

        logger.info(f"Single prediction выполнен за {latency_ms:.2f} мс")

        return result

    def predict_batch(self, transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        start_time = perf_counter()

        if not transactions:
            return []

        feature_df = self._prepare_dataframe(transactions)
        prediction_df = self.predictor.predict_proba(feature_df)

        row_indices = list(range(len(feature_df)))
        explanations = self.explainer.explain_rows(
            feature_df,
            row_indices=row_indices,
            top_k=5,
            background_size=min(100, len(feature_df)),
        )

        explanation_by_row = {item.row_index: item for item in explanations}

        results = []
        for row_index in range(len(prediction_df)):
            prediction_score = float(prediction_df.iloc[row_index]["prediction_score"])
            prediction_label = int(prediction_df.iloc[row_index]["prediction_label"])
            recommendation = self._build_recommendation(prediction_score)

            explanation = explanation_by_row[row_index]

            results.append(
                {
                    "row_index": row_index,
                    "prediction_score": prediction_score,
                    "prediction_label": prediction_label,
                    "recommendation": recommendation,
                    "top_positive_factors": explanation.top_positive_factors,
                    "top_negative_factors": explanation.top_negative_factors,
                    "human_readable_reasons": self.explainer.build_human_readable_reasons(explanation),
                }
            )

        latency_ms = (perf_counter() - start_time) * 1000
        self._save_prediction_log(
            payload={"batch_size": len(transactions)},
            result=results,
            mode="batch",
            latency_ms=latency_ms,
        )

        logger.info(
            f"Batch prediction выполнен: batch_size={len(transactions)}, latency={latency_ms:.2f} мс"
        )

        return results
