from __future__ import annotations

import pandas as pd

from src.utils.io import load_object
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AMLPredictor:
    def __init__(self, bundle_path: str) -> None:
        self.bundle = load_object(bundle_path)

        self.model = self.bundle["model"]
        self.preprocessor = self.bundle["preprocessor"]
        self.feature_columns_before_preprocessing = self.bundle["feature_columns_before_preprocessing"]
        self.threshold = self.bundle["threshold"]

        logger.info(f"Model bundle загружен из {bundle_path}")

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()

        if "fraud_bool" in X.columns:
            X = X.drop(columns=["fraud_bool"])

        if "event_time" in X.columns:
            X = X.drop(columns=["event_time"])

        missing_columns = [col for col in self.feature_columns_before_preprocessing if col not in X.columns]
        if missing_columns:
            raise ValueError(
                f"Во входных данных отсутствуют колонки, необходимые для предсказания: {missing_columns}"
            )

        X = X[self.feature_columns_before_preprocessing].copy()

        X_prepared = self.preprocessor.transform(X)
        probabilities = self.model.predict_proba(X_prepared)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        result = df.copy()
        result["prediction_score"] = probabilities
        result["prediction_label"] = predictions

        return result
