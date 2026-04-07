from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import pandas as pd

from src.data import DataTransformer, DataValidator
from src.features import FeatureEngineer
from src.models.evaluate import calculate_classification_metrics
from src.models.train import AMLModelTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    experiment_name: str
    model_type: str
    use_feature_engineering: bool
    train_rows: int
    test_rows: int
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    train_time_sec: float
    inference_time_sec: float


class AMLBenchmarkRunner:
    def __init__(self) -> None:
        self.validator = DataValidator()
        self.transformer = DataTransformer()
        self.feature_engineer = FeatureEngineer()
        self.trainer = AMLModelTrainer()

    def _prepare_dataset(
        self,
        df: pd.DataFrame,
        use_feature_engineering: bool,
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        self.validator.run_full_validation(df)

        transformed_df = self.transformer.transform(df)

        if use_feature_engineering:
            feature_result = self.feature_engineer.build_features(transformed_df)
            return (
                feature_result.dataframe,
                feature_result.categorical_columns,
                feature_result.numerical_columns,
            )

        raw_df = transformed_df.copy()

        categorical_columns = [
            col for col in [
                "payment_type",
                "employment_status",
                "housing_status",
                "source",
                "device_os",
            ]
            if col in raw_df.columns
        ]

        numerical_columns = [
            col for col in raw_df.columns
            if col not in categorical_columns + ["fraud_bool", "event_time"]
        ]

        return raw_df, categorical_columns, numerical_columns

    def run_single_experiment(
        self,
        experiment_name: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
        model_type: str = "lightgbm",
        use_feature_engineering: bool = True,
    ) -> BenchmarkResult:
        prepared_train_df, categorical_columns, numerical_columns = self._prepare_dataset(
            train_df,
            use_feature_engineering=use_feature_engineering,
        )

        start_train = perf_counter()
        training_result = self.trainer.train(
            df=prepared_train_df,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            model_type=model_type,
        )
        train_time_sec = perf_counter() - start_train

        if test_df is None:
            metrics = training_result.metrics_test
            test_rows = int(len(prepared_train_df) * 0.15)
            inference_time_sec = 0.0
        else:
            prepared_test_df, _, _ = self._prepare_dataset(
                test_df,
                use_feature_engineering=use_feature_engineering,
            )

            from src.models.predict import AMLPredictor

            predictor = AMLPredictor(training_result.bundle_path)

            start_inference = perf_counter()
            prediction_df = predictor.predict_proba(prepared_test_df)
            inference_time_sec = perf_counter() - start_inference

            y_true = prepared_test_df["fraud_bool"].values
            y_proba = prediction_df["prediction_score"].values

            metrics = calculate_classification_metrics(
                y_true=y_true,
                y_proba=y_proba,
                threshold=0.5,
            )
            test_rows = len(prepared_test_df)

        result = BenchmarkResult(
            experiment_name=experiment_name,
            model_type=model_type,
            use_feature_engineering=use_feature_engineering,
            train_rows=len(prepared_train_df),
            test_rows=test_rows,
            roc_auc=metrics["roc_auc"],
            pr_auc=metrics["pr_auc"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            false_positive_rate=metrics["false_positive_rate"],
            train_time_sec=train_time_sec,
            inference_time_sec=inference_time_sec,
        )

        logger.info(
            f"Эксперимент '{experiment_name}' завершён: "
            f"model={model_type}, "
            f"feature_engineering={use_feature_engineering}, "
            f"roc_auc={result.roc_auc:.4f}, "
            f"pr_auc={result.pr_auc:.4f}"
        )

        return result
