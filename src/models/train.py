from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config.settings import settings
from src.models.evaluate import calculate_classification_metrics
from src.models.registry import AMLModelRegistry
from src.utils.io import copy_file, read_yaml, save_json, save_object
from src.utils.logger import get_logger
from src.utils.paths import get_artifacts_dir

logger = get_logger(__name__)


@dataclass
class SplitResult:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass
class TrainingResult:
    model_type: str
    metrics_valid: dict[str, Any]
    metrics_test: dict[str, Any]
    bundle_path: str
    run_id: str
    feature_names_after_preprocessing: list[str]


class AMLModelTrainer:
    def __init__(
        self,
        model_config_path: str = "config/model_config.yaml",
        target_column: str | None = None,
    ) -> None:
        self.target_column = target_column or settings.target_column
        self.model_config = read_yaml(model_config_path)

        self.training_config = self.model_config["training"]
        self.lightgbm_config = self.model_config["lightgbm"]
        self.xgboost_config = self.model_config["xgboost"]

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

    def split_dataset(self, df: pd.DataFrame) -> SplitResult:
        train_size = 0.7
        valid_size = 0.15

        total_rows = len(df)
        train_end = int(total_rows * train_size)
        valid_end = train_end + int(total_rows * valid_size)

        train_df = df.iloc[:train_end].copy()
        valid_df = df.iloc[train_end:valid_end].copy()
        test_df = df.iloc[valid_end:].copy()

        logger.info(
            "Датасет разделён по времени: "
            f"train={train_df.shape}, valid={valid_df.shape}, test={test_df.shape}"
        )

        return SplitResult(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
        )

    def build_preprocessor(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
    ) -> ColumnTransformer:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, numerical_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        logger.info(
            "Собран preprocessing pipeline: "
            f"numerical={len(numerical_columns)}, categorical={len(categorical_columns)}"
        )
        return preprocessor

    def _build_model(self, model_type: str):
        if model_type == "lightgbm":
            return lgb.LGBMClassifier(**self.lightgbm_config)

        if model_type == "xgboost":
            return xgb.XGBClassifier(**self.xgboost_config)

        raise ValueError(f"Неподдерживаемый тип модели: {model_type}")

    def _extract_feature_names(
        self,
        preprocessor: ColumnTransformer,
    ) -> list[str]:
        return list(preprocessor.get_feature_names_out())

    def train(
        self,
        df: pd.DataFrame,
        categorical_columns: list[str],
        numerical_columns: list[str],
        model_type: str = "lightgbm",
    ) -> TrainingResult:
        split_result = self.split_dataset(df)

        train_df = split_result.train_df
        valid_df = split_result.valid_df
        test_df = split_result.test_df

        X_train = train_df.drop(columns=[self.target_column, "event_time"], errors="ignore")
        y_train = train_df[self.target_column].values

        X_valid = valid_df.drop(columns=[self.target_column, "event_time"], errors="ignore")
        y_valid = valid_df[self.target_column].values

        X_test = test_df.drop(columns=[self.target_column, "event_time"], errors="ignore")
        y_test = test_df[self.target_column].values

        categorical_columns = [col for col in categorical_columns if col in X_train.columns]
        numerical_columns = [col for col in numerical_columns if col in X_train.columns]

        preprocessor = self.build_preprocessor(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
        )

        logger.info("Обучение preprocessor на train-части")
        X_train_prepared = preprocessor.fit_transform(X_train)
        X_valid_prepared = preprocessor.transform(X_valid)
        X_test_prepared = preprocessor.transform(X_test)

        feature_names_after_preprocessing = self._extract_feature_names(preprocessor)

        model = self._build_model(model_type)
        threshold = float(self.training_config["threshold"])

        with mlflow.start_run(run_name=f"{model_type}_training") as run:
            run_id = run.info.run_id

            logger.info(f"Старт MLflow run: run_id={run_id}")

            mlflow.log_param("model_type", model_type)
            mlflow.log_param("target_column", self.target_column)
            mlflow.log_param("n_train_rows", len(X_train))
            mlflow.log_param("n_valid_rows", len(X_valid))
            mlflow.log_param("n_test_rows", len(X_test))
            mlflow.log_param("n_features_before_preprocessing", X_train.shape[1])
            mlflow.log_param("n_features_after_preprocessing", len(feature_names_after_preprocessing))
            mlflow.log_param("threshold", threshold)

            if model_type == "lightgbm":
                mlflow.log_params(self.lightgbm_config)
            elif model_type == "xgboost":
                mlflow.log_params(self.xgboost_config)

            logger.info(f"Обучение модели типа '{model_type}'")
            model.fit(X_train_prepared, y_train)

            y_valid_proba = model.predict_proba(X_valid_prepared)[:, 1]
            y_test_proba = model.predict_proba(X_test_prepared)[:, 1]

            metrics_valid = calculate_classification_metrics(
                y_true=y_valid,
                y_proba=y_valid_proba,
                threshold=threshold,
            )
            metrics_test = calculate_classification_metrics(
                y_true=y_test,
                y_proba=y_test_proba,
                threshold=threshold,
            )

            for metric_name, metric_value in metrics_valid.items():
                mlflow.log_metric(f"valid_{metric_name}", metric_value)

            for metric_name, metric_value in metrics_test.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)

            bundle = {
                "model": model,
                "preprocessor": preprocessor,
                "feature_columns_before_preprocessing": list(X_train.columns),
                "feature_names_after_preprocessing": feature_names_after_preprocessing,
                "categorical_columns": categorical_columns,
                "numerical_columns": numerical_columns,
                "target_column": self.target_column,
                "model_type": model_type,
                "threshold": threshold,
            }

            bundle_path = get_artifacts_dir() / "models" / f"{model_type}_bundle.joblib"
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            save_object(bundle, bundle_path)

            mlflow.log_artifact(str(bundle_path), artifact_path="model_bundle")

            logger.info(
                f"Обучение модели '{model_type}' завершено. "
                f"valid_roc_auc={metrics_valid['roc_auc']:.4f}, "
                f"test_roc_auc={metrics_test['roc_auc']:.4f}"
            )

            return TrainingResult(
                model_type=model_type,
                metrics_valid=metrics_valid,
                metrics_test=metrics_test,
                bundle_path=str(bundle_path),
                run_id=run_id,
                feature_names_after_preprocessing=feature_names_after_preprocessing,
            )

    def train_and_select_best(
        self,
        df: pd.DataFrame,
        categorical_columns: list[str],
        numerical_columns: list[str],
        candidate_models: list[str] | None = None,
    ) -> dict[str, Any]:
        candidate_models = candidate_models or ["lightgbm", "xgboost"]

        results: list[TrainingResult] = []

        for model_type in candidate_models:
            result = self.train(
                df=df,
                categorical_columns=categorical_columns,
                numerical_columns=numerical_columns,
                model_type=model_type,
            )
            results.append(result)

        best_result = max(results, key=lambda x: x.metrics_valid["pr_auc"])

        production_bundle_path = get_artifacts_dir() / "models" / "production_bundle.joblib"
        copy_file(best_result.bundle_path, production_bundle_path)

        summary = {
            "best_model_type": best_result.model_type,
            "best_run_id": best_result.run_id,
            "best_bundle_path": best_result.bundle_path,
            "production_bundle_path": str(production_bundle_path),
            "best_valid_metrics": best_result.metrics_valid,
            "best_test_metrics": best_result.metrics_test,
            "all_results": [
                {
                    "model_type": result.model_type,
                    "run_id": result.run_id,
                    "bundle_path": result.bundle_path,
                    "metrics_valid": result.metrics_valid,
                    "metrics_test": result.metrics_test,
                }
                for result in results
            ],
        }

        summary_path = get_artifacts_dir() / "metrics" / "training_summary.json"
        save_json(summary, summary_path)

        try:
            registry = AMLModelRegistry(model_name="aml_detection_model")
            registered = registry.register_run_artifact(
                run_id=best_result.run_id,
                artifact_path=f"model_bundle/{best_result.model_type}_bundle.joblib",
            )
            summary["registry"] = registered
            save_json(summary, summary_path)
        except Exception as exc:
            logger.warning(f"Не удалось зарегистрировать модель в MLflow Registry: {exc}")
            summary["registry"] = None
            save_json(summary, summary_path)

        logger.info(
            "Лучшая модель выбрана по valid PR-AUC: "
            f"{best_result.model_type}, pr_auc={best_result.metrics_valid['pr_auc']:.4f}"
        )
        logger.info(f"Production bundle обновлён: {production_bundle_path}")

        return summary
