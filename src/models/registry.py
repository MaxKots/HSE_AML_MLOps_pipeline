from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AMLModelRegistry:
    def __init__(self, model_name: str = "aml_detection_model") -> None:
        self.model_name = model_name

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    def register_run_artifact(
        self,
        run_id: str,
        artifact_path: str,
    ) -> dict[str, Any]:
        model_uri = f"runs:/{run_id}/{artifact_path}"

        logger.info(
            f"Регистрация модели в MLflow Model Registry: model_name={self.model_name}, model_uri={model_uri}"
        )

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=self.model_name,
        )

        return {
            "model_name": self.model_name,
            "version": registered_model.version,
            "status": registered_model.status,
        }

    def transition_stage(
        self,
        version: str,
        stage: str,
        archive_existing_versions: bool = True,
    ) -> None:
        logger.info(
            f"Перевод модели '{self.model_name}' версии '{version}' в стадию '{stage}'"
        )

        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions,
        )

    def get_latest_versions(self) -> list[dict[str, Any]]:
        versions = self.client.get_latest_versions(self.model_name)

        result = []
        for version in versions:
            result.append(
                {
                    "name": version.name,
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "source": version.source,
                }
            )

        return result
