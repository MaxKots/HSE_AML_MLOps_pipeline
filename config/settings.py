from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="local", alias="APP_ENV")
    app_name: str = Field(default="aml-pipeline", alias="APP_NAME")

    data_raw_dir: str = Field(default="data/raw", alias="DATA_RAW_DIR")
    data_processed_dir: str = Field(default="data/processed", alias="DATA_PROCESSED_DIR")
    data_reference_dir: str = Field(default="data/reference", alias="DATA_REFERENCE_DIR")
    artifacts_dir: str = Field(default="artifacts", alias="ARTIFACTS_DIR")

    base_dataset_path: str = Field(default="data/raw/Base.csv", alias="BASE_DATASET_PATH")
    drift_dataset_path_1: str = Field(default="data/raw/Variant I.csv", alias="DRIFT_DATASET_PATH_1")
    drift_dataset_path_2: str = Field(default="data/raw/Variant II.csv", alias="DRIFT_DATASET_PATH_2")
    synthaml_alerts_path: str = Field(default="data/raw/synthaml_alerts.csv", alias="SYNTHAML_ALERTS_PATH")
    synthaml_transactions_path: str = Field(default="data/raw/synthaml_transactions.csv", alias="SYNTHAML_TRANSACTIONS_PATH")

    target_column: str = Field(default="fraud_bool", alias="TARGET_COLUMN")
    random_seed: int = Field(default=42, alias="RANDOM_SEED")

    mlflow_tracking_uri: str = Field(default="file:./mlruns", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="aml_detection", alias="MLFLOW_EXPERIMENT_NAME")

    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="aml", alias="POSTGRES_DB")
    postgres_user: str = Field(default="aml_user", alias="POSTGRES_USER")
    postgres_password: str = Field(default="aml_password", alias="POSTGRES_PASSWORD")

    minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="mlflow", alias="MINIO_BUCKET")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    dashboard_api_url: str = Field(default="http://localhost:8000", alias="DASHBOARD_API_URL")



    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def base_dataset(self) -> Path:
        return self.project_root / self.base_dataset_path

    @property
    def drift_dataset_1(self) -> Path:
        return self.project_root / self.drift_dataset_path_1

    @property
    def drift_dataset_2(self) -> Path:
        return self.project_root / self.drift_dataset_path_2

    @property
    def synthaml_alerts(self) -> Path:
        return self.project_root / self.synthaml_alerts_path

    @property
    def synthaml_transactions(self) -> Path:
        return self.project_root / self.synthaml_transactions_path


settings = Settings()
