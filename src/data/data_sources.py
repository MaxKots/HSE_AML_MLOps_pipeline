from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from sqlalchemy import create_engine, text

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
SUPPORTED_SOURCES = {"csv", "s3", "postgres", "api"}


@dataclass(frozen=True)
class DatasetSource:
    dataset_name: str
    source: str
    config: dict[str, Any]
    target_column: str | None = None


def _with_http_scheme(endpoint: str) -> str:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return f"http://{endpoint}"


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return settings.project_root / path


def _default_config() -> dict[str, Any]:
    return {
        "default_source": "csv",
        "datasets": {
            "base": {
                "target_column": "fraud_bool",
                "csv": {"path": str(settings.base_dataset_path)},
                "s3": {"uri": "s3://aml-data/raw/Base.csv"},
                "postgres": {"table": "datasets.base"},
            },
            "variant_1": {
                "target_column": "fraud_bool",
                "csv": {"path": str(settings.drift_dataset_path_1)},
                "s3": {"uri": "s3://aml-data/raw/Variant I.csv"},
                "postgres": {"table": "datasets.variant_1"},
            },
            "variant_2": {
                "target_column": "fraud_bool",
                "csv": {"path": str(settings.drift_dataset_path_2)},
                "s3": {"uri": "s3://aml-data/raw/Variant II.csv"},
                "postgres": {"table": "datasets.variant_2"},
            },
            "samld": {
                "target_column": "Is_laundering",
                "csv": {"path": str(settings.samld_dataset_path)},
                "s3": {"uri": "s3://aml-data/raw/SAML-D.csv"},
                "postgres": {"table": "datasets.samld"},
            },
            "synthaml_alerts": {
                "target_column": "fraud_bool",
                "csv": {"path": str(settings.synthaml_alerts_path)},
                "s3": {"uri": "s3://aml-data/raw/synthaml_alerts.csv"},
                "postgres": {"table": "datasets.synthaml_alerts"},
            },
            "synthaml_transactions": {
                "target_column": "fraud_bool",
                "csv": {"path": str(settings.synthaml_transactions_path)},
                "s3": {"uri": "s3://aml-data/raw/synthaml_transactions.csv"},
                "postgres": {"table": "datasets.synthaml_transactions"},
            },
        },
    }


class DataSourceRegistry:
    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = getattr(settings, "data_sources_config_path", "config/data_sources.yaml")
        self.config_path = _project_path(config_path)
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as file:
                self.config = yaml.safe_load(file) or {}
            logger.info(f"Конфиг источников данных загружен: {self.config_path}")
        else:
            self.config = _default_config()
            logger.warning(
                f"Конфиг источников данных не найден: {self.config_path}. "
                "Используется встроенная конфигурация по умолчанию."
            )

    @property
    def default_source(self) -> str:
        return str(
            getattr(settings, "default_data_source", None)
            or self.config.get("default_source")
            or "csv"
        ).lower()

    def dataset_names(self) -> list[str]:
        return sorted((self.config.get("datasets") or {}).keys())

    def resolve(self, dataset_name: str, source_override: str | None = None) -> DatasetSource:
        datasets = self.config.get("datasets") or {}
        if dataset_name not in datasets:
            allowed = ", ".join(sorted(datasets.keys()))
            raise ValueError(f"Неизвестный датасет: {dataset_name}. Допустимые значения: {allowed}")

        dataset_cfg = datasets[dataset_name] or {}
        source = (source_override or self.default_source).lower()
        if source not in SUPPORTED_SOURCES:
            allowed = ", ".join(sorted(SUPPORTED_SOURCES))
            raise ValueError(f"Неподдерживаемый источник данных: {source}. Допустимые значения: {allowed}")

        source_cfg = dataset_cfg.get(source)
        if not source_cfg:
            raise ValueError(
                f"Для датасета '{dataset_name}' не настроен источник '{source}' в {self.config_path}"
            )
        return DatasetSource(dataset_name, source, dict(source_cfg), dataset_cfg.get("target_column"))


class CsvDataSource:
    def load(self, cfg: dict[str, Any], **read_options: Any) -> pd.DataFrame:
        path = _project_path(cfg["path"])
        logger.info(f"Загрузка CSV/Parquet датасета: path={path}")
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, **read_options)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path, **read_options)
        raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")

    def save(self, df: pd.DataFrame, cfg: dict[str, Any], **write_options: Any) -> None:
        path = _project_path(cfg["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        index = bool(write_options.pop("index", False))
        logger.info(f"Сохранение датасета в локальный файл: path={path}")
        if path.suffix.lower() == ".csv":
            df.to_csv(path, index=index, **write_options)
        elif path.suffix.lower() == ".parquet":
            df.to_parquet(path, index=index, **write_options)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")


class S3DataSource:
    def _storage_options(self) -> dict[str, Any]:
        return {
            "key": settings.minio_access_key,
            "secret": settings.minio_secret_key,
            "client_kwargs": {"endpoint_url": _with_http_scheme(settings.minio_endpoint)},
        }

    def load(self, cfg: dict[str, Any], **read_options: Any) -> pd.DataFrame:
        uri = cfg["uri"]
        logger.info(f"Загрузка датасета из S3/MinIO: uri={uri}")
        if uri.lower().endswith(".csv"):
            return pd.read_csv(uri, storage_options=self._storage_options(), **read_options)
        if uri.lower().endswith(".parquet"):
            return pd.read_parquet(uri, storage_options=self._storage_options(), **read_options)
        raise ValueError(f"Неподдерживаемый формат S3-объекта: {uri}")

    def save(self, df: pd.DataFrame, cfg: dict[str, Any], **write_options: Any) -> None:
        uri = cfg["uri"]
        index = bool(write_options.pop("index", False))
        logger.info(f"Сохранение датасета в S3/MinIO: uri={uri}")
        if uri.lower().endswith(".csv"):
            df.to_csv(uri, index=index, storage_options=self._storage_options(), **write_options)
        elif uri.lower().endswith(".parquet"):
            df.to_parquet(uri, index=index, storage_options=self._storage_options(), **write_options)
        else:
            raise ValueError(f"Неподдерживаемый формат S3-объекта: {uri}")


class PostgresDataSource:
    def _engine(self):
        url = (
            f"postgresql+psycopg2://{settings.postgres_user}:"
            f"{settings.postgres_password}@{settings.postgres_host}:"
            f"{settings.postgres_port}/{settings.postgres_db}"
        )
        return create_engine(url)

    @staticmethod
    def _split_table_name(table: str) -> tuple[str | None, str]:
        if "." in table:
            schema, name = table.split(".", 1)
            return schema, name
        return None, table

    def load(self, cfg: dict[str, Any], **read_options: Any) -> pd.DataFrame:
        table = cfg.get("table")
        query = cfg.get("query")
        limit = read_options.pop("limit", None) or read_options.pop("nrows", None)
        engine = self._engine()
        if query:
            sql = str(query)
            if limit:
                sql = f"SELECT * FROM ({sql}) AS source_query LIMIT {int(limit)}"
            return pd.read_sql_query(sql, engine, **read_options)
        if not table:
            raise ValueError("Для postgres-источника нужно указать table или query")
        sql = f"SELECT * FROM {table}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        logger.info(f"Загрузка датасета из PostgreSQL: table={table}, limit={limit}")
        return pd.read_sql_query(sql, engine, **read_options)

    def save(self, df: pd.DataFrame, cfg: dict[str, Any], **write_options: Any) -> None:
        table = cfg.get("table")
        if not table:
            raise ValueError("Для сохранения в postgres нужно указать table")
        if_exists = str(write_options.pop("if_exists", "replace"))
        index = bool(write_options.pop("index", False))
        chunksize = int(write_options.pop("chunksize", 100_000))
        method = write_options.pop("method", None)
        schema, table_name = self._split_table_name(table)
        engine = self._engine()
        if schema:
            with engine.begin() as conn:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        logger.info(f"Сохранение датасета в PostgreSQL: table={table}, rows={len(df)}")
        df.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists=if_exists,
            index=index,
            chunksize=chunksize,
            method=method,
            **write_options,
        )


class ApiDataSource:
    def load(self, cfg: dict[str, Any], **read_options: Any) -> pd.DataFrame:
        url = cfg["url"]
        params = dict(cfg.get("params") or {})
        for key in ("nrows", "limit"):
            if key in read_options:
                params[key] = read_options.pop(key)
        logger.info(f"Загрузка датасета из API: url={url}, params={params}")
        response = requests.get(url, params=params, timeout=int(cfg.get("timeout", 120)))
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get("data") or payload.get("items") or payload.get("records")
        else:
            records = None
        if records is None:
            raise ValueError("API должен вернуть list или dict с ключом data/items/records")
        return pd.DataFrame(records)

    def save(self, df: pd.DataFrame, cfg: dict[str, Any], **write_options: Any) -> None:
        url = cfg["url"]
        records = df.where(pd.notnull(df), None).to_dict(orient="records")
        logger.info(f"Отправка датасета в API: url={url}, rows={len(records)}")
        response = requests.post(url, json={"records": records}, timeout=int(cfg.get("timeout", 120)))
        response.raise_for_status()


class DataSourceManager:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self.registry = DataSourceRegistry(config_path=config_path)
        self.sources = {
            "csv": CsvDataSource(),
            "s3": S3DataSource(),
            "postgres": PostgresDataSource(),
            "api": ApiDataSource(),
        }

    def resolve(self, dataset_name: str, source_override: str | None = None) -> DatasetSource:
        return self.registry.resolve(dataset_name, source_override=source_override)

    def load_dataset(self, dataset_name: str, source_override: str | None = None, **read_options: Any) -> pd.DataFrame:
        resolved = self.resolve(dataset_name, source_override=source_override)
        df = self.sources[resolved.source].load(resolved.config, **read_options)
        logger.info(
            f"Датасет загружен: dataset={dataset_name}, source={resolved.source}, rows={len(df)}, cols={len(df.columns)}"
        )
        return df

    def save_dataset(self, df: pd.DataFrame, dataset_name: str, source_override: str | None = None, **write_options: Any) -> None:
        resolved = self.resolve(dataset_name, source_override=source_override)
        self.sources[resolved.source].save(df, resolved.config, **write_options)
        logger.info(
            f"Датасет сохранён: dataset={dataset_name}, source={resolved.source}, rows={len(df)}, cols={len(df.columns)}"
        )
