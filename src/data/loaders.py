from __future__ import annotations

from pathlib import Path
from typing import Literal, Any

import pandas as pd

from config.settings import settings
from src.data.data_sources import DataSourceManager
from src.utils.io import load_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

DatasetName = Literal[
    "base",
    "variant_1",
    "variant_2",
    "samld",
    "synthaml_alerts",
    "synthaml_transactions",
]


class DataLoader:
    def __init__(
        self,
        source_override: str | None = None,
        data_sources_config_path: str | Path | None = None,
    ) -> None:
        self.source_override = source_override
        self.source_manager = DataSourceManager(config_path=data_sources_config_path)

        # Backward-compatible local paths for old code and direct file loading.
        self.dataset_paths = {
            "base": Path(settings.base_dataset),
            "variant_1": Path(settings.drift_dataset_1),
            "variant_2": Path(settings.drift_dataset_2),
            "samld": Path(settings.samld_dataset),
            "synthaml_alerts": Path(settings.synthaml_alerts),
            "synthaml_transactions": Path(settings.synthaml_transactions),
        }

    def load_from_path(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        logger.info(f"Загрузка датасета из файла: {path}")

        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        df = load_dataframe(path)
        logger.info(f"Датасет загружен: rows={len(df)}, cols={len(df.columns)}")
        return df

    def load_dataset(
        self,
        dataset_name: str,
        source_override: str | None = None,
        **read_options: Any,
    ) -> pd.DataFrame:
        source = source_override or self.source_override
        return self.source_manager.load_dataset(
            dataset_name=dataset_name,
            source_override=source,
            **read_options,
        )

    def save_dataset(self, df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
        """Legacy local save used by existing pipelines."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Сохранение датасета в файл: {path}")

        if path.suffix == ".csv":
            df.to_csv(path, index=index)
        elif path.suffix == ".parquet":
            df.to_parquet(path, index=index)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")

        logger.info(f"Датасет сохранён: rows={len(df)}, cols={len(df.columns)}")

    def save_dataset_to_source(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        source_override: str | None = None,
        **write_options: Any,
    ) -> None:
        source = source_override or self.source_override
        self.source_manager.save_dataset(
            df=df,
            dataset_name=dataset_name,
            source_override=source,
            **write_options,
        )
