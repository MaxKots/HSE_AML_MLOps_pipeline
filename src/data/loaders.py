from pathlib import Path
from typing import Literal

import pandas as pd

from config.settings import settings
from src.utils.io import load_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

DatasetName = Literal["base", "variant_1", "variant_2", "synthaml_alerts", "synthaml_transactions"]


class DataLoader:
    def __init__(self) -> None:
        self.dataset_paths = {
            "base": Path(settings.base_dataset),
            "variant_1": Path(settings.drift_dataset_1),
            "variant_2": Path(settings.drift_dataset_2),
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

    def load_dataset(self, dataset_name: DatasetName) -> pd.DataFrame:
        if dataset_name not in self.dataset_paths:
            allowed = ", ".join(self.dataset_paths.keys())
            raise ValueError(f"Неизвестный датасет: {dataset_name}. Допустимые значения: {allowed}")

        dataset_path = self.dataset_paths[dataset_name]
        logger.info(f"Выбран преднастроенный датасет '{dataset_name}' -> {dataset_path}")

        return self.load_from_path(dataset_path)

    def save_dataset(self, df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
        path = Path(path)

        logger.info(f"Сохранение датасета в файл: {path}")

        if path.suffix == ".csv":
            df.to_csv(path, index=index)
        elif path.suffix == ".parquet":
            df.to_parquet(path, index=index)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")

        logger.info(f"Датасет сохранён: rows={len(df)}, cols={len(df.columns)}")
