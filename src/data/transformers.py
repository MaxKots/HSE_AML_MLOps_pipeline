from typing import Tuple

import numpy as np
import pandas as pd

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataTransformer:
    def __init__(self, target_column: str | None = None) -> None:
        self.target_column = target_column or settings.target_column

        self.columns_with_missing_marker = [
            "prev_address_months_count",
            "current_address_months_count",
            "intended_balcon_amount",
            "bank_months_count",
            "session_length_in_minutes",
        ]

        self.categorical_columns = [
            "payment_type",
            "employment_status",
            "housing_status",
            "source",
            "device_os",
        ]

        self.binary_columns = [
            "email_is_free",
            "phone_home_valid",
            "phone_mobile_valid",
            "has_other_cards",
            "foreign_request",
            "keep_alive_session",
        ]

    def replace_missing_markers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for column in self.columns_with_missing_marker:
            if column in df.columns:
                missing_count = int((df[column] == -1).sum())
                if missing_count > 0:
                    logger.info(
                        f"Замена marker '-1' на NaN в колонке '{column}', количество значений: {missing_count}"
                    )
                    df[column] = df[column].replace(-1, np.nan)

        return df

    def cast_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for column in self.categorical_columns:
            if column in df.columns:
                df[column] = df[column].astype(str)

        return df

    def cast_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for column in self.binary_columns:
            if column in df.columns:
                df[column] = df[column].astype("Int64")

        return df

    def drop_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        duplicates_count = int(df.duplicated().sum())
        if duplicates_count > 0:
            logger.info(f"Удаление дубликатов строк: {duplicates_count}")
            df = df.drop_duplicates()

        return df

    def sort_for_reproducibility(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        sort_candidates = [
            "month",
            "days_since_request",
            "customer_age",
            "income",
        ]
        available_sort_columns = [col for col in sort_candidates if col in df.columns]

        if available_sort_columns:
            df = df.sort_values(by=available_sort_columns).reset_index(drop=True)
            logger.info(
                f"Датасет отсортирован для воспроизводимости по колонкам: {available_sort_columns}"
            )

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Запуск базовой трансформации датасета")

        df = self.drop_duplicate_rows(df)
        df = self.replace_missing_markers(df)
        df = self.cast_categorical_columns(df)
        df = self.cast_binary_columns(df)
        df = self.sort_for_reproducibility(df)

        logger.info(
            f"Базовая трансформация завершена: rows={len(df)}, cols={len(df.columns)}"
        )
        return df

    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if self.target_column not in df.columns:
            raise ValueError(f"Колонка target '{self.target_column}' не найдена")

        X = df.drop(columns=[self.target_column]).copy()
        y = df[self.target_column].copy()

        logger.info(
            f"Признаки и целевая переменная разделены: X_shape={X.shape}, y_shape={y.shape}"
        )
        return X, y
