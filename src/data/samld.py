from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "fraud_bool"

RAW_TARGET_COLUMN = "Is_laundering"
RAW_TIME_COLUMN = "Time"
RAW_DATE_COLUMN = "Date"

TECHNICAL_COLUMNS = [
    "Sender_account",
    "Receiver_account",
]

LEAKAGE_COLUMNS = [
    # Это не признак операции, а типология/описание разметки. Использовать в модели нельзя.
    "Laundering_type",
]

CATEGORICAL_COLUMNS = [
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
]

REQUIRED_COLUMNS = [
    RAW_TIME_COLUMN,
    RAW_DATE_COLUMN,
    "Sender_account",
    "Receiver_account",
    "Amount",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
    RAW_TARGET_COLUMN,
    "Laundering_type",
]


def load_samld_raw(path: str | Path | None = None, max_rows: int | None = None) -> pd.DataFrame:
    """
    Загружает SAML-D из CSV.

    max_rows:
        None или 0 -> весь файл;
        положительное число -> первые max_rows строк.
    """
    if path is None:
        path = settings.samld_dataset

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"SAML-D файл не найден: {path}")

    nrows = None
    if max_rows is not None and int(max_rows) > 0:
        nrows = int(max_rows)

    logger.info(f"Загрузка SAML-D: path={path}, nrows={nrows}")
    df = pd.read_csv(path, nrows=nrows)

    logger.info(f"SAML-D загружен: rows={len(df)}, cols={len(df.columns)}")
    return df


def _validate_samld_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(
            "В SAML-D отсутствуют обязательные колонки: "
            f"{missing_columns}"
        )


def _build_event_time(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        df[RAW_DATE_COLUMN].astype(str) + " " + df[RAW_TIME_COLUMN].astype(str),
        errors="coerce",
    )


def prepare_samld_dataset_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Готовит SAML-D к обучению в общем ML/MLOps-контуре.

    Важные решения:
    - Is_laundering переименовывается в fraud_bool;
    - технические account-id не используются как признаки;
    - Laundering_type удаляется как leakage-признак;
    - Date/Time используются только для event_time и календарных признаков;
    - категориальные признаки оставляются строковыми для OneHotEncoder внутри trainer.
    """
    _validate_samld_columns(df)

    result = df.copy()

    result["event_time"] = _build_event_time(result)
    result = result.sort_values("event_time", na_position="last").reset_index(drop=True)

    result[TARGET_COLUMN] = pd.to_numeric(
        result[RAW_TARGET_COLUMN],
        errors="coerce",
    ).fillna(0).astype(int)

    result["Amount"] = pd.to_numeric(result["Amount"], errors="coerce")
    result["amount_abs"] = result["Amount"].abs()
    result["amount_log1p"] = np.log1p(result["amount_abs"].fillna(0.0))

    result["hour"] = result["event_time"].dt.hour.fillna(-1).astype(int)
    result["day_of_week"] = result["event_time"].dt.dayofweek.fillna(-1).astype(int)
    result["month"] = result["event_time"].dt.month.fillna(-1).astype(int)
    result["day"] = result["event_time"].dt.day.fillna(-1).astype(int)

    result["is_cross_currency"] = (
        result["Payment_currency"].astype(str) != result["Received_currency"].astype(str)
    ).astype(int)

    result["is_cross_border"] = (
        result["Sender_bank_location"].astype(str)
        != result["Receiver_bank_location"].astype(str)
    ).astype(int)

    for column in CATEGORICAL_COLUMNS:
        result[column] = result[column].astype("string").fillna("unknown")

    columns_to_drop = [
        RAW_TIME_COLUMN,
        RAW_DATE_COLUMN,
        RAW_TARGET_COLUMN,
        *TECHNICAL_COLUMNS,
        *LEAKAGE_COLUMNS,
    ]

    result = result.drop(columns=columns_to_drop, errors="ignore")

    # Сначала target и event_time, затем остальные признаки.
    ordered_columns = [TARGET_COLUMN, "event_time"] + [
        column for column in result.columns
        if column not in {TARGET_COLUMN, "event_time"}
    ]

    result = result[ordered_columns]

    logger.info(
        "SAML-D подготовлен: "
        f"rows={len(result)}, cols={len(result.columns)}, "
        f"positive_share={result[TARGET_COLUMN].mean():.6f}"
    )

    return result


def prepare_samld_dataset(
    path: str | Path | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    if max_rows is None:
        max_rows = settings.samld_max_rows

    raw_df = load_samld_raw(path=path, max_rows=max_rows)
    return prepare_samld_dataset_from_frame(raw_df)


def get_samld_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_columns = [
        column for column in CATEGORICAL_COLUMNS
        if column in df.columns
    ]

    excluded_columns = set(categorical_columns + [TARGET_COLUMN, "event_time"])

    numerical_columns = [
        column for column in df.columns
        if column not in excluded_columns
    ]

    return categorical_columns, numerical_columns
