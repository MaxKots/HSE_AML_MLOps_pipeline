from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_synthaml_dataset_from_frames(
    alerts_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    alerts = alerts_df.copy()
    transactions = transactions_df.copy()

    logger.info(
        f"Подготовка SynthAML из датафреймов: "
        f"alerts_rows={len(alerts)}, transactions_rows={len(transactions)}"
    )

    required_alert_cols = {"AlertID", "Date", "Outcome"}
    required_tx_cols = {"AlertID", "Timestamp", "Entry", "Type", "Size"}

    missing_alert_cols = required_alert_cols - set(alerts.columns)
    missing_tx_cols = required_tx_cols - set(transactions.columns)

    if missing_alert_cols:
        raise ValueError(
            f"В таблице alerts отсутствуют обязательные колонки: {sorted(missing_alert_cols)}"
        )

    if missing_tx_cols:
        raise ValueError(
            f"В таблице transactions отсутствуют обязательные колонки: {sorted(missing_tx_cols)}"
        )

    alerts["Date"] = pd.to_datetime(alerts["Date"], errors="coerce")
    transactions["Timestamp"] = pd.to_datetime(transactions["Timestamp"], errors="coerce")
    transactions = transactions.sort_values(["AlertID", "Timestamp"]).reset_index(drop=True)

    outcome_map = {
        "Report": 1,
        "Dismiss": 0,
    }
    alerts["fraud_bool"] = alerts["Outcome"].map(outcome_map)

    if alerts["fraud_bool"].isna().any():
        unknown_outcomes = (
            alerts.loc[alerts["fraud_bool"].isna(), "Outcome"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Обнаружены неизвестные значения Outcome в SynthAML: {unknown_outcomes}"
        )

    tx = transactions.copy()

    # Базовые индикаторы
    tx["is_credit"] = (tx["Entry"] == "Credit").astype(int)
    tx["is_debit"] = (tx["Entry"] == "Debit").astype(int)
    tx["is_wire"] = (tx["Type"] == "Wire").astype(int)
    tx["is_card"] = (tx["Type"] == "Card").astype(int)

    # Размеры операций
    tx["size_abs"] = tx["Size"].abs()
    tx["size_positive"] = (tx["Size"] > 0).astype(int)
    tx["size_negative"] = (tx["Size"] < 0).astype(int)

    # Комбинации направление / тип
    tx["credit_wire"] = ((tx["Entry"] == "Credit") & (tx["Type"] == "Wire")).astype(int)
    tx["credit_card"] = ((tx["Entry"] == "Credit") & (tx["Type"] == "Card")).astype(int)
    tx["debit_wire"] = ((tx["Entry"] == "Debit") & (tx["Type"] == "Wire")).astype(int)
    tx["debit_card"] = ((tx["Entry"] == "Debit") & (tx["Type"] == "Card")).astype(int)

    # Интервалы между транзакциями
    tx["prev_timestamp"] = tx.groupby("AlertID")["Timestamp"].shift(1)
    tx["delta_seconds"] = (tx["Timestamp"] - tx["prev_timestamp"]).dt.total_seconds()
    tx["delta_seconds"] = tx["delta_seconds"].fillna(0)

    # Переключения между состояниями
    tx["prev_entry"] = tx.groupby("AlertID")["Entry"].shift(1)
    tx["prev_type"] = tx.groupby("AlertID")["Type"].shift(1)

    tx["entry_switch"] = (
        tx["prev_entry"].notna() & (tx["Entry"] != tx["prev_entry"])
    ).astype(int)
    tx["type_switch"] = (
        tx["prev_type"].notna() & (tx["Type"] != tx["prev_type"])
    ).astype(int)

    # Позиция транзакции внутри alert
    tx["tx_order"] = tx.groupby("AlertID").cumcount() + 1

    # Ранние транзакции
    tx["is_first_3"] = (tx["tx_order"] <= 3).astype(int)
    tx["is_first_5"] = (tx["tx_order"] <= 5).astype(int)

    # Окна по времени от первой транзакции
    first_ts = tx.groupby("AlertID")["Timestamp"].transform("min")
    tx["seconds_from_start"] = (tx["Timestamp"] - first_ts).dt.total_seconds()

    tx["within_60s"] = (tx["seconds_from_start"] <= 60).astype(int)
    tx["within_300s"] = (tx["seconds_from_start"] <= 300).astype(int)
    tx["within_3600s"] = (tx["seconds_from_start"] <= 3600).astype(int)

    # Отдельные размеры по комбинациям
    tx["size_abs_wire"] = tx["size_abs"] * tx["is_wire"]
    tx["size_abs_card"] = tx["size_abs"] * tx["is_card"]
    tx["size_abs_credit"] = tx["size_abs"] * tx["is_credit"]
    tx["size_abs_debit"] = tx["size_abs"] * tx["is_debit"]

    # Базовые агрегаты
    aggregated = (
        tx.groupby("AlertID")
        .agg(
            n_transactions=("AlertID", "size"),
            n_credit=("is_credit", "sum"),
            n_debit=("is_debit", "sum"),
            n_wire=("is_wire", "sum"),
            n_card=("is_card", "sum"),
            n_credit_wire=("credit_wire", "sum"),
            n_credit_card=("credit_card", "sum"),
            n_debit_wire=("debit_wire", "sum"),
            n_debit_card=("debit_card", "sum"),
            n_positive=("size_positive", "sum"),
            n_negative=("size_negative", "sum"),
            n_entry_switches=("entry_switch", "sum"),
            n_type_switches=("type_switch", "sum"),
            n_within_60s=("within_60s", "sum"),
            n_within_300s=("within_300s", "sum"),
            n_within_3600s=("within_3600s", "sum"),
            size_sum=("Size", "sum"),
            size_mean=("Size", "mean"),
            size_std=("Size", "std"),
            size_min=("Size", "min"),
            size_max=("Size", "max"),
            size_abs_sum=("size_abs", "sum"),
            size_abs_mean=("size_abs", "mean"),
            size_abs_std=("size_abs", "std"),
            size_abs_min=("size_abs", "min"),
            size_abs_max=("size_abs", "max"),
            size_abs_wire_sum=("size_abs_wire", "sum"),
            size_abs_card_sum=("size_abs_card", "sum"),
            size_abs_credit_sum=("size_abs_credit", "sum"),
            size_abs_debit_sum=("size_abs_debit", "sum"),
            delta_mean=("delta_seconds", "mean"),
            delta_std=("delta_seconds", "std"),
            delta_min=("delta_seconds", "min"),
            delta_max=("delta_seconds", "max"),
            first_timestamp=("Timestamp", "min"),
            last_timestamp=("Timestamp", "max"),
        )
        .reset_index()
    )

    # Длительность alert
    aggregated["time_span_seconds"] = (
        aggregated["last_timestamp"] - aggregated["first_timestamp"]
    ).dt.total_seconds()

    # Нормированные доли
    aggregated["credit_share"] = aggregated["n_credit"] / aggregated["n_transactions"]
    aggregated["debit_share"] = aggregated["n_debit"] / aggregated["n_transactions"]
    aggregated["wire_share"] = aggregated["n_wire"] / aggregated["n_transactions"]
    aggregated["card_share"] = aggregated["n_card"] / aggregated["n_transactions"]

    aggregated["credit_wire_share"] = aggregated["n_credit_wire"] / aggregated["n_transactions"]
    aggregated["credit_card_share"] = aggregated["n_credit_card"] / aggregated["n_transactions"]
    aggregated["debit_wire_share"] = aggregated["n_debit_wire"] / aggregated["n_transactions"]
    aggregated["debit_card_share"] = aggregated["n_debit_card"] / aggregated["n_transactions"]

    aggregated["positive_share"] = aggregated["n_positive"] / aggregated["n_transactions"]
    aggregated["negative_share"] = aggregated["n_negative"] / aggregated["n_transactions"]

    aggregated["entry_switch_share"] = aggregated["n_entry_switches"] / aggregated["n_transactions"]
    aggregated["type_switch_share"] = aggregated["n_type_switches"] / aggregated["n_transactions"]

    aggregated["tx_per_minute"] = aggregated["n_transactions"] / (
        aggregated["time_span_seconds"] / 60 + 1e-6
    )
    aggregated["tx_per_hour"] = aggregated["n_transactions"] / (
        aggregated["time_span_seconds"] / 3600 + 1e-6
    )

    # Балансы и асимметрия
    aggregated["credit_to_debit_ratio"] = aggregated["n_credit"] / (aggregated["n_debit"] + 1e-6)
    aggregated["wire_to_card_ratio"] = aggregated["n_wire"] / (aggregated["n_card"] + 1e-6)
    aggregated["positive_to_negative_ratio"] = aggregated["n_positive"] / (aggregated["n_negative"] + 1e-6)
    aggregated["signed_to_abs_ratio"] = aggregated["size_sum"] / (aggregated["size_abs_sum"] + 1e-6)
    aggregated["wire_abs_share"] = aggregated["size_abs_wire_sum"] / (aggregated["size_abs_sum"] + 1e-6)
    aggregated["card_abs_share"] = aggregated["size_abs_card_sum"] / (aggregated["size_abs_sum"] + 1e-6)
    aggregated["credit_abs_share"] = aggregated["size_abs_credit_sum"] / (aggregated["size_abs_sum"] + 1e-6)
    aggregated["debit_abs_share"] = aggregated["size_abs_debit_sum"] / (aggregated["size_abs_sum"] + 1e-6)

    # Простые бинарные rule-like признаки
    aggregated["risk_many_wire_short_span"] = (
        (aggregated["wire_share"] > 0.7) & (aggregated["time_span_seconds"] < 300)
    ).astype(int)

    aggregated["risk_many_debit_short_span"] = (
        (aggregated["debit_share"] > 0.7) & (aggregated["time_span_seconds"] < 300)
    ).astype(int)

    aggregated["risk_high_burst"] = (
        (aggregated["n_transactions"] >= 10) & (aggregated["time_span_seconds"] < 600)
    ).astype(int)

    aggregated["risk_high_switching"] = (
        aggregated["entry_switch_share"] > 0.5
    ).astype(int)

    aggregated["risk_extreme_size_variability"] = (
        aggregated["size_abs_std"] > aggregated["size_abs_mean"]
    ).astype(int)

    # Объединяем с таблицей алертов
    result = alerts.merge(aggregated, on="AlertID", how="left")

    # Календарные признаки алерта
    result["alert_month"] = result["Date"].dt.month
    result["alert_day"] = result["Date"].dt.day
    result["alert_day_of_week"] = result["Date"].dt.dayofweek
    result["alert_is_weekend"] = (result["alert_day_of_week"] >= 5).astype(int)

    # Удаляем сырые datetime и служебные поля
    result = result.drop(
        columns=[
            "Outcome",
            "Date",
            "first_timestamp",
            "last_timestamp",
        ],
        errors="ignore",
    )

    # Заполняем пропуски
    numeric_columns = result.select_dtypes(include=["number"]).columns.tolist()
    for column in numeric_columns:
        if column != "fraud_bool":
            result[column] = result[column].fillna(0)

    logger.info(
        f"SynthAML подготовлен: rows={len(result)}, cols={len(result.columns)}"
    )

    return result
