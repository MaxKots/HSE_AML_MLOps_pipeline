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

    transactions["is_credit"] = (transactions["Entry"] == "Credit").astype(int)
    transactions["is_debit"] = (transactions["Entry"] == "Debit").astype(int)
    transactions["is_wire"] = (transactions["Type"] == "Wire").astype(int)
    transactions["is_card"] = (transactions["Type"] == "Card").astype(int)

    transactions["size_abs"] = transactions["Size"].abs()
    transactions["size_positive"] = (transactions["Size"] > 0).astype(int)
    transactions["size_negative"] = (transactions["Size"] < 0).astype(int)

    aggregated = (
        transactions.groupby("AlertID")
        .agg(
            n_transactions=("AlertID", "size"),
            n_credit=("is_credit", "sum"),
            n_debit=("is_debit", "sum"),
            n_wire=("is_wire", "sum"),
            n_card=("is_card", "sum"),
            size_sum=("Size", "sum"),
            size_mean=("Size", "mean"),
            size_std=("Size", "std"),
            size_min=("Size", "min"),
            size_max=("Size", "max"),
            size_abs_sum=("size_abs", "sum"),
            size_abs_mean=("size_abs", "mean"),
            n_positive=("size_positive", "sum"),
            n_negative=("size_negative", "sum"),
            first_timestamp=("Timestamp", "min"),
            last_timestamp=("Timestamp", "max"),
        )
        .reset_index()
    )

    aggregated["time_span_seconds"] = (
        aggregated["last_timestamp"] - aggregated["first_timestamp"]
    ).dt.total_seconds()

    aggregated["credit_share"] = aggregated["n_credit"] / aggregated["n_transactions"]
    aggregated["debit_share"] = aggregated["n_debit"] / aggregated["n_transactions"]
    aggregated["wire_share"] = aggregated["n_wire"] / aggregated["n_transactions"]
    aggregated["card_share"] = aggregated["n_card"] / aggregated["n_transactions"]

    result = alerts.merge(aggregated, on="AlertID", how="left")

    result["alert_month"] = result["Date"].dt.month
    result["alert_day"] = result["Date"].dt.day
    result["alert_day_of_week"] = result["Date"].dt.dayofweek
    result["alert_is_weekend"] = (result["alert_day_of_week"] >= 5).astype(int)

    result = result.drop(
        columns=[
            "Outcome",
            "Date",
            "first_timestamp",
            "last_timestamp",
        ],
        errors="ignore",
    )

    numeric_columns = result.select_dtypes(include=["number"]).columns.tolist()
    for column in numeric_columns:
        if column != "fraud_bool":
            result[column] = result[column].fillna(0)

    logger.info(
        f"SynthAML подготовлен: rows={len(result)}, cols={len(result.columns)}"
    )

    return result
