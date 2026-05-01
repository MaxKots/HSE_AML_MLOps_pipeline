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
    transactions["Entry"] = transactions["Entry"].astype(str)
    transactions["Type"] = transactions["Type"].astype(str)
    transactions["Size"] = pd.to_numeric(transactions["Size"], errors="coerce")

    transactions = transactions.dropna(subset=["AlertID", "Timestamp", "Size"]).copy()
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
    tx["size_squared"] = tx["Size"] ** 2
    tx["size_abs_log1p"] = tx["size_abs"].clip(lower=0).add(1).apply(pd.np.log) if hasattr(pd, "np") else None
    if tx["size_abs_log1p"] is None:
        import numpy as np
        tx["size_abs_log1p"] = np.log1p(tx["size_abs"].clip(lower=0))

    # Комбинации направление / тип
    tx["credit_wire"] = ((tx["Entry"] == "Credit") & (tx["Type"] == "Wire")).astype(int)
    tx["credit_card"] = ((tx["Entry"] == "Credit") & (tx["Type"] == "Card")).astype(int)
    tx["debit_wire"] = ((tx["Entry"] == "Debit") & (tx["Type"] == "Wire")).astype(int)
    tx["debit_card"] = ((tx["Entry"] == "Debit") & (tx["Type"] == "Card")).astype(int)

    # Отдельные размеры по типам и направлениям
    tx["size_abs_wire"] = tx["size_abs"] * tx["is_wire"]
    tx["size_abs_card"] = tx["size_abs"] * tx["is_card"]
    tx["size_abs_credit"] = tx["size_abs"] * tx["is_credit"]
    tx["size_abs_debit"] = tx["size_abs"] * tx["is_debit"]

    # Порядок транзакций
    tx["tx_order"] = tx.groupby("AlertID").cumcount() + 1
    tx["tx_order_desc"] = tx.groupby("AlertID")["tx_order"].transform("max") - tx["tx_order"] + 1

    tx["is_first_1"] = (tx["tx_order"] == 1).astype(int)
    tx["is_first_3"] = (tx["tx_order"] <= 3).astype(int)
    tx["is_first_5"] = (tx["tx_order"] <= 5).astype(int)
    tx["is_last_1"] = (tx["tx_order_desc"] == 1).astype(int)
    tx["is_last_3"] = (tx["tx_order_desc"] <= 3).astype(int)

    # Ранние/поздние транзакции
    tx["size_abs_first_1"] = tx["size_abs"] * tx["is_first_1"]
    tx["size_abs_first_3"] = tx["size_abs"] * tx["is_first_3"]
    tx["size_abs_first_5"] = tx["size_abs"] * tx["is_first_5"]
    tx["size_abs_last_1"] = tx["size_abs"] * tx["is_last_1"]
    tx["size_abs_last_3"] = tx["size_abs"] * tx["is_last_3"]

    tx["wire_first_3"] = tx["is_wire"] * tx["is_first_3"]
    tx["wire_first_5"] = tx["is_wire"] * tx["is_first_5"]
    tx["debit_first_3"] = tx["is_debit"] * tx["is_first_3"]
    tx["debit_first_5"] = tx["is_debit"] * tx["is_first_5"]

    # Временные признаки
    tx["prev_timestamp"] = tx.groupby("AlertID")["Timestamp"].shift(1)
    tx["next_timestamp"] = tx.groupby("AlertID")["Timestamp"].shift(-1)

    tx["delta_seconds"] = (tx["Timestamp"] - tx["prev_timestamp"]).dt.total_seconds()
    tx["next_delta_seconds"] = (tx["next_timestamp"] - tx["Timestamp"]).dt.total_seconds()

    tx["delta_seconds"] = tx["delta_seconds"].fillna(0)
    tx["next_delta_seconds"] = tx["next_delta_seconds"].fillna(0)

    tx["delta_is_zero"] = (tx["delta_seconds"] == 0).astype(int)
    tx["delta_lt_10s"] = (tx["delta_seconds"] < 10).astype(int)
    tx["delta_lt_60s"] = (tx["delta_seconds"] < 60).astype(int)
    tx["delta_gt_1h"] = (tx["delta_seconds"] > 3600).astype(int)

    # Окна от первой транзакции
    first_ts = tx.groupby("AlertID")["Timestamp"].transform("min")
    tx["seconds_from_start"] = (tx["Timestamp"] - first_ts).dt.total_seconds()

    tx["within_60s"] = (tx["seconds_from_start"] <= 60).astype(int)
    tx["within_300s"] = (tx["seconds_from_start"] <= 300).astype(int)
    tx["within_3600s"] = (tx["seconds_from_start"] <= 3600).astype(int)

    # Переключения между состояниями
    tx["prev_entry"] = tx.groupby("AlertID")["Entry"].shift(1)
    tx["prev_type"] = tx.groupby("AlertID")["Type"].shift(1)
    tx["prev_sign"] = tx.groupby("AlertID")["size_positive"].shift(1)

    tx["entry_switch"] = (
        tx["prev_entry"].notna() & (tx["Entry"] != tx["prev_entry"])
    ).astype(int)
    tx["type_switch"] = (
        tx["prev_type"].notna() & (tx["Type"] != tx["prev_type"])
    ).astype(int)
    tx["sign_switch"] = (
        tx["prev_sign"].notna() & (tx["size_positive"] != tx["prev_sign"])
    ).astype(int)

    # Доли экстремальных значений внутри всего набора
    size_abs_p95 = tx["size_abs"].quantile(0.95)
    size_abs_p99 = tx["size_abs"].quantile(0.99)
    size_abs_p05 = tx["size_abs"].quantile(0.05)

    tx["is_large_abs"] = (tx["size_abs"] > size_abs_p95).astype(int)
    tx["is_very_large_abs"] = (tx["size_abs"] > size_abs_p99).astype(int)
    tx["is_small_abs"] = (tx["size_abs"] < size_abs_p05).astype(int)

    # Внутри-alert z-score по абсолютным размерам
    group_abs_mean = tx.groupby("AlertID")["size_abs"].transform("mean")
    group_abs_std = tx.groupby("AlertID")["size_abs"].transform("std").fillna(0)
    tx["size_abs_zscore"] = (tx["size_abs"] - group_abs_mean) / (group_abs_std + 1e-6)
    tx["is_alert_outlier_abs"] = (tx["size_abs_zscore"].abs() > 2.0).astype(int)

    # Серии одинаковых значений Entry / Type
    tx["entry_run_id"] = (
        tx.groupby("AlertID")["Entry"].transform(lambda s: (s != s.shift()).cumsum())
    )
    tx["type_run_id"] = (
        tx.groupby("AlertID")["Type"].transform(lambda s: (s != s.shift()).cumsum())
    )

    entry_run_lengths = (
        tx.groupby(["AlertID", "entry_run_id"]).size().rename("entry_run_len").reset_index()
    )
    type_run_lengths = (
        tx.groupby(["AlertID", "type_run_id"]).size().rename("type_run_len").reset_index()
    )

    max_entry_run = (
        entry_run_lengths.groupby("AlertID")["entry_run_len"].max().rename("max_entry_run_len")
    )
    max_type_run = (
        type_run_lengths.groupby("AlertID")["type_run_len"].max().rename("max_type_run_len")
    )
    mean_entry_run = (
        entry_run_lengths.groupby("AlertID")["entry_run_len"].mean().rename("mean_entry_run_len")
    )
    mean_type_run = (
        type_run_lengths.groupby("AlertID")["type_run_len"].mean().rename("mean_type_run_len")
    )

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
            n_sign_switches=("sign_switch", "sum"),
            n_within_60s=("within_60s", "sum"),
            n_within_300s=("within_300s", "sum"),
            n_within_3600s=("within_3600s", "sum"),
            n_delta_zero=("delta_is_zero", "sum"),
            n_delta_lt_10s=("delta_lt_10s", "sum"),
            n_delta_lt_60s=("delta_lt_60s", "sum"),
            n_delta_gt_1h=("delta_gt_1h", "sum"),
            n_large_abs=("is_large_abs", "sum"),
            n_very_large_abs=("is_very_large_abs", "sum"),
            n_small_abs=("is_small_abs", "sum"),
            n_alert_outlier_abs=("is_alert_outlier_abs", "sum"),
            size_sum=("Size", "sum"),
            size_mean=("Size", "mean"),
            size_std=("Size", "std"),
            size_min=("Size", "min"),
            size_max=("Size", "max"),
            size_squared_sum=("size_squared", "sum"),
            size_abs_sum=("size_abs", "sum"),
            size_abs_mean=("size_abs", "mean"),
            size_abs_std=("size_abs", "std"),
            size_abs_min=("size_abs", "min"),
            size_abs_max=("size_abs", "max"),
            size_abs_log1p_mean=("size_abs_log1p", "mean"),
            size_abs_wire_sum=("size_abs_wire", "sum"),
            size_abs_card_sum=("size_abs_card", "sum"),
            size_abs_credit_sum=("size_abs_credit", "sum"),
            size_abs_debit_sum=("size_abs_debit", "sum"),
            size_abs_first_1_sum=("size_abs_first_1", "sum"),
            size_abs_first_3_sum=("size_abs_first_3", "sum"),
            size_abs_first_5_sum=("size_abs_first_5", "sum"),
            size_abs_last_1_sum=("size_abs_last_1", "sum"),
            size_abs_last_3_sum=("size_abs_last_3", "sum"),
            n_wire_first_3=("wire_first_3", "sum"),
            n_wire_first_5=("wire_first_5", "sum"),
            n_debit_first_3=("debit_first_3", "sum"),
            n_debit_first_5=("debit_first_5", "sum"),
            delta_mean=("delta_seconds", "mean"),
            delta_std=("delta_seconds", "std"),
            delta_min=("delta_seconds", "min"),
            delta_max=("delta_seconds", "max"),
            next_delta_mean=("next_delta_seconds", "mean"),
            next_delta_std=("next_delta_seconds", "std"),
            first_timestamp=("Timestamp", "min"),
            last_timestamp=("Timestamp", "max"),
            first_entry_credit=("is_credit", "first"),
            first_type_wire=("is_wire", "first"),
            last_entry_credit=("is_credit", "last"),
            last_type_wire=("is_wire", "last"),
        )
        .reset_index()
    )

    # Присоединяем метрики серий
    aggregated = aggregated.merge(max_entry_run, on="AlertID", how="left")
    aggregated = aggregated.merge(max_type_run, on="AlertID", how="left")
    aggregated = aggregated.merge(mean_entry_run, on="AlertID", how="left")
    aggregated = aggregated.merge(mean_type_run, on="AlertID", how="left")

    # Длительность alert
    aggregated["time_span_seconds"] = (
        aggregated["last_timestamp"] - aggregated["first_timestamp"]
    ).dt.total_seconds()

    # Безопасные деления
    denom = aggregated["n_transactions"].replace(0, pd.NA)

    aggregated["credit_share"] = aggregated["n_credit"] / denom
    aggregated["debit_share"] = aggregated["n_debit"] / denom
    aggregated["wire_share"] = aggregated["n_wire"] / denom
    aggregated["card_share"] = aggregated["n_card"] / denom

    aggregated["credit_wire_share"] = aggregated["n_credit_wire"] / denom
    aggregated["credit_card_share"] = aggregated["n_credit_card"] / denom
    aggregated["debit_wire_share"] = aggregated["n_debit_wire"] / denom
    aggregated["debit_card_share"] = aggregated["n_debit_card"] / denom

    aggregated["positive_share"] = aggregated["n_positive"] / denom
    aggregated["negative_share"] = aggregated["n_negative"] / denom

    aggregated["entry_switch_share"] = aggregated["n_entry_switches"] / denom
    aggregated["type_switch_share"] = aggregated["n_type_switches"] / denom
    aggregated["sign_switch_share"] = aggregated["n_sign_switches"] / denom

    aggregated["delta_zero_share"] = aggregated["n_delta_zero"] / denom
    aggregated["delta_lt_10s_share"] = aggregated["n_delta_lt_10s"] / denom
    aggregated["delta_lt_60s_share"] = aggregated["n_delta_lt_60s"] / denom
    aggregated["delta_gt_1h_share"] = aggregated["n_delta_gt_1h"] / denom

    aggregated["large_abs_share"] = aggregated["n_large_abs"] / denom
    aggregated["very_large_abs_share"] = aggregated["n_very_large_abs"] / denom
    aggregated["small_abs_share"] = aggregated["n_small_abs"] / denom
    aggregated["alert_outlier_abs_share"] = aggregated["n_alert_outlier_abs"] / denom

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

    aggregated["credit_minus_debit"] = aggregated["n_credit"] - aggregated["n_debit"]
    aggregated["wire_minus_card"] = aggregated["n_wire"] - aggregated["n_card"]
    aggregated["positive_minus_negative"] = aggregated["n_positive"] - aggregated["n_negative"]
    aggregated["credit_debit_abs_diff"] = (aggregated["n_credit"] - aggregated["n_debit"]).abs()
    aggregated["wire_card_abs_diff"] = (aggregated["n_wire"] - aggregated["n_card"]).abs()

    # Ранние/поздние структуры
    aggregated["wire_first_3_share"] = aggregated["n_wire_first_3"] / 3.0
    aggregated["wire_first_5_share"] = aggregated["n_wire_first_5"] / 5.0
    aggregated["debit_first_3_share"] = aggregated["n_debit_first_3"] / 3.0
    aggregated["debit_first_5_share"] = aggregated["n_debit_first_5"] / 5.0

    aggregated["first_last_entry_same"] = (
        aggregated["first_entry_credit"] == aggregated["last_entry_credit"]
    ).astype(int)
    aggregated["first_last_type_same"] = (
        aggregated["first_type_wire"] == aggregated["last_type_wire"]
    ).astype(int)

    # Взаимодействия
    aggregated["tx_per_minute_x_abs_sum"] = aggregated["tx_per_minute"] * aggregated["size_abs_sum"]
    aggregated["wire_share_x_abs_sum"] = aggregated["wire_share"] * aggregated["size_abs_sum"]
    aggregated["debit_share_x_abs_sum"] = aggregated["debit_share"] * aggregated["size_abs_sum"]
    aggregated["entry_switch_share_x_tx_count"] = aggregated["entry_switch_share"] * aggregated["n_transactions"]
    aggregated["type_switch_share_x_tx_count"] = aggregated["type_switch_share"] * aggregated["n_transactions"]
    aggregated["delta_lt_10s_share_x_tx_count"] = aggregated["delta_lt_10s_share"] * aggregated["n_transactions"]
    aggregated["large_abs_share_x_abs_sum"] = aggregated["large_abs_share"] * aggregated["size_abs_sum"]

    # Энтропии по типам и направлениям
    import numpy as np

    def binary_entropy(p: pd.Series) -> pd.Series:
        p = p.clip(1e-6, 1 - 1e-6)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    aggregated["entry_entropy"] = binary_entropy(aggregated["credit_share"].fillna(0.5))
    aggregated["type_entropy"] = binary_entropy(aggregated["wire_share"].fillna(0.5))
    aggregated["sign_entropy"] = binary_entropy(aggregated["positive_share"].fillna(0.5))

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

    aggregated["risk_instant_dense_flow"] = (
        (aggregated["delta_lt_10s_share"] > 0.6) & (aggregated["n_transactions"] >= 8)
    ).astype(int)

    aggregated["risk_very_large_wire_dominance"] = (
        (aggregated["wire_share"] > 0.8) & (aggregated["very_large_abs_share"] > 0.1)
    ).astype(int)

    aggregated["risk_mostly_debit_and_negative"] = (
        (aggregated["debit_share"] > 0.8) & (aggregated["negative_share"] > 0.8)
    ).astype(int)

    aggregated["risk_many_outliers"] = (
        aggregated["alert_outlier_abs_share"] > 0.2
    ).astype(int)

    # Объединяем с таблицей алертов
    result = alerts.merge(aggregated, on="AlertID", how="left")

    # Календарные признаки алерта
    result["alert_month"] = result["Date"].dt.month
    result["alert_day"] = result["Date"].dt.day
    result["alert_day_of_week"] = result["Date"].dt.dayofweek
    result["alert_is_weekend"] = (result["alert_day_of_week"] >= 5).astype(int)
    result["alert_day_sin"] = np.sin(2 * np.pi * result["alert_day_of_week"] / 7)
    result["alert_day_cos"] = np.cos(2 * np.pi * result["alert_day_of_week"] / 7)
    result["alert_month_sin"] = np.sin(2 * np.pi * result["alert_month"] / 12)
    result["alert_month_cos"] = np.cos(2 * np.pi * result["alert_month"] / 12)

    # Пропусковые флаги
    missing_cols = [
        "size_std",
        "size_abs_std",
        "delta_std",
        "next_delta_std",
        "max_entry_run_len",
        "max_type_run_len",
        "mean_entry_run_len",
        "mean_type_run_len",
    ]
    for col in missing_cols:
        if col in result.columns:
            result[f"{col}_is_missing"] = result[col].isna().astype(int)

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

    # Удаляем возможные дубликаты и проверяем порядок колонок
    result = result.drop_duplicates(subset=["AlertID"]).reset_index(drop=True)

    logger.info(
        f"SynthAML подготовлен: rows={len(result)}, cols={len(result.columns)}"
    )

    return result
