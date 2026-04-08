from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config.settings import settings
from src.utils.io import read_yaml, save_yaml
from src.utils.logger import get_logger
from src.utils.paths import get_artifacts_dir

logger = get_logger(__name__)


@dataclass
class FeatureBuildResult:
    dataframe: pd.DataFrame
    feature_columns: list[str]
    categorical_columns: list[str]
    numerical_columns: list[str]


class FeatureEngineer:
    def __init__(
        self,
        target_column: str | None = None,
        feature_config_path: str | Path | None = None,
    ) -> None:
        self.target_column = target_column or settings.target_column

        if feature_config_path is None:
            feature_config_path = settings.project_root / "config" / "feature_config.yaml"

        self.feature_config = read_yaml(Path(feature_config_path))

        self.base_categorical_columns = self.feature_config["features"]["categorical"]
        self.base_binary_columns = self.feature_config["features"]["binary"]
        self.base_numerical_columns = self.feature_config["features"]["numerical"]

    def build_features(self, df: pd.DataFrame) -> FeatureBuildResult:
        logger.info("Запуск feature engineering")

        features_df = df.copy()

        features_df = self._add_event_order(features_df)
        features_df = self._add_synthetic_timestamp(features_df)
        features_df = self._add_calendar_features(features_df)
        features_df = self._add_missing_flags(features_df)
        features_df = self._add_ratio_features(features_df)
        features_df = self._add_interaction_features(features_df)
        features_df = self._add_velocity_features(features_df)
        features_df = self._add_risk_rule_like_features(features_df)

        feature_columns = [col for col in features_df.columns if col != self.target_column]
        categorical_columns = self._get_categorical_columns(features_df)
        numerical_columns = self._get_numerical_columns(features_df, categorical_columns)

        result = FeatureBuildResult(
            dataframe=features_df,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
        )

        self._save_feature_spec(result)

        logger.info(
            "Feature engineering завершён: "
            f"rows={len(features_df)}, cols={len(features_df.columns)}, "
            f"categorical={len(categorical_columns)}, numerical={len(numerical_columns)}"
        )
        return result

    def _add_event_order(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        sort_columns = [col for col in ["month", "days_since_request", "customer_age", "income"] if col in df.columns]
        df = df.sort_values(sort_columns).reset_index(drop=True)

        df["event_id"] = np.arange(len(df))
        logger.info("Добавлен порядковый идентификатор события 'event_id'")

        return df

    def _add_synthetic_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        base_datetime = datetime(2024, 1, 1, 0, 0, 0)

        month_offsets = df["month"].fillna(1).astype(int).clip(lower=1)
        day_offsets = df["days_since_request"].fillna(0).astype(int).clip(lower=0)
        minute_offsets = df["event_id"].astype(int) * 5

        synthetic_timestamps = []
        for month_offset, day_offset, minute_offset in zip(month_offsets, day_offsets, minute_offsets):
            current_dt = (
                base_datetime
                + pd.DateOffset(month=int(month_offset) - 1)
                + timedelta(days=int(day_offset))
                + timedelta(minutes=int(minute_offset))
            )
            synthetic_timestamps.append(current_dt)

        df["event_time"] = pd.to_datetime(synthetic_timestamps)

        logger.info("Построена синтетическая временная ось 'event_time'")
        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["event_hour"] = df["event_time"].dt.hour
        df["event_day"] = df["event_time"].dt.day
        df["event_day_of_week"] = df["event_time"].dt.dayofweek
        df["event_is_weekend"] = (df["event_day_of_week"] >= 5).astype(int)

        df["event_hour_sin"] = np.sin(2 * np.pi * df["event_hour"] / 24)
        df["event_hour_cos"] = np.cos(2 * np.pi * df["event_hour"] / 24)
        df["event_day_of_week_sin"] = np.sin(2 * np.pi * df["event_day_of_week"] / 7)
        df["event_day_of_week_cos"] = np.cos(2 * np.pi * df["event_day_of_week"] / 7)

        logger.info("Добавлены календарные признаки")
        return df

    def _add_missing_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        columns_to_flag = [
            "prev_address_months_count",
            "current_address_months_count",
            "intended_balcon_amount",
            "bank_months_count",
            "session_length_in_minutes",
        ]

        for column in columns_to_flag:
            if column in df.columns:
                df[f"{column}_is_missing"] = df[column].isna().astype(int)

        logger.info("Добавлены признаки пропусков")
        return df

    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denominator = denominator.replace(0, np.nan)
        result = numerator / denominator
        return result.replace([np.inf, -np.inf], np.nan)

    def _add_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if {"proposed_credit_limit", "income"}.issubset(df.columns):
            df["credit_limit_to_income_ratio"] = self._safe_divide(
                df["proposed_credit_limit"],
                df["income"] + 1e-6,
            )

        if {"velocity_24h", "velocity_6h"}.issubset(df.columns):
            df["velocity_24h_to_6h_ratio"] = self._safe_divide(
                df["velocity_24h"],
                df["velocity_6h"] + 1e-6,
            )

        if {"velocity_4w", "velocity_24h"}.issubset(df.columns):
            df["velocity_4w_to_24h_ratio"] = self._safe_divide(
                df["velocity_4w"],
                df["velocity_24h"] + 1e-6,
            )

        if {"bank_months_count", "customer_age"}.issubset(df.columns):
            df["bank_tenure_to_age_ratio"] = self._safe_divide(
                df["bank_months_count"],
                df["customer_age"] + 1e-6,
            )

        if {"name_email_similarity", "email_is_free"}.issubset(df.columns):
            df["email_name_similarity_weighted"] = df["name_email_similarity"] * (1 + df["email_is_free"].fillna(0))

        logger.info("Добавлены ratio-признаки")
        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if {"income", "credit_risk_score"}.issubset(df.columns):
            df["income_x_credit_risk"] = df["income"] * df["credit_risk_score"]

        if {"proposed_credit_limit", "credit_risk_score"}.issubset(df.columns):
            df["credit_limit_x_risk"] = df["proposed_credit_limit"] * df["credit_risk_score"]

        if {"session_length_in_minutes", "keep_alive_session"}.issubset(df.columns):
            df["session_length_x_keep_alive"] = df["session_length_in_minutes"].fillna(0) * df["keep_alive_session"].fillna(0)

        if {"device_fraud_count", "device_distinct_emails_8w"}.issubset(df.columns):
            df["device_fraud_to_distinct_email_ratio"] = self._safe_divide(
                df["device_fraud_count"],
                df["device_distinct_emails_8w"] + 1e-6,
            )

        if {"zip_count_4w", "bank_branch_count_8w"}.issubset(df.columns):
            df["zip_to_branch_activity_ratio"] = self._safe_divide(
                df["zip_count_4w"],
                df["bank_branch_count_8w"] + 1e-6,
            )

        logger.info("Добавлены interaction-признаки")
        return df

    def _add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if {"velocity_6h", "velocity_24h", "velocity_4w"}.issubset(df.columns):
            df["velocity_mean"] = df[["velocity_6h", "velocity_24h", "velocity_4w"]].mean(axis=1)
            df["velocity_std"] = df[["velocity_6h", "velocity_24h", "velocity_4w"]].std(axis=1)
            df["velocity_max"] = df[["velocity_6h", "velocity_24h", "velocity_4w"]].max(axis=1)

        if {"velocity_6h", "velocity_24h"}.issubset(df.columns):
            df["velocity_delta_24h_6h"] = df["velocity_24h"] - df["velocity_6h"]

        if {"velocity_4w", "velocity_24h"}.issubset(df.columns):
            df["velocity_delta_4w_24h"] = df["velocity_4w"] - df["velocity_24h"]

        logger.info("Добавлены агрегаты по velocity")
        return df

    def _add_risk_rule_like_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if {"foreign_request", "device_fraud_count"}.issubset(df.columns):
            df["risk_foreign_and_device_fraud"] = (
                (df["foreign_request"].fillna(0) == 1) & (df["device_fraud_count"].fillna(0) > 0)
            ).astype(int)

        if {"phone_home_valid", "phone_mobile_valid"}.issubset(df.columns):
            df["risk_both_phones_invalid"] = (
                (df["phone_home_valid"].fillna(0) == 0) & (df["phone_mobile_valid"].fillna(0) == 0)
            ).astype(int)

        if {"name_email_similarity", "email_is_free"}.issubset(df.columns):
            df["risk_low_similarity_free_email"] = (
                (df["name_email_similarity"].fillna(0) < 0.35) & (df["email_is_free"].fillna(0) == 1)
            ).astype(int)

        if {"bank_months_count", "current_address_months_count"}.issubset(df.columns):
            df["risk_new_bank_and_new_address"] = (
                (df["bank_months_count"].fillna(999) < 6) & (df["current_address_months_count"].fillna(999) < 6)
            ).astype(int)

        logger.info("Добавлены rule-like risk признаки")
        return df

    def _get_categorical_columns(self, df: pd.DataFrame) -> list[str]:
        categorical_columns = list(self.base_categorical_columns)

        engineered_categorical = [
            "event_is_weekend",
            "risk_foreign_and_device_fraud",
            "risk_both_phones_invalid",
            "risk_low_similarity_free_email",
            "risk_new_bank_and_new_address",
        ]

        for column in engineered_categorical:
            if column in df.columns:
                categorical_columns.append(column)

        categorical_columns = [col for col in categorical_columns if col in df.columns]
        categorical_columns = sorted(list(set(categorical_columns)))

        return categorical_columns

    def _get_numerical_columns(self, df: pd.DataFrame, categorical_columns: list[str]) -> list[str]:
        excluded_columns = set(categorical_columns + [self.target_column, "event_time"])
        numerical_columns = [col for col in df.columns if col not in excluded_columns]

        return numerical_columns

    def _save_feature_spec(self, result: FeatureBuildResult) -> None:
        feature_spec = {
            "target_column": self.target_column,
            "feature_columns": result.feature_columns,
            "categorical_columns": result.categorical_columns,
            "numerical_columns": result.numerical_columns,
            "generated_at": datetime.utcnow().isoformat(),
        }

        output_path = get_artifacts_dir() / "metrics" / "feature_spec.yaml"
        save_yaml(feature_spec, output_path)

        logger.info(f"Спецификация признаков сохранена в {output_path}")
