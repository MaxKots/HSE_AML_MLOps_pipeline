from dataclasses import asdict

import pandas as pd

from config.settings import settings
from src.data.schemas import DataValidationReport
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    def __init__(self, target_column: str | None = None) -> None:
        self.target_column = target_column or settings.target_column

        self.expected_columns = [
            "fraud_bool",
            "income",
            "name_email_similarity",
            "prev_address_months_count",
            "current_address_months_count",
            "customer_age",
            "days_since_request",
            "intended_balcon_amount",
            "payment_type",
            "zip_count_4w",
            "velocity_6h",
            "velocity_24h",
            "velocity_4w",
            "bank_branch_count_8w",
            "date_of_birth_distinct_emails_4w",
            "employment_status",
            "credit_risk_score",
            "email_is_free",
            "housing_status",
            "phone_home_valid",
            "phone_mobile_valid",
            "bank_months_count",
            "has_other_cards",
            "proposed_credit_limit",
            "foreign_request",
            "source",
            "session_length_in_minutes",
            "device_os",
            "keep_alive_session",
            "device_distinct_emails_8w",
            "device_fraud_count",
            "month",
        ]

    def validate_schema(self, df: pd.DataFrame) -> DataValidationReport:
        missing_columns = [col for col in self.expected_columns if col not in df.columns]
        unexpected_columns = [col for col in df.columns if col not in self.expected_columns]

        is_valid = len(missing_columns) == 0 and self.target_column in df.columns

        report = DataValidationReport(
            is_valid=is_valid,
            row_count=len(df),
            column_count=len(df.columns),
            missing_columns=missing_columns,
            unexpected_columns=unexpected_columns,
            columns_with_nulls=df.columns[df.isnull().any()].tolist(),
            duplicate_rows=int(df.duplicated().sum()),
            message="Схема валидна" if is_valid else "Схема невалидна",
        )

        logger.info(f"Проверка схемы завершена: {asdict(report)}")
        return report

    def validate_target(self, df: pd.DataFrame) -> None:
        if self.target_column not in df.columns:
            raise ValueError(f"Целевая колонка '{self.target_column}' отсутствует в датасете")

        unique_values = sorted(df[self.target_column].dropna().unique().tolist())
        if not set(unique_values).issubset({0, 1}):
            raise ValueError(
                f"Целевая колонка '{self.target_column}' должна быть бинарной. Получены значения: {unique_values}"
            )

        logger.info(
            f"Целевая колонка '{self.target_column}' проверена: уникальные значения = {unique_values}"
        )

    def validate_empty_dataframe(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Датасет пустой")

        logger.info("Проверено, что датасет не пустой")

    def run_full_validation(self, df: pd.DataFrame) -> DataValidationReport:
        self.validate_empty_dataframe(df)
        report = self.validate_schema(df)
        self.validate_target(df)

        if not report.is_valid:
            raise ValueError(
                f"Датасет не прошёл валидацию схемы. "
                f"Отсутствуют колонки: {report.missing_columns}. "
                f"Неожиданные колонки: {report.unexpected_columns}"
            )

        return report
