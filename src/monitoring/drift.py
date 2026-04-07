from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.utils.io import save_yaml
from src.utils.logger import get_logger
from src.utils.paths import get_artifacts_dir

logger = get_logger(__name__)


@dataclass
class DriftCheckResult:
    drift_detected: bool
    share_of_drifted_columns: float
    number_of_drifted_columns: int
    total_columns_checked: int
    drifted_columns: list[str]
    report_path_html: str
    report_path_yaml: str


class AMLDriftDetector:
    def __init__(
        self,
        numerical_drift_threshold: float = 0.1,
        share_of_drifted_columns_threshold: float = 0.3,
        stattest: str = "psi",
    ) -> None:
        self.numerical_drift_threshold = numerical_drift_threshold
        self.share_of_drifted_columns_threshold = share_of_drifted_columns_threshold
        self.stattest = stattest

    def _prepare_columns(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        reference_df = reference_df.copy()
        current_df = current_df.copy()

        columns_to_drop = ["fraud_bool", "event_time"]
        reference_df = reference_df.drop(columns=columns_to_drop, errors="ignore")
        current_df = current_df.drop(columns=columns_to_drop, errors="ignore")

        common_columns = sorted(list(set(reference_df.columns).intersection(set(current_df.columns))))
        if not common_columns:
            raise ValueError("После подготовки данных не осталось общих колонок для проверки дрейфа")

        reference_prepared = reference_df[common_columns].copy()
        current_prepared = current_df[common_columns].copy()

        logger.info(
            "Подготовка данных для drift-check завершена: "
            f"n_columns={len(common_columns)}, "
            f"reference_shape={reference_prepared.shape}, "
            f"current_shape={current_prepared.shape}"
        )

        return reference_prepared, current_prepared

    def detect_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        report_name: str = "drift_report",
    ) -> DriftCheckResult:
        reference_prepared, current_prepared = self._prepare_columns(reference_df, current_df)

        logger.info("Запуск Evidently report для детекции дрейфа")

        report = Report(
            metrics=[
                DataDriftPreset(
                    stattest=self.stattest,
                    drift_share=self.share_of_drifted_columns_threshold,
                )
            ]
        )

        report.run(
            reference_data=reference_prepared,
            current_data=current_prepared,
        )

        report_dict = report.as_dict()

        metrics = report_dict["metrics"][0]["result"]
        drift_detected = bool(metrics["dataset_drift"])
        share_of_drifted_columns = float(metrics["share_of_drifted_columns"])
        number_of_drifted_columns = int(metrics["number_of_drifted_columns"])
        total_columns_checked = int(metrics["number_of_columns"])

        drifted_columns = []
        by_columns = metrics.get("drift_by_columns", {})
        for column_name, column_info in by_columns.items():
            if column_info.get("drift_detected", False):
                drifted_columns.append(column_name)

        report_dir = get_artifacts_dir() / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        report_path_html = report_dir / f"{report_name}.html"
        report_path_yaml = report_dir / f"{report_name}.yaml"

        report.save_html(str(report_path_html))
        save_yaml(report_dict, report_path_yaml)

        logger.info(
            "Drift-check завершён: "
            f"drift_detected={drift_detected}, "
            f"share_of_drifted_columns={share_of_drifted_columns:.4f}, "
            f"drifted_columns={len(drifted_columns)}"
        )

        return DriftCheckResult(
            drift_detected=drift_detected,
            share_of_drifted_columns=share_of_drifted_columns,
            number_of_drifted_columns=number_of_drifted_columns,
            total_columns_checked=total_columns_checked,
            drifted_columns=sorted(drifted_columns),
            report_path_html=str(report_path_html),
            report_path_yaml=str(report_path_yaml),
        )
