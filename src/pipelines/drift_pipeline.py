from __future__ import annotations

from typing import Any

from src.data import DataLoader, DataTransformer, DataValidator
from src.features import FeatureEngineer
from src.monitoring.drift import AMLDriftDetector
from src.utils.logger import get_logger
from src.utils.paths import ensure_directories

logger = get_logger(__name__)


def run_drift_pipeline(
    reference_dataset_name: str = "base",
    current_dataset_name: str = "variant_1",
) -> dict[str, Any]:
    ensure_directories()

    logger.info(
        f"Запуск drift pipeline: reference='{reference_dataset_name}', current='{current_dataset_name}'"
    )

    loader = DataLoader()
    validator = DataValidator()
    transformer = DataTransformer()
    feature_engineer = FeatureEngineer()
    drift_detector = AMLDriftDetector()

    reference_df = loader.load_dataset(reference_dataset_name)
    current_df = loader.load_dataset(current_dataset_name)

    validator.run_full_validation(reference_df)
    validator.run_full_validation(current_df)

    reference_transformed = transformer.transform(reference_df)
    current_transformed = transformer.transform(current_df)

    reference_features = feature_engineer.build_features(reference_transformed)
    current_features = feature_engineer.build_features(current_transformed)

    result = drift_detector.detect_drift(
        reference_df=reference_features.dataframe,
        current_df=current_features.dataframe,
        report_name=f"drift_{reference_dataset_name}_vs_{current_dataset_name}",
    )

    summary = {
        "reference_dataset_name": reference_dataset_name,
        "current_dataset_name": current_dataset_name,
        "drift_detected": result.drift_detected,
        "share_of_drifted_columns": result.share_of_drifted_columns,
        "number_of_drifted_columns": result.number_of_drifted_columns,
        "total_columns_checked": result.total_columns_checked,
        "drifted_columns": result.drifted_columns,
        "report_path_html": result.report_path_html,
        "report_path_yaml": result.report_path_yaml,
    }

    logger.info(
        "Drift pipeline завершён: "
        f"drift_detected={summary['drift_detected']}, "
        f"share_of_drifted_columns={summary['share_of_drifted_columns']:.4f}"
    )

    return summary
