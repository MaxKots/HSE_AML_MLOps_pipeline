from __future__ import annotations

from typing import Any

from config.settings import settings
from src.data import DataLoader, DataTransformer, DataValidator
from src.data.samld import get_samld_feature_columns, prepare_samld_dataset_from_frame
from src.features import FeatureEngineer
from src.models import AMLModelTrainer
from src.utils.logger import get_logger
from src.utils.paths import ensure_directories, get_processed_data_dir

logger = get_logger(__name__)


def _prepare_standard_dataset(dataset_name: str, source: str | None) -> tuple:
    loader = DataLoader()
    validator = DataValidator()
    transformer = DataTransformer()
    feature_engineer = FeatureEngineer()

    df = loader.load_dataset(dataset_name, source_override=source)
    validator.run_full_validation(df)

    transformed_df = transformer.transform(df)
    feature_result = feature_engineer.build_features(transformed_df)

    return feature_result.dataframe, feature_result.categorical_columns, feature_result.numerical_columns


def _prepare_samld_dataset(source: str | None) -> tuple:
    loader = DataLoader()
    raw_df = loader.load_dataset("samld", source_override=source)

    if settings.samld_max_rows and settings.samld_max_rows > 0:
        raw_df = raw_df.head(settings.samld_max_rows).copy()
        logger.info(f"SAML-D ограничен первыми rows={len(raw_df)} по SAMLD_MAX_ROWS")

    samld_df = prepare_samld_dataset_from_frame(raw_df)
    categorical_columns, numerical_columns = get_samld_feature_columns(samld_df)

    return samld_df, categorical_columns, numerical_columns


def run_training_pipeline(dataset_name: str = "base", source: str | None = None) -> dict[str, Any]:
    ensure_directories()

    logger.info(
        f"Запуск training pipeline для датасета '{dataset_name}', "
        f"source='{source or 'config/default'}'"
    )

    loader = DataLoader()
    trainer = AMLModelTrainer()

    if dataset_name == "samld":
        prepared_df, categorical_columns, numerical_columns = _prepare_samld_dataset(source)
    else:
        prepared_df, categorical_columns, numerical_columns = _prepare_standard_dataset(dataset_name, source)

    processed_path = get_processed_data_dir() / f"{dataset_name}_features.csv"
    loader.save_dataset(prepared_df, processed_path)

    training_summary = trainer.train_and_select_best(
        df=prepared_df,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        candidate_models=["lightgbm", "xgboost"],
    )

    training_summary["dataset_name"] = dataset_name
    training_summary["source"] = source or "config/default"

    logger.info(
        f"Training pipeline завершён. Лучшая модель: {training_summary['best_model_type']}"
    )

    return training_summary
