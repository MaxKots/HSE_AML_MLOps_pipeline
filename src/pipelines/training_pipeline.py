from __future__ import annotations

from typing import Any

from src.data import DataLoader, DataTransformer, DataValidator
from src.features import FeatureEngineer
from src.models import AMLModelTrainer
from src.utils.logger import get_logger
from src.utils.paths import ensure_directories, get_processed_data_dir

logger = get_logger(__name__)


def run_training_pipeline(dataset_name: str = "base") -> dict[str, Any]:
    ensure_directories()

    logger.info(f"Запуск training pipeline для датасета '{dataset_name}'")

    loader = DataLoader()
    validator = DataValidator()
    transformer = DataTransformer()
    feature_engineer = FeatureEngineer()
    trainer = AMLModelTrainer()

    df = loader.load_dataset(dataset_name)
    validator.run_full_validation(df)

    transformed_df = transformer.transform(df)
    feature_result = feature_engineer.build_features(transformed_df)

    processed_path = get_processed_data_dir() / f"{dataset_name}_features.csv"
    loader.save_dataset(feature_result.dataframe, processed_path)

    training_summary = trainer.train_and_select_best(
        df=feature_result.dataframe,
        categorical_columns=feature_result.categorical_columns,
        numerical_columns=feature_result.numerical_columns,
        candidate_models=["lightgbm", "xgboost"],
    )

    logger.info(
        f"Training pipeline завершён. Лучшая модель: {training_summary['best_model_type']}"
    )

    return training_summary
