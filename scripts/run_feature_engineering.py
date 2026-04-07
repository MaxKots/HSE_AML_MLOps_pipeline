from src.data import DataLoader, DataTransformer, DataValidator
from src.features import FeatureEngineer
from src.utils.logger import get_logger
from src.utils.paths import ensure_directories, get_processed_data_dir

logger = get_logger(__name__)


def main() -> None:
    ensure_directories()

    loader = DataLoader()
    validator = DataValidator()
    transformer = DataTransformer()
    feature_engineer = FeatureEngineer()

    df = loader.load_dataset("base")
    validator.run_full_validation(df)

    transformed_df = transformer.transform(df)
    feature_result = feature_engineer.build_features(transformed_df)

    output_path = get_processed_data_dir() / "base_features.csv"
    loader.save_dataset(feature_result.dataframe, output_path)

    logger.info(f"Feature dataset сохранён в {output_path}")
    logger.info(f"Получено {len(feature_result.feature_columns)} итоговых признаков")


if __name__ == "__main__":
    main()
