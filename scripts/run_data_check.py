from pathlib import Path

from src.data import DataLoader, DataTransformer, DataValidator
from src.utils.logger import get_logger
from src.utils.paths import ensure_directories, get_processed_data_dir

logger = get_logger(__name__)


def main() -> None:
    ensure_directories()

    loader = DataLoader()
    validator = DataValidator()
    transformer = DataTransformer()

    df = loader.load_dataset("base")
    report = validator.run_full_validation(df)

    logger.info(f"Отчёт валидации: {report}")

    transformed_df = transformer.transform(df)

    output_path = get_processed_data_dir() / "base_processed.csv"
    loader.save_dataset(transformed_df, output_path)

    logger.info(f"Обработанный датасет сохранён в {Path(output_path).resolve()}")


if __name__ == "__main__":
    main()
