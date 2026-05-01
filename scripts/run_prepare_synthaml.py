from config.settings import settings
from src.data.synthaml import prepare_synthaml_dataset
from src.utils.io import save_dataframe
from src.utils.logger import get_logger
from src.utils.paths import get_data_processed_dir

logger = get_logger(__name__)


def main() -> None:
    df = prepare_synthaml_dataset(
        alerts_path=settings.synthaml_alerts,
        transactions_path=settings.synthaml_transactions,
    )

    output_path = get_data_processed_dir() / "synthaml_features.csv"
    save_dataframe(df, output_path, index=False)

    logger.info(f"SynthAML features сохранён в {output_path}")


if __name__ == "__main__":
    main()
