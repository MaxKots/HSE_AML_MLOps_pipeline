from pprint import pprint

from src.pipelines.training_pipeline import run_training_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    summary = run_training_pipeline(dataset_name="base")

    logger.info("Итоговая сводка по обучению")
    pprint(summary)


if __name__ == "__main__":
    main()
