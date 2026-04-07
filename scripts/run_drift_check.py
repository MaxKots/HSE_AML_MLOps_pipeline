from pprint import pprint

from src.pipelines.drift_pipeline import run_drift_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("Запуск drift-check для Variant I")
    summary_variant_1 = run_drift_pipeline(
        reference_dataset_name="base",
        current_dataset_name="variant_1",
    )
    pprint(summary_variant_1)

    logger.info("Запуск drift-check для Variant II")
    summary_variant_2 = run_drift_pipeline(
        reference_dataset_name="base",
        current_dataset_name="variant_2",
    )
    pprint(summary_variant_2)


if __name__ == "__main__":
    main()
