from pprint import pprint

from src.pipelines.benchmark_pipeline import run_benchmark_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    summary = run_benchmark_pipeline()
    logger.info("Benchmark pipeline завершён")
    pprint(summary)


if __name__ == "__main__":
    main()
