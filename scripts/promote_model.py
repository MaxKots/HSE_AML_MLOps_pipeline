from src.models.registry import AMLModelRegistry
from src.utils.io import read_json
from src.utils.logger import get_logger
from src.utils.paths import get_artifacts_dir

logger = get_logger(__name__)


def main() -> None:
    summary_path = get_artifacts_dir() / "metrics" / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Не найден training_summary.json. Сначала выполнить обучение.")

    summary = read_json(summary_path)
    registry_info = summary.get("registry")

    if not registry_info:
        raise ValueError("Модель не была зарегистрирована в MLflow Registry.")

    version = str(registry_info["version"])

    registry = AMLModelRegistry(model_name="aml_detection_model")
    registry.transition_stage(version=version, stage="Production")

    logger.info(f"Модель версии {version} переведена в стадию Production")


if __name__ == "__main__":
    main()
