from pathlib import Path

from config.settings import settings


def get_project_root() -> Path:
    return settings.project_root


def get_raw_data_dir() -> Path:
    return get_project_root() / settings.data_raw_dir


def get_processed_data_dir() -> Path:
    return get_project_root() / settings.data_processed_dir


def get_reference_data_dir() -> Path:
    return get_project_root() / settings.data_reference_dir


def get_artifacts_dir() -> Path:
    return get_project_root() / settings.artifacts_dir


def ensure_directories() -> None:
    directories = [
        get_raw_data_dir(),
        get_processed_data_dir(),
        get_reference_data_dir(),
        get_artifacts_dir(),
        get_artifacts_dir() / "reports",
        get_artifacts_dir() / "metrics",
        get_artifacts_dir() / "shap",
        get_artifacts_dir() / "models",
        get_artifacts_dir() / "predictions",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
