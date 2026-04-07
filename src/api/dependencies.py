from __future__ import annotations

from functools import lru_cache

from src.api.service import AMLInferenceService
from src.utils.paths import get_artifacts_dir


@lru_cache(maxsize=1)
def get_inference_service() -> AMLInferenceService:
    bundle_dir = get_artifacts_dir() / "models"

    candidate_paths = [
        bundle_dir / "production_bundle.joblib",
        bundle_dir / "lightgbm_bundle.joblib",
        bundle_dir / "xgboost_bundle.joblib",
    ]

    bundle_path = None
    for path in candidate_paths:
        if path.exists():
            bundle_path = path
            break

    if bundle_path is None:
        raise FileNotFoundError(
            "Не найден bundle модели. Сначала нужно выполнить обучение pipeline."
        )

    return AMLInferenceService(bundle_path=str(bundle_path))
