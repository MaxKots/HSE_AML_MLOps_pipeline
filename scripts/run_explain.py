from pathlib import Path

from src.data import DataLoader, DataTransformer, DataValidator
from src.explainability.shap_explainer import AMLShapExplainer
from src.features import FeatureEngineer
from src.utils.logger import get_logger
from src.utils.paths import ensure_directories, get_artifacts_dir

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

    bundle_path = get_artifacts_dir() / "models" / "production_bundle.joblib"
    if not bundle_path.exists():
        lightgbm_bundle = get_artifacts_dir() / "models" / "lightgbm_bundle.joblib"
        if lightgbm_bundle.exists():
            bundle_path = lightgbm_bundle
        else:
            raise FileNotFoundError(
                f"Не найден bundle модели: {bundle_path}. Сначала выполнить обучение."
            )

    explainer = AMLShapExplainer(str(bundle_path))
    explanations = explainer.explain_rows(
        feature_result.dataframe.head(10),
        row_indices=[0, 1, 2],
        top_k=5,
    )

    output_path = get_artifacts_dir() / "shap" / "sample_explanations.yaml"
    explainer.export_explanations(explanations, output_path)
    explainer.export_summary_plot(feature_result.dataframe.head(500))

    logger.info(f"Объяснения сохранены в {Path(output_path).resolve()}")

    for explanation in explanations:
        logger.info(
            f"row={explanation.row_index}, "
            f"score={explanation.prediction_score:.4f}, "
            f"label={explanation.prediction_label}"
        )
        for reason in explainer.build_human_readable_reasons(explanation):
            logger.info(f"  - {reason}")


if __name__ == "__main__":
    main()
