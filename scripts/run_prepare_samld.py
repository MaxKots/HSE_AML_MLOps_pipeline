from __future__ import annotations

from pathlib import Path

from config.settings import settings
from src.data.samld import prepare_samld_dataset
from src.utils.logger import get_logger
from src.utils.paths import ensure_directories

logger = get_logger(__name__)


def main() -> None:
    ensure_directories()

    prepared_df = prepare_samld_dataset(
        path=settings.samld_dataset,
        max_rows=settings.samld_max_rows,
    )

    output_path = (
        settings.project_root
        / settings.data_processed_dir
        / "samld_features.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prepared_df.to_csv(output_path, index=False)

    logger.info(
        f"SAML-D features сохранены: {output_path}; "
        f"rows={len(prepared_df)}, cols={len(prepared_df.columns)}"
    )


if __name__ == "__main__":
    main()
