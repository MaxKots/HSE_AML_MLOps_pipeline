import uvicorn

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info(
        f"Запуск API на {settings.api_host}:{settings.api_port}, env={settings.app_env}"
    )

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
