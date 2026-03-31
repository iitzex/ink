import sys
from loguru import logger

def setup_logger(level: str = "INFO"):
    """
    設定統一的日誌格式
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
    )

if __name__ == "__main__":
    setup_logger()
    logger.info("Logger initialized.")
