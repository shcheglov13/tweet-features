"""
Модуль для настройки и управления логированием в пакете tweet-features.
"""
import logging
import sys
from typing import Optional

from tweet_features.config.feature_config import default_config


def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Создает и настраивает логгер с заданным именем и уровнем логирования.

    Args:
        name (str): Имя для логгера.
        level (str, optional): Уровень логирования. По умолчанию используется
            уровень из конфигурации.

    Returns:
        logging.Logger: Настроенный логгер.
    """
    logger = logging.getLogger(name)

    # Используем уровень логирования из конфигурации по умолчанию, если не указан
    log_level = level or default_config.log_level
    logger.setLevel(getattr(logging, log_level))

    # Если у логгера нет обработчиков, добавляем их
    if not logger.handlers:
        # Создаем обработчик для вывода в консоль
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))

        # Создаем форматтер
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        # Добавляем обработчик к логгеру
        logger.addHandler(console_handler)

    return logger


# Корневой логгер пакета
logger = setup_logger('tweet_features')