"""
Модуль для кеширования вычислительно затратных операций.
"""
import os
import pickle
import hashlib
import numpy as np
from typing import Any, Dict, Optional, Union

from tweet_features.config.feature_config import default_config
from tweet_features.utils.logger import setup_logger

logger = setup_logger('tweet_features.utils.caching')


class FeatureCache:
    """
    Класс для кеширования извлеченных признаков.

    Позволяет сохранять и загружать вычислительно затратные признаки,
    такие как эмбеддинги BERT и CLIP.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Инициализирует кеш признаков.

        Args:
            cache_dir (str, optional): Путь до директории кеша.
                По умолчанию используется путь из конфигурации.
        """
        self.cache_dir = cache_dir or default_config.cache_dir
        self.use_cache = default_config.use_cache

        # Создаем директорию кеша, если она не существует
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Создана директория кеша: {self.cache_dir}")

    def get_cache_key(self, data: Union[str, Dict[str, Any], np.ndarray], prefix: str = "") -> str:
        """
        Генерирует ключ кеша на основе входных данных.

        Args:
            data: Данные для генерации ключа (текст, словарь или массив).
            prefix (str): Префикс для ключа кеша.

        Returns:
            str: Ключ кеша.
        """
        if isinstance(data, str):
            raw_data = data.encode('utf-8')
        elif isinstance(data, dict):
            # Сортируем ключи для обеспечения детерминизма
            raw_data = str(sorted(data.items())).encode('utf-8')
        elif isinstance(data, np.ndarray):
            raw_data = data.tobytes()
        else:
            raise TypeError(f"Неподдерживаемый тип данных для кеширования: {type(data)}")

        hash_val = hashlib.md5(raw_data).hexdigest()
        return f"{prefix}_{hash_val}" if prefix else hash_val

    def get_cache_path(self, cache_key: str) -> str:
        """
        Получает путь к файлу кеша по ключу.

        Args:
            cache_key (str): Ключ кеша.

        Returns:
            str: Путь к файлу кеша.
        """
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def exists(self, cache_key: str) -> bool:
        """
        Проверяет, существует ли кеш для данного ключа.

        Args:
            cache_key (str): Ключ кеша.

        Returns:
            bool: True, если кеш существует, иначе False.
        """
        if not self.use_cache:
            return False

        cache_path = self.get_cache_path(cache_key)
        return os.path.exists(cache_path)

    def save(self, cache_key: str, data: Any) -> None:
        """
        Сохраняет данные в кеш.

        Args:
            cache_key (str): Ключ кеша.
            data: Данные для сохранения.
        """
        if not self.use_cache:
            return

        cache_path = self.get_cache_path(cache_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

        logger.debug(f"Кеш сохранен: {cache_path}")

    def load(self, cache_key: str) -> Any:
        """
        Загружает данные из кеша.

        Args:
            cache_key (str): Ключ кеша.

        Returns:
            Данные из кеша.

        Raises:
            FileNotFoundError: Если кеш не найден.
        """
        if not self.use_cache:
            raise FileNotFoundError(f"Кеширование отключено")

        cache_path = self.get_cache_path(cache_key)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Кеш не найден: {cache_path}")

        with open(cache_path, 'rb') as f:
            data = pickle.load(f)

        logger.debug(f"Кеш загружен: {cache_path}")
        return data

    def clear(self, prefix: Optional[str] = None) -> int:
        """
        Очищает кеш.

        Args:
            prefix (str, optional): Если указан, удаляет только файлы с указанным префиксом.

        Returns:
            int: Количество удаленных файлов.
        """
        if not self.use_cache:
            return 0

        count = 0
        for filename in os.listdir(self.cache_dir):
            if prefix is None or filename.startswith(prefix):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    count += 1

        logger.info(f"Удалено {count} файлов кеша")
        return count


# Глобальный экземпляр кеша
cache = FeatureCache()