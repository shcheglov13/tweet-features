"""
Модуль для извлечения визуальных признаков из твитов.
"""
import numpy as np
from typing import Dict, List, Any, Optional

from tweet_features.config.feature_config import default_config, FeatureConfig
from tweet_features.utils.logger import setup_logger
from tweet_features.utils.embeddings import get_clip_embedder

logger = setup_logger('tweet_features.features.image_features')


class ImageFeatureExtractor:
    """
    Класс для извлечения визуальных признаков из твитов.

    Извлекает следующие признаки:
    - Эмбеддинги CLIP для изображений
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Инициализирует экстрактор визуальных признаков.

        Args:
            config (FeatureConfig, optional): Пользовательская конфигурация.
                По умолчанию используется глобальная конфигурация.
        """
        self.config = config or default_config
        self.original_dim = 768  # CLIP vit-large-patch14 возвращает вектор размерности 768

        # Получаем эмбеддер
        self.clip_embedder = get_clip_embedder(config=self.config)

        logger.info(f"Инициализирован экстрактор визуальных признаков")

    def extract(self, tweet: Dict[str, Any]) -> Dict[str, float]:
        """
        Извлекает визуальные признаки из твита.

        Args:
            tweet (Dict[str, Any]): Данные твита.

        Returns:
            Dict[str, float]: Словарь с визуальными признаками.
        """
        features = {}

        # Получаем URL изображения
        image_url = tweet.get('image_url', '')

        # Получаем эмбеддинги или используем нулевой вектор
        image_embeddings = self.clip_embedder.get_embeddings([image_url])[0] if image_url else np.zeros(self.original_dim)

        # Добавляем эмбеддинги в признаки
        for i, val in enumerate(image_embeddings):
            features[f'image_emb_{i}'] = float(val)

        return features

    def batch_extract(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Извлекает визуальные признаки из списка твитов.

        Args:
            tweets (List[Dict[str, Any]]): Список твитов.

        Returns:
            List[Dict[str, float]]: Список словарей с визуальными признаками.
        """
        logger.info(f"Извлечение визуальных признаков для {len(tweets)} твитов")

        # Собираем URL изображений
        image_urls = [tweet.get('image_url', '') for tweet in tweets]

        # Извлекаем эмбеддинги для всех изображений сразу (пакетная обработка)
        all_embeddings = self.clip_embedder.process_batch(image_urls)

        # Формируем признаки для каждого твита
        features = []
        for i, embeddings in enumerate(all_embeddings):
            tweet_features = {}
            for j, val in enumerate(embeddings):
                tweet_features[f'image_emb_{j}'] = float(val)
            features.append(tweet_features)

        return features