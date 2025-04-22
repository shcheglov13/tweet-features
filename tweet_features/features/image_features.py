"""
Модуль для извлечения визуальных признаков из твитов.
"""
import numpy as np
from typing import Dict, List, Any, Optional

from tweet_features.config.feature_config import default_config, FeatureConfig
from tweet_features.utils.logger import setup_logger
from tweet_features.utils.embeddings import get_clip_embedder
from tweet_features.utils.dimensionality_reduction import get_reducer

logger = setup_logger('tweet_features.features.image_features')


class ImageFeatureExtractor:
    """
    Класс для извлечения визуальных признаков из твитов.

    Извлекает следующие признаки:
    - Эмбеддинги CLIP для изображений
    """

    def __init__(self, embedding_dim: Optional[int] = None, config: Optional[FeatureConfig] = None):
        """
        Инициализирует экстрактор визуальных признаков.

        Args:
            embedding_dim (int, optional): Размерность эмбеддингов после снижения размерности.
                По умолчанию используется значение из конфигурации.
        """
        self.config = config or default_config
        self.embedding_dim = embedding_dim or self.config.image_embedding_dim
        self.original_dim = 768  # CLIP vit-large-patch14 возвращает вектор размерности 768

        # Инициализируем редьюсер для снижения размерности
        self.reducer = get_reducer(
            self.config.dim_reduction_method,
            n_components=self.embedding_dim
        )

        # Получаем эмбеддер
        self.clip_embedder = get_clip_embedder(config=self.config)

        logger.info(f"Инициализирован экстрактор визуальных признаков (embedding_dim={self.embedding_dim})")

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

        # Извлекаем эмбеддинги для изображения
        if image_url:
            image_embeddings = self.clip_embedder.get_embeddings([image_url])[0]

            # Добавляем эмбеддинги в признаки
            for i, val in enumerate(image_embeddings):
                features[f'image_emb_{i}'] = float(val)
        else:
            # Для отсутствующего изображения используем нулевой вектор
            for i in range(self.original_dim):
                features[f'image_emb_{i}'] = 0.0

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

        # Если нужно снизить размерность
        if self.config.dim_reduction_method != 'none':
            # Собираем эмбеддинги
            image_embeddings = np.array([
                [features[i][f'image_emb_{j}'] for j in range(self.original_dim)]
                for i in range(len(features))
            ])

            # Снижаем размерность эмбеддингов
            image_embeddings_reduced = self.reducer.fit_transform(image_embeddings)

            # Обновляем признаки с новыми эмбеддингами
            for i in range(len(features)):
                # Удаляем старые эмбеддинги
                for j in range(self.original_dim):
                    if f'image_emb_{j}' in features[i]:
                        del features[i][f'image_emb_{j}']

                # Добавляем новые эмбеддинги
                for j in range(self.embedding_dim):
                    features[i][f'image_emb_reduced_{j}'] = float(image_embeddings_reduced[i][j])

        return features