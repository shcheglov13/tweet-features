"""
Модуль для извлечения текстовых признаков из твитов.
"""
import numpy as np
from typing import Dict, List, Any, Union, Optional

from tweet_features.config.feature_config import default_config
from tweet_features.utils.logger import setup_logger
from tweet_features.utils.embeddings import get_bertweet_embedder
from tweet_features.utils.dimensionality_reduction import get_reducer
from tweet_features.utils.feature_helpers import (
    count_special_elements,
    calculate_densities,
    analyze_text_style
)

logger = setup_logger('tweet_features.features.text_features')


class TextFeatureExtractor:
    """
    Класс для извлечения текстовых признаков из твитов.

    Извлекает следующие признаки:
    - Bertweet эмбеддинги для основного и цитируемого текста
    - Метрики длины: text_length, quoted_text_length, combined_text_length
    - Количество слов: text_word_count, quoted_text_word_count, combined_word_count
    - Средняя длина слов: avg_word_length_text, avg_word_length_quoted
    - Специальные элементы: mention_count, url_count, emoji_count
    - Плотность специальных элементов: mention_density, url_density, emoji_density
    - Стиль текста: uppercase_ratio, word_elongation_count, excessive_punctuation_count
    """

    def __init__(self, embedding_dim: Optional[int] = None, use_embeddings: bool = True):
        """
        Инициализирует экстрактор текстовых признаков.

        Args:
            embedding_dim (int, optional): Размерность эмбеддингов после снижения размерности.
                По умолчанию используется значение из конфигурации.
            use_embeddings (bool): Извлекать ли эмбеддинги BERT.
        """
        self.embedding_dim = embedding_dim or default_config.text_embedding_dim
        self.use_embeddings = use_embeddings
        self.original_dim = 768  # BERTweet возвращает вектор размерности 768

        if self.use_embeddings:
            # Инициализируем редьюсер для снижения размерности
            self.reducer = get_reducer(
                default_config.dim_reduction_method,
                n_components=self.embedding_dim
            )

            # Получаем эмбеддер
            self.bertweet_embedder = get_bertweet_embedder()

        logger.info(
            f"Инициализирован экстрактор текстовых признаков "
            f"(embedding_dim={self.embedding_dim}, use_embeddings={self.use_embeddings})"
        )

    def extract_text_metrics(self, text: str) -> Dict[str, Union[int, float]]:
        """
        Извлекает метрики из текста.

        Args:
            text (str): Текст для анализа.

        Returns:
            Dict[str, Union[int, float]]: Словарь с метриками текста.
        """
        if not text:
            return {
                'length': 0,
                'word_count': 0,
                'avg_word_length': 0,
                'hashtag_count': 0,
                'mention_count': 0,
                'url_count': 0,
                'emoji_count': 0,
                'hashtag_density': 0,
                'mention_density': 0,
                'url_density': 0,
                'emoji_density': 0,
                'uppercase_ratio': 0,
                'word_elongation_count': 0,
                'excessive_punctuation_count': 0
            }

        # Метрики длины
        text_length = len(text)

        # Количество слов
        words = text.split()
        word_count = len(words)

        # Средняя длина слов
        avg_word_length = np.mean([len(word) for word in words]) if words else 0

        # Специальные элементы
        special_elements = count_special_elements(text)

        # Плотность специальных элементов
        densities = calculate_densities(special_elements, text_length)

        # Стиль текста
        style_features = analyze_text_style(text)

        # Объединяем все метрики
        metrics = {
            'length': text_length,
            'word_count': word_count,
            'avg_word_length': float(avg_word_length)
        }
        metrics.update(special_elements)
        metrics.update(densities)
        metrics.update(style_features)

        return metrics

    def extract(self, tweet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлекает текстовые признаки из твита.

        Args:
            tweet (Dict[str, Any]): Данные твита.

        Returns:
            Dict[str, Any]: Словарь с текстовыми признаками.
        """
        features = {}

        # Получаем тексты
        text = tweet.get('text', '')
        quoted_text = tweet.get('quoted_text', '')

        # Извлекаем метрики для основного текста
        text_metrics = self.extract_text_metrics(text)
        features.update({
            'text_' + k: v for k, v in text_metrics.items()
            if k not in ['hashtag_count', 'mention_count', 'url_count', 'emoji_count']
        })

        # Извлекаем метрики для цитируемого текста
        quoted_text_metrics = self.extract_text_metrics(quoted_text)
        features.update({
            'quoted_text_' + k: v for k, v in quoted_text_metrics.items()
            if k not in ['hashtag_count', 'mention_count', 'url_count', 'emoji_count']
        })

        # Общие метрики для обоих текстов
        features.update({
            'combined_text_length': text_metrics['length'] + quoted_text_metrics['length'],
            'combined_word_count': text_metrics['word_count'] + quoted_text_metrics['word_count'],
            'hashtag_count': text_metrics['hashtag_count'] + quoted_text_metrics['hashtag_count'],
            'mention_count': text_metrics['mention_count'] + quoted_text_metrics['mention_count'],
            'url_count': text_metrics['url_count'] + quoted_text_metrics['url_count'],
            'emoji_count': text_metrics['emoji_count'] + quoted_text_metrics['emoji_count']
        })

        # Извлекаем эмбеддинги, если нужно
        if self.use_embeddings:
            # Получаем эмбеддинги для основного текста
            if text:
                text_embeddings = self.bertweet_embedder.get_embeddings([text])[0]

                # Добавляем эмбеддинги в признаки
                for i, val in enumerate(text_embeddings):
                    features[f'text_emb_{i}'] = float(val)
            else:
                # Для пустого текста используем нулевой вектор
                for i in range(self.original_dim):
                    features[f'text_emb_{i}'] = 0.0

            # Получаем эмбеддинги для цитируемого текста
            if quoted_text:
                quoted_text_embeddings = self.bertweet_embedder.get_embeddings([quoted_text])[0]

                # Добавляем эмбеддинги в признаки
                for i, val in enumerate(quoted_text_embeddings):
                    features[f'quoted_text_emb_{i}'] = float(val)
            else:
                # Для пустого текста используем нулевой вектор
                for i in range(self.original_dim):
                    features[f'quoted_text_emb_{i}'] = 0.0

        return features

    def batch_extract(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Извлекает текстовые признаки из списка твитов.

        Args:
            tweets (List[Dict[str, Any]]): Список твитов.

        Returns:
            List[Dict[str, Any]]: Список словарей с текстовыми признаками.
        """
        logger.info(f"Извлечение текстовых признаков для {len(tweets)} твитов")

        # Извлекаем статистические признаки для каждого твита
        features = []
        for tweet in tweets:
            tweet_features = self.extract(tweet)
            features.append(tweet_features)

        # Если нужны эмбеддинги и их нужно снизить размерность
        if self.use_embeddings and default_config.dim_reduction_method != 'none':
            # Собираем эмбеддинги для основного текста
            text_embeddings = np.array([
                [features[i][f'text_emb_{j}'] for j in range(self.original_dim)]
                for i in range(len(features))
            ])

            # Собираем эмбеддинги для цитируемого текста
            quoted_text_embeddings = np.array([
                [features[i][f'quoted_text_emb_{j}'] for j in range(self.original_dim)]
                for i in range(len(features))
            ])

            # Снижаем размерность эмбеддингов основного текста
            text_embeddings_reduced = self.reducer.fit_transform(text_embeddings)

            # Снижаем размерность эмбеддингов цитируемого текста
            quoted_text_embeddings_reduced = self.reducer.fit_transform(quoted_text_embeddings)

            # Обновляем признаки с новыми эмбеддингами
            for i in range(len(features)):
                # Удаляем старые эмбеддинги
                for j in range(self.original_dim):
                    if f'text_emb_{j}' in features[i]:
                        del features[i][f'text_emb_{j}']
                    if f'quoted_text_emb_{j}' in features[i]:
                        del features[i][f'quoted_text_emb_{j}']

                # Добавляем новые эмбеддинги
                for j in range(self.embedding_dim):
                    features[i][f'text_emb_reduced_{j}'] = float(text_embeddings_reduced[i][j])
                    features[i][f'quoted_text_emb_reduced_{j}'] = float(quoted_text_embeddings_reduced[i][j])

        return features