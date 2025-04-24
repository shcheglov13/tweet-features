"""
Модуль для извлечения текстовых признаков из твитов.
"""
import numpy as np
from typing import Dict, List, Any, Union, Optional

from tweet_features.config.feature_config import default_config, FeatureConfig
from tweet_features.utils.logger import setup_logger
from tweet_features.utils.embeddings import get_bertweet_embedder
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

    def __init__(self, use_embeddings: bool = True, config: Optional[FeatureConfig] = None):
        """
        Инициализирует экстрактор текстовых признаков.

        Args:
            use_embeddings (bool): Извлекать ли эмбеддинги BERT.
            config (FeatureConfig, optional): Пользовательская конфигурация.
                По умолчанию используется глобальная конфигурация.
        """
        self.config = config or default_config
        self.use_embeddings = use_embeddings
        self.original_dim = 768  # BERTweet возвращает вектор размерности 768

        if self.use_embeddings:
            # Получаем эмбеддер
            self.bertweet_embedder = get_bertweet_embedder(config=self.config)

        logger.info(
            f"Инициализирован экстрактор текстовых признаков "
            f"(use_embeddings={self.use_embeddings})"
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

        # Обработка эмбеддингов
        if self.use_embeddings:
            # Получаем эмбеддинги или используем нулевые векторы
            text_embeddings = self.bertweet_embedder.get_embeddings([text])[0] if text else np.zeros(self.original_dim)
            quoted_text_embeddings = self.bertweet_embedder.get_embeddings([quoted_text])[0] if quoted_text else np.zeros(self.original_dim)

            # Добавляем эмбеддинги в признаки
            for i, val in enumerate(text_embeddings):
                features[f'text_emb_{i}'] = float(val)

            for i, val in enumerate(quoted_text_embeddings):
                features[f'quoted_text_emb_{i}'] = float(val)

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

        return features