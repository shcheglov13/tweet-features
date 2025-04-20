"""
Модуль для извлечения структурных признаков из твитов.
"""
from typing import Dict, List, Any, Union

from tweet_features.utils.logger import setup_logger
from tweet_features.utils.feature_helpers import (
    extract_time_features,
    detect_media_type,
    create_onehot_encoding
)

logger = setup_logger('tweet_features.features.structural_features')


class StructuralFeatureExtractor:
    """
    Класс для извлечения структурных признаков из твитов.

    Извлекает следующие признаки:
    - Тип твита (one-hot encoding): tweet_type_SINGLE, tweet_type_REPLY, tweet_type_QUOTE, tweet_type_RETWEET
    - Тип медиа: media_type_only_text, media_type_video, media_type_image
    - Флаги наличия контента: has_quoted_text, has_main_text, has_image
    - Соотношения текстов: text_quoted_ratio
    - Временные признаки: hour, day_of_week, is_weekend
    """

    def __init__(self):
        """
        Инициализирует экстрактор структурных признаков.
        """
        # Все возможные типы твитов
        self.tweet_types = ["SINGLE", "REPLY", "QUOTE", "RETWEET"]
        logger.info("Инициализирован экстрактор структурных признаков")

    def extract(self, tweet: Dict[str, Any]) -> Dict[str, Union[int, float]]:
        """
        Извлекает структурные признаки из твита.

        Args:
            tweet (Dict[str, Any]): Данные твита.

        Returns:
            Dict[str, Union[int, float]]: Словарь с структурными признаками.
        """
        features = {}

        # Извлекаем тип твита (one-hot encoding)
        tweet_type = tweet.get('tweet_type', "SINGLE")
        tweet_type_onehot = create_onehot_encoding(tweet_type, self.tweet_types)
        features.update({f"tweet_type_{k}": v for k, v in tweet_type_onehot.items()})

        # Извлекаем флаги наличия контента
        features['has_main_text'] = int(bool(tweet.get('text')))
        features['has_quoted_text'] = int(bool(tweet.get('quoted_text')))
        features['has_image'] = int(bool(tweet.get('image_url')))

        # Определяем тип медиа
        media_type = detect_media_type(tweet)
        features.update(media_type)

        # Рассчитываем соотношение текстов
        main_text = tweet.get('text', '')
        quoted_text = tweet.get('quoted_text', '')

        main_text_len = len(main_text) if main_text else 0
        quoted_text_len = len(quoted_text) if quoted_text else 0

        if quoted_text_len > 0:
            features['text_quoted_ratio'] = main_text_len / quoted_text_len if main_text_len > 0 else 0.0
        else:
            features['text_quoted_ratio'] = 0.0 if main_text_len == 0 else float('inf')

        # Извлекаем временные признаки
        created_at = tweet.get('created_at', '')
        if created_at:
            time_features = extract_time_features(created_at)
            features.update(time_features)
        else:
            features.update({'hour': 0, 'day_of_week': 0, 'is_weekend': 0})

        return features

    def batch_extract(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Union[int, float]]]:
        """
        Извлекает структурные признаки из списка твитов.

        Args:
            tweets (List[Dict[str, Any]]): Список твитов.

        Returns:
            List[Dict[str, Union[int, float]]]: Список словарей с структурными признаками.
        """
        logger.info(f"Извлечение структурных признаков для {len(tweets)} твитов")

        features = []
        for tweet in tweets:
            tweet_features = self.extract(tweet)
            features.append(tweet_features)

        return features