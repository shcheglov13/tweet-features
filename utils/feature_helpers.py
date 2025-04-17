"""
Вспомогательные функции для извлечения признаков.
"""
import re
import emoji
from typing import List, Dict, Union, Any
from datetime import datetime

from utils.logger import setup_logger

logger = setup_logger('tweet_features.utils.feature_helpers')


def count_special_elements(text: str) -> Dict[str, int]:
    """
    Подсчитывает количество специальных элементов в тексте.

    Args:
        text (str): Текст для анализа.

    Returns:
        Dict[str, int]: Словарь с количеством хэштегов, упоминаний, URL и эмодзи.
    """
    if not text:
        return {
            'hashtag_count': 0,
            'mention_count': 0,
            'url_count': 0,
            'emoji_count': 0
        }

    # Регулярные выражения для поиска специальных элементов
    hashtag_pattern = r'#\w+'
    mention_pattern = r'@\w+'
    url_pattern = r'https?://[^\s]+'

    hashtags = re.findall(hashtag_pattern, text)
    mentions = re.findall(mention_pattern, text)
    urls = re.findall(url_pattern, text)
    emojis = [c for c in text if c in emoji.EMOJI_DATA]

    return {
        'hashtag_count': len(hashtags),
        'mention_count': len(mentions),
        'url_count': len(urls),
        'emoji_count': len(emojis)
    }


def calculate_densities(counts: Dict[str, int], text_length: int) -> Dict[str, float]:
    """
    Рассчитывает плотность специальных элементов в тексте.

    Args:
        counts (Dict[str, int]): Словарь с количеством специальных элементов.
        text_length (int): Длина текста.

    Returns:
        Dict[str, float]: Словарь с плотностями специальных элементов.
    """
    if text_length == 0:
        return {
            'hashtag_density': 0.0,
            'mention_density': 0.0,
            'url_density': 0.0,
            'emoji_density': 0.0
        }

    return {
        'hashtag_density': counts['hashtag_count'] / text_length,
        'mention_density': counts['mention_count'] / text_length,
        'url_density': counts['url_count'] / text_length,
        'emoji_density': counts['emoji_count'] / text_length
    }


def analyze_text_style(text: str) -> Dict[str, Union[int, float]]:
    """
    Анализирует стиль текста.

    Args:
        text (str): Текст для анализа.

    Returns:
        Dict[str, Union[int, float]]: Словарь с характеристиками стиля текста.
    """
    if not text:
        return {
            'uppercase_ratio': 0.0,
            'word_elongation_count': 0,
            'excessive_punctuation_count': 0
        }

    # Подсчет доли заглавных букв
    alpha_chars = [c for c in text if c.isalpha()]
    uppercase_chars = [c for c in alpha_chars if c.isupper()]
    uppercase_ratio = len(uppercase_chars) / len(alpha_chars) if alpha_chars else 0

    # Подсчет удлиненных слов (с повторяющимися буквами)
    elongation_pattern = r'\b\w*(\w)\1{2,}\w*\b'
    word_elongations = re.findall(elongation_pattern, text)

    # Подсчет избыточной пунктуации
    punctuation_pattern = r'[!?\.]{2,}|[\.]{3,}'
    excessive_punctuations = re.findall(punctuation_pattern, text)

    return {
        'uppercase_ratio': uppercase_ratio,
        'word_elongation_count': len(word_elongations),
        'excessive_punctuation_count': len(excessive_punctuations)
    }


def extract_time_features(timestamp: str) -> Dict[str, Union[int, bool]]:
    """
    Извлекает временные признаки из метки времени.

    Args:
        timestamp (str): Метка времени в формате 'YYYY-MM-DD HH:MM:SS.ssssss +0000'.

    Returns:
        Dict[str, Union[int, bool]]: Словарь с временными признаками.
    """
    try:
        # Парсим метку времени
        dt = datetime.strptime(timestamp.split('+')[0].strip(), '%Y-%m-%d %H:%M:%S.%f')

        # Извлекаем признаки
        hour = dt.hour
        day_of_week = dt.weekday()  # 0-6, где 0 - понедельник
        is_weekend = day_of_week >= 5  # 5 - суббота, 6 - воскресенье

        return {
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': int(is_weekend)
        }
    except Exception as e:
        logger.warning(f"Ошибка при извлечении временных признаков: {str(e)}")
        return {
            'hour': 0,
            'day_of_week': 0,
            'is_weekend': 0
        }


def detect_media_type(tweet: Dict[str, Any]) -> Dict[str, int]:
    """
    Определяет тип медиа в твите.

    Args:
        tweet (Dict[str, Any]): Данные твита.

    Returns:
        Dict[str, int]: Словарь с типами медиа (0 или 1).
    """
    has_image = bool(tweet.get('image_url'))
    has_text = bool(tweet.get('text'))

    # Проверяем, является ли изображение видео
    is_video = False
    image_url = tweet.get('image_url', '')
    if image_url:
        is_video = 'video_thumb' in image_url.lower() if isinstance(image_url, str) else False

    return {
        'media_type_only_text': int(has_text and not has_image),
        'media_type_video': int(is_video),
        'media_type_image': int(has_image and not is_video)
    }


def create_onehot_encoding(category: str, categories: List[str]) -> Dict[str, int]:
    """
    Создает one-hot encoding для категориальной переменной.

    Args:
        category (str): Значение категории.
        categories (List[str]): Список всех возможных категорий.

    Returns:
        Dict[str, int]: Словарь с one-hot encoding.
    """
    encoding = {}
    for cat in categories:
        encoding[cat] = int(category == cat)
    return encoding