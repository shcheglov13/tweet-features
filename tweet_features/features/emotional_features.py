"""
Модуль для извлечения эмоциональных признаков из твитов.
"""
import torch
from typing import Dict, List, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tweet_features.config.feature_config import default_config, FeatureConfig
from tweet_features.utils.logger import setup_logger
from tweet_features.utils.caching import cache

logger = setup_logger('tweet_features.features.emotional_features')


class EmotionalFeatureExtractor:
    """
    Класс для извлечения эмоциональных признаков из твитов.

    Извлекает следующие признаки:
    - Сентимент-анализ основного и цитируемого текста:
      - Вероятности принадлежности к трем классам: "negative", "neutral", "positive"
    - Определение оскорбительного контента:
      - Вероятность offensive_prob для основного и цитируемого текста
    - Определение иронии:
      - Вероятность irony_prob для основного и цитируемого текста
    """

    def __init__(self, device: Optional[str] = None, config: Optional[FeatureConfig] = None):
        """
        Инициализирует экстрактор эмоциональных признаков.

        Args:
            device (str, optional): Устройство для вычислений ('cpu' или 'cuda').
                По умолчанию используется устройство из конфигурации.
        """
        self.config = config or default_config
        self.device = device or self.config.device
        self.max_length = 128  # Максимальная длина входной последовательности

        # Модели для анализа эмоциональных признаков
        self.models = {
            'sentiment': {
                'name': "cardiffnlp/twitter-roberta-base-sentiment-latest",
                'model': None,
                'tokenizer': None,
                'labels': ["negative", "neutral", "positive"]
            },
            'offensive': {
                'name': "cardiffnlp/twitter-roberta-base-offensive",
                'model': None,
                'tokenizer': None,
                'labels': ["not_offensive", "offensive"]
            },
            'irony': {
                'name': "cardiffnlp/twitter-roberta-base-irony",
                'model': None,
                'tokenizer': None,
                'labels': ["non_irony", "irony"]
            }
        }

        logger.info(f"Инициализирован экстрактор эмоциональных признаков на устройстве {self.device}")

    def _load_model(self, model_type: str) -> None:
        """
        Загружает модель указанного типа.

        Args:
            model_type (str): Тип модели ('sentiment', 'offensive' или 'irony').
        """
        if self.models[model_type]['model'] is None:
            model_name = self.models[model_type]['name']
            logger.info(f"Загрузка модели {model_name} для {model_type} анализа")

            self.models[model_type]['tokenizer'] = AutoTokenizer.from_pretrained(model_name)
            self.models[model_type]['model'] = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.models[model_type]['model'].to(self.device)
            self.models[model_type]['model'].eval()  # Переводим модель в режим оценки

    def _analyze_text(self, text: str, model_type: str) -> Dict[str, float]:
        """
        Анализирует текст с помощью указанной модели.

        Args:
            text (str): Текст для анализа.
            model_type (str): Тип модели ('sentiment', 'offensive' или 'irony').

        Returns:
            Dict[str, float]: Словарь с вероятностями для каждого класса.
        """
        if not text:
            # Для пустого текста возвращаем равномерное распределение вероятностей
            return {label: 1.0 / len(self.models[model_type]['labels']) for label in self.models[model_type]['labels']}

        # Загружаем модель, если она еще не загружена
        self._load_model(model_type)

        # Проверяем, есть ли результат в кеше
        if self.config.use_cache:
            cache_key = cache.get_cache_key(text, prefix=f"{model_type}_analysis")
            if cache.exists(cache_key):
                try:
                    return cache.load(cache_key)
                except Exception as e:
                    logger.warning(f"Ошибка при загрузке кеша для {model_type} анализа: {str(e)}")

        # Токенизируем текст
        tokenizer = self.models[model_type]['tokenizer']
        model = self.models[model_type]['model']
        labels = self.models[model_type]['labels']

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        # Получаем предсказания
        with torch.no_grad():
            outputs = model(**inputs)

        # Применяем softmax для получения вероятностей
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()

        # Формируем результаты
        results = {label: float(prob) for label, prob in zip(labels, probabilities)}

        # Сохраняем результаты в кеш
        if self.config.use_cache:
            cache.save(cache_key, results)

        return results

    def extract(self, tweet: Dict[str, Any]) -> Dict[str, float]:
        """
        Извлекает эмоциональные признаки из твита.

        Args:
            tweet (Dict[str, Any]): Данные твита.

        Returns:
            Dict[str, float]: Словарь с эмоциональными признаками.
        """
        features = {}

        # Получаем тексты
        text = tweet.get('text', '')
        quoted_text = tweet.get('quoted_text', '')

        # Анализируем основной текст
        if text:
            # Сентимент-анализ
            sentiment_results = self._analyze_text(text, 'sentiment')
            for label, prob in sentiment_results.items():
                features[f'text_{label}_prob'] = prob

            # Анализ оскорбительного контента
            offensive_results = self._analyze_text(text, 'offensive')
            features['text_offensive_prob'] = offensive_results.get('offensive', 0.0)

            # Анализ иронии
            irony_results = self._analyze_text(text, 'irony')
            features['text_irony_prob'] = irony_results.get('irony', 0.0)
        else:
            # Для пустого текста
            for label in self.models['sentiment']['labels']:
                features[f'text_{label}_prob'] = 1.0 / len(self.models['sentiment']['labels'])
            features['text_offensive_prob'] = 0.5
            features['text_irony_prob'] = 0.5

        # Анализируем цитируемый текст
        if quoted_text:
            # Сентимент-анализ
            sentiment_results = self._analyze_text(quoted_text, 'sentiment')
            for label, prob in sentiment_results.items():
                features[f'quoted_text_{label}_prob'] = prob

            # Анализ оскорбительного контента
            offensive_results = self._analyze_text(quoted_text, 'offensive')
            features['quoted_text_offensive_prob'] = offensive_results.get('offensive', 0.0)

            # Анализ иронии
            irony_results = self._analyze_text(quoted_text, 'irony')
            features['quoted_text_irony_prob'] = irony_results.get('irony', 0.0)
        else:
            # Для пустого текста
            for label in self.models['sentiment']['labels']:
                features[f'quoted_text_{label}_prob'] = 1.0 / len(self.models['sentiment']['labels'])
            features['quoted_text_offensive_prob'] = 0.5
            features['quoted_text_irony_prob'] = 0.5

        return features

    def batch_extract(self, tweets: List[Dict[str, Any]], batch_size: Optional[int] = None) -> List[Dict[str, float]]:
        """
        Извлекает эмоциональные признаки из списка твитов.

        Args:
            tweets (List[Dict[str, Any]]): Список твитов.
            batch_size (int, optional): Размер батча. По умолчанию используется
                размер из конфигурации.

        Returns:
            List[Dict[str, float]]: Список словарей с эмоциональными признаками.
        """
        logger.info(f"Извлечение эмоциональных признаков для {len(tweets)} твитов")

        batch_size = batch_size or self.config.batch_size

        features = []
        for i in range(0, len(tweets), batch_size):
            batch_tweets = tweets[i:i + batch_size]

            batch_features = []
            for tweet in batch_tweets:
                tweet_features = self.extract(tweet)
                batch_features.append(tweet_features)

            features.extend(batch_features)

        return features