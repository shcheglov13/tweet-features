"""
Модуль для объединения различных экстракторов признаков в единый пайплайн.
"""
import pandas as pd
from typing import Dict, List, Any, Optional, Set

from config.feature_config import default_config, FeatureConfig
from utils.logger import setup_logger
from features.structural_features import StructuralFeatureExtractor
from features.text_features import TextFeatureExtractor
from features.image_features import ImageFeatureExtractor
from features.emotional_features import EmotionalFeatureExtractor

logger = setup_logger('tweet_features.features.feature_pipeline')


class FeaturePipeline:
    """
    Класс для объединения различных экстракторов признаков в единый пайплайн.
    """

    def __init__(
            self,
            config: Optional[FeatureConfig] = None,
            use_structural: bool = True,
            use_text: bool = True,
            use_image: bool = True,
            use_emotional: bool = True,
            use_bert_embeddings: bool = True
    ):
        """
        Инициализирует пайплайн извлечения признаков.

        Args:
            config (FeatureConfig, optional): Конфигурация для пайплайна.
                По умолчанию используется глобальная конфигурация.
            use_structural (bool): Использовать ли структурные признаки.
            use_text (bool): Использовать ли текстовые признаки.
            use_image (bool): Использовать ли визуальные признаки.
            use_emotional (bool): Использовать ли эмоциональные признаки.
            use_bert_embeddings (bool): Извлекать ли BERT эмбеддинги в текстовых признаках.
        """
        self.config = config or default_config

        # Инициализируем экстракторы признаков
        self.extractors = {}

        if use_structural:
            self.extractors['structural'] = StructuralFeatureExtractor()

        if use_text:
            self.extractors['text'] = TextFeatureExtractor(
                embedding_dim=self.config.text_embedding_dim,
                use_embeddings=use_bert_embeddings
            )

        if use_image:
            self.extractors['image'] = ImageFeatureExtractor(
                embedding_dim=self.config.image_embedding_dim
            )

        if use_emotional:
            self.extractors['emotional'] = EmotionalFeatureExtractor()

        logger.info(
            f"Инициализирован пайплайн извлечения признаков с экстракторами: "
            f"{', '.join(self.extractors.keys())}"
        )

    def extract_single(self, tweet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлекает все признаки из одного твита.

        Args:
            tweet (Dict[str, Any]): Данные твита.

        Returns:
            Dict[str, Any]: Словарь со всеми признаками.
        """
        all_features = {}

        # Извлекаем признаки с помощью каждого экстрактора
        for extractor_name, extractor in self.extractors.items():
            logger.debug(f"Извлечение {extractor_name} признаков")
            features = extractor.extract(tweet)
            all_features.update(features)

        return all_features

    def extract_batch(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Извлекает все признаки из списка твитов.

        Args:
            tweets (List[Dict[str, Any]]): Список твитов.

        Returns:
            List[Dict[str, Any]]: Список словарей со всеми признаками.
        """
        if not tweets:
            logger.warning("Получен пустой список твитов")
            return []

        logger.info(f"Извлечение признаков для {len(tweets)} твитов")

        # Инициализируем список для хранения результатов
        all_features = [{} for _ in range(len(tweets))]

        # Извлекаем признаки с помощью каждого экстрактора
        for extractor_name, extractor in self.extractors.items():
            logger.info(f"Извлечение {extractor_name} признаков")

            if hasattr(extractor, 'batch_extract'):
                # Если у экстрактора есть метод batch_extract, используем его
                features_list = extractor.batch_extract(tweets)

                # Объединяем признаки
                for i, features in enumerate(features_list):
                    all_features[i].update(features)
            else:
                # Иначе извлекаем признаки для каждого твита отдельно
                for i, tweet in enumerate(tweets):
                    features = extractor.extract(tweet)
                    all_features[i].update(features)

        return all_features

    def extract_to_dataframe(self, tweets: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Извлекает все признаки из списка твитов и возвращает их в виде DataFrame.

        Args:
            tweets (List[Dict[str, Any]]): Список твитов.

        Returns:
            pd.DataFrame: DataFrame с признаками.
        """
        features_list = self.extract_batch(tweets)
        df = pd.DataFrame(features_list)

        logger.info(f"Создан DataFrame размером {df.shape}")
        return df

    def get_feature_names(self) -> Set[str]:
        """
        Возвращает список всех имен признаков, извлекаемых пайплайном.

        Returns:
            Set[str]: Множество имен признаков.
        """
        # Получаем пример твита
        example_tweet = {
            'id': '1234567890',
            'created_at': '2023-01-01 12:00:00.000000 +00:00',
            'text': 'Пример текста твита',
            'tweet_type': 'SINGLE',
            'tx_count': 42,
            'image_url': 'https://example.com/image.jpg',
            'quoted_text': 'Пример цитируемого текста'
        }

        # Извлекаем признаки для примера
        features = self.extract_single(example_tweet)

        # Возвращаем имена признаков
        return set(features.keys())

    def save_features(self, features: List[Dict[str, Any]], output_path: str) -> None:
        """
        Сохраняет извлеченные признаки в CSV-файл.

        Args:
            features (List[Dict[str, Any]]): Список словарей с признаками.
            output_path (str): Путь для сохранения CSV-файла.
        """
        df = pd.DataFrame(features)
        df.to_csv(output_path, index=False)

        logger.info(f"Признаки сохранены в {output_path}")

    def extract_and_save(self, tweets: List[Dict[str, Any]], output_path: str) -> None:
        """
        Извлекает признаки из твитов и сохраняет их в CSV-файл.

        Args:
            tweets (List[Dict[str, Any]]): Список твитов.
            output_path (str): Путь для сохранения CSV-файла.
        """
        features = self.extract_batch(tweets)
        self.save_features(features, output_path)