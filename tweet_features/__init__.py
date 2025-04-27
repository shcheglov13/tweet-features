"""
Пакет tweet-features для извлечения различных типов признаков из данных Twitter.

Этот пакет предоставляет инструменты для извлечения структурных, текстовых,
визуальных и эмоциональных признаков из твитов. Он также включает функционал
для кеширования вычислительно затратных операций.
"""

from tweet_features.config.feature_config import FeatureConfig, default_config
from tweet_features.features.feature_pipeline import FeaturePipeline
from tweet_features.features.structural_features import StructuralFeatureExtractor
from tweet_features.features.text_features import TextFeatureExtractor
from tweet_features.features.image_features import ImageFeatureExtractor
from tweet_features.features.emotional_features import EmotionalFeatureExtractor
from tweet_features.utils.logger import setup_logger
from tweet_features.utils.caching import FeatureCache, cache
from tweet_features.preprocessing import DimensionalityReducer, FeatureSelector


__version__ = '0.1.0'


__all__ = [
    'FeatureConfig',
    'default_config',
    'FeaturePipeline',
    'StructuralFeatureExtractor',
    'TextFeatureExtractor',
    'ImageFeatureExtractor',
    'EmotionalFeatureExtractor',
    'setup_logger',
    'FeatureCache',
    'cache',
    "DimensionalityReducer",
    "FeatureSelector"
]