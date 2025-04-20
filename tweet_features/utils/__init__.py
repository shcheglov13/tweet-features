"""
Утилиты для извлечения признаков из твитов.
"""

from tweet_features.utils.logger import setup_logger
from tweet_features.utils.caching import FeatureCache, cache
from tweet_features.utils.embeddings import get_bertweet_embedder, get_clip_embedder
from tweet_features.utils.dimensionality_reduction import get_reducer
from tweet_features.utils.feature_helpers import (
    count_special_elements,
    calculate_densities,
    analyze_text_style,
    extract_time_features,
    detect_media_type,
    create_onehot_encoding
)


__all__ = [
    'setup_logger',
    'FeatureCache',
    'cache',
    'get_bertweet_embedder',
    'get_clip_embedder',
    'get_reducer',
    'count_special_elements',
    'calculate_densities',
    'analyze_text_style',
    'extract_time_features',
    'detect_media_type',
    'create_onehot_encoding'
]