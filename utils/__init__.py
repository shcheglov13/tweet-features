"""
Утилиты для извлечения признаков из твитов.
"""

from utils.logger import setup_logger
from utils.caching import FeatureCache, cache
from utils.embeddings import get_bertweet_embedder, get_clip_embedder
from utils.dimensionality_reduction import get_reducer
from utils.feature_helpers import (
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