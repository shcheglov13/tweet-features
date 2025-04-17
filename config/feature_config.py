"""
Модуль конфигурации для пакета tweet-features.
Содержит настройки для извлечения различных типов признаков из данных Twitter.
"""
import os
import torch
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class FeatureConfig:
    """
    Класс конфигурации для пакета извлечения признаков.

    Attributes:
        text_embedding_dim (int): Размерность эмбеддингов текста после снижения размерности.
        image_embedding_dim (int): Размерность эмбеддингов изображений после снижения размерности.
        batch_size (int): Размер пакета для пакетной обработки.
        use_cache (bool): Использовать ли кеширование признаков.
        cache_dir (str): Путь до директории кеша.
        device (str): Устройство для вычислений ('cpu' или 'cuda').
        dim_reduction_method (str): Метод снижения размерности ('pca' или 'none').
        log_level (str): Уровень логирования.
    """
    # Размерность эмбеддингов
    text_embedding_dim: int = 32
    image_embedding_dim: int = 64

    # Параметры обработки
    batch_size: int = 16

    # Кеширование
    use_cache: bool = True
    cache_dir: str = os.path.join(os.path.expanduser('~'), '.tweet_features_cache')

    # Вычислительные ресурсы
    device: Literal['cpu', 'cuda'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Снижение размерности
    dim_reduction_method: Literal['pca', 'none'] = 'pca'

    # Логирование
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'

    def __post_init__(self):
        """Создаёт директорию кеша, если она не существует."""
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)


# Глобальный экземпляр конфигурации с настройками по умолчанию
default_config = FeatureConfig()