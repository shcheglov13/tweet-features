"""
Модуль для работы с эмбеддингами текста и изображений.
"""
import numpy as np
import torch
import requests
from PIL import Image
from io import BytesIO
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer, CLIPProcessor, CLIPModel

from tweet_features.config.feature_config import default_config, FeatureConfig
from tweet_features.utils.logger import setup_logger
from tweet_features.utils.caching import cache, FeatureCache

logger = setup_logger('tweet_features.utils.embeddings')


class BertEmbedder:
    """
    Класс для извлечения эмбеддингов из текста с использованием BERTweet.
    """

    def __init__(
            self,
            model_name: str = "vinai/bertweet-base",
            device: Optional[str] = None,
            config: Optional[FeatureConfig] = None
    ):
        """
        Инициализирует модель BERTweet для извлечения эмбеддингов.

        Args:
            model_name (str): Название модели Hugging Face.
            device (str, optional): Устройство для вычислений ('cpu' или 'cuda').
                По умолчанию используется устройство из конфигурации.
        """
        self.config = config or default_config
        self.model_name = model_name
        self.device = device or self.config.device
        self.max_length = 128  # Максимальная длина входной последовательности

        logger.info(f"Инициализация BERTweet эмбеддера ({model_name}) на устройстве {self.device}")

        # Загружаем токенизатор и модель
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Переводим модель в режим оценки

    def get_embeddings(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Извлекает эмбеддинги из списка текстов.

        Args:
            texts (List[str]): Список текстов для извлечения эмбеддингов.
            use_cache (bool): Использовать ли кеширование.

        Returns:
            np.ndarray: Массив эмбеддингов размера (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])

        # Проверяем кеш для каждого текста
        embeddings = []
        texts_to_process = []
        indices_to_process = []

        if use_cache and self.config.use_cache:
            for i, text in enumerate(texts):
                if text:  # Пропускаем пустые строки
                    cache_key = cache.get_cache_key(text, prefix="bertweet")
                    if cache.exists(cache_key):
                        try:
                            embeddings.append((i, cache.load(cache_key)))
                        except Exception as e:
                            logger.warning(f"Ошибка при загрузке кеша для текста {i}: {str(e)}")
                            texts_to_process.append(text)
                            indices_to_process.append(i)
                    else:
                        texts_to_process.append(text)
                        indices_to_process.append(i)
                else:
                    # Для пустых строк используем нулевой вектор
                    embeddings.append((i, np.zeros(768)))  # BERTweet возвращает вектор размерности 768
        else:
            texts_to_process = texts
            indices_to_process = list(range(len(texts)))

        # Обрабатываем тексты, которых нет в кеше
        if texts_to_process:
            logger.debug(f"Извлечение BERT эмбеддингов для {len(texts_to_process)} текстов")

            # Токенизируем тексты
            inputs = self.tokenizer(
                texts_to_process,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)

            # Извлекаем эмбеддинги
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Используем [CLS] токен как эмбеддинг всего предложения
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # Добавляем новые эмбеддинги и обновляем кеш
            for i, idx in enumerate(indices_to_process):
                embeddings.append((idx, cls_embeddings[i]))

                if use_cache and self.config.use_cache and texts_to_process[i]:
                    cache_key = cache.get_cache_key(texts_to_process[i], prefix="bertweet")
                    cache.save(cache_key, cls_embeddings[i])

        # Сортируем эмбеддинги по индексам и возвращаем только значения
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])

    def process_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Обрабатывает большой список текстов батчами.

        Args:
            texts (List[str]): Список текстов для извлечения эмбеддингов.
            batch_size (int, optional): Размер батча. По умолчанию используется
                размер из конфигурации.

        Returns:
            np.ndarray: Массив эмбеддингов.
        """
        if not texts:
            return np.array([])

        batch_size = batch_size or default_config.batch_size

        # Заменяем None и пустые строки на ""
        processed_texts = [text if text else "" for text in texts]

        # Обрабатываем тексты батчами
        all_embeddings = []
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i + batch_size]
            batch_embeddings = self.get_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])


class CLIPEmbedder:
    """
    Класс для извлечения эмбеддингов из изображений с использованием CLIP.
    """

    def __init__(
            self,
            model_name: str = "openai/clip-vit-large-patch14",
            device: Optional[str] = None,
            config: Optional[FeatureConfig] = None
    ):
        """
        Инициализирует модель CLIP для извлечения эмбеддингов.

        Args:
            model_name (str): Название модели Hugging Face.
            device (str, optional): Устройство для вычислений ('cpu' или 'cuda').
                По умолчанию используется устройство из конфигурации.
        """
        self.config = config or default_config
        self.model_name = model_name
        self.device = device or self.config.device

        logger.info(f"Инициализация CLIP эмбеддера ({model_name}) на устройстве {self.device}")

        # Загружаем процессор и модель
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Переводим модель в режим оценки

    def load_image(self, image_url: str) -> Optional[Image.Image]:
        """
        Загружает изображение по URL.

        Args:
            image_url (str): URL изображения.

        Returns:
            Optional[Image.Image]: Загруженное изображение или None в случае ошибки.
        """
        if self.config.use_cache:
            cache_key = cache.get_cache_key(image_url, prefix="image_download")
            if cache.exists(cache_key):
                try:
                    # Загружаем изображение из кеша
                    image_data = cache.load(cache_key)
                    return Image.open(BytesIO(image_data)).convert('RGB')
                except Exception as e:
                    logger.warning(f"Ошибка при загрузке изображения из кеша {image_url}: {str(e)}")

        try:
            # Скачиваем изображение
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image_data = response.content

            # Сохраняем в кеш
            if self.config.use_cache:
                cache_key = cache.get_cache_key(image_url, prefix="image_download")
                cache.save(cache_key, image_data)
            return Image.open(BytesIO(image_data)).convert('RGB')
        except Exception as e:
            logger.warning(f"Ошибка при загрузке изображения {image_url}: {str(e)}")
            return None

    def get_embeddings(self, image_urls: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Извлекает эмбеддинги из списка URL-адресов изображений.

        Args:
            image_urls (List[str]): Список URL-адресов изображений.
            use_cache (bool): Использовать ли кеширование.

        Returns:
            np.ndarray: Массив эмбеддингов размера (len(image_urls), embedding_dim).
        """
        if not image_urls:
            return np.array([])

        # Проверяем кеш для каждого URL
        embeddings = []
        urls_to_process = []
        indices_to_process = []

        if use_cache and self.config.use_cache:
            for i, url in enumerate(image_urls):
                if url:  # Пропускаем пустые URL
                    cache_key = cache.get_cache_key(url, prefix="clip")
                    if cache.exists(cache_key):
                        try:
                            embeddings.append((i, cache.load(cache_key)))
                        except Exception as e:
                            logger.warning(f"Ошибка при загрузке кеша для URL {i}: {str(e)}")
                            urls_to_process.append(url)
                            indices_to_process.append(i)
                    else:
                        urls_to_process.append(url)
                        indices_to_process.append(i)
                else:
                    # Для пустых URL используем нулевой вектор
                    # CLIP возвращает вектор размерности 768 для vit-large-patch14
                    embeddings.append((i, np.zeros(768)))
        else:
            urls_to_process = [url for url in image_urls if url]
            indices_to_process = [i for i, url in enumerate(image_urls) if url]

        # Обрабатываем URL, которых нет в кеше
        if urls_to_process:
            logger.debug(f"Извлечение CLIP эмбеддингов для {len(urls_to_process)} изображений")

            # Загружаем изображения
            images = [self.load_image(url) for url in urls_to_process]
            valid_images = [(i, img) for i, img in enumerate(images) if img is not None]

            if valid_images:
                valid_indices = [indices_to_process[i] for i, _ in valid_images]
                valid_imgs = [img for _, img in valid_images]

                # Предобрабатываем изображения
                inputs = self.processor(
                    images=valid_imgs,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                # Извлекаем эмбеддинги
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)

                # Нормализуем эмбеддинги
                image_embeddings = outputs.cpu().numpy()

                # Добавляем новые эмбеддинги и обновляем кеш
                for i, idx in enumerate(valid_indices):
                    embeddings.append((idx, image_embeddings[i]))

                    if use_cache and self.config.use_cache:
                        cache_key = cache.get_cache_key(urls_to_process[valid_images[i][0]], prefix="clip")
                        cache.save(cache_key, image_embeddings[i])

            # Для недействительных изображений используем нулевые векторы
            invalid_indices = set(indices_to_process) - set(valid_indices) if valid_images else set(indices_to_process)
            for idx in invalid_indices:
                embeddings.append((idx, np.zeros(768)))  # CLIP возвращает вектор размерности 768

        # Сортируем эмбеддинги по индексам и возвращаем только значения
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])

    def process_batch(self, image_urls: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Обрабатывает большой список URL-адресов изображений батчами.

        Args:
            image_urls (List[str]): Список URL-адресов изображений.
            batch_size (int, optional): Размер батча. По умолчанию используется
                размер из конфигурации.

        Returns:
            np.ndarray: Массив эмбеддингов.
        """
        if not image_urls:
            return np.array([])

        batch_size = batch_size or default_config.batch_size

        # Заменяем None на ""
        processed_urls = [url if url else "" for url in image_urls]

        # Обрабатываем URL батчами
        all_embeddings = []
        for i in range(0, len(processed_urls), batch_size):
            batch_urls = processed_urls[i:i + batch_size]
            batch_embeddings = self.get_embeddings(batch_urls)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])


# Глобальные экземпляры эмбеддеров (синглтоны)
bertweet_embedder = None
clip_embedder = None


def get_bertweet_embedder(config: Optional[FeatureConfig] = None) -> BertEmbedder:
    """
    Получает глобальный экземпляр BertEmbedder (синглтон).

    Args:
        config (FeatureConfig, optional): Пользовательская конфигурация.

    Returns:
        BertEmbedder: Экземпляр BertEmbedder.
    """
    global bertweet_embedder
    if bertweet_embedder is None:
        bertweet_embedder = BertEmbedder(config=config)
    return bertweet_embedder


def get_clip_embedder(config: Optional[FeatureConfig] = None) -> CLIPEmbedder:
    """
    Получает глобальный экземпляр CLIPEmbedder (синглтон).

    Args:
        config (FeatureConfig, optional): Пользовательская конфигурация.

    Returns:
        CLIPEmbedder: Экземпляр CLIPEmbedder.
    """
    global clip_embedder
    if clip_embedder is None:
        clip_embedder = CLIPEmbedder(config=config)
    return clip_embedder