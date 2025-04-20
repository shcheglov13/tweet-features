"""
Модуль для снижения размерности эмбеддингов.
"""
import numpy as np
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod

from tweet_features.utils.logger import setup_logger

logger = setup_logger('tweet_features.utils.dimensionality_reduction')


class DimensionalityReducer(ABC):
    """
    Абстрактный класс для снижения размерности.
    """

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """
        Обучает модель снижения размерности.

        Args:
            data (np.ndarray): Данные для обучения.
        """
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Применяет снижение размерности к данным.

        Args:
            data (np.ndarray): Данные для преобразования.

        Returns:
            np.ndarray: Данные с пониженной размерностью.
        """
        pass

    @abstractmethod
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Обучает модель и применяет снижение размерности к данным.

        Args:
            data (np.ndarray): Данные для обучения и преобразования.

        Returns:
            np.ndarray: Данные с пониженной размерностью.
        """
        pass


class PCAReducer(DimensionalityReducer):
    """
    Класс для снижения размерности с использованием PCA.
    """

    def __init__(self, n_components: int, random_state: int = 42):
        """
        Инициализирует PCA для снижения размерности.

        Args:
            n_components (int): Количество компонент для сохранения.
            random_state (int): Начальное значение для генератора случайных чисел.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.is_fitted = False

    def fit(self, data: np.ndarray) -> None:
        """
        Обучает PCA на данных.

        Args:
            data (np.ndarray): Данные для обучения.
        """
        if data.size == 0 or len(data) == 0:
            logger.warning("Попытка обучить PCA на пустых данных")
            return

        # Проверка на наличие NaN значений
        if np.isnan(data).any():
            logger.warning("Данные содержат NaN значения. Они будут заменены нулями.")
            data = np.nan_to_num(data)

        self.pca.fit(data)
        self.is_fitted = True

        # Логируем объясненную дисперсию
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        logger.info(
            f"Обучен PCA с {self.n_components} компонентами. "
            f"Объясненная дисперсия: {cumulative_variance[-1]:.4f}"
        )

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Применяет PCA к данным.

        Args:
            data (np.ndarray): Данные для преобразования.

        Returns:
            np.ndarray: Данные с пониженной размерностью.

        Raises:
            ValueError: Если PCA не обучен.
        """
        if not self.is_fitted:
            raise ValueError("PCA необходимо сначала обучить с помощью метода fit()")

        if data.size == 0 or len(data) == 0:
            logger.warning("Попытка применить PCA к пустым данным")
            # Возвращаем пустой массив с нужной размерностью
            return np.zeros((0, self.n_components))

        # Проверка на наличие NaN значений
        if np.isnan(data).any():
            logger.warning("Данные содержат NaN значения. Они будут заменены нулями.")
            data = np.nan_to_num(data)

        return self.pca.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Обучает PCA и применяет его к данным.

        Args:
            data (np.ndarray): Данные для обучения и преобразования.

        Returns:
            np.ndarray: Данные с пониженной размерностью.
        """
        if data.size == 0 or len(data) == 0:
            logger.warning("Попытка применить PCA к пустым данным")
            # Возвращаем пустой массив с нужной размерностью
            return np.zeros((0, self.n_components))

        # Проверка на наличие NaN значений
        if np.isnan(data).any():
            logger.warning("Данные содержат NaN значения. Они будут заменены нулями.")
            data = np.nan_to_num(data)

        result = self.pca.fit_transform(data)
        self.is_fitted = True

        # Логируем объясненную дисперсию
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        logger.info(
            f"Обучен PCA с {self.n_components} компонентами. "
            f"Объясненная дисперсия: {cumulative_variance[-1]:.4f}"
        )

        return result


class NoReduction(DimensionalityReducer):
    """
    Класс-заглушка, который не выполняет снижение размерности.
    """

    def fit(self, data: np.ndarray) -> None:
        """
        Ничего не делает.

        Args:
            data (np.ndarray): Данные для обучения.
        """
        pass

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Возвращает данные без изменений.

        Args:
            data (np.ndarray): Данные для преобразования.

        Returns:
            np.ndarray: Исходные данные.
        """
        return data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Возвращает данные без изменений.

        Args:
            data (np.ndarray): Данные для обучения и преобразования.

        Returns:
            np.ndarray: Исходные данные.
        """
        return data


def get_reducer(method: str, n_components: int) -> DimensionalityReducer:
    """
    Фабричный метод для создания объекта снижения размерности.

    Args:
        method (str): Метод снижения размерности ('pca' или 'none').
        n_components (int): Количество компонент для сохранения.

    Returns:
        DimensionalityReducer: Объект для снижения размерности.

    Raises:
        ValueError: Если указан неизвестный метод.
    """
    if method.lower() == 'pca':
        return PCAReducer(n_components=n_components)
    elif method.lower() == 'none':
        return NoReduction()
    else:
        raise ValueError(f"Неизвестный метод снижения размерности: {method}")