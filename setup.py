"""
Скрипт установки пакета tweet_features.
"""
from setuptools import setup, find_packages


setup(
    name="tweet_features",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "transformers>=4.20.0",
        "emoji>=2.0.0",
        "requests>=2.25.0",
        "pillow>=8.0.0",
    ],
    author="Shcheglov Stas",
    description="Пакет для извлечения признаков из данных Twitter",
    url="https://github.com/shcheglov13/tweet_features",
    python_requires=">=3.10",
)