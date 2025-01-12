# setup.py
from setuptools import setup, find_packages

setup(
    name="article-theme-discovery",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.6.0',
        'scikit-learn>=0.24.0',
        'spacy>=3.0.0',
        'nltk>=3.6.0',
        'networkx>=2.5',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0'
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A system for discovering themes in article collections",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/article-theme-discovery",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)