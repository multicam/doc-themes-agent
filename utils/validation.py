# utils/validation.py
from typing import List, Dict, Any
from models.article import Article


class ValidationUtils:
    @staticmethod
    def validate_articles(articles: List[Article]) -> bool:
        """Validate article data structure"""
        if not articles:
            raise ValueError("Empty article list")

        for article in articles:
            if not article.id or not article.path:
                raise ValueError(f"Invalid article data: {article}")
            if not article.keywords:
                raise ValueError(f"Article {article.id} has no keywords")

        return True

    @staticmethod
    def validate_keywords(keywords: List[tuple]) -> bool:
        """Validate keyword format"""
        if not keywords:
            raise ValueError("Empty keywords list")

        for kw in keywords:
            if not isinstance(kw, tuple) or len(kw) != 2:
                raise ValueError(f"Invalid keyword format: {kw}")
            if not isinstance(kw[1], (int, float)):
                raise ValueError(f"Invalid keyword weight: {kw}")

        return True

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration settings"""
        required_keys = ['max_features', 'min_df', 'max_df',
                         'min_coherence', 'assignment_threshold']

        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing configuration key: {key}")

        return True