# config/settings.py
from typing import Dict, Any

class Settings:
    # Feature extraction settings
    FEATURE_EXTRACTION = {
        'max_features': 1000,
        'min_df': 2,
        'max_df': 0.95,
        'ngram_range': (1, 2)
    }

    # Theme discovery settings
    THEME_DISCOVERY = {
        'min_coherence': 0.1,
        'assignment_threshold': 0.2,
        'min_theme_size': 3
    }

    # Visualization settings
    VISUALIZATION = {
        'theme_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        'node_size_factor': 100,
        'edge_width_factor': 2
    }

    # NLP settings
    NLP = {
        'min_keyword_freq': 3,
        'min_relationship_strength': 0.3,
        'custom_stopwords': {'www', 'http', 'https', 'com', 'org'}
    }

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get all settings as dictionary"""
        return {
            'feature_extraction': cls.FEATURE_EXTRACTION,
            'theme_discovery': cls.THEME_DISCOVERY,
            'visualization': cls.VISUALIZATION,
            'nlp': cls.NLP
        }

    @classmethod
    def update(cls, settings: Dict[str, Any]) -> None:
        """Update settings"""
        for category, values in settings.items():
            if hasattr(cls, category.upper()):
                current = getattr(cls, category.upper())
                current.update(values)