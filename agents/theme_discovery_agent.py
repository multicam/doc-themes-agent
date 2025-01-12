# agents/theme_discovery_agent.py
from sklearn.decomposition import NMF
import numpy as np
from typing import List, Dict, Tuple
from models.theme import Theme
from models.feature_matrix import FeatureMatrix


class ThemeDiscoveryAgent:
    def __init__(self,
                 n_themes: int = None,
                 min_theme_coherence: float = 0.1):
        self.n_themes = n_themes
        self.min_theme_coherence = min_theme_coherence
        self.model = None
        self.W = None  # Document-theme matrix
        self.H = None  # Theme-feature matrix

    def determine_n_themes(self, feature_matrix: FeatureMatrix) -> int:
        """Determine optimal number of themes if not specified"""
        if self.n_themes is not None:
            return self.n_themes

        # Use twice the square root of the number of articles as a heuristic
        n = int(np.sqrt(feature_matrix.matrix.shape[0]) * 2)
        return min(n, feature_matrix.matrix.shape[1])

    def discover_themes(self,
                        feature_matrix: FeatureMatrix) -> Tuple[np.ndarray, np.ndarray]:
        """Discover themes using NMF"""
        n_themes = self.determine_n_themes(feature_matrix)

        self.model = NMF(
            n_components=n_themes,
            random_state=42,
            init='nndsvd'
        )

        self.W = self.model.fit_transform(feature_matrix.matrix)  # Document-theme matrix
        self.H = self.model.components_  # Theme-feature matrix

        return self.W, self.H

    def get_theme_terms(self,
                        theme_idx: int,
                        feature_names: List[str],
                        n_terms: int = 10) -> List[Tuple[str, float]]:
        """Get top terms for a theme"""
        theme_vector = self.H[theme_idx]
        top_indices = theme_vector.argsort()[:-n_terms - 1:-1]

        return [(feature_names[i], theme_vector[i])
                for i in top_indices]

    def create_themes(self,
                      feature_matrix: FeatureMatrix,
                      article_assignments: Dict) -> List[Theme]:
        """Create Theme objects from discovered themes"""
        themes = []

        for theme_idx in range(self.H.shape[0]):
            terms = self.get_theme_terms(
                theme_idx,
                feature_matrix.feature_names
            )

            theme = Theme(
                id=theme_idx,
                terms=terms,
                articles=article_assignments.get(theme_idx, []),
                coherence=0.0,  # Will be set by CoherenceAgent
                vector=self.H[theme_idx]
            )

            themes.append(theme)

        return themes

    def get_document_theme_weights(self, doc_idx: int) -> List[Tuple[int, float]]:
        """Get theme weights for a document"""
        doc_weights = self.W[doc_idx]
        theme_weights = [(i, weight) for i, weight in enumerate(doc_weights)]
        return sorted(theme_weights, key=lambda x: x[1], reverse=True)