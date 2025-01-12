# models/theme.py
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import numpy as np


@dataclass
class Theme:
    id: int
    terms: List[Tuple[str, float]]  # List of (term, weight) tuples
    articles: List[Dict]  # List of article assignments
    coherence: float
    vector: np.ndarray = None  # Theme vector from NMF

    @property
    def top_terms(self) -> List[str]:
        return [term for term, _ in sorted(self.terms, key=lambda x: x[1], reverse=True)[:5]]

    @property
    def strength(self) -> float:
        return np.mean([article['score'] for article in self.articles])

    def to_dict(self):
        return {
            'id': self.id,
            'articles': self.articles,
            'strength': self.strength,
            'terms': self.terms,
            'coherence': self.coherence,
            'vector': self.vector.tolist(),  # Convert numpy array to list
            'top_terms': self.top_terms
        }
