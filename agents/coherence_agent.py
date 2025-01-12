# agents/coherence_agent.py
import numpy as np
from scipy.spatial.distance import cosine
from typing import List, Dict
from models.theme import Theme
import spacy


class CoherenceAgent:
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')

    def calculate_term_coherence(self, terms: List[str]) -> float:
        """Calculate semantic coherence between terms using word embeddings"""
        coherence = 0
        count = 0

        docs = [self.nlp(term) for term in terms]

        for i, doc1 in enumerate(docs):
            for doc2 in docs[i + 1:]:
                if doc1.has_vector and doc2.has_vector:
                    similarity = doc1.similarity(doc2)
                    coherence += similarity
                    count += 1

        return coherence / count if count > 0 else 0

    def calculate_theme_coherence(self, theme: Theme) -> float:
        """Calculate overall theme coherence"""
        # Get top terms
        top_terms = [term for term, _ in theme.terms[:10]]

        # Term coherence
        term_coherence = self.calculate_term_coherence(top_terms)

        # Article assignment coherence
        assignment_coherence = np.mean([article['score']
                                        for article in theme.articles]) if theme.articles else 0

        # Combine scores
        return 0.7 * term_coherence + 0.3 * assignment_coherence

    def identify_overlapping_themes(self, themes: List[Theme],
                                    threshold: float = 0.8) -> List[tuple]:
        """Identify themes that might be too similar"""
        overlaps = []

        for i, theme1 in enumerate(themes):
            for j, theme2 in enumerate(themes[i + 1:], i + 1):
                similarity = 1 - cosine(theme1.vector, theme2.vector)
                if similarity > threshold:
                    overlaps.append((i, j, similarity))

        return overlaps