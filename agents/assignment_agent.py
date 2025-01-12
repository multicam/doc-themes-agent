# agents/assignment_agent.py
from collections import defaultdict
from typing import List, Dict
import numpy as np
from models.article import Article
from models.theme import Theme


class AssignmentAgent:
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold  # Minimum score threshold for assignment

    def assign_articles_to_themes(self,
                                  W: np.ndarray,
                                  articles: List[Article]) -> Dict[int, List[Dict]]:
        """Assign articles to themes based on NMF weights"""
        assignments = defaultdict(list)

        for doc_idx, doc_themes in enumerate(W):
            # Get theme contributions
            theme_scores = [(i, score)
                            for i, score in enumerate(doc_themes)]

            # Sort by score
            theme_scores.sort(key=lambda x: x[1], reverse=True)
            max_score = theme_scores[0][1]

            # Assign to themes above threshold
            for theme_idx, score in theme_scores:
                if score >= max_score * self.threshold:
                    assignments[theme_idx].append({
                        'id': articles[doc_idx].id,
                        'path': articles[doc_idx].path,
                        'score': float(score),
                        'explanation': self._generate_assignment_explanation(
                            articles[doc_idx], score, max_score
                        )
                    })

        return dict(assignments)

    def _generate_assignment_explanation(self,
                                         article: Article,
                                         score: float,
                                         max_score: float) -> str:
        """Generate explanation for why article was assigned to theme"""
        confidence = (score / max_score) * 100

        if confidence > 90:
            strength = "very strong"
        elif confidence > 70:
            strength = "strong"
        elif confidence > 50:
            strength = "moderate"
        else:
            strength = "weak"

        return (f"Shows {strength} thematic alignment "
                f"(confidence: {confidence:.1f}%)")

    def get_article_theme_distribution(self,
                                       article_id: int,
                                       W: np.ndarray,
                                       themes: List[Theme]) -> List[Dict]:
        """Get theme distribution for a specific article"""
        doc_idx = None
        for i, theme in enumerate(themes):
            for article in theme.articles:
                if article['id'] == article_id:
                    doc_idx = i
                    break
            if doc_idx is not None:
                break

        if doc_idx is None:
            return []

        theme_weights = W[doc_idx]
        distribution = []

        for theme_idx, weight in enumerate(theme_weights):
            if weight > 0:
                distribution.append({
                    'theme_id': theme_idx,
                    'weight': float(weight),
                    'percentage': float(weight / sum(theme_weights) * 100)
                })

        return sorted(distribution, key=lambda x: x['weight'], reverse=True)