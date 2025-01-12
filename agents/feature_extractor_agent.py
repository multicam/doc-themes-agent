# agents/feature_extractor_agent.py
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Dict, Tuple
from models.article import Article
from models.feature_matrix import FeatureMatrix


class FeatureExtractorAgent:
    def __init__(self, max_features: int = 1000, min_df: int = 2, max_df: float = 0.95):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.title_vectorizer = None
        self.keyword_to_idx = None

    def extract_title_features(self, articles: List[Article]) -> FeatureMatrix:
        """Extract TF-IDF features from article titles"""
        # Initialize vectorizer
        self.title_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=(1, 2)
        )

        # Extract features
        titles = [article.title or article.filename for article in articles]
        title_features = self.title_vectorizer.fit_transform(titles)

        return FeatureMatrix(
            matrix=title_features.toarray(),
            feature_names=self.title_vectorizer.get_feature_names_out(),
            article_indices=[article.id for article in articles]
        )

    def extract_keyword_features(self, articles: List[Article]) -> FeatureMatrix:
        """Create feature matrix from keywords and weights"""
        # Collect all unique keywords
        all_keywords = set()
        for article in articles:
            all_keywords.update(kw[0].lower() for kw in article.keywords)

        # Create mapping
        self.keyword_to_idx = {kw: i for i, kw in enumerate(sorted(all_keywords))}

        # Create feature matrix
        keyword_features = np.zeros((len(articles), len(all_keywords)))

        for i, article in enumerate(articles):
            for kw, weight in article.keywords:
                kw_lower = kw.lower()
                if kw_lower in self.keyword_to_idx:
                    keyword_features[i, self.keyword_to_idx[kw_lower]] = weight

        return FeatureMatrix(
            matrix=keyword_features,
            feature_names=list(self.keyword_to_idx.keys()),
            article_indices=[article.id for article in articles]
        )

    def combine_features(self,
                         title_features: FeatureMatrix,
                         keyword_features: FeatureMatrix) -> FeatureMatrix:
        """Combine title and keyword features"""
        assert (title_features.article_indices ==
                keyword_features.article_indices), "Article indices must match"

        combined_matrix = np.hstack([
            title_features.matrix,
            keyword_features.matrix
        ])

        combined_features = list(title_features.feature_names) + \
                            list(keyword_features.feature_names)

        return FeatureMatrix(
            matrix=combined_matrix,
            feature_names=combined_features,
            article_indices=title_features.article_indices
        )