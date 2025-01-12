# models/feature_matrix.py
from dataclasses import dataclass
import numpy as np
# from scipy.sparse import csr_matrix
from typing import List, Dict


@dataclass
class FeatureMatrix:
    matrix: np.ndarray
    feature_names: List[str]
    article_indices: List[int]

    def get_feature_vector(self, article_idx: int) -> np.ndarray:
        return self.matrix[article_idx]

    def get_feature_importance(self, feature_idx: int) -> np.ndarray:
        return self.matrix[:, feature_idx]