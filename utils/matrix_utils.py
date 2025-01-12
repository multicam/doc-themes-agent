# utils/matrix_utils.py
import numpy as np
from scipy import sparse
from typing import Tuple

class MatrixUtils:
    @staticmethod
    def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        """Normalize matrix rows to unit length"""
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1
        return matrix / norms[:, np.newaxis]

    @staticmethod
    def combine_sparse_dense(sparse_matrix: sparse.csr_matrix,
                           dense_matrix: np.ndarray) -> np.ndarray:
        """Combine sparse and dense matrices"""
        return np.hstack([sparse_matrix.toarray(), dense_matrix])

    @staticmethod
    def matrix_similarity(matrix1: np.ndarray,
                         matrix2: np.ndarray) -> np.ndarray:
        """Calculate similarity between two matrices"""
        return np.dot(matrix1, matrix2.T)