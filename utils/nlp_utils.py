# utils/nlp_utils.py
import re
from typing import List, Set
import numpy as np

class NLPUtils:
    @staticmethod
    def clean_filename(filename: str) -> str:
        """Clean and normalize filename"""
        # Remove file extension
        filename = re.sub(r'\.[^.]+$', '', filename)
        # Remove leading numbers and special characters
        filename = re.sub(r'^\d+[-_]?\s*', '', filename)
        # Replace special characters with spaces
        filename = re.sub(r'[-_]', ' ', filename)
        return filename.strip()

    @staticmethod
    def extract_ngrams(text: str, n: int) -> List[str]:
        """Extract n-grams from text"""
        words = text.split()
        return [' '.join(words[i:i+n])
                for i in range(len(words)-n+1)]

    @staticmethod
    def calculate_similarity(vec1: np.ndarray,
                           vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(vec1, vec2) / (norm1 * norm2)