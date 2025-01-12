# models/article.py
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Article:
    id: int
    path: str
    keywords: List[Tuple[str, float]]  # List of (keyword, weight) tuples
    title: str = None
    content: str = None

    @property
    def filename(self) -> str:
        return self.path.split('/')[-1]

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'path': self.path,
            'keywords': self.keywords,
            'title': self.title,
            'content': self.content
        }