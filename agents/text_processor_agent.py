# agents/text_processor_agent.py
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from typing import List, Set


class TextProcessorAgent:
    def __init__(self):
        # Initialize NLP components
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')

        self.stop_words = set(stopwords.words('english'))
        self.custom_stops = {'www', 'http', 'https', 'com', 'org'}
        self.stop_words.update(self.custom_stops)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        return word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens"""
        return [token for token in tokens if token not in self.stop_words]

    def lemmatize(self, text: str) -> List[str]:
        """Lemmatize text using spaCy"""
        doc = self.nlp(text)
        return [token.lemma_ for token in doc
                if not token.is_stop and token.is_alpha]

    def process_text(self, text: str) -> str:
        """Complete text processing pipeline"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        lemmatized = self.lemmatize(' '.join(tokens))
        return ' '.join(lemmatized)

    def process_title(self, title: str) -> str:
        """Special processing for titles"""
        # Remove common title patterns
        title = re.sub(r'\.md$', '', title)  # Remove .md extension
        title = re.sub(r'^\d+[\s-]+', '', title)  # Remove leading numbers
        return self.process_text(title)

    def get_significant_phrases(self, text: str, n: int = 3) -> List[str]:
        """Extract significant phrases using spaCy"""
        doc = self.nlp(text)
        phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= n:
                phrases.append(chunk.text)
        return phrases