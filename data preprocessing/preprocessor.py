"""
Data Preprocessing Pipeline for NLP Sentiment Analysis
Handles text cleaning, tokenization, and sequence preparation
"""

import re
import string
import numpy as np
import pandas as pd
from collections import Counter
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Handles all text cleaning and normalization tasks."""

    STOPWORDS = {
        'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to', 'for',
        'of', 'and', 'or', 'but', 'not', 'this', 'that', 'with', 'was',
        'are', 'be', 'been', 'by', 'from', 'as', 'so', 'if', 'its'
    }

    def __init__(self, remove_stopwords: bool = False, lowercase: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase

    def clean_text(self, text: str) -> str:
        """Apply full cleaning pipeline to a text string."""
        if not isinstance(text, str):
            return ""
        # Lowercase
        if self.lowercase:
            text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters, keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Optionally remove stopwords
        if self.remove_stopwords:
            tokens = text.split()
            tokens = [t for t in tokens if t not in self.STOPWORDS]
            text = ' '.join(tokens)
        return text

    def process_dataframe(self, df: pd.DataFrame, text_col: str = 'review') -> pd.DataFrame:
        """Clean all text in a DataFrame column."""
        df = df.copy()
        df['cleaned_text'] = df[text_col].apply(self.clean_text)
        df['text_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))
        logger.info(f"Preprocessed {len(df)} rows. Avg length: {df['text_length'].mean():.1f} tokens")
        return df


class SimpleTokenizer:
    """Word-level tokenizer with vocabulary management."""

    PAD_TOKEN = '<PAD>'
    OOV_TOKEN = '<OOV>'

    def __init__(self, max_vocab: int = 10000, max_length: int = 100):
        self.max_vocab = max_vocab
        self.max_length = max_length
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self._fitted = False

    def fit(self, texts: list):
        """Build vocabulary from training texts."""
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        # Reserve 0 for PAD, 1 for OOV
        self.word2idx = {self.PAD_TOKEN: 0, self.OOV_TOKEN: 1}
        for idx, (word, _) in enumerate(counter.most_common(self.max_vocab - 2), start=2):
            self.word2idx[word] = idx

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self._fitted = True
        logger.info(f"Vocabulary built: {self.vocab_size} words")

    def texts_to_sequences(self, texts: list) -> list:
        """Convert list of texts to padded integer sequences."""
        assert self._fitted, "Tokenizer must be fitted before encoding."
        sequences = []
        for text in texts:
            tokens = text.split()[:self.max_length]
            seq = [self.word2idx.get(tok, 1) for tok in tokens]  # 1 = OOV
            # Pad to max_length
            seq = seq + [0] * (self.max_length - len(seq))
            sequences.append(seq)
        return np.array(sequences, dtype=np.int32)

    def save(self, path: str):
        """Persist vocabulary to JSON."""
        with open(path, 'w') as f:
            json.dump({'word2idx': self.word2idx, 'max_length': self.max_length}, f)
        logger.info(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        """Load vocabulary from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        tok = cls()
        tok.word2idx = data['word2idx']
        tok.max_length = data['max_length']
        tok.idx2word = {v: k for k, v in tok.word2idx.items()}
        tok.vocab_size = len(tok.word2idx)
        tok._fitted = True
        return tok


def load_and_split(csv_path: str, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    """Load CSV and split into train/val/test sets."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['review', 'sentiment'])
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n = len(df)
    n_test = int(n * test_size)
    n_val = int(n * val_size)

    test_df = df[:n_test]
    val_df = df[n_test:n_test + n_val]
    train_df = df[n_test + n_val:]

    logger.info(f"Split — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df
