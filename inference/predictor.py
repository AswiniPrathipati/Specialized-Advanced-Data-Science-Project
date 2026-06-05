"""
Inference Engine for Sentiment Analysis
Provides single and batch prediction capabilities
"""

import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Union
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_processing.preprocessor import TextPreprocessor, SimpleTokenizer
from models.lstm_model import SentimentLSTMModel

logger = logging.getLogger(__name__)

LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
EMOJI_MAP = {0: '😠', 1: '😐', 2: '😊'}


class SentimentPredictor:
    """High-level inference class that wraps model + preprocessing."""

    def __init__(self, model_dir: str = 'src/models/saved'):
        self.model_dir = model_dir
        self.preprocessor = TextPreprocessor(remove_stopwords=True)
        self.tokenizer = SimpleTokenizer.load(f'{model_dir}/tokenizer.json')
        self.model = SentimentLSTMModel.load(f'{model_dir}/best_model.pkl')
        self._call_count = 0
        self._total_latency = 0.0
        logger.info("SentimentPredictor ready.")

    def predict_one(self, text: str) -> dict:
        """Predict sentiment for a single text string."""
        t0 = time.time()
        clean = self.preprocessor.clean_text(text)
        seq = self.tokenizer.texts_to_sequences([clean])
        _, probs = self.model.predict(seq)
        probs = probs[0]
        pred_class = int(np.argmax(probs))
        latency_ms = (time.time() - t0) * 1000

        self._call_count += 1
        self._total_latency += latency_ms

        return {
            'text': text,
            'cleaned_text': clean,
            'sentiment': LABEL_MAP[pred_class],
            'sentiment_id': pred_class,
            'emoji': EMOJI_MAP[pred_class],
            'confidence': float(probs[pred_class]),
            'probabilities': {
                'Negative': float(probs[0]),
                'Neutral': float(probs[1]),
                'Positive': float(probs[2]),
            },
            'latency_ms': round(latency_ms, 2),
        }

    def predict_batch(self, texts: list) -> list:
        """Predict sentiment for a list of text strings."""
        t0 = time.time()
        cleans = [self.preprocessor.clean_text(t) for t in texts]
        seqs = self.tokenizer.texts_to_sequences(cleans)
        _, probs = self.model.predict(seqs)
        total_ms = (time.time() - t0) * 1000

        results = []
        for i, (text, clean) in enumerate(zip(texts, cleans)):
            p = probs[i]
            pred_class = int(np.argmax(p))
            results.append({
                'text': text,
                'sentiment': LABEL_MAP[pred_class],
                'sentiment_id': pred_class,
                'emoji': EMOJI_MAP[pred_class],
                'confidence': float(p[pred_class]),
                'probabilities': {
                    'Negative': float(p[0]),
                    'Neutral': float(p[1]),
                    'Positive': float(p[2]),
                },
            })
        logger.info(f"Batch of {len(texts)} processed in {total_ms:.1f}ms")
        return results

    def stats(self) -> dict:
        """Return runtime performance statistics."""
        avg_lat = self._total_latency / self._call_count if self._call_count else 0
        return {
            'total_predictions': self._call_count,
            'avg_latency_ms': round(avg_lat, 2),
            'total_latency_ms': round(self._total_latency, 2),
        }


if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent.parent.parent)

    predictor = SentimentPredictor()

    test_texts = [
        "Great product! Highly recommend!",
        "Average product, nothing special",
        "Worst purchase of my life",
        "It works but could be better",
        "Absolutely fantastic, exceeded all my expectations!",
    ]

    print("\n" + "="*60)
    print("  SAMPLE PREDICTIONS")
    print("="*60)
    for text in test_texts:
        result = predictor.predict_one(text)
        print(f"\n  Text      : {result['text']}")
        print(f"  Sentiment : {result['emoji']} {result['sentiment']} ({result['confidence']*100:.1f}% confidence)")
        print(f"  Scores    : Neg={result['probabilities']['Negative']:.3f} | "
              f"Neu={result['probabilities']['Neutral']:.3f} | "
              f"Pos={result['probabilities']['Positive']:.3f}")
        print(f"  Latency   : {result['latency_ms']}ms")
    print("="*60)
    print("\nPredictor stats:", predictor.stats())
