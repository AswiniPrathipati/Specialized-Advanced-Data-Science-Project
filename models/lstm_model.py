"""
Neural Network Model Architectures for Sentiment Analysis
Implements Bidirectional LSTM and simple Dense classifier
"""

import numpy as np
import json
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def relu(x):
    return np.maximum(0, x)


class EmbeddingLayer:
    """Simple trainable embedding lookup."""

    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        scale = np.sqrt(2.0 / embed_dim)
        self.weights = np.random.randn(vocab_size, embed_dim).astype(np.float32) * scale

    def forward(self, indices: np.ndarray) -> np.ndarray:
        """Shape: (batch, seq_len) -> (batch, seq_len, embed_dim)"""
        return self.weights[indices]


class SimpleLSTMCell:
    """Minimal LSTM cell for inference."""

    def __init__(self, input_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Input gate, forget gate, cell gate, output gate
        n = hidden_dim
        d = input_dim + hidden_dim
        self.W = np.random.randn(d, 4 * n).astype(np.float32) * 0.01
        self.b = np.zeros(4 * n, dtype=np.float32)
        self.b[n:2*n] = 1.0  # Bias forget gate to 1

    def step(self, x, h_prev, c_prev):
        combined = np.concatenate([x, h_prev], axis=-1)
        gates = combined @ self.W + self.b
        n = self.hidden_dim
        i = sigmoid(gates[..., :n])
        f = sigmoid(gates[..., n:2*n])
        g = np.tanh(gates[..., 2*n:3*n])
        o = sigmoid(gates[..., 3*n:])
        c = f * c_prev + i * g
        h = o * np.tanh(c)
        return h, c

    def forward(self, x_seq: np.ndarray, return_sequences: bool = False):
        """x_seq: (batch, seq_len, input_dim)"""
        batch, seq_len, _ = x_seq.shape
        h = np.zeros((batch, self.hidden_dim), dtype=np.float32)
        c = np.zeros((batch, self.hidden_dim), dtype=np.float32)
        hs = []
        for t in range(seq_len):
            h, c = self.step(x_seq[:, t, :], h, c)
            hs.append(h)
        if return_sequences:
            return np.stack(hs, axis=1)
        return h


class DenseLayer:
    """Fully connected layer with optional activation."""

    def __init__(self, input_dim: int, output_dim: int, activation: str = 'relu'):
        scale = np.sqrt(2.0 / input_dim)
        self.W = np.random.randn(input_dim, output_dim).astype(np.float32) * scale
        self.b = np.zeros(output_dim, dtype=np.float32)
        self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.W + self.b
        if self.activation == 'relu':
            return relu(out)
        elif self.activation == 'softmax':
            return softmax(out)
        elif self.activation == 'sigmoid':
            return sigmoid(out)
        return out


class SentimentLSTMModel:
    """
    Bidirectional LSTM model for 3-class sentiment classification.
    Architecture:
      Embedding -> BiLSTM(64) -> LSTM(32) -> Dense(24, relu) -> Dense(3, softmax)
    """

    def __init__(self, vocab_size: int = 10000, embed_dim: int = 64,
                 lstm1_units: int = 64, lstm2_units: int = 32,
                 dense_units: int = 24, num_classes: int = 3,
                 max_length: int = 100):
        self.config = {
            'vocab_size': vocab_size, 'embed_dim': embed_dim,
            'lstm1_units': lstm1_units, 'lstm2_units': lstm2_units,
            'dense_units': dense_units, 'num_classes': num_classes,
            'max_length': max_length
        }
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        # Bidirectional LSTM: forward + backward
        self.lstm1_fwd = SimpleLSTMCell(embed_dim, lstm1_units)
        self.lstm1_bwd = SimpleLSTMCell(embed_dim, lstm1_units)
        # Second LSTM takes concatenated bi-directional output
        self.lstm2 = SimpleLSTMCell(lstm1_units * 2, lstm2_units)
        self.dense1 = DenseLayer(lstm2_units, dense_units, activation='relu')
        self.dense2 = DenseLayer(dense_units, num_classes, activation='softmax')
        self._trained = False

    def forward(self, sequences: np.ndarray) -> np.ndarray:
        """sequences: (batch, seq_len) -> probabilities: (batch, num_classes)"""
        embedded = self.embedding.forward(sequences)           # (B, T, E)
        fwd = self.lstm1_fwd.forward(embedded, return_sequences=True)    # (B, T, H)
        bwd = self.lstm1_bwd.forward(embedded[:, ::-1, :], return_sequences=True)  # reverse
        bi_out = np.concatenate([fwd, bwd[:, ::-1, :]], axis=-1)          # (B, T, 2H)
        lstm2_out = self.lstm2.forward(bi_out)                 # (B, H2)
        d1 = self.dense1.forward(lstm2_out)
        probs = self.dense2.forward(d1)
        return probs

    def predict(self, sequences: np.ndarray) -> tuple:
        """Returns (predicted_classes, probabilities)."""
        probs = self.forward(sequences)
        classes = np.argmax(probs, axis=-1)
        return classes, probs

    def save(self, path: str):
        data = {
            'config': self.config,
            'embedding_weights': self.embedding.weights,
            'lstm1_fwd': (self.lstm1_fwd.W, self.lstm1_fwd.b),
            'lstm1_bwd': (self.lstm1_bwd.W, self.lstm1_bwd.b),
            'lstm2': (self.lstm2.W, self.lstm2.b),
            'dense1': (self.dense1.W, self.dense1.b),
            'dense2': (self.dense2.W, self.dense2.b),
            'trained': self._trained,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'SentimentLSTMModel':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(**data['config'])
        model.embedding.weights = data['embedding_weights']
        model.lstm1_fwd.W, model.lstm1_fwd.b = data['lstm1_fwd']
        model.lstm1_bwd.W, model.lstm1_bwd.b = data['lstm1_bwd']
        model.lstm2.W, model.lstm2.b = data['lstm2']
        model.dense1.W, model.dense1.b = data['dense1']
        model.dense2.W, model.dense2.b = data['dense2']
        model._trained = data.get('trained', False)
        logger.info(f"Model loaded from {path}")
        return model

    def summary(self):
        c = self.config
        total_params = (
            c['vocab_size'] * c['embed_dim'] +
            (c['embed_dim'] + c['lstm1_units']) * 4 * c['lstm1_units'] * 2 +
            (c['lstm1_units'] * 2 + c['lstm2_units']) * 4 * c['lstm2_units'] +
            c['lstm2_units'] * c['dense_units'] + c['dense_units'] +
            c['dense_units'] * c['num_classes'] + c['num_classes']
        )
        print("=" * 55)
        print("  SENTIMENT LSTM MODEL ARCHITECTURE")
        print("=" * 55)
        print(f"  Embedding:        {c['vocab_size']} × {c['embed_dim']}")
        print(f"  BiLSTM Layer 1:   {c['lstm1_units']} units × 2 (fwd+bwd)")
        print(f"  LSTM Layer 2:     {c['lstm2_units']} units")
        print(f"  Dense Layer:      {c['dense_units']} units (ReLU)")
        print(f"  Output Layer:     {c['num_classes']} classes (Softmax)")
        print(f"  Total Params:     {total_params:,}")
        print("=" * 55)
