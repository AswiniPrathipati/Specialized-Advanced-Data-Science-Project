"""
Training Pipeline for Sentiment Analysis Model
Implements mini-batch gradient descent with cross-entropy loss
"""

import numpy as np
import pandas as pd
import json
import logging
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.preprocessor import TextPreprocessor, SimpleTokenizer, load_and_split
from models.lstm_model import SentimentLSTMModel, softmax

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}


def cross_entropy_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean cross-entropy loss."""
    n = len(labels)
    correct_probs = probs[np.arange(n), labels]
    return -np.mean(np.log(correct_probs + 1e-9))


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return np.mean(np.argmax(probs, axis=-1) == labels)


def update_weights_sgd(layer_weights, grads, lr: float):
    """Simple SGD update."""
    for i, (w, g) in enumerate(zip(layer_weights, grads)):
        w -= lr * g


def train_epoch_simple(model: SentimentLSTMModel, X: np.ndarray,
                       y: np.ndarray, lr: float = 0.01, batch_size: int = 32):
    """
    Simplified training step using embedding-level gradient updates.
    For a pure-numpy model this approximates learning via output layer gradients.
    """
    n = len(X)
    indices = np.random.permutation(n)
    X, y = X[indices], y[indices]

    total_loss = 0.0
    total_correct = 0
    n_batches = 0

    for start in range(0, n, batch_size):
        xb = X[start:start + batch_size]
        yb = y[start:start + batch_size]
        batch_n = len(xb)

        probs = model.forward(xb)
        loss = cross_entropy_loss(probs, yb)
        total_loss += loss
        total_correct += np.sum(np.argmax(probs, axis=-1) == yb)
        n_batches += 1

        # Backprop through output layer (Dense2) only for simplicity
        # dL/dz = probs - one_hot
        dz = probs.copy()
        dz[np.arange(batch_n), yb] -= 1
        dz /= batch_n

        # Update dense2 weights
        d1_out = model.dense1.forward(
            model.lstm2.forward(
                np.concatenate([
                    model.lstm1_fwd.forward(model.embedding.forward(xb), return_sequences=True),
                    model.lstm1_bwd.forward(model.embedding.forward(xb)[:, ::-1, :], return_sequences=True)[:, ::-1, :]
                ], axis=-1)
            )
        )
        model.dense2.W -= lr * (d1_out.T @ dz)
        model.dense2.b -= lr * dz.sum(axis=0)

    return total_loss / n_batches, total_correct / n


def evaluate(model: SentimentLSTMModel, X: np.ndarray, y: np.ndarray, batch_size: int = 64):
    """Evaluate model on dataset, return loss and accuracy."""
    all_probs = []
    for start in range(0, len(X), batch_size):
        xb = X[start:start + batch_size]
        probs = model.forward(xb)
        all_probs.append(probs)
    all_probs = np.vstack(all_probs)
    loss = cross_entropy_loss(all_probs, y)
    acc = accuracy(all_probs, y)
    return loss, acc, all_probs


def compute_metrics(probs: np.ndarray, y_true: np.ndarray, num_classes: int = 3):
    """Compute precision, recall, F1 per class."""
    y_pred = np.argmax(probs, axis=-1)
    metrics = {}
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        metrics[LABEL_MAP[c]] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': int(np.sum(y_true == c))}
    return metrics


def train(data_path: str = 'data/customer_reviews.csv',
          model_dir: str = 'src/models/saved',
          epochs: int = 15, lr: float = 0.05,
          batch_size: int = 32, max_vocab: int = 5000, max_length: int = 80):

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # ── Load & preprocess ──────────────────────────────────────────────
    logger.info("Loading data...")
    train_df, val_df, test_df = load_and_split(data_path)

    preprocessor = TextPreprocessor(remove_stopwords=True)
    train_df = preprocessor.process_dataframe(train_df)
    val_df = preprocessor.process_dataframe(val_df)
    test_df = preprocessor.process_dataframe(test_df)

    tokenizer = SimpleTokenizer(max_vocab=max_vocab, max_length=max_length)
    tokenizer.fit(train_df['cleaned_text'].tolist())
    tokenizer.save(f'{model_dir}/tokenizer.json')

    X_train = tokenizer.texts_to_sequences(train_df['cleaned_text'].tolist())
    X_val = tokenizer.texts_to_sequences(val_df['cleaned_text'].tolist())
    X_test = tokenizer.texts_to_sequences(test_df['cleaned_text'].tolist())

    y_train = train_df['sentiment'].values.astype(np.int32)
    y_val = val_df['sentiment'].values.astype(np.int32)
    y_test = test_df['sentiment'].values.astype(np.int32)

    # ── Build model ────────────────────────────────────────────────────
    model = SentimentLSTMModel(vocab_size=tokenizer.vocab_size, max_length=max_length)
    model.summary()

    # ── Training loop ──────────────────────────────────────────────────
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    best_epoch = 0

    logger.info(f"\nStarting training: {epochs} epochs, lr={lr}, batch={batch_size}")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_epoch_simple(model, X_train, y_train, lr=lr, batch_size=batch_size)
        v_loss, v_acc, _ = evaluate(model, X_val, y_val)

        history['train_loss'].append(float(t_loss))
        history['val_loss'].append(float(v_loss))
        history['train_acc'].append(float(t_acc))
        history['val_acc'].append(float(v_acc))

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch = epoch
            model.save(f'{model_dir}/best_model.pkl')

        if epoch % 3 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d}/{epochs} | "
                        f"train_loss={t_loss:.4f} acc={t_acc:.4f} | "
                        f"val_loss={v_loss:.4f} acc={v_acc:.4f}")

    elapsed = time.time() - t0
    logger.info(f"\nTraining complete in {elapsed:.1f}s. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")

    # ── Final evaluation ───────────────────────────────────────────────
    best_model = SentimentLSTMModel.load(f'{model_dir}/best_model.pkl')
    _, test_acc, test_probs = evaluate(best_model, X_test, y_test)
    class_metrics = compute_metrics(test_probs, y_test)

    results = {
        'history': history,
        'best_epoch': best_epoch,
        'best_val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'class_metrics': class_metrics,
        'training_time_seconds': elapsed,
        'config': {'epochs': epochs, 'lr': lr, 'batch_size': batch_size,
                   'max_vocab': max_vocab, 'max_length': max_length}
    }
    with open(f'{model_dir}/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ── Print report ───────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  TRAINING REPORT")
    print("="*55)
    print(f"  Best Validation Accuracy : {best_val_acc*100:.2f}%")
    print(f"  Test Accuracy            : {test_acc*100:.2f}%")
    print(f"  Training Time            : {elapsed:.1f}s")
    print("\n  Per-Class Metrics:")
    for cls, m in class_metrics.items():
        print(f"    {cls:10s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    print("="*55)

    return results


if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent.parent.parent)
    train()
