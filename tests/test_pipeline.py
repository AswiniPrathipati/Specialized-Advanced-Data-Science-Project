"""
Comprehensive Test Suite for Sentiment Analysis Pipeline
Tests preprocessing, model, inference, and monitoring modules
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Adjust path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processing.preprocessor import TextPreprocessor, SimpleTokenizer
from models.lstm_model import SentimentLSTMModel
from monitoring.monitor import ModelMonitor


# ── Preprocessing Tests ──────────────────────────────────────────────────────

class TestTextPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_lowercase(self):
        result = self.preprocessor.clean_text("HELLO WORLD")
        self.assertEqual(result, "hello world")

    def test_remove_url(self):
        result = self.preprocessor.clean_text("Check http://example.com for details")
        self.assertNotIn("http", result)
        self.assertNotIn("example.com", result)

    def test_remove_html(self):
        result = self.preprocessor.clean_text("<b>Hello</b> <i>World</i>")
        self.assertNotIn("<b>", result)
        self.assertIn("hello", result)

    def test_special_chars_removed(self):
        result = self.preprocessor.clean_text("Hello!!! @#$%^&*()")
        self.assertNotIn("@", result)
        self.assertNotIn("!", result)

    def test_empty_string(self):
        result = self.preprocessor.clean_text("")
        self.assertEqual(result, "")

    def test_non_string_input(self):
        result = self.preprocessor.clean_text(None)
        self.assertEqual(result, "")

    def test_whitespace_normalization(self):
        result = self.preprocessor.clean_text("hello    world   test")
        self.assertNotIn("  ", result)

    def test_stopword_removal(self):
        preprocessor_sw = TextPreprocessor(remove_stopwords=True)
        result = preprocessor_sw.clean_text("this is a good product")
        self.assertNotIn(" a ", f" {result} ")

    def test_process_dataframe(self):
        import pandas as pd
        df = pd.DataFrame({'review': ["Great product!", "Terrible!"], 'sentiment': [2, 0]})
        result_df = self.preprocessor.process_dataframe(df)
        self.assertIn('cleaned_text', result_df.columns)
        self.assertIn('text_length', result_df.columns)
        self.assertEqual(len(result_df), 2)


# ── Tokenizer Tests ──────────────────────────────────────────────────────────

class TestSimpleTokenizer(unittest.TestCase):

    def setUp(self):
        texts = ["hello world good product", "bad quality terrible awful",
                 "amazing great fantastic", "okay average decent fine"]
        self.tokenizer = SimpleTokenizer(max_vocab=100, max_length=10)
        self.tokenizer.fit(texts)

    def test_vocab_built(self):
        self.assertGreater(self.tokenizer.vocab_size, 2)

    def test_pad_token_present(self):
        self.assertIn('<PAD>', self.tokenizer.word2idx)
        self.assertEqual(self.tokenizer.word2idx['<PAD>'], 0)

    def test_oov_token_present(self):
        self.assertIn('<OOV>', self.tokenizer.word2idx)

    def test_sequence_length(self):
        seqs = self.tokenizer.texts_to_sequences(["hello world good"])
        self.assertEqual(seqs.shape[1], 10)  # max_length

    def test_padding(self):
        seqs = self.tokenizer.texts_to_sequences(["hi"])  # shorter than max_length
        self.assertEqual(seqs[0, -1], 0)  # padded with 0

    def test_oov_handling(self):
        seqs = self.tokenizer.texts_to_sequences(["xyzabc123 unknownword"])
        # OOV words should map to index 1
        self.assertTrue(any(seqs[0] == 1))

    def test_batch_sequences(self):
        texts = ["good product", "bad item", "okay value"]
        seqs = self.tokenizer.texts_to_sequences(texts)
        self.assertEqual(seqs.shape[0], 3)
        self.assertEqual(seqs.shape[1], 10)

    def test_save_and_load(self, tmp_path='/tmp/test_tok.json'):
        self.tokenizer.save(tmp_path)
        loaded = SimpleTokenizer.load(tmp_path)
        self.assertEqual(loaded.vocab_size, self.tokenizer.vocab_size)
        orig_seq = self.tokenizer.texts_to_sequences(["hello world"])
        load_seq = loaded.texts_to_sequences(["hello world"])
        np.testing.assert_array_equal(orig_seq, load_seq)


# ── Model Tests ──────────────────────────────────────────────────────────────

class TestSentimentLSTMModel(unittest.TestCase):

    def setUp(self):
        self.model = SentimentLSTMModel(
            vocab_size=100, embed_dim=8,
            lstm1_units=8, lstm2_units=4,
            dense_units=8, num_classes=3, max_length=10
        )

    def test_forward_shape(self):
        x = np.random.randint(0, 100, (4, 10))
        probs = self.model.forward(x)
        self.assertEqual(probs.shape, (4, 3))

    def test_probabilities_sum_to_one(self):
        x = np.random.randint(0, 100, (4, 10))
        probs = self.model.forward(x)
        np.testing.assert_allclose(probs.sum(axis=-1), np.ones(4), atol=1e-5)

    def test_probabilities_positive(self):
        x = np.random.randint(0, 100, (4, 10))
        probs = self.model.forward(x)
        self.assertTrue(np.all(probs >= 0))

    def test_predict_returns_valid_classes(self):
        x = np.random.randint(0, 100, (5, 10))
        classes, probs = self.model.predict(x)
        self.assertEqual(classes.shape, (5,))
        self.assertTrue(np.all((classes >= 0) & (classes <= 2)))

    def test_save_and_load(self):
        x = np.random.randint(0, 100, (3, 10))
        orig_probs = self.model.forward(x)
        self.model.save('/tmp/test_model.pkl')
        loaded = SentimentLSTMModel.load('/tmp/test_model.pkl')
        loaded_probs = loaded.forward(x)
        np.testing.assert_allclose(orig_probs, loaded_probs, atol=1e-5)

    def test_batch_consistency(self):
        """Single and batch predictions should give same result."""
        x = np.random.randint(0, 100, (1, 10))
        single = self.model.forward(x)
        batch = self.model.forward(np.repeat(x, 3, axis=0))
        np.testing.assert_allclose(single, batch[:1], atol=1e-5)


# ── Monitoring Tests ─────────────────────────────────────────────────────────

class TestModelMonitor(unittest.TestCase):

    def setUp(self):
        self.monitor = ModelMonitor(log_dir='/tmp/monitor_test')

    def test_record_and_count(self):
        self.monitor.record_prediction('Positive', 0.9, 50.0)
        self.monitor.record_prediction('Negative', 0.8, 40.0)
        self.assertEqual(self.monitor._total_requests, 2)

    def test_label_distribution(self):
        for _ in range(3):
            self.monitor.record_prediction('Positive', 0.9, 30.0)
        for _ in range(1):
            self.monitor.record_prediction('Negative', 0.7, 35.0)
        dist = self.monitor.get_distribution()
        self.assertAlmostEqual(dist['Positive'], 0.75, places=1)

    def test_latency_stats(self):
        latencies = [20, 30, 40, 50, 100]
        for l in latencies:
            self.monitor.record_prediction('Neutral', 0.6, float(l))
        stats = self.monitor.get_latency_stats()
        self.assertIn('p50', stats)
        self.assertIn('p95', stats)
        self.assertGreater(stats['p95'], stats['p50'])

    def test_drift_check_no_drift(self):
        for _ in range(100):
            for label in ['Negative', 'Neutral', 'Positive']:
                self.monitor.record_prediction(label, 0.8, 30.0)
        result = self.monitor.check_drift()
        self.assertFalse(result['drift_detected'])

    def test_error_recording(self):
        self.monitor.record_error()
        self.monitor.record_error()
        self.assertEqual(self.monitor._error_count, 2)

    def test_summary_keys(self):
        self.monitor.record_prediction('Positive', 0.9, 50.0)
        s = self.monitor.summary()
        expected_keys = ['total_requests', 'error_rate', 'label_distribution',
                         'latency_stats', 'drift_analysis']
        for key in expected_keys:
            self.assertIn(key, s)


# ── Run all tests ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestTextPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestSimpleTokenizer))
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentLSTMModel))
    suite.addTests(loader.loadTestsFromTestCase(TestModelMonitor))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures : {len(result.failures)}")
    print(f"Errors   : {len(result.errors)}")
    print(f"Status   : {'✅ ALL PASSED' if result.wasSuccessful() else '❌ SOME FAILED'}")
    print("="*50)
