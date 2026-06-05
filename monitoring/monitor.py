"""
Model Monitoring & Performance Tracking
Tracks prediction distribution, latency, and data drift signals
"""

import json
import time
import logging
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}


class ModelMonitor:
    """
    Lightweight monitor that tracks:
    - Prediction distribution per label
    - Average confidence per label
    - Latency percentiles (p50, p95, p99)
    - Request volume over time
    - Simple drift alerts via KL divergence
    """

    BASELINE_DISTRIBUTION = {'Negative': 0.33, 'Neutral': 0.33, 'Positive': 0.34}
    DRIFT_THRESHOLD = 0.15  # KL divergence threshold

    def __init__(self, window_size: int = 500, log_dir: str = 'monitoring/logs'):
        self.window_size = window_size
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._label_counts = defaultdict(int)
        self._label_confidence = defaultdict(list)
        self._latencies = deque(maxlen=window_size)
        self._request_log = []
        self._total_requests = 0
        self._error_count = 0
        self._start_time = time.time()

    def record_prediction(self, sentiment: str, confidence: float, latency_ms: float):
        """Record a single prediction event."""
        self._total_requests += 1
        self._label_counts[sentiment] += 1
        self._label_confidence[sentiment].append(confidence)
        self._latencies.append(latency_ms)
        self._request_log.append({
            'timestamp': time.time(),
            'sentiment': sentiment,
            'confidence': confidence,
            'latency_ms': latency_ms,
        })

    def record_error(self):
        self._error_count += 1

    def get_distribution(self) -> dict:
        """Fraction of each label in recent predictions."""
        total = max(self._total_requests, 1)
        return {label: self._label_counts[label] / total for label in LABEL_MAP.values()}

    def get_avg_confidence(self) -> dict:
        return {
            label: float(np.mean(confs)) if confs else 0.0
            for label, confs in self._label_confidence.items()
        }

    def get_latency_stats(self) -> dict:
        if not self._latencies:
            return {'p50': 0, 'p95': 0, 'p99': 0, 'mean': 0}
        lats = list(self._latencies)
        return {
            'p50': float(np.percentile(lats, 50)),
            'p95': float(np.percentile(lats, 95)),
            'p99': float(np.percentile(lats, 99)),
            'mean': float(np.mean(lats)),
        }

    def check_drift(self) -> dict:
        """Compute KL divergence against baseline distribution."""
        current = self.get_distribution()
        kl = 0.0
        for label in LABEL_MAP.values():
            p = current.get(label, 0) + 1e-9
            q = self.BASELINE_DISTRIBUTION.get(label, 0.33) + 1e-9
            kl += p * np.log(p / q)
        drift_detected = kl > self.DRIFT_THRESHOLD
        return {
            'kl_divergence': round(kl, 4),
            'drift_detected': drift_detected,
            'message': '⚠️ Drift detected!' if drift_detected else '✅ No significant drift',
        }

    def summary(self) -> dict:
        uptime = time.time() - self._start_time
        error_rate = self._error_count / max(self._total_requests, 1)
        rps = self._total_requests / max(uptime, 1)
        return {
            'uptime_seconds': round(uptime, 2),
            'total_requests': self._total_requests,
            'error_count': self._error_count,
            'error_rate': round(error_rate, 4),
            'requests_per_second': round(rps, 4),
            'label_distribution': self.get_distribution(),
            'avg_confidence': self.get_avg_confidence(),
            'latency_stats': self.get_latency_stats(),
            'drift_analysis': self.check_drift(),
        }

    def save_snapshot(self):
        """Persist current state to JSON."""
        path = self.log_dir / f'snapshot_{int(time.time())}.json'
        def _to_serializable(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            return obj

        import json as _json
        summary = self.summary()
        with open(path, 'w') as f:
            _json.dump(summary, f, indent=2, default=_to_serializable)
        logger.info(f"Monitor snapshot saved: {path}")
        return str(path)

    def print_dashboard(self):
        s = self.summary()
        print("\n" + "="*55)
        print("  📊 MONITORING DASHBOARD")
        print("="*55)
        print(f"  Requests Served  : {s['total_requests']}")
        print(f"  Error Rate       : {s['error_rate']*100:.2f}%")
        print(f"  Avg Latency      : {s['latency_stats'].get('mean', 0):.1f}ms")
        print(f"  p95 Latency      : {s['latency_stats'].get('p95', 0):.1f}ms")
        print(f"\n  Label Distribution:")
        for label, frac in s['label_distribution'].items():
            bar = '█' * int(frac * 30)
            print(f"    {label:10s}: {bar:<30} {frac*100:.1f}%")
        print(f"\n  Drift Analysis   : {s['drift_analysis']['message']}")
        print(f"  KL Divergence    : {s['drift_analysis']['kl_divergence']}")
        print("="*55)
