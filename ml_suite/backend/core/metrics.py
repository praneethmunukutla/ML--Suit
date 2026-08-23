"""A tiny in-process metrics registry exposed in Prometheus text format.
Swapping in prometheus_client later means changing only this module."""
from __future__ import annotations

import threading
import time
from collections import defaultdict

_lock = threading.Lock()
_counters: dict[tuple[str, tuple], float] = defaultdict(float)
_gauges: dict[tuple[str, tuple], float] = {}
_histograms: dict[tuple[str, tuple], list[float]] = defaultdict(list)
_started = time.time()


def _key(name: str, labels: dict | None) -> tuple[str, tuple]:
    return name, tuple(sorted((labels or {}).items()))


def inc(name: str, labels: dict | None = None, value: float = 1.0) -> None:
    with _lock:
        _counters[_key(name, labels)] += value


def gauge(name: str, value: float, labels: dict | None = None) -> None:
    with _lock:
        _gauges[_key(name, labels)] = value


def observe(name: str, value: float, labels: dict | None = None) -> None:
    with _lock:
        bucket = _histograms[_key(name, labels)]
        bucket.append(value)
        if len(bucket) > 1000:
            del bucket[:-1000]


def _fmt_labels(labels: tuple) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in labels)
    return "{" + inner + "}"


def render() -> str:
    lines = [
        "# HELP mlsuite_uptime_seconds Seconds since process start",
        "# TYPE mlsuite_uptime_seconds gauge",
        f"mlsuite_uptime_seconds {time.time() - _started:.1f}",
    ]
    with _lock:
        for (name, labels), value in sorted(_counters.items()):
            lines += [f"# TYPE {name} counter", f"{name}{_fmt_labels(labels)} {value:g}"]
        for (name, labels), value in sorted(_gauges.items()):
            lines += [f"# TYPE {name} gauge", f"{name}{_fmt_labels(labels)} {value:g}"]
        for (name, labels), values in sorted(_histograms.items()):
            if not values:
                continue
            ordered = sorted(values)
            p50 = ordered[len(ordered) // 2]
            p95 = ordered[int(len(ordered) * 0.95) - 1 if len(ordered) > 1 else 0]
            label_str = _fmt_labels(labels)
            lines += [
                f"# TYPE {name} summary",
                f"{name}_count{label_str} {len(ordered)}",
                f"{name}_sum{label_str} {sum(ordered):g}",
                f"{name}_p50{label_str} {p50:g}",
                f"{name}_p95{label_str} {p95:g}",
            ]
    return "\n".join(lines) + "\n"


def snapshot() -> dict:
    with _lock:
        return {
            "uptime_seconds": round(time.time() - _started, 1),
            "counters": {f"{n}{_fmt_labels(l)}": v for (n, l), v in _counters.items()},
            "gauges": {f"{n}{_fmt_labels(l)}": v for (n, l), v in _gauges.items()},
        }
