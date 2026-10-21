"""Gymnasium environment for latency-aware web prefetching.

The MSNBC trace supplies only section-level navigation. Network conditions,
payload sizes, think-time and cache capacity are therefore simulated and are
kept explicitly separate from the real navigation data.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import Env, spaces


CATEGORIES = [
    "frontpage", "news", "tech", "local", "opinion", "on-air", "misc",
    "weather", "msn-news", "health", "living", "business", "msn-sports",
    "sports", "summary", "bbs", "travel",
]
N_PAGES = len(CATEGORIES)

# Simulated payload weights. These are deliberately category-level, not claimed
# measurements from the MSNBC trace. Values are KB and kept moderate so that
# network/cache trade-offs remain visible on a laptop.
PAGE_SIZE_KB = np.array([
    900, 1800, 2400, 1500, 1700, 1300, 1000, 1600, 1900,
    2100, 2300, 2600, 2200, 2400, 1200, 1100, 2000,
], dtype=np.float32)

ACTION_SKIP = 0
ACTION_PREFETCH = 1
ACTION_NAMES = {ACTION_SKIP: "DON'T PREFETCH", ACTION_PREFETCH: "PREFETCH"}

STATE_NAMES = [
    "current_page", "hist_1", "hist_2", "predicted_next", "confidence",
    "resource_size", "bandwidth", "rtt", "cache_occupancy",
    "predicted_cached", "prev_action", "predictability",
]
STATE_DIM = len(STATE_NAMES)


SCENARIOS = {
    "fast": {
        "title": "Fast network",
        "description": "High bandwidth, low RTT; speculative bytes are relatively costly.",
        "bandwidth_mbps": 50.0, "rtt_ms": 18.0, "cache_mb": 32.0,
        "bw_jitter": 0.08, "rtt_jitter": 0.08, "think_median_s": 2.4,
        "predictor_temperature": 1.0,
    },
    "moderate": {
        "title": "Moderate network",
        "description": "Balanced bandwidth/latency where prefetching can substantially reduce load time.",
        "bandwidth_mbps": 8.0, "rtt_ms": 60.0, "cache_mb": 16.0,
        "bw_jitter": 0.10, "rtt_jitter": 0.10, "think_median_s": 2.4,
        "predictor_temperature": 1.0,
    },
    "slow": {
        "title": "Slow network",
        "description": "Low bandwidth and higher RTT make wrong speculative transfers expensive.",
        "bandwidth_mbps": 1.6, "rtt_ms": 140.0, "cache_mb": 16.0,
        "bw_jitter": 0.12, "rtt_jitter": 0.12, "think_median_s": 2.4,
        "predictor_temperature": 1.0,
    },
    "high_latency": {
        "title": "High-latency network",
        "description": "Bandwidth is reasonable but round-trip delay is dominant.",
        "bandwidth_mbps": 10.0, "rtt_ms": 260.0, "cache_mb": 16.0,
        "bw_jitter": 0.10, "rtt_jitter": 0.10, "think_median_s": 2.4,
        "predictor_temperature": 1.0,
    },
    "variable": {
        "title": "Variable network",
        "description": "Moderate bandwidth with stronger bandwidth/RTT jitter.",
        "bandwidth_mbps": 8.0, "rtt_ms": 80.0, "cache_mb": 16.0,
        "bw_jitter": 0.35, "rtt_jitter": 0.35, "think_median_s": 2.4,
        "predictor_temperature": 1.0,
    },
    "limited_cache": {
        "title": "Limited cache",
        "description": "Moderate network with a small cache, increasing eviction pressure.",
        "bandwidth_mbps": 8.0, "rtt_ms": 60.0, "cache_mb": 4.0,
        "bw_jitter": 0.10, "rtt_jitter": 0.10, "think_median_s": 2.4,
        "predictor_temperature": 1.0,
    },
    "low_predictability": {
        "title": "Low predictability",
        "description": "The train-only predictor is deliberately tempered, making predictions less confident.",
        "bandwidth_mbps": 8.0, "rtt_ms": 60.0, "cache_mb": 16.0,
        "bw_jitter": 0.10, "rtt_jitter": 0.10, "think_median_s": 2.4,
        "predictor_temperature": 2.5,
    },
}


@dataclass
class NetworkConfig:
    bandwidth_mbps: float = 8.0
    rtt_ms: float = 60.0
    cache_mb: float = 16.0
    bw_jitter: float = 0.10
    rtt_jitter: float = 0.10
    think_median_s: float = 2.4
    predictor_temperature: float = 1.0
    randomize: bool = False
    jitter_ms: float = 0.0

    @classmethod
    def from_scenario(cls, scenario: str) -> "NetworkConfig":
        if scenario not in SCENARIOS:
            raise KeyError(f"Unknown scenario: {scenario}")
        params = {k: v for k, v in SCENARIOS[scenario].items() if k in cls.__dataclass_fields__}
        return cls(**params)


class LRUCache:
    """Byte-capacity LRU cache used for speculative objects."""

    def __init__(self, capacity_mb: float) -> None:
        self.capacity_bytes = max(float(capacity_mb), 0.25) * 1024.0 * 1024.0
        self._items: OrderedDict[object, float] = OrderedDict()
        self.used_bytes = 0.0

    @property
    def occupancy(self) -> float:
        return float(np.clip(self.used_bytes / self.capacity_bytes, 0.0, 1.0))

    def contains(self, key: object) -> bool:
        """Check membership and mark the object as recently used."""
        if key in self._items:
            size = self._items.pop(key)
            self._items[key] = size
            return True
        return False

    def peek(self, key: object) -> bool:
        """Check membership without changing LRU recency."""
        return key in self._items

    def get(self, key: object) -> Optional[float]:
        if key not in self._items:
            return None
        size = self._items.pop(key)
        self._items[key] = size
        return size

    def remove(self, key: object) -> Optional[float]:
        size = self._items.pop(key, None)
        if size is not None:
            self.used_bytes -= size
        return size

    def put(self, key: object, size_bytes: float) -> bool:
        size = float(max(size_bytes, 0.0))
        if size <= 0.0 or size > self.capacity_bytes:
            return False
        if key in self._items:
            self.used_bytes -= self._items.pop(key)
        while self._items and self.used_bytes + size > self.capacity_bytes:
            _, old = self._items.popitem(last=False)
            self.used_bytes -= old
        self._items[key] = size
        self.used_bytes += size
        return True

    def clear(self) -> None:
        self._items.clear()
        self.used_bytes = 0.0

    def snapshot(self) -> Dict[str, float]:
        return {
            "capacity_mb": self.capacity_bytes / (1024.0 * 1024.0),
            "used_mb": self.used_bytes / (1024.0 * 1024.0),
            "occupancy": self.occupancy,
            "objects": float(len(self._items)),
        }


# Alias retained for simple external imports.
LRU = LRUCache


def build_transition_counts(sessions: Sequence[Sequence[int]], n_pages: int = N_PAGES) -> np.ndarray:
    """Build a Laplace-smoothed first-order transition probability matrix."""
    counts = np.ones((n_pages, n_pages), dtype=np.float64)  # Laplace smoothing.
    for session in sessions:
        for a, b in zip(session[:-1], session[1:]):
            if 0 <= int(a) < n_pages and 0 <= int(b) < n_pages:
                counts[int(a), int(b)] += 1.0
    counts /= counts.sum(axis=1, keepdims=True)
    return counts


class WebPrefetchEnv(Env):
    """Session-level MDP for deciding whether to prefetch the predicted next resource."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        sessions: Sequence[Sequence[int]],
        transition_counts: np.ndarray,
        network: Optional[NetworkConfig] = None,
        reward_mode: str = "latency",
        seed: Optional[int] = None,
        episode_shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.sessions = [list(map(int, s)) for s in sessions if len(s) >= 2]
        if not self.sessions:
            raise ValueError("WebPrefetchEnv requires at least one session of length >= 2")
        self.transition_counts = np.asarray(transition_counts, dtype=np.float64)
        if self.transition_counts.shape != (N_PAGES, N_PAGES):
            raise ValueError(f"transition_counts must have shape {(N_PAGES, N_PAGES)}")
        self.base_network = network or NetworkConfig.from_scenario("moderate")
        self.network = self.base_network
        self.reward_mode = reward_mode
        if reward_mode not in {"latency", "simple"}:
            raise ValueError("reward_mode must be 'latency' or 'simple'")
        self.episode_shuffle = episode_shuffle
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(0.0, 1.0, shape=(STATE_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.session: List[int] = []
        self.session_index = 0
        self.t = 0
        self.prev_action = 0
        self.last_predicted = 0
        self.last_confidence = 0.0
        self.last_think_time = 0.0
        self.cache = LRUCache(self.network.cache_mb)
        self.bw = float(self.network.bandwidth_mbps)
        self.rtt = float(self.network.rtt_ms)

    # ------------------------------------------------------------------
    # RNG/network helpers
    # ------------------------------------------------------------------
    def _configure_network(self) -> None:
        cfg = self.base_network
        if cfg.randomize:
            bw = cfg.bandwidth_mbps * float(np.exp(self.rng.normal(0.0, 0.16)))
            rtt = cfg.rtt_ms * float(np.exp(self.rng.normal(0.0, 0.12)))
            cache_mb = cfg.cache_mb * float(np.exp(self.rng.normal(0.0, 0.10)))
        else:
            bw = cfg.bandwidth_mbps * float(max(0.5, 1.0 + self.rng.normal(0.0, cfg.bw_jitter)))
            rtt = cfg.rtt_ms * float(max(0.35, 1.0 + self.rng.normal(0.0, cfg.rtt_jitter)))
            cache_mb = cfg.cache_mb
        self.bw = max(0.25, bw)
        self.rtt = max(1.0, rtt)
        self.network = NetworkConfig(**{**cfg.__dict__, "cache_mb": cache_mb})
        self.cache = LRUCache(cache_mb)

    def _effective_bandwidth(self) -> float:
        if self.network.randomize:
            return max(0.2, self.bw * float(np.exp(self.rng.normal(0.0, 0.05))))
        return max(0.2, self.bw * float(max(0.4, 1.0 + self.rng.normal(0.0, self.network.bw_jitter * 0.25))))

    def _effective_rtt(self) -> float:
        if self.network.randomize:
            return max(1.0, self.rtt * float(np.exp(self.rng.normal(0.0, 0.04))))
        return max(1.0, self.rtt * float(max(0.5, 1.0 + self.rng.normal(0.0, self.network.rtt_jitter * 0.25))))

    def _think_time(self) -> float:
        median = max(0.2, self.network.think_median_s)
        return float(self.rng.lognormal(np.log(median), 0.45))

    def _download_latency(self, size_bytes: float) -> float:
        """Simulated full download latency in milliseconds."""
        bw = self._effective_bandwidth() * 1_000_000.0 / 8.0
        rtt = self._effective_rtt()
        transfer_ms = float(size_bytes / bw * 1000.0)
        jitter = float(self.rng.normal(0.0, self.network.jitter_ms)) if self.network.jitter_ms else 0.0
        return max(8.0, rtt + transfer_ms + jitter)

    # ------------------------------------------------------------------
    # Predictor/state
    # ------------------------------------------------------------------
    def _prediction_distribution(self, current: int) -> np.ndarray:
        p = np.asarray(self.transition_counts[int(current)], dtype=np.float64).copy()
        temp = max(float(self.network.predictor_temperature), 1e-6)
        if temp != 1.0:
            logits = np.log(np.clip(p, 1e-12, None)) / temp
            logits -= np.max(logits)
            p = np.exp(logits)
            p /= p.sum()
        return p

    def _predict(self, current: int) -> Tuple[int, float, np.ndarray]:
        p = self._prediction_distribution(current)
        pred = int(np.argmax(p))
        return pred, float(p[pred]), p

    def _predictability(self, p: np.ndarray) -> float:
        entropy = -float(np.sum(p * np.log(np.clip(p, 1e-12, None))))
        return float(np.clip(1.0 - entropy / np.log(N_PAGES), 0.0, 1.0))

    def _state(self) -> np.ndarray:
        current = int(self.session[self.t])
        hist1 = int(self.session[self.t - 1]) if self.t >= 1 else current
        hist2 = int(self.session[self.t - 2]) if self.t >= 2 else hist1
        pred, conf, p = self._predict(current)
        size_norm = float(np.clip(PAGE_SIZE_KB[pred] / 3000.0, 0.0, 1.0))
        # Log scaling keeps common bandwidths informative without saturating.
        bw_norm = float(np.clip(np.log1p(self.bw) / np.log1p(50.0), 0.0, 1.0))
        rtt_norm = float(np.clip(self.rtt / 300.0, 0.0, 1.0))
        pred_cached = 1.0 if self.cache.peek(("spec", pred)) else 0.0
        state = np.array([
            current / max(N_PAGES - 1, 1),
            hist1 / max(N_PAGES - 1, 1),
            hist2 / max(N_PAGES - 1, 1),
            pred / max(N_PAGES - 1, 1),
            conf,
            size_norm,
            bw_norm,
            rtt_norm,
            self.cache.occupancy,
            pred_cached,
            float(self.prev_action),
            self._predictability(p),
        ], dtype=np.float32)
        self.last_predicted = pred
        self.last_confidence = conf
        return state

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and options.get("session") is not None:
            session = list(map(int, options["session"]))
        elif self.episode_shuffle:
            session = self.sessions[int(self.rng.integers(0, len(self.sessions)))]
        else:
            session = self.sessions[self.session_index % len(self.sessions)]
            self.session_index += 1
        if len(session) < 2:
            raise ValueError("Episode session must contain at least two requests")
        self.session = session
        self.t = 0
        self.prev_action = 0
        self.last_step_info = None
        self._configure_network()
        obs = self._state()
        info = self.describe_observation(obs)
        return obs, info

    def step(self, action: int):
        action = int(action)
        if action not in (ACTION_SKIP, ACTION_PREFETCH):
            raise ValueError("action must be 0 (skip) or 1 (prefetch)")
        if self.t >= len(self.session) - 1:
            raise RuntimeError("step() called after episode termination; call reset()")

        current = int(self.session[self.t])
        actual = int(self.session[self.t + 1])
        predicted, confidence, pred_dist = self._predict(current)
        pred_size_bytes = float(PAGE_SIZE_KB[predicted]) * 1024.0
        actual_size_bytes = float(PAGE_SIZE_KB[actual]) * 1024.0

        # User reads/thinks while a speculative transfer can proceed.
        think_s = self._think_time()
        think_ms = think_s * 1000.0

        # A speculative object is represented by category. This is intentional:
        # the trace has no URL identities. It is consumed by the next matching
        # request, so repeated visits to the same section are not free hits.
        predicted_key = ("spec", predicted)
        had_spec = self.cache.contains(predicted_key)
        if action == ACTION_PREFETCH and not had_spec:
            prefetch_full_ms = self._download_latency(pred_size_bytes)
            transfer_rate = pred_size_bytes / max(prefetch_full_ms - self.rtt, 1.0)
            transferred = min(pred_size_bytes, transfer_rate * think_ms)
            prefetch_complete = prefetch_full_ms <= think_ms
            if prefetch_complete:
                self.cache.put(predicted_key, pred_size_bytes)
                prefetch_progress = pred_size_bytes
            else:
                prefetch_progress = transferred
        else:
            prefetch_full_ms = 0.0
            prefetch_complete = had_spec
            prefetch_progress = pred_size_bytes if had_spec else 0.0

        prediction_correct = predicted == actual
        baseline_ms = self._download_latency(actual_size_bytes)
        cache_hit = False
        prefetch_useful = False
        prefetch_wasted = False
        bytes_used = 0.0
        latency_ms = baseline_ms
        latency_saved_ms = 0.0

        if prediction_correct and (prefetch_complete or had_spec):
            # Consume the speculative object as the distinct article requested now.
            self.cache.remove(predicted_key)
            cache_hit = True
            prefetch_useful = action == ACTION_PREFETCH or had_spec
            latency_ms = 8.0
            latency_saved_ms = max(0.0, baseline_ms - latency_ms)
            # Only charge bytes for a prefetch transfer that actually occurred on
            # this step. If the object was already cached, no new bytes are spent.
            bytes_used = pred_size_bytes if (action == ACTION_PREFETCH and not had_spec) else 0.0
        elif prediction_correct and action == ACTION_PREFETCH and prefetch_progress > 0.0:
            # Partial speculative transfer: only the remainder blocks the click.
            remaining = max(0.0, actual_size_bytes - prefetch_progress)
            remainder_ms = self._download_latency(remaining) if remaining > 0 else 8.0
            latency_ms = remainder_ms
            latency_saved_ms = max(0.0, baseline_ms - latency_ms)
            prefetch_useful = True
            bytes_used = prefetch_progress + max(0.0, remaining)
        else:
            # Wrong speculation consumes bytes but does not make the actual request faster.
            if action == ACTION_PREFETCH and not had_spec:
                bytes_used += pred_size_bytes
                prefetch_wasted = True
            # The actual object must be downloaded on demand.
            bytes_used += actual_size_bytes
            latency_ms = baseline_ms

        # A prefetch that was correct but did not finish before the click is useful
        # in the sense that it reduces remaining latency; count it as useful.
        if action == ACTION_PREFETCH and prediction_correct and latency_saved_ms > 0:
            prefetch_useful = True
        if action == ACTION_PREFETCH and not prediction_correct:
            prefetch_wasted = True

        # Store the requested article as a non-speculative object only when it was
        # actually downloaded. This object is uniquely keyed by session position,
        # preventing repeated category visits from becoming free cache hits.
        article_key = ("article", id(self.session), self.t + 1)
        if not cache_hit:
            self.cache.put(article_key, actual_size_bytes)

        # Reward is intentionally aligned with the README formulation.
        latency_norm = min(latency_ms / 1000.0, 1.5)
        bytes_norm = min(bytes_used / (3.0 * 1024.0 * 1024.0), 1.5)
        hit_bonus = 1.0 if cache_hit else 0.0
        waste_penalty = 1.0 if prefetch_wasted else 0.0
        if self.reward_mode == "simple":
            reward = 1.0 if prefetch_useful else (-1.0 if prefetch_wasted else 0.0)
        else:
            reward = (
                -1.50 * latency_norm
                -0.80 * bytes_norm
                -0.45 * waste_penalty
                +0.15 * hit_bonus
                -0.10 * max(self.cache.occupancy - 0.85, 0.0)
            )
            reward = float(np.clip(reward, -4.0, 1.5))

        self.t += 1
        self.prev_action = action
        terminated = self.t >= len(self.session) - 1
        if terminated:
            next_obs = np.zeros(STATE_DIM, dtype=np.float32)
        else:
            next_obs = self._state()

        info = {
            "latency_ms": float(latency_ms),
            "baseline_latency_ms": float(baseline_ms),
            "latency_saved_ms": float(latency_saved_ms),
            "bytes_used": float(bytes_used),
            "cache_hit": bool(cache_hit),
            "prefetch_useful": bool(prefetch_useful),
            "prefetch_wasted": bool(prefetch_wasted),
            "prediction_correct": bool(prediction_correct),
            "predicted": int(predicted),
            "actual": int(actual),
            "predicted_name": CATEGORIES[predicted],
            "actual_name": CATEGORIES[actual],
            "action": int(action),
            "action_name": ACTION_NAMES[action],
            "confidence": float(confidence),
            "think_time_s": float(think_s),
            "prefetch_complete": bool(prefetch_complete),
            "prefetch_progress_bytes": float(prefetch_progress),
            "reward": float(reward),
            "cache": self.cache.snapshot(),
        }
        self.last_step_info = info
        return next_obs, float(reward), bool(terminated), False, info

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------
    def describe_observation(self, obs: Optional[np.ndarray] = None) -> Dict[str, object]:
        if not self.session:
            return {}
        if self.t >= len(self.session):
            return {}
        current = int(self.session[min(self.t, len(self.session) - 1)])
        pred, conf, _ = self._predict(current)
        return {
            "current": current,
            "current_name": CATEGORIES[current],
            "predicted": pred,
            "predicted_name": CATEGORIES[pred],
            "confidence": conf,
            "predicted_size_kb": float(PAGE_SIZE_KB[pred]),
            "bandwidth_mbps": float(self.bw),
            "rtt_ms": float(self.rtt),
            "predicted_cached": bool(self.cache.peek(("spec", pred))),
            "cache": self.cache.snapshot(),
            "step": int(self.t),
            "session_length": int(len(self.session)),
            "remaining": int(max(len(self.session) - 1 - self.t, 0)),
        }


__all__ = [
    "ACTION_SKIP", "ACTION_PREFETCH", "ACTION_NAMES", "CATEGORIES", "N_PAGES",
    "PAGE_SIZE_KB", "SCENARIOS", "STATE_NAMES", "STATE_DIM", "NetworkConfig",
    "LRUCache", "LRU", "WebPrefetchEnv", "build_transition_counts",
]
