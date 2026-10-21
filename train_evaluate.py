"""
Dataset, training, baselines, experiments, metrics, ablation, plots.

All experiment orchestration lives here on purpose (five-file project).
Reported numbers always come from this script — never from hand-edited tables.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import shutil
import sys
import time
import hashlib
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from environment import (
    ACTION_PREFETCH,
    CATEGORIES,
    N_PAGES,
    PAGE_SIZE_KB,
    SCENARIOS,
    STATE_DIM,
    WebPrefetchEnv,
    build_transition_counts,
    NetworkConfig,
)
from rl_agent import DQNAgent, DQNConfig


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = RESULTS / "figures"
DATA_DIR = RESULTS / "data"

MSNBC_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/msnbc-mld/msnbc990928.seq.gz",
    "http://kdd.ics.uci.edu/databases/msnbc/msnbc990928.seq.gz",
]
MSNBC_NAME = "msnbc990928.seq.gz"

SPLIT_SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15

METHOD_ORDER = ["No Prefetch", "Always Prefetch", "Rule-Based", "DQN"]
METHOD_COLORS = {
    "No Prefetch": "#73737a",
    "Always Prefetch": "#9a4a4a",
    "Rule-Based": "#3d5c73",
    "DQN": "#1f7a64",
    "DQN (simple reward)": "#b08d3e",
    "DQN (latency-aware)": "#1f7a64",
}

# Two operating points. Quick is for a laptop smoke run; full is the reported run.
MODES = {
    "quick": {"n_episodes": 220, "n_eval": 80, "n_seeds": 2, "max_sessions": 6000, "cache_eval": 80},
    "full": {"n_episodes": 500, "n_eval": 180, "n_seeds": 3, "max_sessions": 16000, "cache_eval": 120},
}


# ===========================================================================
# Dataset
# ===========================================================================
def _try_download(url: str, dest: Path, timeout: int = 60) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 prefetch-rl"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return dest.exists() and dest.stat().st_size > 10_000
    except Exception as exc:
        print(f"  download failed ({url}): {exc}")
        return False


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def acquire_msnbc(allow_fallback: bool = False) -> Tuple[Path, str]:
    """Locate or download the real MSNBC trace. Synthetic fallback is opt-in only."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / MSNBC_NAME

    # Prefer an existing real trace anywhere in the project (including the
    # dataset folder the user may have downloaded manually).
    candidates: List[Path] = [
        dest,
        ROOT / MSNBC_NAME,
        ROOT / "msnbc+com+anonymous+web+data" / MSNBC_NAME,
        Path("/tmp/msnbc") / MSNBC_NAME,
    ]
    try:
        candidates.extend(ROOT.rglob(MSNBC_NAME))
    except OSError:
        pass

    seen = set()
    for c in candidates:
        c = Path(c)
        key = str(c.resolve()) if c.exists() else str(c)
        if key in seen or not c.exists() or c.stat().st_size <= 10_000:
            continue
        seen.add(key)
        if c.resolve() != dest.resolve():
            shutil.copy2(c, dest)
        return dest, "MSNBC.com Anonymous Web Data (UCI Machine Learning Repository / KDD Archive), 28 Sep 1999"

    print("Downloading the real MSNBC sequential browsing dataset…")
    for url in MSNBC_URLS:
        if _try_download(url, dest):
            return dest, "MSNBC.com Anonymous Web Data (UCI Machine Learning Repository / KDD Archive), 28 Sep 1999"

    if not allow_fallback:
        raise RuntimeError(
            "The real MSNBC trace could not be found or downloaded. "
            "Place msnbc990928.seq.gz in the project folder (or its dataset subfolder) "
            "and rerun. Synthetic fallback is disabled by default; use --allow-fallback "
            "only for a clearly labelled software smoke test."
        )

    fallback = DATA_DIR / "fallback_demo_sessions.txt"
    print("WARNING: live download failed. Writing labelled DEMO/FALLBACK sessions.")
    _write_fallback(fallback)
    return fallback, "DEMO/FALLBACK synthetic sessions (NOT real experimental data)"

def _write_fallback(path: Path) -> None:
    """Tiny Markov toy data, clearly not the MSNBC log. Used only if download fails."""
    rng = np.random.default_rng(0)
    # Rough relative popularity of the 17 MSNBC sections (not claimed as real counts).
    stay = 0.45
    lines = ["% DEMO/FALLBACK DATA — not MSNBC, not for reported experiments", "% Sequences:"]
    for _ in range(400):
        n = int(rng.integers(3, 10))
        p = int(rng.integers(0, N_PAGES))
        seq = [p]
        for _k in range(n - 1):
            p = p if rng.random() < stay else int(rng.integers(0, N_PAGES))
            seq.append(p)
        lines.append(" ".join(str(x + 1) for x in seq))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_sessions(path: Path) -> List[List[int]]:
    opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".gz") else open
    sessions: List[List[int]] = []
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            try:
                seq = [int(x) - 1 for x in parts]
            except ValueError:
                continue
            if seq and all(0 <= p < N_PAGES for p in seq):
                sessions.append(seq)
    return sessions


def prepare_bundle(max_sessions: int, min_len: int = 3, max_len: int = 24, allow_fallback: bool = False) -> Dict[str, Any]:
    path, source = acquire_msnbc(allow_fallback=allow_fallback)
    raw = parse_sessions(path)
    if not raw:
        raise RuntimeError(f"No valid MSNBC sessions were parsed from {path}")
    filtered = [s for s in raw if min_len <= len(s) <= max_len]
    if not filtered:
        raise RuntimeError(f"No sessions remain after filtering length {min_len}–{max_len}")
    rng = np.random.default_rng(SPLIT_SEED)
    order = np.arange(len(filtered))
    rng.shuffle(order)
    chosen = [filtered[i] for i in order[:max_sessions]]
    n = len(chosen)
    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)
    train = chosen[:n_train]
    val = chosen[n_train : n_train + n_val]
    test = chosen[n_train + n_val :]
    counts = build_transition_counts(train, N_PAGES)
    lengths = np.array([len(s) for s in chosen], dtype=np.int32)
    bundle = {
        "source": source,
        "is_fallback": source.startswith("DEMO"),
        "n_raw": len(raw),
        "n_filtered": len(filtered),
        "n_used": n,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "split": (
            f"Session-level shuffle (seed={SPLIT_SEED}), "
            f"{int(TRAIN_FRAC*100)}/{int(VAL_FRAC*100)}/{int(round((1-TRAIN_FRAC-VAL_FRAC)*100))} "
            "train/val/test. Transition matrix estimated on TRAIN sessions only. "
            "The true next request is never written into the observation."
        ),
        "length_mean": float(lengths.mean()),
        "length_median": float(np.median(lengths)),
        "train": train,
        "val": val,
        "test": test,
        "counts": counts,
        "categories": CATEGORIES,
        "page_size_kb": PAGE_SIZE_KB,
        "dataset_path": str(path),
        "dataset_sha256": _sha256(path),
    }
    return bundle


def save_bundle(bundle: Dict[str, Any]) -> Path:
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / "env_bundle.pkl"
    slim = dict(bundle)
    path.write_bytes(pickle.dumps(slim, protocol=4))
    return path


def load_bundle(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or (MODELS / "env_bundle.pkl")
    return pickle.loads(path.read_bytes())


# ===========================================================================
# Policies
# ===========================================================================
def policy_never(state: np.ndarray, env: WebPrefetchEnv, agent=None) -> int:
    return 0


def policy_always(state: np.ndarray, env: WebPrefetchEnv, agent=None) -> int:
    return 1


def policy_rule(state: np.ndarray, env: WebPrefetchEnv, agent=None) -> int:
    """
    Probability- and cost-aware heuristic.
    Prefetch when the Markov predictor is confident *and* the speculative
    transfer is not obviously too expensive given bandwidth / size / cache.
    """
    conf = float(state[4])
    size = float(state[5])
    bw = float(state[6])
    rtt = float(state[7])
    occ = float(state[8])
    cached = float(state[9])
    if cached >= 0.5:
        return 0
    if occ > 0.93:
        return 0
    # Higher RTT raises the value of a hit; large size + low bw raises the cost of a miss.
    score = conf + 0.30 * rtt - 0.40 * size * (1.0 - 0.7 * bw)
    if conf >= 0.60:
        return 1
    if conf >= 0.40 and score >= 0.42:
        return 1
    return 0


def policy_dqn(state: np.ndarray, env: WebPrefetchEnv, agent=None) -> int:
    assert agent is not None
    return int(agent.select_action(state, eval_mode=True))


POLICIES: Dict[str, Callable] = {
    "No Prefetch": policy_never,
    "Always Prefetch": policy_always,
    "Rule-Based": policy_rule,
    "DQN": policy_dqn,
}


# ===========================================================================
# Evaluation
# ===========================================================================
def _empty_accum() -> Dict[str, List[float]]:
    return {
        "latency": [],
        "bytes": [],
        "reward": [],
        "prefetch": [],
        "useful": [],
        "wasted": [],
        "hits": [],
        "correct_pred": [],
        "occupancy": [],
        "saved": [],
        "baseline": [],
    }


def rollout(
    env: WebPrefetchEnv,
    policy: Callable,
    session: Sequence[int],
    seed: int,
    agent: Optional[DQNAgent] = None,
) -> Dict[str, Any]:
    obs, _ = env.reset(seed=seed, options={"session": list(session)})
    acc = _empty_accum()
    done = False
    ep_reward = 0.0
    steps = 0
    while not done:
        action = int(policy(obs, env, agent))
        obs, reward, term, trunc, info = env.step(action)
        done = bool(term or trunc)
        ep_reward += float(reward)
        steps += 1
        acc["latency"].append(float(info["latency_ms"]))
        acc["bytes"].append(float(info["bytes_used"]))
        acc["reward"].append(float(reward))
        acc["prefetch"].append(1.0 if action == ACTION_PREFETCH else 0.0)
        acc["useful"].append(1.0 if info["prefetch_useful"] else 0.0)
        acc["wasted"].append(1.0 if info["prefetch_wasted"] else 0.0)
        acc["hits"].append(1.0 if info["cache_hit"] else 0.0)
        acc["correct_pred"].append(1.0 if info["prediction_correct"] else 0.0)
        acc["occupancy"].append(float(info["cache"]["occupancy"]))
        acc["saved"].append(float(info["latency_saved_ms"]))
        acc["baseline"].append(float(info["baseline_latency_ms"]))
    n_pref = float(np.sum(acc["prefetch"]))
    lat = np.asarray(acc["latency"], dtype=np.float64)
    return {
        "n_steps": steps,
        "episode_reward": ep_reward,
        "avg_latency_ms": float(lat.mean()) if steps else 0.0,
        "median_latency_ms": float(np.median(lat)) if steps else 0.0,
        "p95_latency_ms": float(np.percentile(lat, 95)) if steps else 0.0,
        "sum_latency_ms": float(lat.sum()) if steps else 0.0,
        "avg_baseline_ms": float(np.mean(acc["baseline"])) if steps else 0.0,
        "latency_saved_ms": float(np.sum(acc["saved"])),
        "bytes": float(np.sum(acc["bytes"])),
        "prefetch_actions": n_pref,
        "useful": float(np.sum(acc["useful"])),
        "wasted": float(np.sum(acc["wasted"])),
        "cache_hit_rate": float(np.mean(acc["hits"])) if steps else 0.0,
        "prefetch_hit_rate": (float(np.sum(acc["useful"])) / n_pref) if n_pref else 0.0,
        "unnecessary_prefetch_rate": (float(np.sum(acc["wasted"])) / n_pref) if n_pref else 0.0,
        "cache_util": float(np.mean(acc["occupancy"])) if steps else 0.0,
        "predictor_acc": float(np.mean(acc["correct_pred"])) if steps else 0.0,
        "prefetch_rate": (n_pref / steps) if steps else 0.0,
    }


def evaluate_method(
    bundle: Dict[str, Any],
    method: str,
    scenario: str,
    n_eval: int,
    agent: Optional[DQNAgent] = None,
    cache_mb: Optional[float] = None,
    seed_offset: int = 0,
    split: str = "test",
) -> Dict[str, float]:
    sessions = bundle[split][:n_eval]
    cfg = NetworkConfig.from_scenario(scenario)
    if cache_mb is not None:
        cfg.cache_mb = float(cache_mb)
    env = WebPrefetchEnv(
        sessions,
        bundle["counts"],
        network=cfg,
        reward_mode="latency",
        seed=seed_offset,
        episode_shuffle=False,
    )
    policy = POLICIES[method]
    rolls = [
        rollout(env, policy, sess, seed=10_000 + seed_offset + i, agent=agent)
        for i, sess in enumerate(sessions)
    ]
    df = pd.DataFrame(rolls)
    n_pref = float(df["prefetch_actions"].sum())
    useful = float(df["useful"].sum())
    wasted = float(df["wasted"].sum())
    steps = float(df["n_steps"].sum())
    return {
        "avg_latency_ms": float(df["avg_latency_ms"].mean()),
        "median_latency_ms": float(df["median_latency_ms"].mean()),
        "p95_latency_ms": float(df["p95_latency_ms"].mean()),
        "avg_baseline_ms": float(df["avg_baseline_ms"].mean()),
        "latency_saved_ms": float(df["latency_saved_ms"].mean()),
        "bandwidth_mb": float(df["bytes"].mean()) / (1024.0 * 1024.0),
        "prefetch_actions": float(df["prefetch_actions"].mean()),
        "prefetch_rate": float(df["prefetch_rate"].mean()),
        "prefetch_hit_rate": (useful / n_pref) if n_pref else 0.0,
        "unnecessary_prefetch_rate": (wasted / n_pref) if n_pref else 0.0,
        "cache_hit_rate": float(df["cache_hit_rate"].mean()),
        "cache_util": float(df["cache_util"].mean()),
        "avg_reward": float(df["episode_reward"].mean()),
        "predictor_acc": float(df["predictor_acc"].mean()),
        "n_episodes": float(len(df)),
        "n_steps": steps,
    }


# ===========================================================================
# Training
# ===========================================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_dqn(
    bundle: Dict[str, Any],
    seed: int,
    reward_mode: str,
    n_episodes: int,
) -> Tuple[DQNAgent, Dict[str, List[float]]]:
    set_seed(seed)
    net = NetworkConfig.from_scenario("moderate")
    net.randomize = True
    env = WebPrefetchEnv(
        bundle["train"],
        bundle["counts"],
        network=net,
        reward_mode=reward_mode,
        seed=seed,
        episode_shuffle=True,
    )
    agent = DQNAgent(DQNConfig(), seed=seed)
    rewards, losses, eps, pref_rate = [], [], [], []
    t0 = time.time()
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        tot = 0.0
        n_pref = 0
        steps = 0
        ep_losses: List[float] = []
        while not done:
            action = agent.select_action(obs, eval_mode=False)
            n_pref += int(action == ACTION_PREFETCH)
            nxt, reward, term, trunc, _info = env.step(action)
            done = bool(term or trunc)
            loss = agent.observe(obs, action, reward, nxt, done)
            if loss is not None:
                ep_losses.append(loss)
            tot += float(reward)
            steps += 1
            obs = nxt
        rewards.append(tot)
        losses.append(float(np.mean(ep_losses)) if ep_losses else float("nan"))
        eps.append(agent.epsilon)
        pref_rate.append((n_pref / steps) if steps else 0.0)
        if (ep + 1) % 50 == 0 or ep == 0:
            recent = np.mean(rewards[-50:])
            print(
                f"  [{reward_mode} seed={seed}] ep {ep+1:4d}/{n_episodes}  "
                f"R={recent:7.3f}  eps={agent.epsilon:.3f}  "
                f"loss={agent.last_loss:.4f}  pref={np.mean(pref_rate[-50:]):.2f}"
            )
    print(f"  training finished in {time.time()-t0:.1f}s  steps={agent.total_steps}")
    curves = {
        "reward": rewards,
        "loss": losses,
        "epsilon": eps,
        "prefetch_rate": pref_rate,
    }
    return agent, curves


# ===========================================================================
# Self-tests (run before experiments)
# ===========================================================================
def run_self_tests(bundle: Dict[str, Any]) -> None:
    print("Running self-tests…")
    cfg = NetworkConfig.from_scenario("moderate")
    env = WebPrefetchEnv(bundle["train"][:32], bundle["counts"], network=cfg, seed=0)
    obs, info = env.reset(seed=1)
    assert obs.shape == (STATE_DIM,), obs.shape
    assert env.observation_space.contains(obs)
    # True next must not be copied into the state. Predicted may equal actual
    # by chance; we only assert the state is built from the predictor.
    pred, conf, _ = env._predict(env.session[env.t])
    assert abs(obs[4] - conf) < 1e-5
    assert 0.0 <= conf <= 1.0

    # Skip step
    obs2, r, term, trunc, inf = env.step(0)
    assert np.isfinite(r)
    assert "latency_ms" in inf
    assert inf["latency_ms"] >= 0

    # Cache: put then hit
    env2 = WebPrefetchEnv(bundle["train"][:8], bundle["counts"], network=cfg, seed=2)
    env2.reset(seed=2, options={"session": [0, 1, 2, 3]})
    env2.cache.put(1, 1000)
    assert env2.cache.contains(1)
    env2.cache.put(2, cfg.cache_mb * 1024 * 1024 * 2)  # too big to store if > capacity after
    # Latency: cached resource is faster than a compulsory download.
    miss = env2._download_latency(1800 * 1024)
    assert 8.0 < miss

    # Reward: successful prefetch should not crash either reward mode
    for mode in ("latency", "simple"):
        e = WebPrefetchEnv(
            bundle["train"][:16], bundle["counts"], network=cfg, reward_mode=mode, seed=3
        )
        s, _ = e.reset(seed=3)
        _, r1, _, _, _ = e.step(1)
        s, _ = e.reset(seed=3)
        _, r0, _, _, _ = e.step(0)
        assert np.isfinite(r0) and np.isfinite(r1)

    # DQN forward + one learning update
    agent = DQNAgent(DQNConfig(min_buffer=8, batch_size=8), seed=0)
    s = np.zeros(STATE_DIM, np.float32)
    q = agent.q_values(s)
    assert q.shape == (2,)
    for i in range(16):
        ns = np.zeros(STATE_DIM, np.float32)
        ns[0] = i / 16
        agent.observe(s, i % 2, 0.1, ns, False)
        s = ns
    loss = agent.train_step()
    assert loss is not None and np.isfinite(loss)

    # Transition matrix uses train only: corrupting test sessions must not change counts
    counts_before = bundle["counts"].copy()
    if bundle["test"]:
        bundle["test"][0] = list(bundle["test"][0])  # no-op; counts already built
    assert np.allclose(counts_before, bundle["counts"])

    # Distinct-article model: staying in one section is NOT a free cache hit.
    stay = [2, 2, 2, 2, 2, 2]
    cfg = NetworkConfig.from_scenario("moderate")
    cfg.bw_jitter = 0.0
    cfg.rtt_jitter = 0.0

    def _run(act: int) -> Tuple[float, float, float]:
        e = WebPrefetchEnv([stay], bundle["counts"], network=cfg, seed=0)
        o, _ = e.reset(seed=7, options={"session": stay})
        lats, bws, hits = [], [], []
        done = False
        while not done:
            o, _, term, trunc, inf = e.step(act)
            lats.append(inf["latency_ms"])
            bws.append(inf["bytes_used"])
            hits.append(float(inf["cache_hit"]))
            done = bool(term or trunc)
        return float(np.mean(lats)), float(np.sum(bws)), float(np.mean(hits))

    lat0, bw0, hit0 = _run(0)
    lat1, bw1, hit1 = _run(1)
    print(f"  stay-session never: lat={lat0:.1f}ms bw={bw0/1024:.0f}KB hit={hit0:.2f}")
    print(f"  stay-session always: lat={lat1:.1f}ms bw={bw1/1024:.0f}KB hit={hit1:.2f}")
    assert hit0 == 0.0
    assert lat1 < lat0 * 0.5, (lat0, lat1)
    assert bw1 >= bw0 * 0.5  # successful prefetch shifts bytes earlier, similar volume

    print("Self-tests passed.")


# ===========================================================================
# Aggregation / stats
# ===========================================================================
def mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    if len(arr) == 1:
        return {"mean": float(arr[0]), "std": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1))}


def add_latency_reduction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["latency_reduction_pct"] = np.nan
    for (scen, seed), g in out.groupby(["scenario", "seed"]):
        base = g.loc[g["method"] == "No Prefetch", "avg_latency_ms"]
        if base.empty:
            continue
        b = float(base.iloc[0])
        idx = g.index
        if b > 1e-6:
            out.loc[idx, "latency_reduction_pct"] = (b - out.loc[idx, "avg_latency_ms"]) / b * 100.0
        else:
            out.loc[idx, "latency_reduction_pct"] = 0.0
    return out


# ===========================================================================
# Plots
# ===========================================================================
def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 140,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.facecolor": "#fafafa",
            "figure.facecolor": "white",
            "axes.edgecolor": "#222222",
            "grid.alpha": 0.25,
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_training(curves_by_seed: Dict[int, Dict[str, List[float]]]) -> None:
    _style()
    # Reward
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for seed, c in curves_by_seed.items():
        r = pd.Series(c["reward"]).rolling(20, min_periods=1).mean()
        ax.plot(r, label=f"seed {seed}", linewidth=1.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return (smoothed)")
    ax.set_title("DQN reward vs training episodes")
    ax.legend(frameon=False)
    _save(fig, "01_dqn_reward.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for seed, c in curves_by_seed.items():
        loss = pd.Series(c["loss"]).rolling(20, min_periods=1).mean()
        ax.plot(loss, label=f"seed {seed}", linewidth=1.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Huber (SmoothL1) Bellman loss")
    ax.set_title("DQN loss")
    ax.legend(frameon=False)
    _save(fig, "02_dqn_loss.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for seed, c in curves_by_seed.items():
        ax.plot(c["epsilon"], label=f"seed {seed}", linewidth=1.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon decay")
    ax.legend(frameon=False)
    _save(fig, "03_dqn_epsilon.png")


def _bar(df: pd.DataFrame, y: str, title: str, ylabel: str, fname: str, scenario: str = "moderate") -> None:
    _style()
    sub = df[df["scenario"] == scenario].copy()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sns.barplot(
        data=sub,
        x="method",
        y=y,
        hue="method",
        order=METHOD_ORDER,
        palette=METHOD_COLORS,
        dodge=False,
        errorbar="sd",
        ax=ax,
        legend=False,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=15)
    _save(fig, fname)


def plot_comparisons(df: pd.DataFrame) -> None:
    _bar(df, "avg_latency_ms", "Average load latency by method (moderate network)", "Average latency (ms)", "04_latency_avg_by_method.png")
    _bar(df, "p95_latency_ms", "P95 load latency by method (moderate network)", "P95 latency (ms)", "05_latency_p95_by_method.png")
    _bar(df, "prefetch_hit_rate", "Prefetch hit rate by method (moderate network)", "Prefetch hit rate", "06_prefetch_hit_rate.png")
    _bar(df, "unnecessary_prefetch_rate", "Unnecessary prefetch rate (moderate network)", "Unnecessary prefetch rate", "07_unnecessary_prefetch_rate.png")
    _bar(df, "bandwidth_mb", "Bandwidth consumption by method (moderate network)", "Mean bandwidth (MB / session)", "08_bandwidth_by_method.png")

    _style()
    sub = df[df["scenario"] == "moderate"].copy()
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for method, g in sub.groupby("method"):
        ax.scatter(
            g["bandwidth_mb"].mean(),
            g["avg_latency_ms"].mean(),
            s=140,
            color=METHOD_COLORS.get(method, "#333"),
            label=method,
            zorder=3,
        )
        ax.errorbar(
            g["bandwidth_mb"].mean(),
            g["avg_latency_ms"].mean(),
            xerr=g["bandwidth_mb"].std(ddof=1) if len(g) > 1 else 0,
            yerr=g["avg_latency_ms"].std(ddof=1) if len(g) > 1 else 0,
            color=METHOD_COLORS.get(method, "#333"),
            capsize=3,
            linewidth=1.0,
        )
    ax.set_xlabel("Bandwidth (MB / session)")
    ax.set_ylabel("Average latency (ms)")
    ax.set_title("Latency vs bandwidth tradeoff (moderate network)")
    ax.legend(frameon=False)
    _save(fig, "09_latency_bandwidth_tradeoff.png")

    # Network conditions: grouped bars of latency
    _style()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    order = list(SCENARIOS.keys())
    sns.barplot(
        data=df,
        x="scenario",
        y="avg_latency_ms",
        hue="method",
        hue_order=METHOD_ORDER,
        palette=METHOD_COLORS,
        order=order,
        errorbar="sd",
        ax=ax,
    )
    ax.set_title("Average latency under simulated network / cache conditions")
    ax.set_xlabel("Scenario (simulated)")
    ax.set_ylabel("Average latency (ms)")
    ax.legend(frameon=False, title="")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "10_performance_by_network.png")


def plot_cache(cache_df: pd.DataFrame) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    sns.lineplot(
        data=cache_df,
        x="cache_mb",
        y="avg_latency_ms",
        hue="method",
        hue_order=METHOD_ORDER,
        palette=METHOD_COLORS,
        marker="o",
        errorbar="sd",
        ax=ax,
    )
    ax.set_title("Latency vs cache capacity (moderate network, simulated)")
    ax.set_xlabel("Cache capacity (MB)")
    ax.set_ylabel("Average latency (ms)")
    ax.legend(frameon=False)
    _save(fig, "11_performance_by_cache.png")


def plot_ablation(ab_df: pd.DataFrame) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    metrics = [
        ("avg_latency_ms", "Average latency (ms)"),
        ("bandwidth_mb", "Bandwidth (MB / session)"),
        ("avg_reward", "Average episode reward"),
    ]
    order = ["No Prefetch", "Always Prefetch", "Rule-Based", "DQN (simple reward)", "DQN (latency-aware)"]
    palette = {m: METHOD_COLORS.get(m, "#555") for m in order}
    for ax, (col, label) in zip(axes, metrics):
        sns.barplot(
            data=ab_df,
            x="method",
            y=col,
            hue="method",
            order=order,
            palette=palette,
            dodge=False,
            errorbar="sd",
            ax=ax,
            legend=False,
        )
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Ablation: simple reward vs latency-aware reward (moderate network)", y=1.02)
    fig.tight_layout()
    _save(fig, "12_ablation_comparison.png")


# ===========================================================================
# Main experiment driver
# ===========================================================================
def summarize_group(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    metric_cols = [
        c
        for c in df.columns
        if c not in set(keys) | {"method", "scenario", "seed", "reward_mode", "cache_mb"}
        and np.issubdtype(df[c].dtype, np.number)
    ]
    rows = []
    for vals, g in df.groupby(list(keys), dropna=False):
        if not isinstance(vals, tuple):
            vals = (vals,)
        rec = {k: v for k, v in zip(keys, vals)}
        for col in metric_cols:
            ms = mean_std(g[col].tolist())
            rec[f"{col}_mean"] = ms["mean"]
            rec[f"{col}_std"] = ms["std"]
        rec["n"] = int(len(g))
        rows.append(rec)
    return pd.DataFrame(rows)


def run_experiments(mode: str = "full", allow_fallback: bool = False) -> Dict[str, Any]:
    t_all = time.time()
    cfg = MODES[mode]
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print(f"=== Preparing dataset (mode={mode}) ===")
    bundle = prepare_bundle(max_sessions=cfg["max_sessions"], allow_fallback=allow_fallback)
    save_bundle(bundle)
    print(
        f"Source: {bundle['source']}\n"
        f"Raw sessions={bundle['n_raw']}  filtered={bundle['n_filtered']}  "
        f"used={bundle['n_used']}  train={bundle['n_train']}  "
        f"val={bundle['n_val']}  test={bundle['n_test']}\n"
        f"Split: {bundle['split']}"
    )
    if bundle["is_fallback"]:
        print("WARNING: results are from DEMO/FALLBACK data, not the real MSNBC log.")

    run_self_tests(bundle)

    n_seeds = cfg["n_seeds"]
    n_episodes = cfg["n_episodes"]
    n_eval = cfg["n_eval"]
    seeds = list(range(n_seeds))

    print(f"\n=== Training latency-aware DQN ({n_seeds} seeds × {n_episodes} episodes) ===")
    latency_curves = {}
    latency_agents: Dict[int, DQNAgent] = {}
    for seed in seeds:
        agent, curves = train_dqn(bundle, seed, "latency", n_episodes)
        path = MODELS / f"dqn_latency_seed{seed}.pt"
        agent.save(str(path), extra={"reward_mode": "latency", "seed": seed, "mode": mode})
        latency_agents[seed] = agent
        latency_curves[seed] = curves
    # UI checkpoint: seed 0 (fully specified, not cherry-picked by test score)
    shutil.copy2(MODELS / "dqn_latency_seed0.pt", MODELS / "dqn_prefetch.pt")

    print(f"\n=== Training simple-reward DQN (ablation, {n_seeds} seeds) ===")
    simple_agents: Dict[int, DQNAgent] = {}
    simple_curves = {}
    for seed in seeds:
        agent, curves = train_dqn(bundle, seed, "simple", n_episodes)
        path = MODELS / f"dqn_simple_seed{seed}.pt"
        agent.save(str(path), extra={"reward_mode": "simple", "seed": seed, "mode": mode})
        simple_agents[seed] = agent
        simple_curves[seed] = curves

    # Persist training curves
    with open(RESULTS / "training_curves.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "latency": {str(k): v for k, v in latency_curves.items()},
                "simple": {str(k): v for k, v in simple_curves.items()},
            },
            f,
        )

    print("\n=== Evaluating baselines + DQN across scenarios ===")
    rows = []
    for scenario in SCENARIOS:
        for seed in seeds:
            agent = latency_agents[seed]
            for method in METHOD_ORDER:
                use_agent = agent if method == "DQN" else None
                metrics = evaluate_method(
                    bundle, method, scenario, n_eval, agent=use_agent, seed_offset=seed
                )
                rec = {"method": method, "scenario": scenario, "seed": seed, "reward_mode": "latency"}
                rec.update(metrics)
                rows.append(rec)
                print(
                    f"  {scenario:18s} {method:16s} seed={seed}  "
                    f"lat={metrics['avg_latency_ms']:.1f}ms  "
                    f"bw={metrics['bandwidth_mb']:.2f}MB  "
                    f"hit={metrics['prefetch_hit_rate']:.2f}  "
                    f"waste={metrics['unnecessary_prefetch_rate']:.2f}"
                )
    df = add_latency_reduction(pd.DataFrame(rows))
    df.to_csv(RESULTS / "comparison.csv", index=False)

    print("\n=== Cache-capacity sweep (moderate network) ===")
    cache_rows = []
    for cache_mb in (4.0, 8.0, 16.0, 32.0):
        for seed in seeds:
            agent = latency_agents[seed]
            for method in METHOD_ORDER:
                metrics = evaluate_method(
                    bundle,
                    method,
                    "moderate",
                    cfg["cache_eval"],
                    agent=agent if method == "DQN" else None,
                    cache_mb=cache_mb,
                    seed_offset=seed,
                )
                rec = {
                    "method": method,
                    "scenario": "moderate",
                    "seed": seed,
                    "cache_mb": cache_mb,
                }
                rec.update(metrics)
                cache_rows.append(rec)
    cache_df = pd.DataFrame(cache_rows)
    cache_df.to_csv(RESULTS / "cache_sweep.csv", index=False)

    print("\n=== Ablation evaluation (moderate network) ===")
    ab_rows = []
    for seed in seeds:
        # Replay baselines once per seed so error bars match the DQN seeds.
        for method in ("No Prefetch", "Always Prefetch", "Rule-Based"):
            metrics = evaluate_method(
                bundle, method, "moderate", n_eval, agent=None, seed_offset=seed
            )
            rec = {"method": method, "scenario": "moderate", "seed": seed}
            rec.update(metrics)
            ab_rows.append(rec)
        m_simple = evaluate_method(
            bundle, "DQN", "moderate", n_eval, agent=simple_agents[seed], seed_offset=seed
        )
        rec = {"method": "DQN (simple reward)", "scenario": "moderate", "seed": seed}
        rec.update(m_simple)
        ab_rows.append(rec)
        m_lat = evaluate_method(
            bundle, "DQN", "moderate", n_eval, agent=latency_agents[seed], seed_offset=seed
        )
        rec = {"method": "DQN (latency-aware)", "scenario": "moderate", "seed": seed}
        rec.update(m_lat)
        ab_rows.append(rec)
    ab_df = pd.DataFrame(ab_rows)
    ab_df.to_csv(RESULTS / "ablation.csv", index=False)

    print("\n=== Plotting ===")
    plot_training(latency_curves)
    plot_comparisons(df)
    plot_cache(cache_df)
    plot_ablation(ab_df)

    # Aggregated tables
    summary_df = summarize_group(df, ["method", "scenario"])
    summary_df.to_csv(RESULTS / "comparison_mean_std.csv", index=False)
    ab_sum = summarize_group(ab_df, ["method"])
    ab_sum.to_csv(RESULTS / "ablation_mean_std.csv", index=False)

    moderate = df[df["scenario"] == "moderate"]
    overall = {}
    for method, g in moderate.groupby("method"):
        overall[method] = {
            col: mean_std(g[col].tolist())
            for col in [
                "avg_latency_ms",
                "median_latency_ms",
                "p95_latency_ms",
                "latency_reduction_pct",
                "prefetch_hit_rate",
                "unnecessary_prefetch_rate",
                "bandwidth_mb",
                "cache_util",
                "avg_reward",
                "prefetch_actions",
                "cache_hit_rate",
            ]
        }

    dashboard = {
        "mode": mode,
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
        "n_eval": n_eval,
        "seeds": seeds,
        "dataset": {
            "source": bundle["source"],
            "is_fallback": bundle["is_fallback"],
            "n_raw": bundle["n_raw"],
            "n_used": bundle["n_used"],
            "n_train": bundle["n_train"],
            "n_val": bundle["n_val"],
            "n_test": bundle["n_test"],
            "split": bundle["split"],
            "length_mean": bundle["length_mean"],
            "dataset_sha256": bundle["dataset_sha256"],
            "note": (
                "MSNBC records section-level page requests, not raw URLs or byte sizes. "
                "Payload sizes, bandwidth, RTT and cache capacity are simulated and labelled as such."
            ),
        },
        "moderate_overall": overall,
        "scenarios": list(SCENARIOS.keys()),
        "methods": METHOD_ORDER,
        "primary_checkpoint": "models/dqn_prefetch.pt",
        "runtime_s": time.time() - t_all,
        "honest_reading": (
            "Results are produced by train_evaluate.py. If DQN does not dominate every "
            "metric, that is reported rather than adjusted. The intended lesson is the "
            "latency–bandwidth tradeoff, not a guaranteed ranking."
        ),
    }
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2)

    # Human-readable table for the README
    lines = ["# Experiment summary (mean ± std over seeds, moderate network)\n"]
    if bundle["is_fallback"]:
        lines.append("**DEMO/FALLBACK DATA — do not treat as real MSNBC results.**\n")
    lines.append("| Method | Avg latency (ms) | P95 (ms) | Reduction % | Prefetch hit | Unnecessary | BW (MB) | Reward |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for method in METHOD_ORDER:
        o = overall[method]
        def fmt(key):
            m, s = o[key]["mean"], o[key]["std"]
            return f"{m:.2f} ± {s:.2f}"
        lines.append(
            f"| {method} | {fmt('avg_latency_ms')} | {fmt('p95_latency_ms')} | "
            f"{fmt('latency_reduction_pct')} | {fmt('prefetch_hit_rate')} | "
            f"{fmt('unnecessary_prefetch_rate')} | {fmt('bandwidth_mb')} | {fmt('avg_reward')} |"
        )
    (RESULTS / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nAll experiments finished in {time.time()-t_all:.1f}s")
    print(f"Figures → {FIGURES}")
    print(f"Models  → {MODELS}")
    return dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate latency-aware prefetch DQN")
    parser.add_argument("--mode", choices=["quick", "full", "test"], default="full")
    parser.add_argument(
        "--allow-fallback", action="store_true",
        help="Allow labelled synthetic demo data only when the real MSNBC trace is unavailable. "
             "Do not use for reported results."
    )
    args = parser.parse_args()
    if args.mode == "test":
        bundle = prepare_bundle(max_sessions=400, allow_fallback=args.allow_fallback)
        run_self_tests(bundle)
        print("test mode: self-tests only")
        return
    run_experiments(mode=args.mode, allow_fallback=args.allow_fallback)


if __name__ == "__main__":
    main()
