"""
Streamlit demonstration UI for latency-aware web resource prefetching.

Presentation dashboard: live simulation, DQN decisions, baseline comparison,
scenario controls, and experiment plots. Everything required by the project
lives in this file — no extra frontend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from environment import (
    ACTION_NAMES,
    CATEGORIES,
    PAGE_SIZE_KB,
    SCENARIOS,
    STATE_NAMES,
    WebPrefetchEnv,
    NetworkConfig,
)
from rl_agent import DQNAgent
from train_evaluate import POLICIES, load_bundle

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = RESULTS / "figures"

st.set_page_config(
    page_title="Latency-Aware Web Prefetching",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg: #0a0a0b;
  --bg-elevated: #121214;
  --bg-subtle: #1a1a1e;
  --fg: #f4f4f5;
  --fg-muted: #a1a1aa;
  --fg-subtle: #71717a;
  --border: rgba(244,244,245,0.10);
  --border-strong: rgba(244,244,245,0.18);
  --accent: #c8ccd4;
  --accent-fg: #0a0a0b;
  --good: #3f8f78;
  --warn: #c4a35a;
  --bad: #b45c5c;
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 20px;
  --font: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--bg) !important;
  color: var(--fg) !important;
  font-family: var(--font) !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

.block-container {
  padding-top: 1.4rem !important;
  padding-bottom: 3rem !important;
  max-width: 1320px;
}

[data-testid="stSidebar"] {
  background: #0d0d0f !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--fg) !important; font-family: var(--font) !important; }

h1, h2, h3, h4 { font-family: var(--font) !important; letter-spacing: -0.03em; color: var(--fg) !important; }
h1 { font-weight: 600 !important; font-size: 2.05rem !important; line-height: 1.15 !important; }
h2 { font-weight: 600 !important; font-size: 1.15rem !important; margin-top: 0.4rem !important; }
p, label, span, div { font-family: var(--font); }

.hero-kicker {
  font-size: 0.72rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--fg-subtle);
  font-weight: 500;
  margin-bottom: 0.35rem;
}
.hero-sub {
  color: var(--fg-muted);
  font-size: 1.02rem;
  max-width: 46rem;
  line-height: 1.5;
  margin: 0.4rem 0 1.1rem 0;
}

.panel {
  background: var(--bg-elevated);
  box-shadow: 0 0 0 1px rgba(255,255,255,0.08);
  border-radius: 20px;
  padding: 18px 18px 16px 18px;
  margin-bottom: 12px;
}
.panel h3 {
  font-size: 0.72rem !important;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--fg-subtle) !important;
  font-weight: 500 !important;
  margin: 0 0 12px 0 !important;
}

.kv { display: flex; justify-content: space-between; gap: 12px; padding: 7px 0; border-bottom: 1px solid var(--border); }
.kv:last-child { border-bottom: none; }
.kv .k { color: var(--fg-subtle); font-size: 0.84rem; }
.kv .v { color: var(--fg); font-family: var(--mono); font-size: 0.86rem; font-weight: 500; }

.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
@media (max-width: 900px) { .metric-grid { grid-template-columns: repeat(2, 1fr); } }
.metric-card {
  background: var(--bg-elevated);
  box-shadow: 0 0 0 1px rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 14px 14px 12px 14px;
}
.metric-card .lbl { color: var(--fg-subtle); font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; }
.metric-card .val { font-family: var(--mono); font-size: 1.28rem; font-weight: 500; margin-top: 6px; color: var(--fg); font-variant-numeric: tabular-nums; }
.metric-card .hint { color: var(--fg-muted); font-size: 0.75rem; margin-top: 4px; }

.badge {
  display: inline-block;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 600;
  padding: 5px 10px;
  border-radius: 999px;
  font-family: var(--font);
}
.badge-good { background: rgba(63,143,120,0.18); color: #8fd0bc; }
.badge-bad  { background: rgba(180,92,92,0.18); color: #e0a0a0; }
.badge-warn { background: rgba(196,163,90,0.16); color: #e4cc8a; }
.badge-neutral { background: rgba(244,244,245,0.08); color: var(--fg-muted); }
.badge-go { background: var(--accent); color: var(--accent-fg); }
.badge-stop { background: transparent; color: var(--fg); box-shadow: 0 0 0 1px var(--border-strong); }

.decision-box { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin: 8px 0 14px 0; }
.decision-name { font-size: 1.55rem; font-weight: 600; letter-spacing: -0.03em; }

.qrow { display: grid; grid-template-columns: 150px 1fr 64px; gap: 10px; align-items: center; margin: 8px 0; }
.qrow .lab { font-size: 0.8rem; color: var(--fg-muted); }
.qrow .num { font-family: var(--mono); font-size: 0.85rem; text-align: right; font-variant-numeric: tabular-nums; }
.qbar { height: 8px; background: #222228; border-radius: 999px; overflow: hidden; }
.qbar > span { display: block; height: 100%; background: var(--accent); border-radius: 999px; }

.outcome {
  border-radius: 16px;
  padding: 16px 18px;
  margin-top: 8px;
}
.outcome.hit { background: rgba(63,143,120,0.12); box-shadow: 0 0 0 1px rgba(63,143,120,0.28); }
.outcome.miss { background: rgba(180,92,92,0.10); box-shadow: 0 0 0 1px rgba(180,92,92,0.28); }
.outcome.idle { background: var(--bg-subtle); box-shadow: 0 0 0 1px var(--border); }
.outcome .ttl { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--fg-subtle); }
.outcome .big { font-size: 1.25rem; font-weight: 600; margin: 4px 0 8px 0; }
.outcome p { color: var(--fg-muted); margin: 0; font-size: 0.92rem; line-height: 1.45; }

div.stButton > button {
  background: var(--accent) !important;
  color: var(--accent-fg) !important;
  border: none !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  font-family: var(--font) !important;
  height: 44px !important;
  padding: 0 18px !important;
}
div.stButton > button:hover { filter: brightness(1.06); }
div.stButton > button:active { transform: scale(0.98); }

[data-testid="stTabs"] button { font-family: var(--font) !important; color: var(--fg-muted) !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--fg) !important; }

[data-testid="stDataFrame"] { background: var(--bg-elevated); }

.small-note { color: var(--fg-subtle); font-size: 0.8rem; line-height: 1.45; }
hr.sep { border: none; border-top: 1px solid var(--border); margin: 8px 0 16px 0; }

.story {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  margin: 8px 0 18px 0;
}
@media (max-width: 900px) { .story { grid-template-columns: 1fr 1fr; } }
.story .s {
  background: var(--bg-elevated);
  box-shadow: 0 0 0 1px rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 14px;
}
.story .s .t { font-size: 0.74rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--fg-subtle); }
.story .s .h { font-weight: 600; margin: 6px 0 4px 0; }
.story .s p { color: var(--fg-muted); font-size: 0.86rem; margin: 0; line-height: 1.4; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def html(fragment: str) -> None:
    st.markdown(fragment, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_bundle() -> Dict[str, Any]:
    path = MODELS / "env_bundle.pkl"
    if not path.exists():
        raise FileNotFoundError(
            "models/env_bundle.pkl is missing. Run: python train_evaluate.py --mode full"
        )
    return load_bundle(path)


@st.cache_resource(show_spinner=False)
def get_agent() -> DQNAgent:
    path = MODELS / "dqn_prefetch.pt"
    if not path.exists():
        raise FileNotFoundError(
            "models/dqn_prefetch.pt is missing. Run: python train_evaluate.py --mode full"
        )
    return DQNAgent.from_checkpoint(str(path), seed=0)


@st.cache_data(show_spinner=False)
def get_summary() -> Dict[str, Any]:
    path = RESULTS / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt_ms(x: float) -> str:
    if x >= 1000:
        return f"{x/1000:.2f} s"
    return f"{x:.0f} ms"


def fmt_mb(bytes_or_mb: float, already_mb: bool = False) -> str:
    mb = bytes_or_mb if already_mb else bytes_or_mb / (1024 * 1024)
    if mb < 0.1:
        return f"{mb * 1024:.0f} KB"
    return f"{mb:.2f} MB"


def mean_std_fmt(d: Dict[str, float], digits: int = 1) -> str:
    if not d:
        return "—"
    return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f}"


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------
def make_env(bundle: Dict[str, Any], scenario: str, session_idx: int) -> WebPrefetchEnv:
    sessions = bundle["test"]
    idx = session_idx % max(len(sessions), 1)
    cfg = NetworkConfig.from_scenario(scenario)
    env = WebPrefetchEnv(
        [sessions[idx]],
        bundle["counts"],
        network=cfg,
        reward_mode="latency",
        seed=session_idx,
        episode_shuffle=False,
    )
    return env


def q_bar_html(q: np.ndarray, action: int) -> str:
    qmin, qmax = float(np.min(q)), float(np.max(q))
    span = max(qmax - qmin, 1e-6)

    def row(i: int, label: str) -> str:
        pct = 12 + 88 * ((float(q[i]) - qmin) / span)
        weight = "font-weight:600;color:var(--fg);" if i == action else "color:var(--fg-muted);"
        return (
            f'<div class="qrow"><div class="lab" style="{weight}">{label}</div>'
            f'<div class="qbar"><span style="width:{pct:.1f}%"></span></div>'
            f'<div class="num">{q[i]:+.3f}</div></div>'
        )

    return row(0, "Don't prefetch") + row(1, "Prefetch")


def decision_rationale(state: np.ndarray, q: np.ndarray, action: int) -> str:
    conf = float(state[4])
    size = float(state[5])
    bw = float(state[6])
    rtt = float(state[7])
    occ = float(state[8])
    cached = float(state[9]) > 0.5
    margin = float(q[1] - q[0])
    if cached:
        why = "Predicted resource is already in cache, so a prefetch would not buy extra latency."
    elif action == 1:
        bits = []
        if conf >= 0.45:
            bits.append(f"predictor confidence is {conf:.0%}")
        if rtt >= 0.4:
            bits.append("round-trip delay is high")
        if size < 0.35:
            bits.append("the predicted payload is relatively small")
        if not bits:
            bits.append("the Q-network estimates a net benefit")
        why = "Prefetch selected because " + ", ".join(bits) + "."
    else:
        bits = []
        if conf < 0.4:
            bits.append(f"confidence is only {conf:.0%}")
        if size > 0.6 and bw < 0.45:
            bits.append("a large payload on a constrained link is expensive if wrong")
        if occ > 0.85:
            bits.append("the cache is nearly full")
        if not bits:
            bits.append("expected waste outweighs the latency gain")
        why = "Holding back because " + ", ".join(bits) + "."
    why += f" Q-margin (prefetch − skip) = {margin:+.3f}."
    return why


def init_session(bundle: Dict[str, Any], scenario: str, session_idx: int) -> None:
    env = make_env(bundle, scenario, session_idx)
    obs, info = env.reset(seed=session_idx)
    st.session_state.env = env
    st.session_state.obs = obs
    st.session_state.info = info
    st.session_state.done = False
    st.session_state.last_step = None
    st.session_state.scenario = scenario
    st.session_state.session_idx = session_idx
    st.session_state.log = []
    st.session_state.live = {
        "steps": 0,
        "lat_sum": 0.0,
        "saved_sum": 0.0,
        "bytes": 0.0,
        "useful": 0,
        "wasted": 0,
        "prefetch": 0,
        "reward": 0.0,
        "hits": 0,
    }


def ensure_sim(bundle: Dict[str, Any], scenario: str) -> None:
    if "env" not in st.session_state or st.session_state.get("scenario") != scenario:
        init_session(bundle, scenario, st.session_state.get("session_idx", 0))


def step_sim(policy_name: str, agent: DQNAgent) -> None:
    env: WebPrefetchEnv = st.session_state.env
    if st.session_state.done:
        return
    obs = st.session_state.obs
    policy = POLICIES[policy_name]
    action = int(policy(obs, env, agent if policy_name == "DQN" else None))
    nxt, reward, term, trunc, info = env.step(action)
    live = st.session_state.live
    live["steps"] += 1
    live["lat_sum"] += float(info["latency_ms"])
    live["saved_sum"] += float(info["latency_saved_ms"])
    live["bytes"] += float(info["bytes_used"])
    live["useful"] += int(info["prefetch_useful"])
    live["wasted"] += int(info["prefetch_wasted"])
    live["prefetch"] += int(action == 1)
    live["reward"] += float(reward)
    live["hits"] += int(info["cache_hit"])
    st.session_state.obs = nxt
    st.session_state.info = env.describe_observation(nxt) if not (term or trunc) else info
    st.session_state.done = bool(term or trunc)
    st.session_state.last_step = info
    st.session_state.log.append(info)


# ---------------------------------------------------------------------------
# Layout pieces
# ---------------------------------------------------------------------------
def render_header() -> None:
    html(
        """
        <div class="hero-kicker">Final-year reinforcement learning project</div>
        <h1>Latency-Aware Web Prefetching</h1>
        <p class="hero-sub">
            A Deep Q-Network that learns <em>when</em> to prefetch the predicted next
            resource — reducing load latency without blindly spending bandwidth.
        </p>
        <div class="story">
          <div class="s"><div class="t">No prefetch</div><div class="h">Safe, slow</div><p>Every first request pays full RTT plus transfer.</p></div>
          <div class="s"><div class="t">Always prefetch</div><div class="h">Fast, wasteful</div><p>Wrong guesses still consume cache and bandwidth.</p></div>
          <div class="s"><div class="t">Rule-based</div><div class="h">A fixed compromise</div><p>Confidence threshold with hand-tuned cost guards.</p></div>
          <div class="s"><div class="t">DQN</div><div class="h">Learns the tradeoff</div><p>Acts only when preparing for the request is actually worth it.</p></div>
        </div>
        """
    )


def render_live_metrics() -> None:
    live = st.session_state.live
    steps = max(live["steps"], 1)
    avg_lat = live["lat_sum"] / steps if live["steps"] else 0.0
    hit = (live["useful"] / live["prefetch"]) if live["prefetch"] else 0.0
    waste = (live["wasted"] / live["prefetch"]) if live["prefetch"] else 0.0
    occ = st.session_state.env.cache.occupancy
    html(
        f"""
        <div class="metric-grid">
          <div class="metric-card"><div class="lbl">Average latency</div><div class="val">{fmt_ms(avg_lat)}</div><div class="hint">{live['steps']} requests this session</div></div>
          <div class="metric-card"><div class="lbl">Latency saved</div><div class="val">{fmt_ms(live['saved_sum'])}</div><div class="hint">versus downloading on demand</div></div>
          <div class="metric-card"><div class="lbl">Prefetch hit rate</div><div class="val">{hit*100:.0f}%</div><div class="hint">{live['useful']} useful / {live['prefetch']} prefetches</div></div>
          <div class="metric-card"><div class="lbl">Bandwidth used</div><div class="val">{fmt_mb(live['bytes'])}</div><div class="hint">payload bytes this session</div></div>
          <div class="metric-card"><div class="lbl">Wasted prefetches</div><div class="val">{live['wasted']}</div><div class="hint">{waste*100:.0f}% of prefetch actions</div></div>
          <div class="metric-card"><div class="lbl">Cache utilisation</div><div class="val">{occ*100:.0f}%</div><div class="hint">LRU occupancy</div></div>
          <div class="metric-card"><div class="lbl">Cumulative reward</div><div class="val">{live['reward']:+.2f}</div><div class="hint">latency-aware return</div></div>
          <div class="metric-card"><div class="lbl">Cache hits</div><div class="val">{live['hits']}</div><div class="hint">speculative resources that were prepared ahead of the click</div></div>
        </div>
        """
    )


def render_simulation(bundle: Dict[str, Any], agent: DQNAgent, scenario: str, policy_name: str) -> None:
    ensure_sim(bundle, scenario)
    env: WebPrefetchEnv = st.session_state.env
    obs = st.session_state.obs
    q = agent.q_values(obs)
    dqn_action = int(np.argmax(q))
    if policy_name == "DQN":
        action = dqn_action
    else:
        action = int(POLICIES[policy_name](obs, env, None))

    left, right = st.columns((1.15, 1.0), gap="large")

    with left:
        desc = env.describe_observation(obs) if not st.session_state.done else st.session_state.info
        cache = desc.get("cache") or env.cache.snapshot()
        html(
            f"""
            <div class="panel">
              <h3>Live simulation</h3>
              <div class="kv"><div class="k">Current page</div><div class="v">{desc.get('current_name', '—')}</div></div>
              <div class="kv"><div class="k">Predicted next resource</div><div class="v">{desc.get('predicted_name', '—')}</div></div>
              <div class="kv"><div class="k">Prediction confidence</div><div class="v">{desc.get('confidence', 0)*100:.1f}%</div></div>
              <div class="kv"><div class="k">Predicted size</div><div class="v">{desc.get('predicted_size_kb', 0):.0f} KB</div></div>
              <div class="kv"><div class="k">Network bandwidth</div><div class="v">{desc.get('bandwidth_mbps', env.bw):.2f} Mbps</div></div>
              <div class="kv"><div class="k">Network latency (RTT)</div><div class="v">{desc.get('rtt_ms', env.rtt):.0f} ms</div></div>
              <div class="kv"><div class="k">Cache capacity</div><div class="v">{cache.get('capacity_mb', 0):.1f} MB</div></div>
              <div class="kv"><div class="k">Cache usage</div><div class="v">{cache.get('used_mb', 0):.2f} MB · {cache.get('occupancy', 0)*100:.0f}%</div></div>
              <div class="kv"><div class="k">Predicted already cached</div><div class="v">{'yes' if desc.get('predicted_cached') else 'no'}</div></div>
              <div class="kv"><div class="k">Session progress</div><div class="v">{desc.get('step', 0)} / {max(desc.get('session_length', 1)-1, 1)} remaining {desc.get('remaining', 0)}</div></div>
            </div>
            """
        )

        badge_class = "badge-go" if action == 1 else "badge-stop"
        html(
            f"""
            <div class="panel">
              <h3>RL decision</h3>
              <div class="decision-box">
                <div class="decision-name">{ACTION_NAMES[action]}</div>
                <span class="badge {badge_class}">{policy_name}</span>
              </div>
              {q_bar_html(q, dqn_action)}
              <p class="small-note" style="margin-top:12px">{decision_rationale(obs, q, dqn_action)}</p>
              <p class="small-note">Q-values shown here always come from the trained DQN; the selected policy controls the simulator action.</p>
            </div>
            """
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            go = st.button("Next user request", use_container_width=True, disabled=st.session_state.done)
        with c2:
            nxt_sess = st.button("New session", use_container_width=True)
        with c3:
            reset = st.button("Reset session", use_container_width=True)

        if go:
            step_sim(policy_name, agent)
            st.rerun()
        if nxt_sess:
            init_session(bundle, scenario, st.session_state.session_idx + 1)
            st.rerun()
        if reset:
            init_session(bundle, scenario, st.session_state.session_idx)
            st.rerun()

    with right:
        last = st.session_state.last_step
        html('<div class="panel"><h3>User request simulation</h3>')
        if last is None:
            html(
                """
                <div class="outcome idle">
                  <div class="ttl">Waiting</div>
                  <div class="big">No request yet</div>
                  <p>Press <strong>Next user request</strong>. The agent decides whether to prefetch
                  the predicted resource <em>before</em> the true click is revealed. That is the
                  leakage boundary: the policy never sees the future page in its state.</p>
                </div>
                """
            )
        else:
            hit = bool(last["cache_hit"])
            cls = "hit" if hit else "miss"
            title = "Cache hit" if hit else "Cache miss"
            if last["prefetch_useful"]:
                body = (
                    f"Resource <strong>{last['actual_name']}</strong> was prefetched successfully. "
                    f"Estimated latency saved: <strong>{fmt_ms(last['latency_saved_ms'])}</strong>."
                )
            elif last["prefetch_wasted"]:
                body = (
                    f"Prefetched <strong>{last.get('predicted_name', '—')}</strong> but the user "
                    f"requested <strong>{last['actual_name']}</strong>. Bandwidth spent on a miss, "
                    f"then a normal download of {fmt_ms(last['latency_ms'])}."
                )
            elif hit:
                body = (
                    f"Resource <strong>{last['actual_name']}</strong> was already in cache "
                    f"(earlier visit or completed prefetch). Latency {fmt_ms(last['latency_ms'])}."
                )
            else:
                body = (
                    f"Resource <strong>{last['actual_name']}</strong> was not prefetched. "
                    f"Normal download required. Latency: <strong>{fmt_ms(last['latency_ms'])}</strong>."
                )
            html(
                f"""
                <div class="outcome {cls}">
                  <div class="ttl">{title}</div>
                  <div class="big">{last['action_name']} → {last['actual_name']}</div>
                  <p>{body}</p>
                  <p style="margin-top:8px">
                    Reward {last['reward']:+.3f} · bandwidth {fmt_mb(last['bytes_used'])} ·
                    predictor {'correct' if last['prediction_correct'] else 'wrong'}
                    ({last['confidence']*100:.0f}% confidence)
                  </p>
                </div>
                """
            )
        html("</div>")

        spec = SCENARIOS[scenario]
        html(
            f"""
            <div class="panel">
              <h3>Scenario</h3>
              <div class="kv"><div class="k">Name</div><div class="v">{spec['title']}</div></div>
              <div class="kv"><div class="k">Description</div><div class="v">{spec['description']}</div></div>
              <div class="kv"><div class="k">Nominal bandwidth</div><div class="v">{spec['bandwidth_mbps']} Mbps</div></div>
              <div class="kv"><div class="k">Nominal RTT</div><div class="v">{spec['rtt_ms']} ms</div></div>
              <div class="kv"><div class="k">Cache</div><div class="v">{spec['cache_mb']} MB</div></div>
            </div>
            """
        )

    html("<div style='height:8px'></div>")
    html("<h2>Live metrics</h2>")
    render_live_metrics()

    if st.session_state.log:
        html("<h2 style='margin-top:22px'>Request log</h2>")
        log_df = pd.DataFrame(
            [
                {
                    "predicted": e["predicted_name"],
                    "actual": e["actual_name"],
                    "action": e["action_name"],
                    "hit": e["cache_hit"],
                    "latency_ms": round(e["latency_ms"], 1),
                    "saved_ms": round(e["latency_saved_ms"], 1),
                    "KB": round(e["bytes_used"] / 1024.0, 1),
                    "reward": round(e["reward"], 3),
                }
                for e in st.session_state.log
            ]
        )
        st.dataframe(log_df, use_container_width=True, hide_index=True)


def _agg_moderate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sub = df[df["scenario"] == "moderate"] if "scenario" in df.columns else df
    cols = [
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
    ]
    present = [c for c in cols if c in sub.columns]
    g = sub.groupby("method")[present].agg(["mean", "std"]).reset_index()
    g.columns = ["method"] + [f"{a}_{b}" for a, b in g.columns[1:]]
    return g


def render_comparison() -> None:
    summary = get_summary()
    df = get_csv("comparison.csv")
    html(
        """
        <h2>Model comparison</h2>
        <p class="hero-sub">Held-out MSNBC test sessions, moderate network (simulated).
        Figures are mean ± standard deviation across random seeds. No metric was hand-edited.</p>
        """
    )
    overall = summary.get("moderate_overall") or {}
    if overall:
        order = ["No Prefetch", "Always Prefetch", "Rule-Based", "DQN"]
        rows_html = []
        for m in order:
            o = overall.get(m, {})
            def cell(key, d=1):
                return mean_std_fmt(o.get(key, {}), d)
            rows_html.append(
                f"<tr><td>{m}</td><td>{cell('avg_latency_ms')}</td><td>{cell('p95_latency_ms')}</td>"
                f"<td>{cell('latency_reduction_pct')}</td><td>{cell('prefetch_hit_rate', 2)}</td>"
                f"<td>{cell('unnecessary_prefetch_rate', 2)}</td><td>{cell('bandwidth_mb', 2)}</td>"
                f"<td>{cell('avg_reward', 2)}</td></tr>"
            )
        html(
            """
            <style>
              .cmp { width:100%; border-collapse: collapse; font-size:0.88rem; }
              .cmp th { text-align:left; color:var(--fg-subtle); font-weight:500; font-size:0.72rem;
                        letter-spacing:0.08em; text-transform:uppercase; padding:8px 10px; border-bottom:1px solid var(--border); }
              .cmp td { padding:10px; border-bottom:1px solid var(--border); font-family:var(--mono); font-variant-numeric: tabular-nums; }
              .cmp td:first-child { font-family:var(--font); font-weight:500; }
            </style>
            """
            + "<div class='panel'><table class='cmp'><thead><tr>"
            + "<th>Method</th><th>Avg latency (ms)</th><th>P95 (ms)</th><th>Reduction %</th>"
            + "<th>Prefetch hit</th><th>Unnecessary</th><th>BW (MB)</th><th>Reward</th>"
            + "</tr></thead><tbody>"
            + "".join(rows_html)
            + "</tbody></table></div>"
        )
    elif not df.empty:
        st.dataframe(_agg_moderate(df), use_container_width=True, hide_index=True)
    else:
        st.info("Run python train_evaluate.py --mode full to generate comparison tables.")

    fig = FIGURES / "09_latency_bandwidth_tradeoff.png"
    if fig.exists():
        st.image(str(fig), caption="Latency versus bandwidth on the moderate network (simulated).")
    c1, c2 = st.columns(2)
    with c1:
        p = FIGURES / "04_latency_avg_by_method.png"
        if p.exists():
            st.image(str(p))
    with c2:
        p = FIGURES / "08_bandwidth_by_method.png"
        if p.exists():
            st.image(str(p))


def render_experiments() -> None:
    html(
        """
        <h2>Simulated operating conditions</h2>
        <p class="hero-sub">Bandwidth, RTT, cache size and predictor temperature are simulation
        knobs. Navigation sequences come from held-out MSNBC sessions. The DQN is trained with
        domain-randomised networks so it must read those knobs from the state.</p>
        """
    )
    for fname, caption in [
        ("10_performance_by_network.png", "Average latency by method under each simulated scenario."),
        ("11_performance_by_cache.png", "Cache-capacity sweep on the moderate network."),
        ("05_latency_p95_by_method.png", "Tail latency (P95) on the moderate network."),
        ("06_prefetch_hit_rate.png", "Fraction of prefetch actions that were useful."),
        ("07_unnecessary_prefetch_rate.png", "Fraction of prefetch actions that were wasted."),
    ]:
        p = FIGURES / fname
        if p.exists():
            st.image(str(p), caption=caption)

    html("<h2 style='margin-top:8px'>Ablation — does the latency-aware reward matter?</h2>")
    html(
        "<p class='hero-sub'>Same DQN, same data split. The simple agent gets +1 / −1 for "
        "useful vs wasted prefetches. The latency-aware agent is charged for milliseconds, "
        "bytes, waste and cache pressure.</p>"
    )
    p = FIGURES / "12_ablation_comparison.png"
    if p.exists():
        st.image(str(p), caption="Simple reward versus latency-aware reward (moderate network).")
    ab = get_csv("ablation.csv")
    if not ab.empty:
        show = (
            ab.groupby("method")[["avg_latency_ms", "bandwidth_mb", "prefetch_hit_rate", "unnecessary_prefetch_rate", "avg_reward"]]
            .agg(["mean", "std"])
            .round(3)
        )
        st.dataframe(show, use_container_width=True)


def render_analytics() -> None:
    html(
        """
        <h2>Training diagnostics</h2>
        <p class="hero-sub">DQN on CPU: Q-network, target network, replay buffer, epsilon-greedy,
        SmoothL1 Bellman loss, Adam. Curves are per random seed.</p>
        """
    )
    for fname, caption in [
        ("01_dqn_reward.png", "Episode return while training the latency-aware agent."),
        ("02_dqn_loss.png", "Huber Bellman loss."),
        ("03_dqn_epsilon.png", "Exploration schedule."),
    ]:
        p = FIGURES / fname
        if p.exists():
            st.image(str(p), caption=caption)

    html("<h2>All generated figures</h2>")
    if FIGURES.exists():
        files = sorted(FIGURES.glob("*.png"))
        cols = st.columns(2)
        for i, f in enumerate(files):
            with cols[i % 2]:
                st.image(str(f), caption=f.name)
    else:
        st.info("No figures yet. Run training first.")


def render_about(bundle: Optional[Dict[str, Any]]) -> None:
    summary = get_summary()
    ds = (summary.get("dataset") or {}) if summary else {}
    html(
        f"""
        <h2>What this system does</h2>
        <p class="hero-sub">Traditional prefetching asks “what will the user request next?”
        This project asks a different question: <strong>is preparing for that request actually
        beneficial</strong> given confidence, payload size, bandwidth, RTT and cache pressure?</p>
        <div class="panel">
          <h3>RL formulation</h3>
          <div class="kv"><div class="k">Algorithm</div><div class="v">Deep Q-Network (Mnih et al., 2015)</div></div>
          <div class="kv"><div class="k">State</div><div class="v">12-D vector — page, history, prediction, confidence, size, bandwidth, RTT, cache, previous action, predictability</div></div>
          <div class="kv"><div class="k">Actions</div><div class="v">0 = don't prefetch · 1 = prefetch predicted resource</div></div>
          <div class="kv"><div class="k">Reward</div><div class="v">−latency − bandwidth − waste + hit − cache pressure</div></div>
          <div class="kv"><div class="k">Environment</div><div class="v">Gymnasium WebPrefetchEnv</div></div>
        </div>
        <div class="panel">
          <h3>Dataset and split</h3>
          <div class="kv"><div class="k">Source</div><div class="v">{ds.get('source') or (bundle or {}).get('source', '—')}</div></div>
          <div class="kv"><div class="k">Sessions used</div><div class="v">{ds.get('n_used', '—')} (train {ds.get('n_train', '—')} / val {ds.get('n_val', '—')} / test {ds.get('n_test', '—')})</div></div>
          <div class="kv"><div class="k">Split</div><div class="v">{ds.get('split', 'Session-level 70/15/15, transitions from train only.')}</div></div>
          <div class="kv"><div class="k">Fallback?</div><div class="v">{'YES — demo data, not real experiments' if ds.get('is_fallback') else 'No, real MSNBC sessions'}</div></div>
        </div>
        <p class="small-note">{ds.get('note', '')}</p>
        <p class="small-note">This is not a claim of novelty for web prefetching itself. It is a
        latency-aware RL approach to the prefetch / don't-prefetch decision, evaluated against
        standard strategies under varying simulated network and cache conditions.</p>
        """
    )
    html(
        """
        <div class="panel">
          <h3>State vector</h3>
        """
        + "".join(
            f"<div class='kv'><div class='k'>[{i:02d}] {name}</div><div class='v'>see environment.py</div></div>"
            for i, name in enumerate(STATE_NAMES)
        )
        + "</div>"
    )


def main() -> None:
    inject_css()
    render_header()

    missing = []
    if not (MODELS / "dqn_prefetch.pt").exists():
        missing.append("models/dqn_prefetch.pt")
    if not (MODELS / "env_bundle.pkl").exists():
        missing.append("models/env_bundle.pkl")
    if missing:
        st.warning(
            "Trained artefacts are missing: "
            + ", ".join(missing)
            + ". From this folder run `python train_evaluate.py --mode full`, then refresh."
        )
        render_about(None)
        return

    bundle = get_bundle()
    agent = get_agent()
    agent.q.eval()
    if bundle.get("is_fallback"):
        st.warning("This dashboard is using labelled fallback/demo data. Do not use these results as the reported MSNBC experiment.")

    with st.sidebar:
        html('<p class="hero-kicker">Controls</p><h2>Scenario selector</h2>')
        scenario_labels = {k: v["title"] for k, v in SCENARIOS.items()}
        scenario_key = st.selectbox(
            "Network / cache condition (simulated)",
            list(SCENARIOS.keys()),
            index=list(SCENARIOS.keys()).index("moderate"),
            format_func=lambda k: scenario_labels[k],
        )
        policy_name = st.selectbox(
            "Decision policy",
            ["DQN", "Rule-Based", "Always Prefetch", "No Prefetch"],
            index=0,
        )
        st.caption(
            "Network knobs are simulated. Page sequences are held-out MSNBC sessions. "
            "The DQN never observes the true next page before acting. Network conditions and cache pressure are simulated."
        )
        html("<hr class='sep'>")
        st.caption("Checkpoints")
        st.code("models/dqn_prefetch.pt", language=None)
        summary = get_summary()
        if summary:
            st.caption(
                f"{summary.get('n_seeds')} seeds · {summary.get('n_episodes')} episodes · "
                f"eval {summary.get('n_eval')} sessions / cell"
            )

    tabs = st.tabs(
        ["Live simulation", "Model comparison", "Experiments", "Analytics", "About"]
    )
    with tabs[0]:
        render_simulation(bundle, agent, scenario_key, policy_name)
    with tabs[1]:
        render_comparison()
    with tabs[2]:
        render_experiments()
    with tabs[3]:
        render_analytics()
    with tabs[4]:
        render_about(bundle)


if __name__ == "__main__":
    main()
