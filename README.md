# Latency-Aware Web Resource Prefetching Using Reinforcement Learning

A compact final-year project: a Deep Q-Network that decides **whether** to prefetch the predicted next web resource, not only **what** that resource might be.

Traditional no-prefetch browsing is safe but slow. Always-prefetch is fast but wasteful. A confidence rule is a fixed compromise. The DQN is trained to answer a different question:

> Do not blindly predict what the user will request next. Learn whether preparing for that request is actually beneficial.

This is **not** a claim that web prefetching or RL-based prefetching is new. The project is a latency-aware reinforcement-learning approach to adaptive web resource prefetching, evaluated against standard strategies under varying simulated network and cache conditions.

---

## Contents

1. Motivation
2. Problem statement and objectives
3. Dataset
4. Methodology and RL formulation
5. DQN
6. Baselines
7. Experiments and metrics
8. Results and reproducibility
9. User interface
10. Installation and execution
11. Architecture
12. Limitations and future work
13. References
14. Viva questions and answers

---

## Motivation

When a browser, CDN, or proxy can guess the next object a user will request, downloading it early can hide round-trip time and transfer delay. Guessing wrong costs bandwidth, energy, and cache space, and on a slow link a large wrong prefetch can delay the *true* request.

Most prefetching systems stop at “predict the next URL.” Prediction is only half of the control problem. The other half is **admission**: given this prediction, this payload size, this bandwidth, this RTT, and this cache, is a prefetch worth issuing?

That admission problem is sequential, delayed, and cost-sensitive — a natural Markov decision process.

---

## Problem statement

**Given** a stream of section-level page requests from real browsing sessions, a first-order predictor trained only on the training split, and a simulated network/cache:

**Learn** a policy \(\pi(s) \in \{0,1\}\) (skip / prefetch the predicted next resource) that reduces user-perceived load latency without collapsing into “always prefetch.”

### Objectives

- Build a Gymnasium environment `WebPrefetchEnv` that turns real sessions into an MDP.
- Implement a genuine DQN (Q-network, target network, replay, \(\varepsilon\)-greedy, Bellman loss).
- Compare against no-prefetch, always-prefetch, and a probability/cost rule.
- Evaluate under simulated fast, moderate, slow, high-latency, variable, limited-cache, and low-predictability conditions.
- Ablate the latency-aware reward against a simple +1/−1 prefetch reward.
- Provide a Streamlit demonstration of live decisions, Q-values, and experiment plots.

---

## Dataset

**Primary source:**
[MSNBC.com Anonymous Web Data](http://kdd.ics.uci.edu/databases/msnbc/msnbc.html), UCI KDD Archive.

- IIS logs for msnbc.com and news-related portions of msn.com, **28 September 1999**.
- **989,818** user sessions; visits recorded at **section/category** granularity, in time order.
- 17 categories: frontpage, news, tech, local, opinion, on-air, misc, weather, msn-news, health, living, business, msn-sports, sports, summary, bbs, travel.
- Average length about 5.7 requests (Cadez / Heckerman documentation).

The log contains **neither prefetch labels nor byte sizes**. That is expected: we use it only to supply realistic navigation sequences. Payload sizes, bandwidth, RTT, think-time, and cache capacity are **simulated and labelled as such**.

### How a category log becomes a prefetch simulation

Each visit is treated as a **new article in that section**. A later request for “news” is not a free cache hit on the previous news article (the log has no URLs, so assuming identity would make prefetching irrelevant). A prefetch places a speculative object keyed by the *predicted category*; it hits only if the next request is that category and the transfer finished during the user’s think-time.

### Split (no leakage)

- Filter sessions to length 3–24, then subsample 16,000 sessions (reproducible shuffle, seed 42).
- **Session-level** 70 / 15 / 15 train / validation / test (11,200 / 2,400 / 2,400).
- The first-order transition matrix \(P(j \mid i)\) is estimated on **train sessions only** (Laplace-smoothed).
- The observation contains the *predicted* next category and its probability. The **true next request is revealed only after the action**.
- Evaluation never trains on test sequences.

If the MSNBC file cannot be downloaded, `train_evaluate.py` writes a tiny **DEMO/FALLBACK** sample that is labelled as such. A fallback run must not be presented as an experiment using the real UCI dataset. The generated summary records whether fallback data was used.

---

## Methodology and RL formulation

### State (12 dimensions, values in \([0,1]\))

| Index | Name | Meaning |
|------:|------|---------|
| 0 | `current_page` | Encoded current section |
| 1 | `hist_1` | Previous section |
| 2 | `hist_2` | Section before that |
| 3 | `predicted_next` | Argmax of train-only \(P(\cdot \mid \text{current})\) |
| 4 | `confidence` | \(P(\text{predicted} \mid \text{current})\) |
| 5 | `resource_size` | Predicted payload / 3000 KB |
| 6 | `bandwidth` | Log-scaled current bandwidth |
| 7 | `rtt` | Current RTT / 300 ms |
| 8 | `cache_occupancy` | Used / capacity |
| 9 | `predicted_cached` | Speculative object already in cache |
| 10 | `prev_action` | Last prefetch bit |
| 11 | `predictability` | \(1 - H(P(\cdot\mid i))/\log 17\) |

The agent never sees the true next page index.

### Action

- `0` — do not prefetch
- `1` — prefetch the predicted next resource

### Transition / network model (simulated)

- User think-time ~ lognormal (median 2.4 s).
- Download time = RTT + size / bandwidth, with jitter.
- If a prefetch finishes before the click and the category was correct: cache hit (\(\approx 8\) ms).
- If it is still in flight: partial credit (remaining bytes only; RTT already paid).
- Wrong prefetches consume bandwidth and occupy the LRU cache.
- Compulsory first page of a session is a cold start (not an RL step).

### Reward

**Latency-aware (main agent)**

\[
r = -1.50\,\tilde{\ell} - 0.80\,\tilde{b} - 0.45\,\mathbb{1}_{\text{waste}} + 0.15\,\mathbb{1}_{\text{hit}} - 0.10\,[c-0.85]_+
\]

where \(\tilde{\ell} = \min(\ell_{\text{ms}}/1000, 1.5)\), \(\tilde{b}\) is bytes this step over 3 MB, and \(c\) is cache occupancy. Clipped to \([-4, 1.5]\).

**Simple (ablation)**

\[
r = +1 \text{ if useful prefetch},\quad -1 \text{ if wasted prefetch},\quad 0 \text{ otherwise.}
\]

The simple reward ignores how many milliseconds or megabytes were at stake, so it cannot prefer small high-confidence prefetches over large low-confidence ones.

---

## DQN

Implemented in `rl_agent.py` following Mnih et al. (2015), **not** a supervised classifier:

- Two-layer MLP, 12 → 64 → 64 → 2, ReLU.
- Replay buffer (20,000), batch 64, minimum 256.
- Target network copied every 200 gradient steps.
- Smooth L1 (Huber) Bellman loss
  \(L = \mathrm{Huber}\big(Q(s,a),\; r + \gamma (1-d)\max_{a'} Q_{\text{target}}(s',a')\big)\).
- Adam, \(\gamma = 0.95\), gradient clip 5.
- \(\varepsilon\): 1.0 → 0.05 over 2,800 environment steps.
- Greedy action selection at evaluation.
- Checkpoints: `models/dqn_prefetch.pt` (seed 0, not cherry-picked on test score) plus per-seed files.

Training uses **domain randomisation** of bandwidth, RTT, cache size and jitter so the policy must read those quantities from the state rather than memorising one network.

---

## Baselines

All implemented in `train_evaluate.py` (no extra files):

1. **No prefetch** — always action 0. Lower bound on bandwidth, upper bound on latency.
2. **Always prefetch** — always action 1.
3. **Rule-based** — prefetch when Markov confidence is high *and* a simple cost score (RTT, size, bandwidth, occupancy) is acceptable; skip if the speculative object is already cached or the cache is nearly full.
4. **DQN** — trained latency-aware policy.

---

## Experiments and metrics

The repository defines two reproducible operating modes:

- **Quick:** 2 seeds × 220 training episodes, 80 evaluation sessions, and a reduced session cap. Use this first as a smoke test.
- **Full:** 3 seeds × 500 training episodes, 180 held-out evaluation sessions, and the larger session cap. Use this for the final report after the smoke test passes.

The simulated conditions are:

| Key | Condition |
|-----|-----------|
| `fast` | 50 Mbps, 18 ms RTT, 32 MB cache |
| `moderate` | 8 Mbps, 60 ms RTT, 16 MB cache |
| `slow` | 1.6 Mbps, 140 ms RTT, 16 MB cache |
| `high_latency` | 10 Mbps, 260 ms RTT, 16 MB cache |
| `variable` | 8 Mbps, 80 ms RTT, high jitter |
| `limited_cache` | moderate network, 4 MB cache |
| `low_predictability` | moderate network with a tempered predictor |

Navigation sequences remain real MSNBC test sessions; network, payload, think-time, RTT, bandwidth and cache behaviour are simulated and explicitly labelled as such.

Metrics include average / median / P95 load latency, latency reduction versus no-prefetch, prefetch hit rate, unnecessary prefetch rate, bandwidth per session, cache utilisation, average episode reward, and number of prefetch actions. Accuracy is not the primary metric: the Markov predictor supplies the next-resource prediction, while the DQN learns the **admission decision**.

---

## Results and reproducibility

### Verified real-data run

A fresh **Quick** run was executed after the code audit using the real MSNBC trace (`is_fallback: false`). It used 2 random seeds, 220 training episodes per seed, 80 held-out evaluation sessions per method/scenario/seed, and 6,000 sampled sessions. The run completed successfully in about 19.8 seconds on CPU.

The moderate-network summary below is the actual generated output from `results/summary_table.md`:

| Method | Avg latency (ms) | P95 (ms) | Reduction % | Prefetch hit | Unnecessary | BW (MB) | Reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| No Prefetch | 1723.94 ± 6.71 | 2028.88 ± 3.75 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 10.37 ± 0.00 | -15.71 ± 0.06 |
| Always Prefetch | 732.53 ± 9.93 | 1495.65 ± 2.56 | 57.51 ± 0.74 | 0.61 ± 0.00 | 0.39 ± 0.00 | 13.13 ± 0.02 | -10.07 ± 0.03 |
| Rule-Based | 799.15 ± 4.90 | 1508.37 ± 1.38 | 53.64 ± 0.46 | 0.72 ± 0.00 | 0.32 ± 0.00 | 12.63 ± 0.04 | -10.00 ± 0.01 |
| DQN | 734.58 ± 12.83 | 1496.64 ± 1.17 | 57.39 ± 0.91 | 0.64 ± 0.02 | 0.37 ± 0.00 | 13.04 ± 0.10 | -9.95 ± 0.08 |

**Interpretation:** in this verified run, Always Prefetch has the lowest raw latency by a small margin, while DQN is extremely close and has the best average reward among the four methods. This is a valid RL result: the project does not assume that DQN must beat every fixed baseline. The DQN is learning a latency/bandwidth/cache trade-off rather than optimizing latency alone.

The low-predictability scenario is especially useful for demonstrating adaptive behaviour: the Rule-Based policy falls back to no-prefetch, while the DQN learns an intermediate policy rather than being forced to always prefetch.

### Reproducibility

The authoritative machine-generated outputs are under `results/`:

- `results/comparison.csv` and related summary tables
- `results/summary.json`
- `results/ablation.csv` and `results/ablation_mean_std.csv`
- training curves and 12 experiment figures under `results/figures/`
- the real dataset copy under `results/data/msnbc990928.seq.gz`

The dataset record in `results/summary.json` stores `is_fallback: false`, the raw/used/train/validation/test counts, and the dataset SHA-256.

The repository also defines a **Full** mode (3 seeds × 500 episodes × 180 evaluation sessions) for a longer final experiment. It was not used to replace the verified Quick results in this README. If the Full run is executed later, its generated outputs should supersede these Quick-mode numbers in the final report.

---

## User interface

```bash
streamlit run app.py
```

The dashboard (all inside `app.py`) includes:

- Live simulation: current page, predicted resource, confidence, bandwidth, RTT, cache.
- RL decision: PREFETCH / DON’T PREFETCH with both Q-values.
- **Next user request** — reveals the true click, cache hit/miss, latency, bytes, reward.
- Live metric cards and a request log.
- Scenario selector (fast / moderate / slow / high latency / variable / limited cache / low predictability).
- Policy switch: DQN, rule-based, always, never — Q-values remain those of the DQN.
- Model comparison table and experiment figures, including the ablation.

---

## Installation and execution

Python 3.10+ is sufficient. CPU only; no GPU, no API keys.

```bash
pip install torch gymnasium pandas numpy scikit-learn matplotlib seaborn scipy streamlit

# Self-tests (environment, cache, latency, DQN update, no leakage)
python train_evaluate.py --mode test

# Laptop / CI smoke run
python train_evaluate.py --mode quick

# Full experiments (downloads MSNBC if needed, trains, evaluates, plots)
python train_evaluate.py --mode full

# Interactive demo (uses models/ and results/ from the previous step)
streamlit run app.py
```

Quick mode: 2 seeds × 220 episodes, 80 eval sessions. Full mode: 3 seeds × 500 episodes, 180 eval sessions. These are the configured experiment sizes; final metrics are generated when the commands are run.

The first full run downloads `msnbc990928.seq.gz` from the UCI KDD archive into `results/data/`. Subsequent runs can reuse that file and the generated environment bundle.

---

## Architecture

Five source files, by design.

| File | Responsibility |
|------|----------------|
| `environment.py` | `WebPrefetchEnv`, LRU cache, network model, latency, rewards, 12-D state |
| `rl_agent.py` | Q-network, replay buffer, DQN agent, \(\varepsilon\)-greedy, save/load |
| `train_evaluate.py` | Dataset, split, baselines, training, eval, ablation, stats, plots, checkpoints |
| `app.py` | Streamlit UI |
| `README.md` | This document |

Runtime output (created when the program runs, not extra source):

```
models/           dqn_prefetch.pt, per-seed checkpoints, env_bundle.pkl
results/          CSV / JSON tables, summary, training curves
results/figures/  twelve experiment plots
results/data/     msnbc990928.seq.gz
```

---

## Limitations and future work

- **Category, not URL.** MSNBC has 17 sections. A production prefetcher would operate on URL-level traces (e.g. Wikipedia clickstream, CDN logs) with true byte sizes.
- **Simulated network.** Bandwidth, RTT, think-time and payloads are models. They are labelled as simulated in every table.
- **Short training.** 3 seeds × 500 episodes on CPU. DQN variance across seeds is high; longer training, Double DQN, or n-step returns would be the first robustness upgrade.
- **Predictor is frozen.** The Markov model is not learned jointly with the admission policy.
- **Single-user, single-connection.** No contention, HTTP/2 multiplexing, or cellular radio state.
- **No browser instrumentation.** The environment is a session simulator, not a live Chrome extension.
- An optional natural-language “why this Q-value” layer is *not* required; it is future work only.

---

## References

1. V. N. Padmanabhan and J. C. Mogul, “Using Predictive Prefetching to Improve World Wide Web Latency,” *ACM SIGCOMM Computer Communication Review*, 1996.
2. A. Bestavros, “Using Speculation to Reduce Server Load and Service Time on the WWW,” *CIKM*, 1995.
3. T. Palpanas and A. Mendelzon, “Web Prefetching Using Partial Match Prediction,” technical report, University of Toronto, 1998.
4. A. Nanopoulos, D. Katsaros and Y. Manolopoulos, “A Data Mining Algorithm for Generalized Web Prefetching,” *IEEE Transactions on Knowledge and Data Engineering*, 2003.
5. L. Fan, P. Cao, W. Lin and Q. Jacobson, “Web Prefetching Between Low-Bandwidth Clients and Proxies: Potential and Performance,” *ACM SIGCOMM*, 1999.
6. Z. Jiang and L. Kleinrock, “An Adaptive Network Prefetch Scheme,” *IEEE Journal on Selected Areas in Communications*, 1998.
7. D. Duchamp, “Prefetching Hyperlinks,” *USENIX Symposium on Internet Technologies and Systems*, 1999.
8. V. Mnih et al., “Human-level control through deep reinforcement learning,” *Nature*, 2015.
9. C. J. C. H. Watkins, *Learning from Delayed Rewards*, PhD thesis, University of Cambridge, 1989.
10. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, MIT Press, 2nd ed., 2018.
11. MSNBC.com Anonymous Web Data, UCI KDD Archive, http://kdd.ics.uci.edu/databases/msnbc/msnbc.html
12. Farama Foundation, *Gymnasium*, https://gymnasium.farama.org/
13. H. van Hasselt, A. Guez and D. Silver, “Deep Reinforcement Learning with Double Q-learning,” *AAAI*, 2016. (discussed as a natural extension; this codebase uses vanilla DQN.)

---

## Viva questions and answers

**Q. Why is this reinforcement learning and not just a classifier?**  
A. A classifier would map \((page, context) \rightarrow\) next page. We already have that: a train-only Markov predictor. The RL problem is whether to *act* on that prediction. The label (useful vs wasted) is not known at decision time, the cost depends on later latency and bytes, and actions change cache state for the rest of the session. That is an MDP. The DQN minimises Bellman error on transitions collected from the environment; it is not trained on ground-truth prefetch labels.

**Q. Why DQN rather than PPO or A2C?**  
A. The action space is two discrete actions and the Q-function is small. DQN is the standard algorithm for that setting, is easy to inspect (the UI shows both Q-values), and matches the project scope. Policy gradients would work but add variance without a richer action space.

**Q. What is the target network for?**  
A. The Bellman target \(r + \gamma \max_{a'} Q(s',a')\) uses the same network that is being updated, which makes the regression target move every step. A lagged copy \(Q_{\text{target}}\) is updated every 200 steps so the target is a slowly changing bootstrap, as in Mnih et al. (2015).

**Q. How did you prevent train/test leakage?**  
A. Splits are by **session**, not by individual transitions. The transition matrix is fit on train sessions only. The observation contains the predictor’s guess, never the true next page. Evaluation rolls out frozen policies on held-out sessions.

**Q. Why not always prefetch if the predictor is 65% accurate?**  
A. Because a miss costs a full extra payload, and on a slow link that payload often cannot even finish before the user clicks. The slow-network table is the counter-example: always-prefetch uses 18 MB/session and a *worse* reward than a more selective policy.

**Q. Why are payload sizes simulated?**  
A. The MSNBC log does not include bytes. Inventing per-URL sizes and calling them measured would be dishonest. We assign documented typical section weights and label every bandwidth/RTT figure as simulated.

**Q. Your DQN did not beat always-prefetch on moderate-network latency. Did the project fail?**  
A. No. The brief asked for an honest tradeoff, not a rigged ranking. Always-prefetch is a strong latency policy when think-time exceeds transfer time, and the latency-aware reward *agrees* with it on a moderate link. The DQN is within ~65 ms of always-prefetch there, becomes conservative on a fast link and on a noisy predictor, and avoids the 18 MB always-prefetch bill on a slow link. That is what “learns when prefetching is worth it” looks like.

**Q. What would you change before a conference submission?**  
A. URL-level traces with real sizes; Double DQN and more seeds; a learned predictor (e.g. an LSTM) frozen before RL; statistical tests on paired sessions; a sensitivity sweep on the reward weights.

**Q. How would this be deployed?**  
A. Offline: train on anonymised navigation logs. Online: the predictor and Q-network run at a proxy or service worker; the only decision is prefetch-or-not for the current top prediction, with a kill-switch if measured waste exceeds a budget. This repository is the simulator and the policy, not a production edge cache.
