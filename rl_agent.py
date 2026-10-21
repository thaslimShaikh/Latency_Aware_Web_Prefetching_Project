"""Deep Q-Network agent for the latency-aware web-prefetching environment.

This is a genuine DQN implementation: replay memory, epsilon-greedy
exploration, a lagged target network, Bellman targets, Huber loss, Adam,
gradient clipping, and checkpoint persistence.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from environment import STATE_DIM

DEVICE = torch.device("cpu")
N_ACTIONS = 2


@dataclass
class DQNConfig:
    state_dim: int = STATE_DIM
    n_actions: int = N_ACTIONS
    hidden: int = 64
    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 20_000
    min_buffer: int = 256
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 2_800
    target_update_every: int = 200
    grad_clip: float = 5.0
    train_every: int = 1
    updates_per_step: int = 1

    def validate(self) -> None:
        if self.state_dim <= 0 or self.n_actions <= 1:
            raise ValueError("state_dim must be positive and n_actions must be > 1")
        if self.batch_size <= 0 or self.buffer_size < self.batch_size:
            raise ValueError("Invalid replay-buffer/batch configuration")
        if self.min_buffer < self.batch_size:
            raise ValueError("min_buffer must be >= batch_size")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.lr <= 0 or self.hidden <= 0:
            raise ValueError("hidden and lr must be positive")
        if self.target_update_every <= 0 or self.train_every <= 0 or self.updates_per_step <= 0:
            raise ValueError("training intervals must be positive")
        if not 0.0 <= self.eps_end <= self.eps_start <= 1.0:
            raise ValueError("epsilon bounds must satisfy 0 <= end <= start <= 1")


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-size uniform experience replay buffer."""

    def __init__(self, capacity: int, state_dim: int) -> None:
        if capacity <= 0 or state_dim <= 0:
            raise ValueError("capacity and state_dim must be positive")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.position

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        if state.shape != (self.state_dim,) or next_state.shape != (self.state_dim,):
            raise ValueError(f"states must have shape ({self.state_dim},)")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(next_state)):
            raise ValueError("Replay transition contains non-finite state values")
        if not np.isfinite(reward):
            raise ValueError("Replay transition contains non-finite reward")

        i = self.position
        self.states[i] = state
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_states[i] = next_state
        self.dones[i] = 1.0 if done else 0.0
        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[torch.Tensor, ...]:
        n = len(self)
        if batch_size > n:
            raise ValueError(f"Cannot sample {batch_size} transitions from {n}")
        indices = rng.choice(n, size=batch_size, replace=False)
        return (
            torch.from_numpy(self.states[indices]).to(DEVICE),
            torch.from_numpy(self.actions[indices]).to(DEVICE),
            torch.from_numpy(self.rewards[indices]).to(DEVICE),
            torch.from_numpy(self.next_states[indices]).to(DEVICE),
            torch.from_numpy(self.dones[indices]).to(DEVICE),
        )


class DQNAgent:
    def __init__(self, config: Optional[DQNConfig] = None, seed: int = 0) -> None:
        self.config = config or DQNConfig()
        self.config.validate()
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        cfg = self.config
        self.q = QNetwork(cfg.state_dim, cfg.n_actions, cfg.hidden).to(DEVICE)
        self.target = QNetwork(cfg.state_dim, cfg.n_actions, cfg.hidden).to(DEVICE)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.optim = Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, cfg.state_dim)

        self.epsilon = cfg.eps_start
        self.total_steps = 0
        self.train_steps = 0
        self.last_loss = 0.0
        self.loss_history: List[float] = []
        self.eps_history: List[float] = []

    def _validate_state(self, state: np.ndarray) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32)
        if arr.shape != (self.config.state_dim,):
            raise ValueError(
                f"Expected state shape ({self.config.state_dim},), got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("State contains non-finite values")
        return arr

    # ---------------------------------------------------------------- policy
    def q_values(self, state: np.ndarray) -> np.ndarray:
        arr = self._validate_state(state)
        x = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
        self.q.eval()
        with torch.no_grad():
            values = self.q(x).squeeze(0).cpu().numpy()
        return values.astype(np.float32)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.config.n_actions))
        return int(np.argmax(self.q_values(state)))

    def decay_epsilon(self) -> None:
        cfg = self.config
        frac = min(1.0, self.total_steps / max(cfg.eps_decay_steps, 1))
        self.epsilon = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

    # -------------------------------------------------------------- learning
    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        cfg = self.config
        if len(self.buffer) < max(cfg.min_buffer, cfg.batch_size):
            return None

        self.q.train()
        states, actions, rewards, next_states, dones = self.buffer.sample(
            cfg.batch_size, self.rng
        )
        q_sa = self.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1).values
            targets = rewards + cfg.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(q_sa, targets)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), cfg.grad_clip)
        self.optim.step()

        self.train_steps += 1
        if self.train_steps % cfg.target_update_every == 0:
            self.target.load_state_dict(self.q.state_dict())

        value = float(loss.detach().cpu().item())
        self.last_loss = value
        self.loss_history.append(value)
        return value

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        self.push(state, action, reward, next_state, done)
        self.total_steps += 1
        self.decay_epsilon()
        self.eps_history.append(self.epsilon)

        if self.total_steps % self.config.train_every != 0:
            return None

        loss = None
        for _ in range(self.config.updates_per_step):
            loss = self.train_step()
        return loss

    # ----------------------------------------------------------- persistence
    def save(self, path: str, extra: Optional[Dict] = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 2,
            "q": self.q.state_dict(),
            "target": self.target.state_dict(),
            "config": asdict(self.config),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "train_steps": self.train_steps,
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, target)

    def load(self, path: str, map_location: str = "cpu") -> Dict:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        cfg_dict = payload.get("config") or {}
        current = asdict(self.config)
        for key, value in cfg_dict.items():
            if key in current:
                current[key] = value
        loaded_cfg = DQNConfig(**current)
        loaded_cfg.validate()

        # Check architecture compatibility before loading weights.
        if loaded_cfg.state_dim != self.config.state_dim or loaded_cfg.n_actions != self.config.n_actions:
            raise ValueError("Checkpoint state/action dimensions do not match the current environment")
        if loaded_cfg.hidden != self.config.hidden:
            raise ValueError("Checkpoint hidden-layer width does not match the current agent")

        self.config = loaded_cfg
        self.q.load_state_dict(payload["q"])
        self.target.load_state_dict(payload.get("target", payload["q"]))
        self.epsilon = float(payload.get("epsilon", self.config.eps_end))
        self.total_steps = int(payload.get("total_steps", 0))
        self.train_steps = int(payload.get("train_steps", 0))
        self.q.eval()
        self.target.eval()
        return payload.get("extra") or {}

    @classmethod
    def from_checkpoint(cls, path: str, seed: int = 0) -> "DQNAgent":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = DQNConfig(**(payload.get("config") or {}))
        agent = cls(config=cfg, seed=seed)
        agent.q.load_state_dict(payload["q"])
        agent.target.load_state_dict(payload.get("target", payload["q"]))
        agent.epsilon = float(payload.get("epsilon", cfg.eps_end))
        agent.total_steps = int(payload.get("total_steps", 0))
        agent.train_steps = int(payload.get("train_steps", 0))
        agent.q.eval()
        agent.target.eval()
        return agent


__all__ = ["DQNConfig", "QNetwork", "ReplayBuffer", "DQNAgent", "DEVICE"]
