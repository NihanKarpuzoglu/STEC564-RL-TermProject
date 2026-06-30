"""Shared utilities for the STEK 564 Term Project.

Covers:
  - flatten_obs : Dict obs -> 181-dim vector (matches offline.py/_flatten_obs)
  - build_mlp   : quick MLP factory
  - OfflineBuffer : wraps D_logs for mini-batch sampling
  - Logger      : CSV writer for training curves
  - compute_depletion_cost : safe-RL cost signal from (obs, next_obs)
"""
from __future__ import annotations

import os
import csv
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────
# Observation layout (from env_dispatch.py / config)
# ──────────────────────────────────────────────────
N_DRONES = 8
N_FEATURES_PER_DRONE = 10   # x, y, soc, alive, status_onehot(5), has_order
K_MAX = 20
N_FEATURES_PER_ORDER = 5    # ox, oy, dx, dy, age
OBS_DIM = N_DRONES * N_FEATURES_PER_DRONE + K_MAX * N_FEATURES_PER_ORDER + 1  # 181
N_ACTIONS = 169              # 8*20 + 8 + 1

# Index of "alive" bit for drone d in the flattened obs
def drone_alive_idx(d: int) -> int:
    return d * N_FEATURES_PER_DRONE + 3

# Index of SoC for drone d
def drone_soc_idx(d: int) -> int:
    return d * N_FEATURES_PER_DRONE + 2


def flatten_obs(obs: dict) -> np.ndarray:
    """Flatten the Gymnasium Dict obs to a fixed 181-dim vector.
    Identical to offline.py:_flatten_obs so online and offline share a format."""
    return np.concatenate([
        obs["drones"].flatten(),
        obs["orders"].flatten(),
        obs["time"].astype(np.float32),
    ]).astype(np.float32)


# ──────────────────────────────────────────────────
# Network builder
# ──────────────────────────────────────────────────

def build_mlp(in_dim: int, out_dim: int,
              hidden_sizes=(256, 256),
              activation=nn.ReLU,
              output_activation=None) -> nn.Sequential:
    dims = [in_dim] + list(hidden_sizes) + [out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────
# Offline replay buffer
# ──────────────────────────────────────────────────

class OfflineBuffer:
    """Wraps the D_logs .npz and exposes mini-batch sampling.

    Optionally injects a *synthetic* reward column so the same buffer
    can be reused for IRL-policy training (pass `reward_override`).
    """

    def __init__(self, path: str, device: str = "cpu",
                 reward_override: Optional[np.ndarray] = None):
        d = np.load(path)
        self.obs     = torch.tensor(d["observations"],     dtype=torch.float32).to(device)
        self.act     = torch.tensor(d["actions"],          dtype=torch.long).to(device)
        self.rew     = torch.tensor(d["rewards"],          dtype=torch.float32).to(device)
        self.next_obs= torch.tensor(d["next_observations"],dtype=torch.float32).to(device)
        # terminal = true done; timeout = truncation (not truly terminal for bootstrap)
        self.done    = torch.tensor(d["terminals"] | d["timeouts"], dtype=torch.float32).to(device)
        self.terminal= torch.tensor(d["terminals"],        dtype=torch.float32).to(device)

        if reward_override is not None:
            self.rew = torch.tensor(reward_override, dtype=torch.float32).to(device)

        self.N = len(self.act)
        self.device = device
        # Pre-compute depletion cost for CMDP
        self._depletion_cost: Optional[torch.Tensor] = None

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.N, (batch_size,))
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.next_obs[idx], self.done[idx], self.terminal[idx])

    @property
    def depletion_cost(self) -> torch.Tensor:
        """Per-transition cost = 1 if any drone just depleted (alive bit fell)."""
        if self._depletion_cost is None:
            alive_before = torch.stack([self.obs[:, drone_alive_idx(d)]
                                        for d in range(N_DRONES)], dim=1)
            alive_after  = torch.stack([self.next_obs[:, drone_alive_idx(d)]
                                        for d in range(N_DRONES)], dim=1)
            self._depletion_cost = (alive_after < alive_before).any(dim=1).float()
        return self._depletion_cost

    def sample_with_cost(self, batch_size: int):
        idx = torch.randint(0, self.N, (batch_size,))
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.next_obs[idx], self.done[idx], self.terminal[idx],
                self.depletion_cost[idx])


# ──────────────────────────────────────────────────
# CSV Logger
# ──────────────────────────────────────────────────

class Logger:
    """Lightweight CSV logger for training curves."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self._rows: list[dict] = []
        self._headers: Optional[list[str]] = None
        self._start = time.time()

    def log(self, **kwargs):
        kwargs.setdefault("wall_time", round(time.time() - self._start, 1))
        if self._headers is None:
            self._headers = list(kwargs.keys())
        self._rows.append(kwargs)

    def flush(self):
        if not self._rows:
            return
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._headers)
            w.writeheader()
            w.writerows(self._rows)

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass


# ──────────────────────────────────────────────────
# Soft target update
# ──────────────────────────────────────────────────

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)
