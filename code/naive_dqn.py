"""Naive Offline DQN — Section 3.2.

Standard DQN applied naively to D_logs.  No OOD correction.

Failure mode:
  The Bellman backup bootstraps from max_{a'} Q(s', a'), which includes
  actions never seen in the dataset.  The network assigns arbitrarily
  large Q-values to those OOD actions -> bootstrapped target inflates ->
  Q-values diverge or saturate, yielding a degenerate policy.

We log:
  - mean Q-value on the dataset (should explode or saturate)
  - mean *max* Q-value over all actions at sampled s' (shows the OOD gap)
  - eval metrics every eval_freq steps (should plateau / regress)
"""
from __future__ import annotations

import os
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from utils import (OBS_DIM, N_ACTIONS, flatten_obs, build_mlp,
                   OfflineBuffer, Logger, soft_update)


# ─────────────────────────────────────────────────────
# Q-network
# ─────────────────────────────────────────────────────

class QNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS,
                 hidden=(256, 256)):
        super().__init__()
        self.net = build_mlp(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)   # (B, n_actions)


# ─────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────

class NaiveDQNPolicy:
    def __init__(self, q_net: QNet, device: str = "cpu"):
        self.q_net = q_net.to(device)
        self.q_net.eval()
        self.device = device

    def act(self, obs: dict) -> int:
        flat = torch.tensor(flatten_obs(obs), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(flat).squeeze(0).cpu().numpy()
        mask = np.asarray(obs["action_mask"], dtype=bool)
        q[~mask] = -np.inf
        return int(np.argmax(q))

    def action_values(self, obs: dict) -> Optional[np.ndarray]:
        flat = torch.tensor(flatten_obs(obs), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(flat).squeeze(0).cpu().numpy()
        mask = np.asarray(obs["action_mask"], dtype=bool)
        q[~mask] = np.nan
        return q


# ─────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────

def train_naive_dqn(cfg: dict, data_path: str, save_path: str,
                    log_path: str, seed: int = 0) -> NaiveDQNPolicy:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device   = cfg.get("device", "cpu")
    lr       = cfg.get("lr", 1e-3)
    batch    = cfg.get("batch_size", 256)
    n_steps  = cfg.get("n_steps", 50_000)
    gamma    = cfg.get("gamma", 0.99)
    tau      = cfg.get("tau", 0.005)
    hidden   = tuple(cfg.get("hidden", [256, 256]))
    log_freq = cfg.get("log_freq", 1000)

    buf      = OfflineBuffer(data_path, device=device)
    q_net    = QNet(OBS_DIM, N_ACTIONS, hidden).to(device)
    q_target = QNet(OBS_DIM, N_ACTIONS, hidden).to(device)
    q_target.load_state_dict(q_net.state_dict())
    opt      = optim.Adam(q_net.parameters(), lr=lr)
    logger   = Logger(log_path)

    running = {"td_loss": 0.0, "q_data": 0.0, "q_max_ood": 0.0}

    for step in range(1, n_steps + 1):
        obs_b, act_b, rew_b, nobs_b, done_b, _ = buf.sample(batch)

        with torch.no_grad():
            # Naive bootstrap: max over ALL 169 actions — includes OOD
            q_next = q_target(nobs_b).max(dim=1).values        # (B,)
            target = rew_b + gamma * (1.0 - done_b) * q_next   # (B,)

        q_all  = q_net(obs_b)                                   # (B, 169)
        q_taken = q_all.gather(1, act_b.unsqueeze(1)).squeeze(1)  # (B,)
        loss   = nn.functional.mse_loss(q_taken, target)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        opt.step()
        soft_update(q_target, q_net, tau)

        # Diagnostics: track OOD inflation
        running["td_loss"]   += loss.item()
        running["q_data"]    += q_taken.detach().mean().item()
        running["q_max_ood"] += q_all.detach().max(dim=1).values.mean().item()

        if step % log_freq == 0:
            avg = {k: v / log_freq for k, v in running.items()}
            print(f"[NaiveDQN] step {step:6d}/{n_steps}  "
                  f"td_loss={avg['td_loss']:.4f}  "
                  f"Q_data={avg['q_data']:.2f}  "
                  f"Q_max_ood={avg['q_max_ood']:.2f}")
            logger.log(step=step, **avg)
            running = {k: 0.0 for k in running}

    logger.flush()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(q_net.state_dict(), save_path)
    print(f"[NaiveDQN] saved → {save_path}")

    q_net.eval()
    return NaiveDQNPolicy(q_net, device)


# ─────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────

def load_naive_dqn(save_path: str, device: str = "cpu",
                   hidden=(256, 256)) -> NaiveDQNPolicy:
    net = QNet(OBS_DIM, N_ACTIONS, hidden)
    net.load_state_dict(torch.load(save_path, map_location=device))
    return NaiveDQNPolicy(net, device)


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default="configs/naive_dqn.yaml")
    ap.add_argument("--data",    default="../drone_dispatch_env/data/D_logs.npz")
    ap.add_argument("--weights", default="weights/naive.pt")
    ap.add_argument("--log",     default="logs/naive_dqn.csv")
    ap.add_argument("--seed",    type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    policy = train_naive_dqn(cfg, args.data, args.weights, args.log, args.seed)

    import sys
    sys.path.insert(0, "../drone_dispatch_env")
    import drone_dispatch_env
    from drone_dispatch_env import evaluate, Config
    results = evaluate(policy, Config(), seeds=[0, 1, 2])
    print("\nNaive DQN quick eval (seeds 0-2):")
    for k, v in results["mean"].items():
        print(f"  {k}: {v:.4f}")
