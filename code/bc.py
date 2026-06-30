"""Behavioral Cloning (BC) — Section 3.1.

Trains a policy by supervised imitation of the dataset's actions via
cross-entropy loss (multi-class classification). At inference time the
valid action mask is applied so only legal moves are chosen.

Architecture: MLP(181 → 256 → 256 → 169)
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
                   OfflineBuffer, Logger)


# ─────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────

class BCNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS,
                 hidden=(256, 256)):
        super().__init__()
        self.net = build_mlp(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)          # raw logits, shape (B, 169)


# ─────────────────────────────────────────────────────
# Policy (implements the agent_interface.Policy protocol)
# ─────────────────────────────────────────────────────

class BCPolicy:
    """Wraps a trained BCNet and satisfies the Policy protocol.

    Acts greedily (argmax of masked logits) — deterministic for eval.
    """

    def __init__(self, net: BCNet, device: str = "cpu"):
        self.net = net.to(device)
        self.net.eval()
        self.device = device

    def act(self, obs: dict) -> int:
        flat = torch.tensor(flatten_obs(obs), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.net(flat).squeeze(0).cpu().numpy()          # (169,)
        mask = np.asarray(obs["action_mask"], dtype=bool)
        logits[~mask] = -np.inf
        return int(np.argmax(logits))

    # Optional: action probability overlay for the visualizer
    def action_probs(self, obs: dict) -> Optional[np.ndarray]:
        flat = torch.tensor(flatten_obs(obs), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.net(flat).squeeze(0)
        mask = torch.tensor(np.asarray(obs["action_mask"], dtype=bool))
        logits[~mask] = -1e9
        return torch.softmax(logits, dim=-1).cpu().numpy()


# ─────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────

def train_bc(cfg: dict, data_path: str, save_path: str,
             log_path: str, seed: int = 0) -> BCPolicy:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device   = cfg.get("device", "cpu")
    lr       = cfg.get("lr", 3e-4)
    batch    = cfg.get("batch_size", 256)
    n_steps  = cfg.get("n_steps", 50_000)
    hidden   = tuple(cfg.get("hidden", [256, 256]))
    log_freq = cfg.get("log_freq", 1000)

    buf  = OfflineBuffer(data_path, device=device)
    net  = BCNet(OBS_DIM, N_ACTIONS, hidden).to(device)
    opt  = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    logger = Logger(log_path)

    net.train()
    running_loss = 0.0
    for step in range(1, n_steps + 1):
        obs_b, act_b, *_ = buf.sample(batch)
        logits = net(obs_b)                           # (B, 169)
        loss   = loss_fn(logits, act_b)               # CE over observed actions

        opt.zero_grad()
        loss.backward()
        opt.step()

        running_loss += loss.item()
        if step % log_freq == 0:
            avg = running_loss / log_freq
            print(f"[BC] step {step:6d}/{n_steps}  loss={avg:.4f}")
            logger.log(step=step, loss=avg)
            running_loss = 0.0

    logger.flush()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(net.state_dict(), save_path)
    print(f"[BC] saved → {save_path}")

    net.eval()
    return BCPolicy(net, device)


# ─────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────

def load_bc(save_path: str, device: str = "cpu",
            hidden=(256, 256)) -> BCPolicy:
    net = BCNet(OBS_DIM, N_ACTIONS, hidden)
    net.load_state_dict(torch.load(save_path, map_location=device))
    return BCPolicy(net, device)


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default="configs/bc.yaml")
    ap.add_argument("--data",    default="../drone_dispatch_env/data/D_logs.npz")
    ap.add_argument("--weights", default="weights/bc.pt")
    ap.add_argument("--log",     default="logs/bc.csv")
    ap.add_argument("--seed",    type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    policy = train_bc(cfg, args.data, args.weights, args.log, args.seed)

    # Quick smoke-test eval on 3 seeds
    import sys
    sys.path.insert(0, "../drone_dispatch_env")
    import drone_dispatch_env
    from drone_dispatch_env import evaluate, Config
    results = evaluate(policy, Config(), seeds=[0, 1, 2])
    print("\nBC quick eval (seeds 0-2):")
    for k, v in results["mean"].items():
        print(f"  {k}: {v:.4f}")
