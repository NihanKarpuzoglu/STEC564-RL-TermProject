"""Preference-Based Reward Model (Bradley-Terry IRL) — Section 3.4.

We construct trajectory segment pairs from D_logs: segment A is preferred
over segment B if sum(r_A) > sum(r_B).  A reward model r_θ(s,a) is trained
to explain these preferences via the Bradley-Terry model:

    P(A ≻ B) = σ( sum_{t∈A} r_θ(s_t, a_t) − sum_{t∈B} r_θ(s_t, a_t) )

Loss = -E[ label·log P(A≻B) + (1-label)·log P(B≻A) ]

This is the direct bridge to the RLHF pipeline (Ch.20, reward model step):
the same loss is used in InstructGPT, RLHF-from-scratch, etc., just with
human labels instead of return-derived labels.

After training, we:
  1. Score every transition in D_logs with r_θ to get recovered_rewards.
  2. Re-run IQL with those recovered rewards to get an IRL policy.
  3. Compare IRL policy vs true-reward IQL and analyse what r_θ captured
     (reward hacking / mis-specification).
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

from utils import OBS_DIM, N_ACTIONS, build_mlp, Logger


# ─────────────────────────────────────────────────────
# Reward model
# ─────────────────────────────────────────────────────

class RewardModel(nn.Module):
    """r_θ(s, a) → scalar reward.

    Input: [flat_obs (181) || one_hot_action (169)] = 350-dim.
    Output: scalar (unbounded; we do NOT apply tanh, letting scale emerge).
    """

    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS,
                 hidden=(256, 128)):
        super().__init__()
        in_dim = obs_dim + n_actions
        self.net = build_mlp(in_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, act_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act_onehot], dim=-1)
        return self.net(x).squeeze(-1)   # (B,)


def _act_onehot(actions: torch.Tensor, n_actions: int) -> torch.Tensor:
    return torch.zeros(actions.shape[0], n_actions,
                       dtype=torch.float32, device=actions.device
                       ).scatter_(1, actions.unsqueeze(1), 1.0)


# ─────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────

def train_reward_model(cfg: dict, data_path: str,
                       save_path: str, log_path: str,
                       seed: int = 0) -> RewardModel:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device   = cfg.get("device", "cpu")
    lr       = cfg.get("lr", 3e-4)
    batch    = cfg.get("batch_size", 256)
    n_steps  = cfg.get("n_steps", 20_000)
    n_pairs  = cfg.get("n_pairs", 2000)
    hidden   = tuple(cfg.get("hidden", [256, 128]))
    log_freq = cfg.get("log_freq", 500)
    seg_seed = cfg.get("seg_seed", 42)

    # Build preference pairs
    import sys, os as _os
    sys.path.insert(0, _os.path.abspath("../drone_dispatch_env"))
    from drone_dispatch_env import make_preference_pairs
    print(f"[IRL] building {n_pairs} preference pairs …")
    pairs = make_preference_pairs(data_path, n_pairs=n_pairs, seed=seg_seed)
    print(f"[IRL] done. seg_len=25, {n_pairs} pairs")

    # Convert to tensors
    obs_a   = torch.tensor(np.stack([p["obs_a"] for p in pairs]),
                           dtype=torch.float32, device=device)   # (N, 25, 181)
    act_a   = torch.tensor(np.stack([p["act_a"] for p in pairs]),
                           dtype=torch.long, device=device)       # (N, 25)
    obs_b   = torch.tensor(np.stack([p["obs_b"] for p in pairs]),
                           dtype=torch.float32, device=device)
    act_b   = torch.tensor(np.stack([p["act_b"] for p in pairs]),
                           dtype=torch.long, device=device)
    labels  = torch.tensor([p["label"] for p in pairs],
                           dtype=torch.float32, device=device)    # (N,)

    N, T, _ = obs_a.shape
    model  = RewardModel(OBS_DIM, N_ACTIONS, hidden).to(device)
    opt    = optim.Adam(model.parameters(), lr=lr)
    logger = Logger(log_path)
    rng    = np.random.default_rng(seed)

    running = {"loss": 0.0, "acc": 0.0}

    for step in range(1, n_steps + 1):
        idx = torch.from_numpy(rng.integers(0, N, batch)).long()

        # Flatten (B, T, 181) and (B, T) → (B*T, 181), (B*T,)
        ob_a_flat = obs_a[idx].reshape(-1, OBS_DIM)
        ac_a_flat = act_a[idx].reshape(-1)
        ob_b_flat = obs_b[idx].reshape(-1, OBS_DIM)
        ac_b_flat = act_b[idx].reshape(-1)

        oh_a = _act_onehot(ac_a_flat, N_ACTIONS)
        oh_b = _act_onehot(ac_b_flat, N_ACTIONS)

        r_a = model(ob_a_flat, oh_a).reshape(batch, T).sum(dim=1)  # (B,)
        r_b = model(ob_b_flat, oh_b).reshape(batch, T).sum(dim=1)  # (B,)

        lbl = labels[idx]  # (B,) ∈ {0,1}

        # Bradley-Terry: P(A≻B) = σ(R_A - R_B)
        logit = r_a - r_b
        loss  = nn.functional.binary_cross_entropy_with_logits(logit, lbl)

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            acc = ((logit > 0).float() == lbl).float().mean().item()

        running["loss"] += loss.item()
        running["acc"]  += acc

        if step % log_freq == 0:
            avg = {k: v / log_freq for k, v in running.items()}
            print(f"[IRL] step {step:5d}/{n_steps}  "
                  f"loss={avg['loss']:.4f}  acc={avg['acc']:.3f}")
            logger.log(step=step, **avg)
            running = {k: 0.0 for k in running}

    logger.flush()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[IRL] saved reward model → {save_path}")
    return model


# ─────────────────────────────────────────────────────
# Score the full dataset with r_θ
# ─────────────────────────────────────────────────────

@torch.no_grad()
def score_dataset(model: RewardModel, data_path: str,
                  device: str = "cpu",
                  batch_size: int = 2048) -> np.ndarray:
    """Return r_θ(s,a) for every transition in D_logs."""
    d = np.load(data_path)
    obs  = torch.tensor(d["observations"],  dtype=torch.float32)
    acts = torch.tensor(d["actions"],       dtype=torch.long)
    N = len(acts)
    scores = []
    model.eval()
    model.to(device)
    for i in range(0, N, batch_size):
        ob  = obs[i:i+batch_size].to(device)
        ac  = acts[i:i+batch_size].to(device)
        oh  = _act_onehot(ac, N_ACTIONS)
        scores.append(model(ob, oh).cpu().numpy())
    return np.concatenate(scores)


# ─────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────

def load_reward_model(save_path: str, device: str = "cpu",
                      hidden=(256, 128)) -> RewardModel:
    model = RewardModel(OBS_DIM, N_ACTIONS, hidden)
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default="configs/irl.yaml")
    ap.add_argument("--data",    default="../drone_dispatch_env/data/D_logs.npz")
    ap.add_argument("--weights", default="weights/reward_model.pt")
    ap.add_argument("--log",     default="logs/irl_reward.csv")
    ap.add_argument("--seed",    type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = train_reward_model(cfg, args.data, args.weights, args.log, args.seed)

    # Score the dataset and save for IRL-policy training
    scores = score_dataset(model, args.data)
    np.save("weights/irl_recovered_rewards.npy", scores)
    print(f"[IRL] recovered rewards: mean={scores.mean():.4f}  "
          f"std={scores.std():.4f}  "
          f"corr-with-env={np.corrcoef(scores, np.load(args.data)['rewards'])[0,1]:.3f}")
