"""Implicit Q-Learning (IQL) — Section 3.3.

IQL (Kostrikov et al., 2021) avoids ever querying Q at OOD (s', a') pairs.
The key idea:

  1. V(s) ← expectile regression of Q(s,a) at level tau:
       L_V = E[ L_tau( Q(s,a) - V(s) ) ]
       where L_tau(u) = |tau - 1(u<0)| * u^2
     For tau=0.7 this learns the ~70th-percentile of Q under the data
     distribution — implicit pessimism without querying OOD actions.

  2. Q(s,a) ← one-step TD with V as the bootstrap target (NO max_{a'}):
       L_Q = E[ 0.5 * (Q(s,a) - (r + γ·V(s')))^2 ]

  3. Policy ← advantage-weighted regression (AWR):
       L_π = E[ -exp(β·(Q(s,a)-V(s))) · log π(a|s) ]
     β controls how sharply the policy concentrates on high-A actions.

Justification for IQL over CQL on this dataset:
  - The dataset is *mixed-quality* (60% greedy, 40% random).  CQL's global
    logsumexp penalty is most important when the dataset has heavy OOD
    gaps; IQL's expectile gives a softer, more data-adaptive pessimism
    that suits heterogeneous support better.
  - IQL has only one sensitive hyperparameter (tau) vs. CQL's alpha, and
    tau is interpretable as a quantile level.
"""
from __future__ import annotations

import os
import argparse
import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from utils import (OBS_DIM, N_ACTIONS, flatten_obs, build_mlp,
                   OfflineBuffer, Logger, soft_update)


# ─────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────

class VNet(nn.Module):
    """State-value function V(s) → scalar."""
    def __init__(self, obs_dim=OBS_DIM, hidden=(256, 256)):
        super().__init__()
        self.net = build_mlp(obs_dim, 1, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)   # (B,)


class QNet(nn.Module):
    """Action-value function Q(s,·) → (n_actions,) vector."""
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS,
                 hidden=(256, 256)):
        super().__init__()
        self.net = build_mlp(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)   # (B, n_actions)


class PolicyNet(nn.Module):
    """Discrete policy π(·|s) → log-probs."""
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS,
                 hidden=(256, 256)):
        super().__init__()
        self.net = build_mlp(obs_dim, n_actions, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.net(obs), dim=-1)   # (B, n_actions)


# ─────────────────────────────────────────────────────
# IQL training
# ─────────────────────────────────────────────────────

def _expectile_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric L2: |tau - 1(u<0)| * u^2"""
    weight = torch.where(u < 0, torch.full_like(u, tau),
                         torch.full_like(u, 1.0 - tau))
    return (weight * u.pow(2)).mean()


def train_iql(cfg: dict, data_path: str,
              save_path_q: str, save_path_v: str, save_path_pi: str,
              log_path: str, seed: int = 0,
              reward_override: Optional[np.ndarray] = None):
    """Train IQL on the offline buffer.

    `reward_override` lets us reuse this function for IRL-policy training
    (replace env rewards with recovered rewards).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device    = cfg.get("device", "cpu")
    lr        = cfg.get("lr", 3e-4)
    batch     = cfg.get("batch_size", 256)
    n_steps   = cfg.get("n_steps", 100_000)
    gamma     = cfg.get("gamma", 0.99)
    tau_soft  = cfg.get("tau_soft", 0.005)
    expectile = cfg.get("expectile", 0.7)
    beta      = cfg.get("beta", 3.0)      # AWR temperature
    hidden    = tuple(cfg.get("hidden", [256, 256]))
    log_freq  = cfg.get("log_freq", 1000)

    buf  = OfflineBuffer(data_path, device=device,
                         reward_override=reward_override)

    v_net  = VNet(OBS_DIM, hidden).to(device)
    q_net  = QNet(OBS_DIM, N_ACTIONS, hidden).to(device)
    pi_net = PolicyNet(OBS_DIM, N_ACTIONS, hidden).to(device)
    q_tgt  = copy.deepcopy(q_net)

    opt_v  = optim.Adam(v_net.parameters(),  lr=lr)
    opt_q  = optim.Adam(q_net.parameters(),  lr=lr)
    opt_pi = optim.Adam(pi_net.parameters(), lr=lr)
    logger = Logger(log_path)

    running = {"loss_v": 0.0, "loss_q": 0.0, "loss_pi": 0.0,
               "q_data": 0.0, "adv_mean": 0.0}

    for step in range(1, n_steps + 1):
        obs_b, act_b, rew_b, nobs_b, done_b, term_b = buf.sample(batch)

        # ── 1. V update (expectile regression on Q-targets) ─────────────
        with torch.no_grad():
            q_all   = q_tgt(obs_b)                          # (B, 169)
            q_taken = q_all.gather(1, act_b.unsqueeze(1)).squeeze(1)  # (B,)
        v_pred = v_net(obs_b)
        loss_v = _expectile_loss(q_taken - v_pred, expectile)

        opt_v.zero_grad()
        loss_v.backward()
        opt_v.step()

        # ── 2. Q update (TD with V bootstrap, no OOD max) ───────────────
        with torch.no_grad():
            v_next  = v_net(nobs_b)
            # Only use V bootstrap when NOT truly terminal
            td_target = rew_b + gamma * (1.0 - term_b) * v_next

        q_all_cur = q_net(obs_b)
        q_taken_cur = q_all_cur.gather(1, act_b.unsqueeze(1)).squeeze(1)
        loss_q = 0.5 * nn.functional.mse_loss(q_taken_cur, td_target)

        opt_q.zero_grad()
        loss_q.backward()
        opt_q.step()

        # ── 3. Policy update (AWR) ──────────────────────────────────────
        with torch.no_grad():
            adv = q_taken - v_pred.detach()                 # A(s,a) = Q - V
            # Clip advantage weights to avoid instability
            weight = torch.exp(beta * adv).clamp(max=100.0)

        log_pi = pi_net(obs_b)                              # (B, 169)
        log_pi_taken = log_pi.gather(1, act_b.unsqueeze(1)).squeeze(1)
        loss_pi = -(weight * log_pi_taken).mean()

        opt_pi.zero_grad()
        loss_pi.backward()
        opt_pi.step()

        # ── Soft target update ──────────────────────────────────────────
        soft_update(q_tgt, q_net, tau_soft)

        # ── Logging ─────────────────────────────────────────────────────
        running["loss_v"]   += loss_v.item()
        running["loss_q"]   += loss_q.item()
        running["loss_pi"]  += loss_pi.item()
        running["q_data"]   += q_taken_cur.detach().mean().item()
        running["adv_mean"] += adv.mean().item()

        if step % log_freq == 0:
            avg = {k: v / log_freq for k, v in running.items()}
            print(f"[IQL] step {step:6d}/{n_steps}  "
                  f"L_V={avg['loss_v']:.4f}  "
                  f"L_Q={avg['loss_q']:.4f}  "
                  f"L_π={avg['loss_pi']:.4f}  "
                  f"Q={avg['q_data']:.2f}  "
                  f"A={avg['adv_mean']:.3f}")
            logger.log(step=step, **avg)
            running = {k: 0.0 for k in running}

    logger.flush()
    for path, net in [(save_path_q, q_net),
                      (save_path_v, v_net),
                      (save_path_pi, pi_net)]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(net.state_dict(), path)
    print(f"[IQL] saved Q→{save_path_q}  V→{save_path_v}  π→{save_path_pi}")

    return IQLPolicy(q_net, pi_net, device)


# ─────────────────────────────────────────────────────
# Policy wrapper
# ─────────────────────────────────────────────────────

class IQLPolicy:
    """Greedy-argmax over masked Q-values at eval time."""

    def __init__(self, q_net: QNet, pi_net: PolicyNet, device: str = "cpu"):
        self.q_net  = q_net.to(device).eval()
        self.pi_net = pi_net.to(device).eval()
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

    def action_probs(self, obs: dict) -> Optional[np.ndarray]:
        flat = torch.tensor(flatten_obs(obs), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        with torch.no_grad():
            log_pi = self.pi_net(flat).squeeze(0)
        mask = torch.tensor(np.asarray(obs["action_mask"], dtype=bool))
        log_pi[~mask] = -1e9
        return torch.softmax(log_pi, dim=-1).cpu().numpy()


# ─────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────

def load_iql(q_path: str, v_path: str, pi_path: str,
             device: str = "cpu", hidden=(256, 256)) -> IQLPolicy:
    q_net  = QNet(OBS_DIM, N_ACTIONS, hidden)
    v_net  = VNet(OBS_DIM, hidden)
    pi_net = PolicyNet(OBS_DIM, N_ACTIONS, hidden)
    q_net.load_state_dict(torch.load(q_path,  map_location=device))
    v_net.load_state_dict(torch.load(v_path,  map_location=device))
    pi_net.load_state_dict(torch.load(pi_path, map_location=device))
    return IQLPolicy(q_net, pi_net, device)


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default="configs/iql.yaml")
    ap.add_argument("--data",    default="../drone_dispatch_env/data/D_logs.npz")
    ap.add_argument("--save-q",  default="weights/iql_q.pt")
    ap.add_argument("--save-v",  default="weights/iql_v.pt")
    ap.add_argument("--save-pi", default="weights/iql_pi.pt")
    ap.add_argument("--log",     default="logs/iql.csv")
    ap.add_argument("--seed",    type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    policy = train_iql(cfg, args.data,
                       args.save_q, args.save_v, args.save_pi,
                       args.log, args.seed)

    import sys
    sys.path.insert(0, "../drone_dispatch_env")
    import drone_dispatch_env
    from drone_dispatch_env import evaluate, Config
    results = evaluate(policy, Config(), seeds=[0, 1, 2])
    print("\nIQL quick eval (seeds 0-2):")
    for k, v in results["mean"].items():
        print(f"  {k}: {v:.4f}")
