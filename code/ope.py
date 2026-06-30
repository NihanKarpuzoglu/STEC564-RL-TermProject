"""Off-Policy Evaluation (OPE) — Section 3.6.

We estimate the value of the learned IQL policy from D_logs alone,
BEFORE running it live.  Two estimators:

  IS  (Importance Sampling / Horvath-Thompson):
      V̂_IS = (1/n) Σ_τ [ (Π_t ρ_t) · G_τ ]
      where ρ_t = π(a_t|s_t) / μ(a_t|s_t) and G_τ = Σ γ^t r_t

  PDIS (Per-Decision IS, Thomas & Brunskill 2016):
      V̂_PDIS = (1/n) Σ_τ Σ_t [ γ^t · (Π_{k≤t} ρ_k) · r_t ]
      Lower variance than IS because the ratio only covers relevant steps.

  DR  (Doubly Robust, Jiang & Li 2016) — bonus estimator:
      V̂_DR = V̂_DM + (1/n) Σ_τ Σ_t [ γ^t · w_t · (r_t + γ·Q̂(s_{t+1},·) − Q̂(s_t,a_t)) ]
      Combines a direct model (DM) with IS correction; more robust when
      either the reward model or IS weights are imperfect.

Why OPE is hard here:
  - The behavior policy is *mixed*: 60% greedy, 40% random.  We do not
    have access to the exact greedy probabilities, only the observed actions.
    We approximate μ(a|s) with a BC model trained on the data.
  - The dataset has large coverage gaps (some regions never visited by
    greedy, some by random), so IS ratios can be very large or zero,
    causing high variance.
  - Trajectories are length ~100 steps each; cumulative products of ρ
    can underflow to zero or overflow, making naive IS unreliable.
    We use weight clipping (ρ_max=20) to reduce variance at the cost of bias.
"""
from __future__ import annotations

import os
import argparse
from typing import Optional

import numpy as np
import torch
import yaml

from utils import OBS_DIM, N_ACTIONS
from bc import BCNet
from iql import QNet, VNet, PolicyNet


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

def _get_episode_slices(terminals: np.ndarray, timeouts: np.ndarray):
    """Return list of (start, end+1) slices for each episode."""
    boundary = terminals | timeouts
    ends = list(np.flatnonzero(boundary))
    if not boundary[-1]:
        ends.append(len(boundary) - 1)
    starts = [0] + [e + 1 for e in ends[:-1]]
    return [(s, e + 1) for s, e in zip(starts, ends) if e + 1 > s]


@torch.no_grad()
def _policy_probs(net: PolicyNet, obs: np.ndarray,
                  device: str, batch: int = 2048) -> np.ndarray:
    """π(·|s) for every row in obs, shape (N, n_actions)."""
    net.eval()
    out = []
    obs_t = torch.tensor(obs, dtype=torch.float32)
    for i in range(0, len(obs_t), batch):
        out.append(torch.softmax(
            torch.exp(net(obs_t[i:i+batch].to(device))),
            dim=-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def _bc_probs(bc_net: BCNet, obs: np.ndarray,
              device: str, batch: int = 2048) -> np.ndarray:
    """μ(·|s) via BC model."""
    bc_net.eval()
    out = []
    obs_t = torch.tensor(obs, dtype=torch.float32)
    for i in range(0, len(obs_t), batch):
        out.append(torch.softmax(
            bc_net(obs_t[i:i+batch].to(device)),
            dim=-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def _q_values(q_net: QNet, obs: np.ndarray,
              device: str, batch: int = 2048) -> np.ndarray:
    """Q(s,·) for every row, shape (N, n_actions)."""
    q_net.eval()
    out = []
    obs_t = torch.tensor(obs, dtype=torch.float32)
    for i in range(0, len(obs_t), batch):
        out.append(q_net(obs_t[i:i+batch].to(device)).cpu().numpy())
    return np.concatenate(out)


# ─────────────────────────────────────────────────────
# Main OPE function
# ─────────────────────────────────────────────────────

def run_ope(data_path: str,
            pi_net: PolicyNet,
            bc_net: BCNet,
            q_net:  QNet,
            device: str = "cpu",
            gamma:  float = 0.99,
            rho_clip: float = 20.0,
            max_episodes: Optional[int] = None) -> dict:
    """
    Returns a dict with keys: IS, PDIS, DR, n_episodes.
    """
    d   = np.load(data_path)
    obs  = d["observations"]
    acts = d["actions"].astype(int)
    rews = d["rewards"]
    nobs = d["next_observations"]
    term = d["terminals"]
    tout = d["timeouts"]

    episodes = _get_episode_slices(term, tout)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    print(f"[OPE] scoring {len(episodes)} episodes …")
    pi_all  = _policy_probs(pi_net, obs, device)   # (N, 169)
    mu_all  = _bc_probs(bc_net,  obs, device)      # (N, 169)
    q_all   = _q_values(q_net,   obs, device)      # (N, 169)
    q_all_n = _q_values(q_net,  nobs, device)      # (N, 169)

    # Importance ratios: ρ_t = π(a_t|s_t) / μ(a_t|s_t)
    pi_a = pi_all[np.arange(len(acts)), acts]       # (N,)
    mu_a = mu_all[np.arange(len(acts)), acts]       # (N,)
    rho  = np.clip(pi_a / (mu_a + 1e-8), 0.0, rho_clip)

    # Q values along the trajectory
    q_sa  = q_all[np.arange(len(acts)), acts]       # (N,)
    # V(s') = E_{a'~π}[Q(s', a')] (direct model value)
    v_sp  = (pi_all * q_all_n).sum(axis=1)          # (N,)
    # V(s) = E_{a~π}[Q(s, a)]
    v_s   = (pi_all * q_all).sum(axis=1)            # (N,)

    IS_list, PDIS_list, DR_list = [], [], []

    for s, e in episodes:
        T    = e - s
        r    = rews[s:e]
        rho_ep = rho[s:e]
        q_ep = q_sa[s:e]
        v_sp_ep = v_sp[s:e]
        v_s_ep  = v_s[s:e]

        gammas = gamma ** np.arange(T)

        # ── IS ──────────────────────────────────────────────────────────
        cum_rho = np.cumprod(rho_ep)[-1]
        G = (gammas * r).sum()
        IS_list.append(cum_rho * G)

        # ── PDIS ────────────────────────────────────────────────────────
        cum_rho_t = np.cumprod(rho_ep)   # (T,)
        PDIS_list.append((gammas * cum_rho_t * r).sum())

        # ── DR (doubly-robust) ─────────────────────────────────────────
        # V̂_DR = V(s_0) + Σ_t γ^t · w_t · δ_t
        # where w_t = Π_{k≤t} ρ_k, δ_t = r_t + γ·V(s_{t+1}) - Q(s_t, a_t)
        delta = r + gamma * v_sp_ep - q_ep
        DR_list.append(v_s_ep[0] + (gammas * cum_rho_t * delta).sum())

    return {
        "IS":   float(np.mean(IS_list)),
        "IS_std": float(np.std(IS_list)),
        "PDIS": float(np.mean(PDIS_list)),
        "PDIS_std": float(np.std(PDIS_list)),
        "DR":   float(np.mean(DR_list)),
        "DR_std": float(np.std(DR_list)),
        "n_episodes": len(episodes),
    }


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",      default="../drone_dispatch_env/data/D_logs.npz")
    ap.add_argument("--pi",        default="weights/iql_pi.pt")
    ap.add_argument("--bc",        default="weights/bc.pt")
    ap.add_argument("--q",         default="weights/iql_q.pt")
    ap.add_argument("--gamma",     type=float, default=0.99)
    ap.add_argument("--rho-clip",  type=float, default=20.0)
    ap.add_argument("--log",       default="logs/ope.csv")
    ap.add_argument("--device",    default="cpu")
    args = ap.parse_args()

    pi_net = PolicyNet(OBS_DIM, N_ACTIONS)
    pi_net.load_state_dict(torch.load(args.pi, map_location=args.device))

    bc_net = BCNet(OBS_DIM, N_ACTIONS)
    bc_net.load_state_dict(torch.load(args.bc, map_location=args.device))

    q_net = QNet(OBS_DIM, N_ACTIONS)
    q_net.load_state_dict(torch.load(args.q,  map_location=args.device))

    result = run_ope(args.data, pi_net, bc_net, q_net,
                     device=args.device,
                     gamma=args.gamma,
                     rho_clip=args.rho_clip)

    print("\n── Off-Policy Evaluation Results ────────────────────────────")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Save
    import csv, os
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    with open(args.log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        w.writeheader(); w.writerow(result)
    print(f"[OPE] saved → {args.log}")
