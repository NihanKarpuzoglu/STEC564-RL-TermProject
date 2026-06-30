"""train_all.py — trains every method in order.

Usage:
    cd code/
    python train_all.py [--data PATH] [--seeds 0,1,2] [--device cpu]

Runs BC → NaiveDQN → IQL (3 seeds) → IRL reward model → IRL policy →
CMDP (3 seeds) → OPE.  Saves all weights/ logs/ along the way.
"""
from __future__ import annotations

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

# ─────────────────────────────────────────────────────
# Add drone_dispatch_env to path
# ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "drone_dispatch_env"))
import drone_dispatch_env  # noqa: F401 (registers gym ids)
from drone_dispatch_env import evaluate, Config

from bc         import train_bc, load_bc
from naive_dqn  import train_naive_dqn, load_naive_dqn
from iql        import train_iql, load_iql, PolicyNet, QNet, VNet
from irl_reward import train_reward_model, score_dataset, load_reward_model
from cmdp       import train_cmdp, load_cmdp
from ope        import run_ope, BCNet
from utils      import OBS_DIM, N_ACTIONS

import torch


def load_cfg(name: str) -> dict:
    path = ROOT / "configs" / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def fmt(d: dict) -> str:
    return "  " + "\n  ".join(f"{k}: {v:.4f}" for k, v in d.items())


def eval_multi(policy, cfg_env, seeds, label):
    """Evaluate over multiple seeds; return mean dict and per-seed list."""
    from drone_dispatch_env import evaluate
    results = evaluate(policy, cfg_env, seeds=list(seeds))
    mean = results["mean"]
    stds = {}
    keys = results["per_seed"][0].keys()
    for k in keys:
        vals = [s[k] for s in results["per_seed"]]
        stds[k] = float(np.std(vals))
    print(f"\n{'─'*55}")
    print(f"[EVAL] {label}  (seeds {list(seeds)})")
    for k in mean:
        print(f"  {k}: {mean[k]:.4f} ± {stds[k]:.4f}")
    return mean, stds, results["per_seed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default=str(ROOT / "drone_dispatch_env/data/D_logs.npz"))
    ap.add_argument("--seeds",  default="0,1,2")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = args.device
    data   = args.data
    W = ROOT / "weights"
    L = ROOT / "logs"

    cfg_env = Config()
    all_results: dict = {}

    # ──────────────────────────────────────────────────
    # 0. Greedy Nearest baseline
    # ──────────────────────────────────────────────────
    print("\n═══ 0. GreedyNearest baseline ═══")
    from drone_dispatch_env import GreedyNearest
    gn_mean, gn_std, gn_per = eval_multi(GreedyNearest(cfg_env), cfg_env, seeds, "GreedyNearest")
    all_results["greedy_nearest"] = {"mean": gn_mean, "std": gn_std}

    # ──────────────────────────────────────────────────
    # 1. Behavioral Cloning
    # ──────────────────────────────────────────────────
    print("\n═══ 1. Behavioral Cloning ═══")
    bc_cfg = load_cfg("bc")
    bc_cfg["device"] = device
    from bc import train_bc
    bc_policy = train_bc(bc_cfg, data,
                         str(W / "bc.pt"), str(L / "bc_seed0.csv"), seed=seeds[0])
    bc_mean, bc_std, bc_per = eval_multi(bc_policy, cfg_env, seeds, "BC")
    all_results["bc"] = {"mean": bc_mean, "std": bc_std}

    # ──────────────────────────────────────────────────
    # 2. Naive offline DQN
    # ──────────────────────────────────────────────────
    print("\n═══ 2. Naive Offline DQN ═══")
    dqn_cfg = load_cfg("naive_dqn")
    dqn_cfg["device"] = device
    dqn_policy = train_naive_dqn(dqn_cfg, data,
                                  str(W / "naive.pt"),
                                  str(L / "naive_dqn_seed0.csv"), seed=seeds[0])
    dqn_mean, dqn_std, _ = eval_multi(dqn_policy, cfg_env, seeds, "NaiveDQN")
    all_results["naive_dqn"] = {"mean": dqn_mean, "std": dqn_std}

    # ──────────────────────────────────────────────────
    # 3. IQL (3 seeds)
    # ──────────────────────────────────────────────────
    print("\n═══ 3. IQL — 3 seeds ═══")
    iql_cfg = load_cfg("iql")
    iql_cfg["device"] = device
    iql_results = []
    for i, seed in enumerate(seeds):
        print(f"\n── IQL seed {seed} ──")
        pol = train_iql(iql_cfg, data,
                        str(W / f"iql_q_seed{seed}.pt"),
                        str(W / f"iql_v_seed{seed}.pt"),
                        str(W / f"iql_pi_seed{seed}.pt"),
                        str(L / f"iql_seed{seed}.csv"),
                        seed=seed)
        m = evaluate(pol, cfg_env, seeds=[seed])["mean"]
        iql_results.append(m)
    # Copy seed-0 weights as canonical
    import shutil
    for suffix in ["q", "v", "pi"]:
        shutil.copy(str(W / f"iql_{suffix}_seed{seeds[0]}.pt"),
                    str(W / f"iql_{suffix}.pt"))
    # Reload canonical for further use
    iql_policy = load_iql(str(W / "iql_q.pt"), str(W / "iql_v.pt"),
                           str(W / "iql_pi.pt"), device=device)
    iql_mean_full, iql_std_full, _ = eval_multi(
        iql_policy, cfg_env, seeds, "IQL (canonical seed)")
    # Aggregate across seeds
    keys = iql_results[0].keys()
    iql_mean_agg = {k: float(np.mean([r[k] for r in iql_results])) for k in keys}
    iql_std_agg  = {k: float(np.std( [r[k] for r in iql_results])) for k in keys}
    all_results["iql"] = {"mean": iql_mean_agg, "std": iql_std_agg,
                          "per_seed": iql_results}
    print("\n[IQL] Across 3 training seeds:")
    for k in ["cost_per_order", "success_rate", "depletion_events"]:
        print(f"  {k}: {iql_mean_agg[k]:.4f} ± {iql_std_agg[k]:.4f}")

    # ──────────────────────────────────────────────────
    # 4. IRL: reward model + IRL policy
    # ──────────────────────────────────────────────────
    print("\n═══ 4. IRL — Bradley–Terry reward model ═══")
    irl_cfg = load_cfg("irl")
    irl_cfg["device"] = device
    reward_model = train_reward_model(irl_cfg, data,
                                      str(W / "reward_model.pt"),
                                      str(L / "irl_reward.csv"),
                                      seed=seeds[0])
    recovered = score_dataset(reward_model, data, device=device)
    np.save(str(W / "irl_recovered_rewards.npy"), recovered)
    corr = np.corrcoef(recovered, np.load(data)["rewards"])[0, 1]
    print(f"[IRL] recovered ↔ env reward correlation: {corr:.3f}")

    print("\n── IRL policy (IQL on recovered rewards) ──")
    irl_policy_cfg = load_cfg("iql")   # same architecture
    irl_policy_cfg["device"] = device
    irl_policy_cfg["n_steps"] = irl_cfg.get("policy_steps", 60_000)
    irl_policy = train_iql(irl_policy_cfg, data,
                           str(W / "irl_policy_q.pt"),
                           str(W / "irl_policy_v.pt"),
                           str(W / "irl_policy_pi.pt"),
                           str(L / "irl_policy.csv"),
                           seed=seeds[0],
                           reward_override=recovered)
    irl_mean, irl_std, _ = eval_multi(irl_policy, cfg_env, seeds, "IRL policy")
    all_results["irl_policy"] = {"mean": irl_mean, "std": irl_std, "corr": corr}

    # ──────────────────────────────────────────────────
    # 5. CMDP (3 seeds)
    # ──────────────────────────────────────────────────
    print("\n═══ 5. CMDP (Lagrangian IQL) ═══")
    cmdp_cfg = load_cfg("cmdp")
    cmdp_cfg["device"] = device
    cmdp_results = []
    for i, seed in enumerate(seeds):
        print(f"\n── CMDP seed {seed} ──")
        pol = train_cmdp(cmdp_cfg, data,
                         str(W / f"cmdp_q_seed{seed}.pt"),
                         str(W / f"cmdp_v_seed{seed}.pt"),
                         str(W / f"cmdp_pi_seed{seed}.pt"),
                         str(W / f"cmdp_qc_seed{seed}.pt"),
                         str(L / f"cmdp_seed{seed}.csv"),
                         seed=seed)
        m = evaluate(pol, cfg_env, seeds=[seed])["mean"]
        cmdp_results.append(m)
    shutil.copy(str(W / f"cmdp_q_seed{seeds[0]}.pt"),  str(W / "cmdp_q.pt"))
    shutil.copy(str(W / f"cmdp_v_seed{seeds[0]}.pt"),  str(W / "cmdp_v.pt"))
    shutil.copy(str(W / f"cmdp_pi_seed{seeds[0]}.pt"), str(W / "cmdp_pi.pt"))
    shutil.copy(str(W / f"cmdp_qc_seed{seeds[0]}.pt"), str(W / "cmdp_qc.pt"))
    cmdp_mean_agg = {k: float(np.mean([r[k] for r in cmdp_results]))
                     for k in cmdp_results[0]}
    cmdp_std_agg  = {k: float(np.std( [r[k] for r in cmdp_results]))
                     for k in cmdp_results[0]}
    all_results["cmdp"] = {"mean": cmdp_mean_agg, "std": cmdp_std_agg,
                           "per_seed": cmdp_results}
    print("\n[CMDP] Across 3 training seeds:")
    for k in ["cost_per_order", "depletion_events", "success_rate"]:
        print(f"  {k}: {cmdp_mean_agg[k]:.4f} ± {cmdp_std_agg[k]:.4f}")

    # ──────────────────────────────────────────────────
    # 6. OPE
    # ──────────────────────────────────────────────────
    print("\n═══ 6. Off-Policy Evaluation ═══")
    pi_net = PolicyNet(OBS_DIM, N_ACTIONS)
    q_net  = QNet(OBS_DIM, N_ACTIONS)
    bc_net = BCNet(OBS_DIM, N_ACTIONS)
    pi_net.load_state_dict(torch.load(str(W / "iql_pi.pt"), map_location=device))
    q_net.load_state_dict(torch.load(str(W / "iql_q.pt"),  map_location=device))
    bc_net.load_state_dict(torch.load(str(W / "bc.pt"),     map_location=device))

    ope_result = run_ope(data, pi_net, bc_net, q_net, device=device)
    true_value = iql_mean_full["episode_return"]
    print(f"\n  OPE IS  estimate:   {ope_result['IS']:.2f} ± {ope_result['IS_std']:.2f}")
    print(f"  OPE PDIS estimate:  {ope_result['PDIS']:.2f} ± {ope_result['PDIS_std']:.2f}")
    print(f"  OPE DR  estimate:   {ope_result['DR']:.2f} ± {ope_result['DR_std']:.2f}")
    print(f"  True simulated V:   {true_value:.2f}")
    print(f"  PDIS gap:           {abs(ope_result['PDIS'] - true_value):.2f}")
    all_results["ope"] = {**ope_result, "true_episode_return": true_value}

    # Save summary
    with open(str(L / "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[DONE] Results saved to {L}/all_results.json")

    # Final comparison table
    print("\n" + "═"*55)
    print("FINAL COMPARISON  (seeds: {})".format(seeds))
    print("─"*55)
    header = f"{'Method':<20}  {'cost/order':>10}  {'success':>8}  {'depletion':>10}"
    print(header)
    print("─"*55)
    for name, res in [
        ("GreedyNearest", all_results["greedy_nearest"]),
        ("BC",            all_results["bc"]),
        ("NaiveDQN",      all_results["naive_dqn"]),
        ("IQL",           all_results["iql"]),
        ("IRL-policy",    all_results["irl_policy"]),
        ("CMDP",          all_results["cmdp"]),
    ]:
        m = res["mean"]
        s = res.get("std", {})
        print(f"{name:<20}  "
              f"{m['cost_per_order']:>6.3f}±{s.get('cost_per_order',0):.2f}  "
              f"{m['success_rate']:>6.3f}  "
              f"{m['depletion_events']:>7.2f}±{s.get('depletion_events',0):.2f}")


if __name__ == "__main__":
    main()
