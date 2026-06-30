"""Record one episode and open the interactive replay viewer.

Run from the project root:
    python replay.py                              # greedy_nearest (default)
    python replay.py --policy random
    python replay.py --policy bc
    python replay.py --policy naive_dqn
    python replay.py --policy iql
    python replay.py --policy irl
    python replay.py --policy cmdp
    python replay.py --policy cmdp --lam 2.5      # override Lagrange λ
    python replay.py --save logs/replay.gif        # export GIF instead
    python replay.py --seed 5                      # different episode seed

Requires matplotlib with a GUI backend (Tk, Qt, etc.) for the interactive
window. If no display is available, use --save to export a GIF.

Available policies: greedy_nearest, random, milp, bc, naive_dqn, iql, irl, cmdp
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "drone_dispatch_env"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))

import torch

from drone_dispatch_env.config import Config
from drone_dispatch_env.env_dispatch import DroneDispatchEnv
from drone_dispatch_env.baselines import make_baseline
from drone_dispatch_env.visualize import Recorder, Replayer, record_episode

W = "weights"  # shorthand for default weight paths


def _hidden_from_ckpt(path: str) -> tuple:
    """Read hidden layer sizes from a checkpoint without instantiating the model."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    h1 = sd["net.0.weight"].shape[0]
    h2 = sd["net.2.weight"].shape[0]
    return (h1, h2)


def load_policy(name: str, args):
    """Return a policy object for the given name."""
    name = name.lower()

    if name == "bc":
        from bc import load_bc  # type: ignore[import]
        path = args.weights or f"{W}/bc.pt"
        return load_bc(path, hidden=_hidden_from_ckpt(path))

    if name == "naive_dqn":
        from naive_dqn import load_naive_dqn  # type: ignore[import]
        path = args.weights or f"{W}/naive.pt"
        return load_naive_dqn(path, hidden=_hidden_from_ckpt(path))

    if name == "iql":
        from iql import load_iql  # type: ignore[import]
        return load_iql(
            q_path=f"{W}/iql_q.pt",
            v_path=f"{W}/iql_v.pt",
            pi_path=f"{W}/iql_pi.pt",
            hidden=_hidden_from_ckpt(f"{W}/iql_q.pt"),
        )

    if name == "irl":
        from iql import load_iql  # type: ignore[import]
        return load_iql(
            q_path=f"{W}/irl_policy_q.pt",
            v_path=f"{W}/irl_policy_v.pt",
            pi_path=f"{W}/irl_policy_pi.pt",
            hidden=_hidden_from_ckpt(f"{W}/irl_policy_q.pt"),
        )

    if name == "cmdp":
        from cmdp import load_cmdp  # type: ignore[import]
        return load_cmdp(
            q_path=f"{W}/cmdp_q.pt",
            v_path=f"{W}/cmdp_v.pt",
            pi_path=f"{W}/cmdp_pi.pt",
            qc_path=f"{W}/cmdp_qc.pt",
            lam=args.lam,
            hidden=_hidden_from_ckpt(f"{W}/cmdp_q.pt"),
        )

    # built-in baselines (greedy_nearest, random, milp)
    cfg = args._cfg
    return make_baseline(name, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",    type=int, default=0,
                    help="Episode seed (default 0)")
    ap.add_argument("--policy",  default="greedy_nearest",
                    help="greedy_nearest | random | milp | bc | naive_dqn "
                         "| iql | irl | cmdp  (default: greedy_nearest)")
    ap.add_argument("--weights", default=None,
                    help="Override weights file for bc or naive_dqn "
                         "(defaults: weights/bc.pt, weights/naive.pt)")
    ap.add_argument("--lam",     type=float, default=1.0,
                    help="Lagrange multiplier λ for cmdp (default 1.0)")
    ap.add_argument("--save",    default=None,
                    help="Path to save GIF or MP4 instead of opening a window "
                         "(e.g. logs/replay.gif or logs/replay.mp4)")
    args = ap.parse_args()

    cfg = Config()
    env = DroneDispatchEnv(cfg)
    args._cfg = cfg  # pass cfg through to load_policy for baseline fallback

    policy = load_policy(args.policy, args)

    print("Recording episode (seed={}, policy={}) ...".format(args.seed, args.policy))
    rec = record_episode(policy, env, seed=args.seed)
    print("Captured {} frames.".format(len(rec)))

    replayer = Replayer(rec)

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        if args.save.endswith(".mp4"):
            replayer.to_mp4(args.save, fps=10)
        else:
            replayer.to_gif(args.save, fps=5)
        print("Saved -> " + args.save)
    else:
        print("Opening interactive viewer ...")
        print("  Controls: play/pause button, step-forward, step-back, scrub slider")
        replayer.play()


if __name__ == "__main__":
    main()
