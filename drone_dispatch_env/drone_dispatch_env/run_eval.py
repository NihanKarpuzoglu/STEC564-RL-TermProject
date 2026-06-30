"""CLI: evaluate a baseline (or random) over seeds with an overridable config.
Used by reproduce.sh so the instructor can swap held-out config/seeds."""
from __future__ import annotations

import argparse
import json

from .config import Config
from .baselines import make_baseline
from .evaluate import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval_standard.yaml")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--policy", default="greedy_nearest")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    policy = make_baseline(args.policy, cfg)
    results = evaluate(policy, cfg, seeds)
    print(json.dumps(results["mean"], indent=2))


if __name__ == "__main__":
    main()
