#!/usr/bin/env bash
# reproduce.sh — load saved weights and print evaluation results.
#
# Usage:
#   bash reproduce.sh [CONFIG] [SEEDS] [POLICY]
#
# Defaults:
#   CONFIG = configs/eval_standard.yaml
#   SEEDS  = 0,1,2
#   POLICY = all   (or: greedy | bc | naive | iql | irl | cmdp)
#
# Example (grader overrides):
#   bash reproduce.sh configs/eval_standard.yaml "7,8,9" iql

set -e
CONFIG="${1:-configs/eval_standard.yaml}"
SEEDS="${2:-0,1,2}"
POLICY="${3:-all}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Install env if not already installed
pip install -e drone_dispatch_env -q --break-system-packages 2>/dev/null || true
pip install torch numpy gymnasium pyyaml -q --break-system-packages 2>/dev/null || true

python - <<PYEOF
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join("${SCRIPT_DIR}", "drone_dispatch_env"))
import drone_dispatch_env
from drone_dispatch_env import evaluate, Config

cfg_env = Config.from_yaml("${SCRIPT_DIR}/${CONFIG}")
seeds   = [int(s) for s in "${SEEDS}".split(",")]
filt    = "${POLICY}"
W       = "${SCRIPT_DIR}/weights"

sys.path.insert(0, os.path.join("${SCRIPT_DIR}", "code"))
from bc        import load_bc
from naive_dqn import load_naive_dqn
from iql       import load_iql
from cmdp      import load_cmdp

H_bc  = (256, 256)   # BC trained with configs/bc.yaml hidden=[256,256]
H_iql = (128, 128)   # IQL/IRL/CMDP trained with 128x128

def run(name, policy):
    if filt not in ("all", name):
        return
    r   = evaluate(policy, cfg_env, seeds=seeds)
    m   = r["mean"]
    std = {k: float(np.std([s[k] for s in r["per_seed"]])) for k in m}
    print(f"\n{'─'*54}")
    print(f"  {name}   seeds={seeds}")
    print(f"{'─'*54}")
    for k, v in m.items():
        print(f"  {k:<28} {v:>8.4f} ± {std[k]:.4f}")

from drone_dispatch_env import GreedyNearest
run("greedy_nearest", GreedyNearest(cfg_env))

if filt in ("all", "bc"):
    try:   run("bc", load_bc(f"{W}/bc.pt", hidden=H_bc))
    except FileNotFoundError: print("[WARN] bc.pt not found")

if filt in ("all", "naive"):
    try:   run("naive_dqn", load_naive_dqn(f"{W}/naive.pt", hidden=H_bc))
    except FileNotFoundError: print("[WARN] naive.pt not found")

if filt in ("all", "iql"):
    try:   run("iql", load_iql(f"{W}/iql_q.pt", f"{W}/iql_v.pt", f"{W}/iql_pi.pt", hidden=H_iql))
    except FileNotFoundError: print("[WARN] iql weights not found")

if filt in ("all", "irl"):
    try:   run("irl_policy", load_iql(f"{W}/irl_policy_q.pt", f"{W}/irl_policy_v.pt", f"{W}/irl_policy_pi.pt", hidden=H_iql))
    except FileNotFoundError: print("[WARN] irl_policy weights not found")

if filt in ("all", "cmdp"):
    try:
        with open(f"{W}/cmdp_lambda.json") as f:
            lam = json.load(f)["lam"]
        run("cmdp", load_cmdp(f"{W}/cmdp_q.pt", f"{W}/cmdp_v.pt",
                               f"{W}/cmdp_pi.pt", f"{W}/cmdp_qc.pt",
                               lam=lam, hidden=H_iql))
    except FileNotFoundError: print("[WARN] cmdp weights not found")

PYEOF
