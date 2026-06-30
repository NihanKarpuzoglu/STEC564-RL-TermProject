"""Evaluation harness and metrics (Spec Section 7)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

from .config import Config
from .env_dispatch import DroneDispatchEnv


@dataclass
class Metrics:
    cost_per_order: float          # primary score
    success_rate: float
    ontime_rate: float
    mean_delivery_time: float
    energy_per_order: float
    depletion_events: float
    idle_pct: float
    charger_utilization: float
    n_delivered: float
    n_dropped: float
    episode_return: float

    def to_dict(self):
        return asdict(self)


def run_episode(policy, env: DroneDispatchEnv, seed: int, recorder=None):
    obs, info = env.reset(seed=seed)
    ep_return = 0.0
    if recorder is not None:
        recorder.capture(env)
    done = False
    while not done:
        a = policy.act(obs)
        obs, r, term, trunc, info = env.step(a)
        ep_return += r
        if recorder is not None:
            recorder.capture(env)
        done = term or trunc
    return env.stats, ep_return


def _metrics_from_stats(s: dict, ep_return: float) -> Metrics:
    delivered = max(s["delivered"], 0)
    total_orders = delivered + s["dropped"]
    cost = s["energy"] + s["late_cost"] + s["drop_cost"] + s["depletion_cost"]
    denom = delivered if delivered > 0 else 1
    return Metrics(
        cost_per_order=cost / denom,
        success_rate=delivered / total_orders if total_orders else 0.0,
        ontime_rate=s["ontime"] / delivered if delivered else 0.0,
        mean_delivery_time=s["sum_delivery_time"] / delivered if delivered else 0.0,
        energy_per_order=s["energy"] / denom,
        depletion_events=s.get("depletion_events", 0),
        idle_pct=s["idle_steps"] / s["drone_steps"] if s["drone_steps"] else 0.0,
        charger_utilization=s["charging_steps"] / s["drone_steps"] if s["drone_steps"] else 0.0,
        n_delivered=delivered,
        n_dropped=s["dropped"],
        episode_return=ep_return,
    )


def evaluate(policy, config: Optional[Config] = None, seeds=range(10)):
    """Run episodes over `seeds`; return mean metrics and per-seed list."""
    cfg = config or Config()
    env = DroneDispatchEnv(cfg)
    per_seed = []
    for seed in seeds:
        stats, ret = run_episode(policy, env, int(seed))
        per_seed.append(_metrics_from_stats(stats, ret))

    keys = per_seed[0].to_dict().keys()
    mean = {k: float(np.mean([getattr(m, k) for m in per_seed])) for k in keys}
    return {"mean": mean, "per_seed": [m.to_dict() for m in per_seed]}
