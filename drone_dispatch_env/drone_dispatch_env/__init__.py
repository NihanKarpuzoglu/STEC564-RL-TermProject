"""drone_dispatch_env — operational drone-dispatch RL simulator.

Registers three gymnasium env ids behind one install:
  DroneDispatch-v0     centralized dispatcher (discrete, masked)
  DroneControl-v0      single-drone continuous control
  DroneDispatchMA-v0   decentralized multi-agent (parallel, dict-keyed)
"""
from gymnasium.envs.registration import register

from .config import Config, RewardWeights
from .env_dispatch import DroneDispatchEnv
from .env_control import DroneControlEnv
from .env_ma import DroneDispatchMAEnv
from .baselines import RandomPolicy, GreedyNearest, MILPRolling, make_baseline
from .evaluate import evaluate, run_episode, Metrics
from .offline import (generate_offline_dataset, load_offline_dataset,
                      make_preference_pairs)
from . import visualize

register(id="DroneDispatch-v0",
         entry_point="drone_dispatch_env.env_dispatch:DroneDispatchEnv")
register(id="DroneControl-v0",
         entry_point="drone_dispatch_env.env_control:DroneControlEnv")
register(id="DroneDispatchMA-v0",
         entry_point="drone_dispatch_env.env_ma:DroneDispatchMAEnv")

__all__ = [
    "Config", "RewardWeights",
    "DroneDispatchEnv", "DroneControlEnv", "DroneDispatchMAEnv",
    "RandomPolicy", "GreedyNearest", "MILPRolling", "make_baseline",
    "evaluate", "run_episode", "Metrics",
    "generate_offline_dataset", "load_offline_dataset", "make_preference_pairs",
    "visualize",
]
