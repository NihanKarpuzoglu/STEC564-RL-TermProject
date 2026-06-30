# drone_dispatch_env — course simulator (student release)

The pre-built drone-delivery RL simulator for the term project. You do **not**
modify the simulator; you write policies that run inside it. The full API you may
rely on is in the **Simulator Specification, Section 12** (handed out separately
on Teams) — code against that, not against the internals.

## Install
```bash
cd drone_dispatch_env          # this folder (has pyproject.toml)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```
This installs the three Gymnasium env ids: `DroneDispatch-v0` (centralized
dispatcher, discrete), `DroneControl-v0` (continuous control), and
`DroneDispatchMA-v0` (multi-agent).

## Sanity check (do this Day 1)
```bash
python -m pytest tests -q                                   # 14 tests, all pass
bash reproduce.sh configs/eval_standard.yaml "0,1,2" greedy_nearest
```
The second command prints the baseline scores on the standard eval config. Your
learned method must beat `greedy_nearest` on `cost_per_order`.

## Minimal usage
```python
import gymnasium as gym
import drone_dispatch_env                       # registers the env ids
from drone_dispatch_env import (Config, evaluate,
    RandomPolicy, GreedyNearest, MILPRolling)   # shipped baselines

env = gym.make("DroneDispatch-v0")
obs, info = env.reset(seed=0)
# obs is a Dict; obs["action_mask"] (also info["action_mask"]) marks valid actions.
# Implement the Policy protocol in agent_interface.py: act(obs) -> action.

print(evaluate(GreedyNearest(Config()), Config(), seeds=[0,1,2])["mean"])
```

## Offline dataset (`data/D_logs.npz`)
Required for the offline-RL component. ~200k transitions from a mixed behavior
policy (≈60% greedy, ≈40% noisy/random).
```python
from drone_dispatch_env import load_offline_dataset, make_preference_pairs
d = load_offline_dataset("data/D_logs.npz")        # observations, actions, rewards,
#   next_observations, terminals, timeouts, episode_returns
pairs = make_preference_pairs("data/D_logs.npz", n_pairs=1000)  # for preference/RLHF
```
Episode boundaries are `terminals | timeouts` (most episodes end by truncation).

## Visualizer
`drone_dispatch_env.visualize` provides `render_frame`, `Recorder`, `Replayer`
(GIF/MP4 export + interactive play/pause/step/scrub), `compare(a, b, seed)`, and
`metrics_dashboard(results)`. Use it to debug what your policy actually does.

## Grading note
You are evaluated on **held-out seeds and a held-out config you have not seen**.
`reproduce.sh` takes config/seeds as overridable arguments — do not tune to one
specific random setup. The Section 7 metrics (primary: `cost_per_order`) are
returned by `evaluate(...)` and exposed in `info["metrics"]` every step.
