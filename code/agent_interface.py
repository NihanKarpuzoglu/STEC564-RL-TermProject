"""Agent <-> visualizer contract for drone_dispatch_env.

Freeze this file before students start (Simulator Spec, Section 8.5). It defines
the only coupling between a learning method and the visualizer overlays.

Two structural protocols (typing.Protocol -> no inheritance required, just
implement the methods):

  Policy         REQUIRED. Every baseline and learned agent implements act().
  Introspectable OPTIONAL. Implement any subset to unlock the matching
                 Section 8.2 overlay. Any method may return None to skip it.

Overlays that do NOT come from the agent (the visualizer reads them elsewhere):
  - reward-term stream, action mask, chosen assignments -> from step() `info`
  - dataset-coverage heatmap -> from the offline dataset D_logs
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable
import numpy as np

Obs = dict[str, Any]   # gymnasium Dict observation (Spec Section 3.1)
Action = Any           # int (discrete dispatch) or np.ndarray (continuous control)


@runtime_checkable
class Policy(Protocol):
    def act(self, obs: Obs) -> Action:
        """Return one action for `obs`. In discrete modes the action MUST be
        valid under info['action_mask'] (also exposed in obs)."""
        ...


@runtime_checkable
class Introspectable(Protocol):
    """Hooks the visualizer calls at a selected step to draw overlays.
    Return None from any method to skip that overlay. Arrays are plain numpy."""

    def action_values(self, obs: Obs) -> Optional[np.ndarray]:
        """Q-value per action, shape (n_actions,); masked actions may be NaN.
        Powers the per-action Q overlay (value-based methods)."""
        ...

    def action_probs(self, obs: Obs) -> Optional[np.ndarray]:
        """Action distribution, shape (n_actions,), sums to 1.
        Powers the action-probability overlay (policy-based methods)."""
        ...

    def state_values(self, obs: Obs) -> Optional[np.ndarray]:
        """V(s) per grid cell, shape (H, W). Powers the state-value heatmap.
        Return None if the method has no value estimate."""
        ...


# --- Visualizer side: how overlays are gathered (provided by instructor) ---
def gather_overlays(agent: Policy, obs: Obs) -> dict[str, np.ndarray]:
    out: dict[str, Optional[np.ndarray]] = {}
    if isinstance(agent, Introspectable):
        out["q"] = agent.action_values(obs)
        out["pi"] = agent.action_probs(obs)
        out["v"] = agent.state_values(obs)
    return {k: v for k, v in out.items() if v is not None}


# --- Student side: example adapter wrapping a trained value network ---
class DQNAdapter:
    """Minimal example. A student wraps their trained net so it satisfies both
    Policy and Introspectable. `q_net(obs) -> np.ndarray (n_actions,)`."""

    def __init__(self, q_net, mask_fn):
        self.q_net = q_net          # callable: obs -> Q-values
        self.mask_fn = mask_fn      # callable: obs -> bool array of valid actions

    def act(self, obs: Obs) -> Action:
        q = np.asarray(self.q_net(obs), dtype=float)
        q = np.where(self.mask_fn(obs), q, -np.inf)
        return int(np.argmax(q))

    def action_values(self, obs: Obs) -> Optional[np.ndarray]:
        q = np.asarray(self.q_net(obs), dtype=float)
        return np.where(self.mask_fn(obs), q, np.nan)  # unlocks the Q overlay

    def action_probs(self, obs: Obs) -> Optional[np.ndarray]:
        return None   # value-based: no distribution

    def state_values(self, obs: Obs) -> Optional[np.ndarray]:
        return None   # optional; implement if a grid V-estimate is available
