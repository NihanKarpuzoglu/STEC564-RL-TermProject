"""DroneControl-v0 — single drone, continuous speed/heading (for DDPG).

One active delivery: reach the target cell while managing energy and avoiding
no-fly cells. Position is continuous; the occupied cell is floor(position).
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import Config, NOFLY
from .world import make_grid, Router


class DroneControlEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config: Optional[Config] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or Config()
        self.render_mode = render_mode
        self.max_speed = 1.0  # cells per step at speed=1

        # obs: (dx, dy, soc, dist_to_target, heading, nearest_nofly_dx, nearest_nofly_dy)
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, 0, 0, -np.pi, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, np.sqrt(2), np.pi, 1, 1], dtype=np.float32))
        # action: (speed in [0,1], heading_delta in [-1,1] -> scaled to +-pi)
        self.action_space = spaces.Box(low=np.array([0.0, -1.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0], dtype=np.float32))
        self.grid = None

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        c = self.cfg
        self.grid, self.hubs = make_grid(c, self.np_random)
        self.router = Router(self.grid, c.neighborhood)
        self.pos = np.array(self._free_cell(), dtype=np.float32)
        self.target = np.array(self._free_cell(), dtype=np.float32)
        self.heading = float(self.np_random.uniform(-np.pi, np.pi))
        self.soc = c.init_soc
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        c = self.cfg
        speed = float(np.clip(action[0], 0.0, 1.0))
        self.heading += float(np.clip(action[1], -1.0, 1.0)) * np.pi
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

        prev_dist = float(np.linalg.norm(self.target - self.pos))
        step_vec = speed * self.max_speed * np.array([np.cos(self.heading),
                                                      np.sin(self.heading)])
        new_pos = self.pos + step_vec
        cell = (int(np.floor(new_pos[0])), int(np.floor(new_pos[1])))

        terminated = False
        reward = -0.01  # time penalty

        in_bounds = 0 <= cell[0] < c.H and 0 <= cell[1] < c.W
        if not in_bounds or self.grid[cell] == NOFLY:
            reward += c.reward.r_depletion / 2.0  # large penalty, stay put
        else:
            self.pos = new_pos

        energy = c.e_move * speed + c.e_idle
        self.soc -= energy
        reward += c.reward.r_energy * energy

        new_dist = float(np.linalg.norm(self.target - self.pos))
        reward += (prev_dist - new_dist)  # progress

        if self.soc <= 0.0:
            self.soc = 0.0
            reward += c.reward.r_depletion
            terminated = True
        if new_dist < 0.7:
            reward += c.reward.r_delivered + c.reward.r_ontime_bonus
            terminated = True

        self.t += 1
        truncated = self.t >= c.T_max
        return self._obs(), float(reward), terminated, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            from .visualize import render_control_frame
            return render_control_frame(self)
        return None

    def _free_cell(self):
        while True:
            x = int(self.np_random.integers(0, self.cfg.H))
            y = int(self.np_random.integers(0, self.cfg.W))
            if self.grid[x, y] != NOFLY:
                return x, y

    def _nearest_nofly_offset(self):
        nf = np.argwhere(self.grid == NOFLY)
        if len(nf) == 0:
            return 0.0, 0.0
        d = nf - np.array([self.pos[0], self.pos[1]])
        idx = int(np.argmin(np.linalg.norm(d, axis=1)))
        off = (nf[idx] - self.pos) / max(self.cfg.H, self.cfg.W)
        return float(off[0]), float(off[1])

    def _obs(self):
        c = self.cfg
        diff = (self.target - self.pos) / np.array([c.H, c.W], dtype=np.float32)
        dist = float(np.linalg.norm(self.target - self.pos)) / np.sqrt(c.H**2 + c.W**2)
        nfx, nfy = self._nearest_nofly_offset()
        return np.array([diff[0], diff[1], self.soc, dist, self.heading, nfx, nfy],
                        dtype=np.float32)
