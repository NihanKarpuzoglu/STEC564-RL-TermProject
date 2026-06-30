"""DroneDispatchMA-v0 — decentralized multi-agent (Ch. 21).

Parallel API with dict-keyed observations/actions (one entry per agent
`drone_i`). Per-agent discrete action: {0 accept nearest order, 1 move toward
assigned target, 2 go charge, 3 idle}. Supports parameter sharing: every agent
shares the same observation/action shape so one network keyed by agent id works.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import (Config, NOFLY, CHARGER, IDLE, TO_PICKUP, TO_DROPOFF,
                     TO_CHARGER, CHARGING)
from .world import Drone, Order, make_grid, Router

ACCEPT, MOVE, CHARGE, STAY = 0, 1, 2, 3
PATCH = 3  # local grid patch half-width


class DroneDispatchMAEnv(gym.Env):
    """Parallel multi-agent env. reset/step use dict-keyed agent maps."""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(self, config: Optional[Config] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or Config()
        self.render_mode = render_mode
        self.agents = [f"drone_{i}" for i in range(self.cfg.n_drones)]
        side = 2 * PATCH + 1
        own = 4              # x, y, soc, status_norm
        near_order = 4       # nearest pending order ox,oy,dx,dy (relative)
        near_drone = 2       # nearest other drone dx,dy
        obs_dim = own + near_order + near_drone + side * side
        self.single_observation_space = spaces.Box(-1.0, max(self.cfg.H, self.cfg.W),
                                                    shape=(obs_dim,), dtype=np.float32)
        self.single_action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {a: self.single_observation_space for a in self.agents})
        self.action_space = spaces.Dict(
            {a: self.single_action_space for a in self.agents})

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        c = self.cfg
        self.grid, self.hubs = make_grid(c, self.np_random)
        self.router = Router(self.grid, c.neighborhood)
        self.t = 0
        self._next_oid = 0
        self.pending: list[Order] = []
        self.drones = []
        for i in range(c.n_drones):
            hx, hy = self.hubs[i % len(self.hubs)]
            self.drones.append(Drone(hx, hy, c.init_soc, IDLE))
        return self._obs_all(), {a: {} for a in self.agents}

    def step(self, actions: dict):
        c = self.cfg
        rewards = {a: 0.0 for a in self.agents}

        self._spawn_orders()
        for i, a in enumerate(self.agents):
            d = self.drones[i]
            if d.lost:
                continue
            act = int(actions.get(a, STAY))
            rewards[a] += self._apply_agent(i, d, act)

        # arrivals, deliveries, drops shared but credited to acting drone
        self._resolve(rewards)
        self._drop_expired(rewards)
        self.t += 1

        obs = self._obs_all()
        terminated = {a: self.drones[i].lost for i, a in enumerate(self.agents)}
        done = all(terminated.values()) or self.t >= c.T_max
        terms = {a: terminated[a] or done for a in self.agents}
        truncs = {a: (self.t >= c.T_max) for a in self.agents}
        return obs, rewards, terms, truncs, {a: {} for a in self.agents}

    def render(self):
        if self.render_mode == "rgb_array":
            from .visualize import render_frame
            return render_frame(self)
        return None

    # ---- dynamics ----
    def _apply_agent(self, i, d, act):
        c = self.cfg
        r = 0.0
        if act == ACCEPT and d.status == IDLE:
            cand = [o for o in self.pending if not o.picked and o.drone is None]
            if cand:
                o = min(cand, key=lambda o: self.router.dist(d.pos, (o.ox, o.oy)))
                o.drone = i
                d.order_id = o.oid
                d.status = TO_PICKUP
                d.path = self.router.path(d.pos, (o.ox, o.oy))
        elif act == CHARGE and d.status in (IDLE, TO_PICKUP):
            # release an in-progress (not yet picked) assignment back to the pool;
            # a drone carrying a picked payload (TO_DROPOFF) finishes first.
            if d.order_id is not None:
                o = self._oid(d.order_id)
                if o is not None and not o.picked:
                    o.drone = None
                    d.order_id = None
            if d.order_id is None:
                tgt = min(self.hubs, key=lambda h: self.router.dist(d.pos, h))
                d.status = TO_CHARGER
                d.path = self.router.path(d.pos, tgt)

        if d.status in (TO_PICKUP, TO_DROPOFF, TO_CHARGER) and d.path:
            d.x, d.y = d.path.pop(0)
            d.soc -= c.e_move
            r += c.reward.r_energy * c.e_move
        elif d.status == CHARGING:
            d.soc = min(1.0, d.soc + c.c_rate)
            if d.soc >= 1.0:
                d.status = IDLE
        else:
            d.soc -= c.e_idle
            r += c.reward.r_energy * c.e_idle

        if d.soc <= 0.0 and d.status in (TO_PICKUP, TO_DROPOFF, TO_CHARGER):
            d.lost = True
            d.soc = 0.0
            r += c.reward.r_depletion
            if d.order_id is not None:
                o = self._oid(d.order_id)
                if o:
                    o.drone = None
                    o.picked = False
                d.order_id = None
        return r

    def _resolve(self, rewards):
        c = self.cfg
        for i, a in enumerate(self.agents):
            d = self.drones[i]
            if d.lost or d.path:
                continue
            if d.status == TO_PICKUP:
                o = self._oid(d.order_id)
                if o and (d.x, d.y) == (o.ox, o.oy):
                    o.picked = True
                    d.status = TO_DROPOFF
                    d.path = self.router.path(d.pos, (o.dx, o.dy))
            elif d.status == TO_DROPOFF:
                o = self._oid(d.order_id)
                if o and (d.x, d.y) == (o.dx, o.dy):
                    o.delivered = True
                    rewards[a] += c.reward.r_delivered
                    rewards[a] += (c.reward.r_ontime_bonus if self.t <= o.deadline
                                   else c.reward.r_late_penalty * (self.t - o.deadline))
                    d.status = IDLE
                    d.order_id = None
            elif d.status == TO_CHARGER and self.grid[d.x, d.y] == CHARGER:
                d.status = CHARGING

    def _drop_expired(self, rewards):
        c = self.cfg
        survivors = []
        for o in self.pending:
            if o.delivered:
                continue
            if not o.picked and self.t > o.deadline:
                o.dropped = True
                if o.drone is not None:
                    rewards[self.agents[o.drone]] += c.reward.r_dropped
                    self.drones[o.drone].status = IDLE
                    self.drones[o.drone].order_id = None
                else:
                    for a in self.agents:
                        rewards[a] += c.reward.r_dropped / len(self.agents)
                continue
            survivors.append(o)
        self.pending = survivors

    def _spawn_orders(self):
        c = self.cfg
        for _ in range(int(self.np_random.poisson(c.lam))):
            if len(self.pending) >= c.k_max:
                break
            ox, oy = self._free_cell()
            dx, dy = self._free_cell()
            self.pending.append(Order(self._next_oid, ox, oy, dx, dy,
                                      self.t, self.t + c.sla_steps))
            self._next_oid += 1

    def _free_cell(self):
        while True:
            x = int(self.np_random.integers(0, self.cfg.H))
            y = int(self.np_random.integers(0, self.cfg.W))
            if self.grid[x, y] != NOFLY:
                return x, y

    def _oid(self, oid):
        for o in self.pending:
            if o.oid == oid:
                return o
        return None

    # ---- per-agent local observation ----
    def _obs_all(self):
        return {a: self._agent_obs(i) for i, a in enumerate(self.agents)}

    def _agent_obs(self, i):
        c = self.cfg
        d = self.drones[i]
        own = [d.x / c.H, d.y / c.W, d.soc, d.status / 4.0]

        cand = [o for o in self.pending if not o.picked and o.drone is None]
        if cand:
            o = min(cand, key=lambda o: self.router.dist(d.pos, (o.ox, o.oy)))
            near_o = [(o.ox - d.x) / c.H, (o.oy - d.y) / c.W,
                      (o.dx - d.x) / c.H, (o.dy - d.y) / c.W]
        else:
            near_o = [0.0, 0.0, 0.0, 0.0]

        others = [(j, self.drones[j]) for j in range(c.n_drones)
                  if j != i and not self.drones[j].lost]
        if others:
            _, nd = min(others, key=lambda t: abs(t[1].x - d.x) + abs(t[1].y - d.y))
            near_d = [(nd.x - d.x) / c.H, (nd.y - d.y) / c.W]
        else:
            near_d = [0.0, 0.0]

        patch = np.zeros((2 * PATCH + 1, 2 * PATCH + 1), dtype=np.float32)
        for px in range(-PATCH, PATCH + 1):
            for py in range(-PATCH, PATCH + 1):
                gx, gy = d.x + px, d.y + py
                if 0 <= gx < c.H and 0 <= gy < c.W:
                    patch[px + PATCH, py + PATCH] = self.grid[gx, gy]
                else:
                    patch[px + PATCH, py + PATCH] = NOFLY
        return np.array(own + near_o + near_d + patch.flatten().tolist(),
                        dtype=np.float32)
