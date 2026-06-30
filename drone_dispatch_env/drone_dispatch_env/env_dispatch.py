"""DroneDispatch-v0 — centralized dispatcher (discrete, masked).

Stepping convention (frozen): assignment / charge actions update state without
advancing simulation time; `no-op` advances one tick. The env auto-advances
ticks whenever no drone is idle OR no order is pending, so step() always returns
control at a decision epoch (or termination). info carries the action mask,
chosen assignments, and the per-term reward decomposition.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import (Config, FREE, NOFLY, HUB, CHARGER,
                     IDLE, TO_PICKUP, TO_DROPOFF, TO_CHARGER, CHARGING, N_STATUS)
from .world import Drone, Order, make_grid, Router

REWARD_TERMS = ["delivered", "ontime", "late", "dropped", "energy", "depletion", "noop"]


class DroneDispatchEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(self, config: Optional[Config] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or Config()
        self.render_mode = render_mode
        c = self.cfg

        self.observation_space = spaces.Dict({
            "drones": spaces.Box(-1.0, max(c.H, c.W),
                                 shape=(c.n_drones, 4 + N_STATUS + 1), dtype=np.float32),
            "orders": spaces.Box(-1.0, float(max(c.H, c.W, c.T_max)),
                                 shape=(c.k_max, 5), dtype=np.float32),
            "grid": spaces.Box(0, 3, shape=(c.H, c.W), dtype=np.int8),
            "time": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(c.n_actions,), dtype=np.int8),
        })
        self.action_space = spaces.Discrete(c.n_actions)

        self._np_random = None
        self.grid = None
        self.router = None
        self.drones: list[Drone] = []
        self.pending: list[Order] = []     # active, not delivered/dropped
        self.t = 0
        self._next_oid = 0
        self._last_reward_terms = {k: 0.0 for k in REWARD_TERMS}
        self._last_assignments: list[tuple] = []

    @staticmethod
    def _zero_stats():
        return dict(delivered=0, ontime=0, dropped=0, sum_delivery_time=0.0,
                    energy=0.0, late_cost=0.0, drop_cost=0.0, depletion_cost=0.0,
                    idle_steps=0, charging_steps=0, drone_steps=0, depletion_events=0)

    # ---------- gym API ----------
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        c = self.cfg
        self.grid, self.hubs = make_grid(c, rng)
        self.router = Router(self.grid, c.neighborhood)
        self.t = 0
        self._next_oid = 0
        self.pending = []
        self._last_assignments = []
        self._last_reward_terms = {k: 0.0 for k in REWARD_TERMS}
        self.stats = self._zero_stats()

        # drones start at hubs (cycled), full battery
        self.drones = []
        for i in range(c.n_drones):
            hx, hy = self.hubs[i % len(self.hubs)]
            self.drones.append(Drone(hx, hy, c.init_soc, IDLE))

        # advance to first decision epoch (also seeds the order stream), then
        # zero stats so episode metrics start at the agent's first decision
        self._auto_advance(reward_terms=None)
        self.stats = self._zero_stats()
        return self._obs(), self._info()

    def step(self, action: int):
        c = self.cfg
        terms = {k: 0.0 for k in REWARD_TERMS}
        self._last_assignments = []
        mask = self._action_mask()

        decoded = c.decode(int(action))
        valid = bool(mask[int(action)])

        if not valid or decoded[0] == "noop":
            # no-op / mask-ignoring action: small penalty and advance time (defer)
            terms["noop"] += c.reward.r_noop
            self._tick(terms)
        elif decoded[0] == "assign":
            _, d, slot = decoded
            self._assign(d, slot)
        else:  # charge
            self._send_to_charge(decoded[1])

        # auto-advance through ticks with no decision to make
        self._auto_advance(terms)

        self._last_reward_terms = terms
        reward = float(sum(terms.values()))
        terminated = all(d.lost for d in self.drones)
        truncated = self.t >= c.T_max
        return self._obs(), reward, terminated, truncated, self._info()

    def render(self):
        if self.render_mode == "rgb_array":
            from .visualize import render_frame
            return render_frame(self)
        return None

    # ---------- dynamics ----------
    def _assign(self, d: int, slot: int):
        drone = self.drones[d]
        order = self._visible_orders()[slot]
        order.drone = d
        drone.order_id = order.oid
        drone.status = TO_PICKUP
        drone.path = self.router.path(drone.pos, (order.ox, order.oy))
        self._last_assignments.append((d, order.oid))

    def _send_to_charge(self, d: int):
        drone = self.drones[d]
        target = min(self.hubs, key=lambda h: self.router.dist(drone.pos, h))
        drone.status = TO_CHARGER
        drone.path = self.router.path(drone.pos, target)

    def _order_by_id(self, oid):
        for o in self.pending:
            if o.oid == oid:
                return o
        return None

    def _spawn_orders(self):
        c = self.cfg
        n_new = int(self.np_random.poisson(c.lam))
        for _ in range(n_new):
            ox, oy = self._free_cell()
            dx, dy = self._free_cell()
            self.pending.append(Order(self._next_oid, ox, oy, dx, dy,
                                      self.t, self.t + c.sla_steps))
            self._next_oid += 1

    def _visible_orders(self):
        """Assignable orders (unpicked, unassigned), most-urgent first, capped at
        k_max. This is the observation/action window; orders beyond it still
        exist in `pending` and are dropped on deadline (no silent loss)."""
        avail = [o for o in self.pending if not o.picked and o.drone is None]
        avail.sort(key=lambda o: o.deadline)   # smallest deadline = most urgent
        return avail[:self.cfg.k_max]

    def _free_cell(self):
        while True:
            x = int(self.np_random.integers(0, self.cfg.H))
            y = int(self.np_random.integers(0, self.cfg.W))
            if self.grid[x, y] != NOFLY:
                return x, y

    def _tick(self, terms: dict):
        """Advance physical simulation by one step, accruing reward terms."""
        c = self.cfg
        # 1. new orders
        self._spawn_orders()

        # 2. move / act drones
        for d in self.drones:
            if d.lost:
                continue
            self.stats["drone_steps"] += 1
            if d.status in (TO_PICKUP, TO_DROPOFF, TO_CHARGER) and d.path:
                d.x, d.y = d.path.pop(0)
                d.soc -= c.e_move
                terms["energy"] += c.reward.r_energy * c.e_move
                self.stats["energy"] += c.e_move
            elif d.status == CHARGING:
                d.soc = min(1.0, d.soc + c.c_rate)
                self.stats["charging_steps"] += 1
                if d.soc >= 1.0:
                    d.status = IDLE
            else:
                d.soc -= c.e_idle
                terms["energy"] += c.reward.r_energy * c.e_idle
                self.stats["energy"] += c.e_idle
                if d.status == IDLE:
                    self.stats["idle_steps"] += 1

            # depletion check
            if d.soc <= 0.0 and not d.lost:
                if d.status in (TO_PICKUP, TO_DROPOFF, TO_CHARGER):
                    d.lost = True
                    d.soc = 0.0
                    terms["depletion"] += c.reward.r_depletion
                    self.stats["depletion_cost"] += -c.reward.r_depletion
                    self.stats["depletion_events"] += 1
                    o = self._order_by_id(d.order_id) if d.order_id is not None else None
                    if o is not None:
                        o.drone = None       # cargo lost with the drone; order
                        o.picked = False     # returns to the pool (re-assignable
                        d.order_id = None    # or droppable on deadline)
                    continue
                else:
                    d.soc = 0.0

            # arrival handling
            if not d.path:
                if d.status == TO_PICKUP:
                    o = self._order_by_id(d.order_id)
                    if o is not None and (d.x, d.y) == (o.ox, o.oy):
                        o.picked = True
                        d.status = TO_DROPOFF
                        d.path = self.router.path(d.pos, (o.dx, o.dy))
                elif d.status == TO_DROPOFF:
                    o = self._order_by_id(d.order_id)
                    if o is not None and (d.x, d.y) == (o.dx, o.dy):
                        o.delivered = True
                        terms["delivered"] += c.reward.r_delivered
                        self.stats["delivered"] += 1
                        self.stats["sum_delivery_time"] += self.t - o.created
                        if self.t <= o.deadline:
                            terms["ontime"] += c.reward.r_ontime_bonus
                            self.stats["ontime"] += 1
                        else:
                            late = self.t - o.deadline
                            terms["late"] += c.reward.r_late_penalty * late
                            self.stats["late_cost"] += -c.reward.r_late_penalty * late
                        d.status = IDLE
                        d.order_id = None
                elif d.status == TO_CHARGER:
                    if self.grid[d.x, d.y] == CHARGER:
                        d.status = CHARGING

        # 3. drop expired orders
        survivors = []
        for o in self.pending:
            if o.delivered:
                continue
            if not o.picked and self.t > o.deadline:
                o.dropped = True
                terms["dropped"] += c.reward.r_dropped
                self.stats["dropped"] += 1
                self.stats["drop_cost"] += -c.reward.r_dropped
                if o.drone is not None:
                    self.drones[o.drone].status = IDLE
                    self.drones[o.drone].order_id = None
                continue
            survivors.append(o)
        self.pending = survivors

        self.t += 1

    def _decision_epoch(self) -> bool:
        idle = any(d.status == IDLE and not d.lost for d in self.drones)
        has_order = any((not o.picked and o.drone is None) for o in self.pending)
        return idle and has_order

    def _auto_advance(self, reward_terms: Optional[dict]):
        """Advance ticks until a decision epoch or termination."""
        c = self.cfg
        terms = reward_terms if reward_terms is not None else {k: 0.0 for k in REWARD_TERMS}
        while (not self._decision_epoch()
               and self.t < c.T_max
               and not all(d.lost for d in self.drones)):
            self._tick(terms)

    # ---------- observation / info ----------
    def _action_mask(self):
        c = self.cfg
        mask = np.zeros(c.n_actions, dtype=np.int8)
        n_visible = len(self._visible_orders())
        for d_i, drone in enumerate(self.drones):
            if drone.lost:
                continue
            if drone.status == IDLE:
                for slot in range(n_visible):
                    mask[c.assign_index(d_i, slot)] = 1
                if drone.soc < 1.0:
                    mask[c.charge_index(d_i)] = 1
        mask[c.noop_index] = 1
        return mask

    def _obs(self):
        c = self.cfg
        dr = np.zeros((c.n_drones, 4 + N_STATUS + 1), dtype=np.float32)
        for i, d in enumerate(self.drones):
            onehot = np.zeros(N_STATUS, dtype=np.float32)
            onehot[d.status] = 1.0
            dr[i, 0] = d.x
            dr[i, 1] = d.y
            dr[i, 2] = d.soc
            dr[i, 3] = 0.0 if d.lost else 1.0
            dr[i, 4:4 + N_STATUS] = onehot
            dr[i, 4 + N_STATUS] = 1.0 if d.order_id is not None else 0.0

        orr = np.zeros((c.k_max, 5), dtype=np.float32)
        for i, o in enumerate(self._visible_orders()):
            orr[i] = [o.ox, o.oy, o.dx, o.dy, self.t - o.created]

        return {
            "drones": dr,
            "orders": orr,
            "grid": self.grid.copy(),
            "time": np.array([self.t / c.T_max], dtype=np.float32),
            "action_mask": self._action_mask(),
        }

    def _info(self):
        return {
            "action_mask": self._action_mask(),
            "assignments": list(self._last_assignments),
            "reward_terms": dict(self._last_reward_terms),
            "t": self.t,
            "metrics": self._episode_metrics(),
        }

    def _episode_metrics(self):
        return {
            "depletion_events": sum(1 for d in self.drones if d.lost),
            "n_pending": len(self.pending),
        }
