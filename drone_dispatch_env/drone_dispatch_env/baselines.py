"""Built-in baselines (Spec Section 6). Each implements act(obs) -> action,
matching the Policy protocol in agent_interface.py."""
from __future__ import annotations

import numpy as np

from .config import Config
from .world import Router


class _RoutedDist:
    """Cached no-fly-aware routed distances for the baselines.

    Replaces the Manhattan proxy that mis-ranked drone/order pairs separated by
    no-fly walls (Spec Section 11 #1). A `Router` is built once per world and a
    single-source BFS field is memoized per source cell, so distances are exact
    routed costs while BFS runs at most once per distinct cell per reset. The
    grid is read from the observation, so this stays inside the frozen `Policy`
    contract (no env internals). The cache rebuilds automatically when the grid
    changes (i.e., a new episode/reset)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._grid_key = None
        self._router = None
        self._fields: dict = {}

    def _ensure(self, grid):
        grid = np.asarray(grid)
        key = grid.tobytes()
        if key != self._grid_key:
            self._grid_key = key
            self._router = Router(grid, self.cfg.neighborhood)
            self._fields = {}

    def dist(self, grid, source, target) -> float:
        """Routed distance from `source` to `target` (cells). Falls back to
        Manhattan only if the routed distance is infinite (should not happen
        under the connectivity guarantee, but keeps the baseline robust)."""
        self._ensure(grid)
        source = (int(source[0]), int(source[1]))
        field = self._fields.get(source)
        if field is None:
            field = self._router.dist_field(source)
            self._fields[source] = field
        tx, ty = int(target[0]), int(target[1])
        d = float(field[tx, ty])
        if not np.isfinite(d):
            return abs(source[0] - tx) + abs(source[1] - ty)
        return d


def _mask(obs):
    return np.asarray(obs["action_mask"])


def _valid_assignments(obs, cfg: Config):
    """Yield (action_index, drone, slot) for valid assignment actions."""
    mask = _mask(obs)
    out = []
    for a in range(cfg.n_drones * cfg.k_max):
        if mask[a]:
            out.append((a, a // cfg.k_max, a % cfg.k_max))
    return out


class RandomPolicy:
    """Random valid assignment / action."""

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        valid = np.flatnonzero(_mask(obs))
        return int(self.rng.choice(valid))


class GreedyNearest:
    """Assign each pending order to the nearest valid idle drone; send
    below-threshold idle drones to charge. The primary bar."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._routed = _RoutedDist(cfg)

    def act(self, obs):
        c = self.cfg
        mask = _mask(obs)
        drones = obs["drones"]
        orders = obs["orders"]
        grid = obs["grid"]

        # 1. charge a critically low idle drone
        for d in range(c.n_drones):
            if mask[c.charge_index(d)] and drones[d, 2] < c.charge_threshold:
                return c.charge_index(d)

        # 2. nearest (drone, order) over all valid assignments, ranked by true
        #    routed (no-fly-aware) distance rather than the Manhattan proxy
        best, best_d = None, np.inf
        for a, d, slot in _valid_assignments(obs, c):
            if drones[d, 2] < c.charge_threshold:
                continue  # too low to take a job
            # source = order origin: one BFS field serves every drone for that
            # order, so the per-order field is reused across the drone loop
            dist = self._routed.dist(grid, (orders[slot, 0], orders[slot, 1]),
                                     (drones[d, 0], drones[d, 1]))
            if dist < best_d:
                best_d, best = dist, a
        if best is not None:
            return int(best)

        # 3. nothing useful -> defer
        return c.noop_index


class MILPRolling:
    """Rolling-horizon assignment MILP (PuLP). Minimizes total Manhattan travel
    of an assignment between idle drones and pending orders, recomputed each
    decision epoch. Returns one assignment per call (the env consumes one per
    step); subsequent calls re-solve on the updated state."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._routed = _RoutedDist(cfg)

    def act(self, obs):
        c = self.cfg
        mask = _mask(obs)
        # charge a critically low idle drone first (energy management)
        for d in range(c.n_drones):
            if mask[c.charge_index(d)] and obs["drones"][d, 2] < c.charge_threshold:
                return c.charge_index(d)
        valid = [(a, d, s) for (a, d, s) in _valid_assignments(obs, c)
                 if obs["drones"][d, 2] >= c.charge_threshold]
        if not valid:
            return c.noop_index

        import pulp
        drones = sorted({d for _, d, _ in valid})
        slots = sorted({s for _, _, s in valid})
        dr, orr = obs["drones"], obs["orders"]
        grid = obs["grid"]
        cost = {}
        for a, d, s in valid:
            # true routed distance (no-fly-aware), cached per order origin
            cost[(d, s)] = self._routed.dist(grid, (orr[s, 0], orr[s, 1]),
                                             (dr[d, 0], dr[d, 1]))

        prob = pulp.LpProblem("assign", pulp.LpMinimize)
        xs = {(d, s): pulp.LpVariable(f"x_{d}_{s}", cat="Binary")
              for (d, s) in cost}
        # max-cardinality, min-cost matching: reward each assignment by M so the
        # solver assigns as many pairs as possible, breaking ties by travel cost.
        M = max(cost.values()) + 1.0
        prob += pulp.lpSum((cost[k] - M) * xs[k] for k in xs)
        for d in drones:
            prob += pulp.lpSum(xs[(d, s)] for s in slots if (d, s) in xs) <= 1
        for s in slots:
            prob += pulp.lpSum(xs[(d, s)] for d in drones if (d, s) in xs) <= 1
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        chosen = [(d, s) for (d, s), v in xs.items() if v.value() and v.value() > 0.5]
        if not chosen:
            return c.noop_index
        # return the cheapest selected pair this step
        d, s = min(chosen, key=lambda k: cost[k])
        return c.assign_index(d, s)


def make_baseline(name: str, cfg: Config, seed: int = 0):
    name = name.lower()
    if name == "random":
        return RandomPolicy(cfg, seed)
    if name in ("greedy", "greedy_nearest"):
        return GreedyNearest(cfg)
    if name in ("milp", "milp_rolling"):
        return MILPRolling(cfg)
    raise ValueError(f"unknown baseline: {name}")
