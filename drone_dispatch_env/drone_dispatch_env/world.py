"""World model: deterministic grid, drones, order stream, no-fly-aware router."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .config import Config, FREE, NOFLY, HUB, CHARGER


@dataclass
class Drone:
    x: int
    y: int
    soc: float
    status: int
    order_id: Optional[int] = None
    path: list = field(default_factory=list)   # remaining cells to traverse
    lost: bool = False

    @property
    def pos(self):
        return (self.x, self.y)


@dataclass
class Order:
    oid: int
    ox: int
    oy: int
    dx: int
    dy: int
    created: int
    deadline: int
    picked: bool = False
    delivered: bool = False
    dropped: bool = False
    drone: Optional[int] = None


def _moves_for(neighborhood: int):
    if neighborhood == 8:
        return [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
    return [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _free_connected(grid: np.ndarray, moves, start) -> bool:
    """True iff every non-no-fly cell is reachable from `start`."""
    H, W = grid.shape
    total = int((grid != NOFLY).sum())
    seen = {start}
    q = deque([start])
    while q:
        cx, cy = q.popleft()
        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx, ny] != NOFLY \
                    and (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny))
    return len(seen) == total


def make_grid(cfg: Config, rng: np.random.Generator):
    """Deterministically place hubs/chargers and no-fly cells. No-fly cells are
    placed only if they keep the free space fully connected, so every hub,
    pickup, and charger is always reachable (matters under high-no-fly configs)."""
    grid = np.full((cfg.H, cfg.W), FREE, dtype=np.int8)

    # hubs on a coarse lattice, chargers co-located
    hubs = []
    rows = int(np.ceil(np.sqrt(cfg.n_hubs)))
    cols = int(np.ceil(cfg.n_hubs / rows))
    placed = 0
    for i in range(rows):
        for j in range(cols):
            if placed >= cfg.n_hubs:
                break
            hx = int((i + 1) * cfg.H / (rows + 1))
            hy = int((j + 1) * cfg.W / (cols + 1))
            grid[hx, hy] = CHARGER          # charger co-located at hub
            hubs.append((hx, hy))
            placed += 1

    # no-fly cells, avoiding hubs, preserving connectivity of the free space
    moves = _moves_for(cfg.neighborhood)
    count = 0
    tries = 0
    while count < cfg.n_nofly and tries < cfg.n_nofly * 100:
        tries += 1
        x = int(rng.integers(0, cfg.H))
        y = int(rng.integers(0, cfg.W))
        if grid[x, y] != FREE:
            continue
        grid[x, y] = NOFLY
        if _free_connected(grid, moves, hubs[0]):
            count += 1
        else:
            grid[x, y] = FREE               # revert: would disconnect the map
    return grid, hubs


class Router:
    """BFS shortest path that never enters no-fly cells."""

    def __init__(self, grid: np.ndarray, neighborhood: int = 4):
        self.grid = grid
        self.H, self.W = grid.shape
        if neighborhood == 8:
            self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def passable(self, x, y):
        return 0 <= x < self.H and 0 <= y < self.W and self.grid[x, y] != NOFLY

    def path(self, start, goal):
        """List of cells from the cell AFTER start up to goal (empty if unreachable
        or already there)."""
        if start == goal:
            return []
        if not self.passable(*goal):
            return []
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            cx, cy = cur
            for dx, dy in self.moves:
                nx, ny = cx + dx, cy + dy
                if self.passable(nx, ny) and (nx, ny) not in prev:
                    prev[(nx, ny)] = cur
                    q.append((nx, ny))
        if goal not in prev:
            return []
        cells = []
        node = goal
        while node != start:
            cells.append(node)
            node = prev[node]
        cells.reverse()
        return cells

    def dist(self, start, goal):
        p = self.path(start, goal)
        if start == goal:
            return 0
        return len(p) if p else np.inf

    def dist_field(self, source):
        """Single-source BFS: routed distance from `source` to every reachable
        cell. Returns a float array `[H, W]`; unreachable / no-fly cells are
        `inf`. One BFS yields distances to all drones/targets at once, so callers
        cache one field per source instead of re-running BFS per pair."""
        dist = np.full((self.H, self.W), np.inf, dtype=np.float64)
        if not self.passable(*source):
            return dist
        dist[source] = 0.0
        q = deque([source])
        while q:
            cx, cy = q.popleft()
            d0 = dist[cx, cy]
            for dx, dy in self.moves:
                nx, ny = cx + dx, cy + dy
                if self.passable(nx, ny) and dist[nx, ny] == np.inf:
                    dist[nx, ny] = d0 + 1.0
                    q.append((nx, ny))
        return dist
