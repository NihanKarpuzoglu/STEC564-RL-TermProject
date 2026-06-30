"""Environment configuration (Simulator Spec, all defaults from the spec)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# grid cell codes
FREE, NOFLY, HUB, CHARGER = 0, 1, 2, 3

# drone status codes (index used for one-hot in observation)
IDLE, TO_PICKUP, TO_DROPOFF, TO_CHARGER, CHARGING = 0, 1, 2, 3, 4
N_STATUS = 5


@dataclass
class RewardWeights:
    r_delivered: float = 10.0
    r_ontime_bonus: float = 5.0
    r_late_penalty: float = -0.1      # per step late
    r_dropped: float = -15.0
    r_energy: float = -1.0            # x SoC consumed
    r_depletion: float = -50.0
    r_noop: float = -0.01


@dataclass
class Config:
    H: int = 20
    W: int = 20
    T_max: int = 500
    n_drones: int = 8
    k_max: int = 20                   # max simultaneously tracked pending orders
    lam: float = 0.3                  # Poisson order arrival rate per step
    sla_steps: int = 60
    neighborhood: int = 4             # 4 or 8
    e_move: float = 0.01
    e_idle: float = 0.001
    c_rate: float = 0.05
    charge_threshold: float = 0.30    # used by greedy/baselines, not by env mask
    n_nofly: int = 24
    n_hubs: int = 4                   # chargers co-located at hubs
    init_soc: float = 1.0
    reward: RewardWeights = field(default_factory=RewardWeights)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d)
        rw = d.pop("reward", None)
        cfg = cls(**d)
        if rw:
            cfg.reward = RewardWeights(**rw)
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    # discrete action-space layout (frozen convention)
    @property
    def n_actions(self) -> int:
        return self.n_drones * self.k_max + self.n_drones + 1

    @property
    def noop_index(self) -> int:
        return self.n_actions - 1

    def charge_index(self, drone: int) -> int:
        return self.n_drones * self.k_max + drone

    def assign_index(self, drone: int, order_slot: int) -> int:
        return drone * self.k_max + order_slot

    def decode(self, a: int):
        """Return ('assign', drone, slot) | ('charge', drone) | ('noop',)."""
        if a == self.noop_index:
            return ("noop",)
        if a >= self.n_drones * self.k_max:
            return ("charge", a - self.n_drones * self.k_max)
        return ("assign", a // self.k_max, a % self.k_max)
