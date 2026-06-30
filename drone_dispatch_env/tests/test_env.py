"""Unit tests (Spec Section 10): no-fly enforcement, battery depletion, deadline
drop, charging, action masking, deterministic reset."""
import numpy as np
import gymnasium as gym
import drone_dispatch_env  # registers ids
from drone_dispatch_env import Config, DroneDispatchEnv, GreedyNearest, RandomPolicy
from drone_dispatch_env.config import NOFLY, CHARGER, CHARGING, TO_CHARGER, IDLE
from drone_dispatch_env.world import make_grid, Router


def test_registration():
    for eid in ("DroneDispatch-v0", "DroneControl-v0", "DroneDispatchMA-v0"):
        env = gym.make(eid)
        env.reset(seed=0)
        env.close()


def test_deterministic_reset_and_rollout():
    env = DroneDispatchEnv(Config(T_max=120))

    def rollout(seed):
        obs, _ = env.reset(seed=seed)
        rs, g0 = [], obs["grid"].copy()
        for _ in range(40):
            a = int(env._action_mask().argmax())
            obs, r, term, trunc, _ = env.step(a)
            rs.append(r)
            if term or trunc:
                break
        return g0, np.array(rs)

    g1, r1 = rollout(42)
    g2, r2 = rollout(42)
    assert np.array_equal(g1, g2)
    assert np.array_equal(r1, r2), "same seed must reproduce the reward sequence"
    g3, _ = rollout(7)
    assert not np.array_equal(g1, g3), "different seed should give a different world"


def test_no_global_state_across_resets():
    env = DroneDispatchEnv(Config())
    env.reset(seed=1)
    g1 = env.grid.copy()
    for _ in range(20):
        env.step(int(env._action_mask().argmax()))
    env.reset(seed=1)
    assert np.array_equal(env.grid, g1)
    assert env.t == 0 or env._decision_epoch()


def test_no_orphaned_picked_orders():
    # carrying drones deplete mid-delivery; picked orders must not clog pending
    cfg = Config(T_max=300, e_move=0.06, lam=0.6, n_drones=4)
    env = DroneDispatchEnv(cfg)
    env.reset(seed=2)
    for _ in range(300):
        _, _, term, trunc, _ = env.step(int(env._action_mask().argmax()))
        if term or trunc:
            break
    stuck = [o for o in env.pending
             if o.picked and not o.delivered and o.drone is None]
    assert len(stuck) == 0, f"{len(stuck)} orphaned picked orders"


def test_grid_connectivity_under_stress():
    from collections import deque
    from drone_dispatch_env.world import make_grid, _moves_for
    cfg = Config(n_nofly=60)  # stress-like density
    for seed in range(30):
        rng = np.random.default_rng(seed)
        grid, hubs = make_grid(cfg, rng)
        moves = _moves_for(cfg.neighborhood)
        total = int((grid != NOFLY).sum())
        seen = {hubs[0]}
        q = deque([hubs[0]])
        while q:
            cx, cy = q.popleft()
            for dx, dy in moves:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < cfg.H and 0 <= ny < cfg.W and grid[nx, ny] != NOFLY \
                        and (nx, ny) not in seen:
                    seen.add((nx, ny)); q.append((nx, ny))
        assert len(seen) == total, f"seed {seed}: free space not fully connected"


def test_preference_pairs_within_episode():
    from drone_dispatch_env import (generate_offline_dataset, load_offline_dataset,
                                     make_preference_pairs)
    import tempfile, os
    cfg = Config(T_max=80)
    path = os.path.join(tempfile.gettempdir(), "_pref_test.npz")
    generate_offline_dataset(path, cfg, min_transitions=2000, base_seed=900)
    d = load_offline_dataset(path)
    assert "timeouts" in d, "dataset must record timeouts for episode splits"
    assert (d["terminals"] | d["timeouts"]).sum() >= 1
    pairs = make_preference_pairs(path, n_pairs=100, seed=1)
    assert len(pairs) == 100 and pairs[0]["obs_a"].shape[0] == 25


def test_overflow_orders_not_silently_lost():
    # under heavy demand, pending must be able to exceed the observation window
    # k_max, and overflow must expire (drop) rather than vanish unpenalized.
    cfg = Config(T_max=300, lam=2.0, k_max=20, n_drones=8)
    env = DroneDispatchEnv(cfg)
    obs, _ = env.reset(seed=0)
    max_pending = 0
    g = GreedyNearest(cfg)
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(g.act(obs))
        max_pending = max(max_pending, len(env.pending))
        done = term or trunc
    assert max_pending > cfg.k_max, "orders should accumulate beyond the obs window"
    assert env.stats["dropped"] > 0, "overflow demand must be dropped, not discarded"
    # observation window never exceeds k_max
    assert obs["orders"].shape[0] == cfg.k_max


def test_obs_within_space():
    env = DroneDispatchEnv(Config())
    obs, _ = env.reset(seed=0)
    sp = env.observation_space
    for _ in range(200):
        for k, v in obs.items():
            assert sp.spaces[k].contains(v), f"obs[{k}] out of space"
        obs, _, term, trunc, _ = env.step(int(env._action_mask().argmax()))
        if term or trunc:
            break


def test_nofly_enforcement():
    cfg = Config()
    env = DroneDispatchEnv(cfg)
    env.reset(seed=3)
    # router never returns a path that steps onto a no-fly cell
    r: Router = env.router
    for _ in range(50):
        start = (int(np.random.randint(cfg.H)), int(np.random.randint(cfg.W)))
        goal = (int(np.random.randint(cfg.H)), int(np.random.randint(cfg.W)))
        if env.grid[start] == NOFLY or env.grid[goal] == NOFLY:
            continue
        for cell in r.path(start, goal):
            assert env.grid[cell] != NOFLY


def test_action_masking():
    cfg = Config()
    env = DroneDispatchEnv(cfg)
    env.reset(seed=5)
    mask = env._action_mask()
    assert mask[cfg.noop_index] == 1            # no-op always valid
    # a mask-ignoring action is treated as a no-op: penalty + time advance (no hang)
    invalid = np.flatnonzero(mask == 0)
    if len(invalid):
        t0 = env.t
        _, r, _, _, _ = env.step(int(invalid[0]))
        assert env.t >= t0
        assert r <= 0


def test_battery_depletion():
    cfg = Config(e_move=0.5, e_idle=0.1, T_max=200, n_drones=2, lam=1.0)
    env = DroneDispatchEnv(cfg)
    env.reset(seed=9)
    saw_loss = False
    for _ in range(400):
        a = env._action_mask().argmax()  # take an assignment if any, else noop
        _, _, term, trunc, info = env.step(int(a))
        if any(d.lost for d in env.drones):
            saw_loss = True
            break
        if term or trunc:
            break
    assert saw_loss, "high energy cost should produce a depletion event"


def test_deadline_drop():
    cfg = Config(sla_steps=2, lam=1.0, n_drones=1, T_max=60)
    env = DroneDispatchEnv(cfg)
    env.reset(seed=11)
    drops_before = env.stats["dropped"]
    for _ in range(60):
        _, _, term, trunc, _ = env.step(cfg.noop_index)  # never assign -> orders expire
        if term or trunc:
            break
    assert env.stats["dropped"] > drops_before


def test_charging_restores_soc():
    cfg = Config()
    env = DroneDispatchEnv(cfg)
    env.reset(seed=13)
    d0 = env.drones[0]
    d0.soc = 0.4
    a = cfg.charge_index(0)
    assert env._action_mask()[a]
    env.step(a)
    # advance until the drone reaches the charger and starts charging
    started = False
    for _ in range(80):
        env.step(cfg.noop_index)
        if env.drones[0].status == CHARGING:
            started = True
            break
    assert started, "drone should reach charger and begin charging"
    soc_at_charger = env.drones[0].soc
    env.step(cfg.noop_index)  # one charging tick
    assert env.drones[0].soc > soc_at_charger or env.drones[0].soc >= 1.0


def test_greedy_beats_random():
    cfg = Config(T_max=300)
    from drone_dispatch_env import evaluate
    g = evaluate(GreedyNearest(cfg), cfg, seeds=[0, 1, 2])["mean"]
    r = evaluate(RandomPolicy(cfg), cfg, seeds=[0, 1, 2])["mean"]
    assert g["episode_return"] >= r["episode_return"]


if __name__ == "__main__":
    import sys, traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
