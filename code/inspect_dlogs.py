"""Inspect D_logs.npz: return distribution and behavior-policy coverage.

Run from the project root or from code/:
    python code/inspect_dlogs.py
    python inspect_dlogs.py   (from inside code/)
"""
from __future__ import annotations

import sys
import os
import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH    = os.path.join(PROJECT_ROOT, "drone_dispatch_env", "data", "D_logs.npz")

# action-space constants (from config.py / utils.py)
N_DRONES  = 8
K_MAX     = 20
N_ACTIONS = N_DRONES * K_MAX + N_DRONES + 1  # 169
N_ASSIGN  = N_DRONES * K_MAX                  # 160  (actions 0-159)
N_CHARGE  = N_DRONES                          # 8    (actions 160-167)
NOOP_IDX  = N_ACTIONS - 1                     # 168

# obs layout
N_FEAT_DRONE  = 10
N_FEAT_ORDER  = 5
OBS_DRONE_END = N_DRONES * N_FEAT_DRONE            # 80
OBS_ORDER_END = OBS_DRONE_END + K_MAX * N_FEAT_ORDER  # 180
SOC_INDICES   = [d * N_FEAT_DRONE + 2 for d in range(N_DRONES)]
ALIVE_INDICES = [d * N_FEAT_DRONE + 3 for d in range(N_DRONES)]


def sep(title="", width=70):
    if title:
        side = (width - len(title) - 2) // 2
        print("\n" + "-" * side + " " + title + " " + "-" * side)
    else:
        print("\n" + "-" * width)


def pct(arr, q):
    return np.percentile(arr, q)


def main():
    if not os.path.exists(DATA_PATH):
        sys.exit("[ERROR] Dataset not found at:\n  " + DATA_PATH)

    print("Loading  " + DATA_PATH)
    d          = np.load(DATA_PATH)
    obs        = d["observations"]
    actions    = d["actions"]
    rewards    = d["rewards"]
    terminals  = d["terminals"]
    timeouts   = d["timeouts"]
    ep_returns = d["episode_returns"]

    N     = len(actions)
    E     = len(ep_returns)
    dones = terminals | timeouts

    # ------------------------------------------------------------------
    sep("1. BASIC DATASET STATS")
    print("  Total transitions  : {:,}".format(N))
    print("  Total episodes     : {:,}".format(E))
    print("  Mean episode len   : {:.1f} steps".format(N / E))
    mem_mb = sum(d[k].nbytes for k in d.files) / 1024**2
    print("  Memory (all arrays): {:.1f} MB".format(mem_mb))
    print("  obs shape          : {}  dtype={}".format(obs.shape, obs.dtype))
    print("  actions unique     : {}  of {} possible".format(
          np.unique(actions).size, N_ACTIONS))
    print("  terminals          : {:,}  ({:.1f}%)".format(
          terminals.sum(), 100 * terminals.mean()))
    print("  timeouts (truncate): {:,}  ({:.1f}%)".format(
          timeouts.sum(), 100 * timeouts.mean()))

    # ------------------------------------------------------------------
    sep("2. PER-STEP REWARD DISTRIBUTION")
    pos_mask = rewards > 0
    neg_mask = rewards < 0
    zer_mask = rewards == 0
    print("  Positive rewards: {:>8,}  ({:.2f}%)   mean={:.2f}  max={:.2f}".format(
          pos_mask.sum(), 100 * pos_mask.mean(),
          rewards[pos_mask].mean(), rewards[pos_mask].max()))
    print("  Zero     rewards: {:>8,}  ({:.2f}%)".format(
          zer_mask.sum(), 100 * zer_mask.mean()))
    print("  Negative rewards: {:>8,}  ({:.2f}%)   mean={:.2f}  min={:.2f}".format(
          neg_mask.sum(), 100 * neg_mask.mean(),
          rewards[neg_mask].mean(), rewards[neg_mask].min()))
    print("  Overall  mean={:.4f}  std={:.4f}".format(
          rewards.mean(), rewards.std()))

    qs = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    print("\n  Per-step reward percentiles:")
    print("  " + "  ".join("p{:3d}={:7.2f}".format(q, pct(rewards, q)) for q in qs))

    # ------------------------------------------------------------------
    sep("2b. EPISODE RETURN DISTRIBUTION")
    print("  Episodes     : {:,}".format(E))
    print("  Mean return  : {:.2f}".format(ep_returns.mean()))
    print("  Std  return  : {:.2f}".format(ep_returns.std()))

    print("\n  Episode return percentiles:")
    print("  " + "  ".join("p{:3d}={:8.2f}".format(q, pct(ep_returns, q)) for q in qs))

    lo, hi   = ep_returns.min(), ep_returns.max()
    n_bins   = 10
    edges    = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(ep_returns, bins=edges)
    bar_max  = 30
    print("\n  Return histogram ({} bins):".format(n_bins))
    for i, c in enumerate(counts):
        bar = "#" * int(bar_max * c / counts.max())
        print("    [{:8.1f}, {:8.1f})  {:<{}}  {:5,}".format(
              edges[i], edges[i + 1], bar, bar_max, c))

    th_high = np.percentile(ep_returns, 75)
    th_low  = np.percentile(ep_returns, 25)
    print("\n  Top-25% episodes (return >= {:.1f}): {:,}".format(
          th_high, (ep_returns >= th_high).sum()))
    print("  Bot-25% episodes (return <= {:.1f}): {:,}".format(
          th_low, (ep_returns <= th_low).sum()))

    # ------------------------------------------------------------------
    sep("3. ACTION COVERAGE")
    act_counts    = np.bincount(actions, minlength=N_ACTIONS)
    n_seen        = (act_counts > 0).sum()
    assign_counts = act_counts[:N_ASSIGN]
    charge_counts = act_counts[N_ASSIGN:N_ASSIGN + N_CHARGE]
    noop_count    = act_counts[NOOP_IDX]

    print("  Actions seen       : {} / {}  ({:.1f}% coverage)".format(
          n_seen, N_ACTIONS, 100 * n_seen / N_ACTIONS))
    print("  Never-seen actions : {}".format(N_ACTIONS - n_seen))
    print()
    print("  ASSIGN (0-{})   seen {}/{}  total={:,}  ({:.1f}% of steps)".format(
          N_ASSIGN - 1, assign_counts.astype(bool).sum(), N_ASSIGN,
          assign_counts.sum(), 100 * assign_counts.sum() / N))
    print("  CHARGE ({}-{})   seen {}/{}  total={:,}  ({:.1f}% of steps)".format(
          N_ASSIGN, N_ASSIGN + N_CHARGE - 1,
          charge_counts.astype(bool).sum(), N_CHARGE,
          charge_counts.sum(), 100 * charge_counts.sum() / N))
    print("  NOOP   ({})        total={:,}  ({:.1f}% of steps)".format(
          NOOP_IDX, noop_count, 100 * noop_count / N))

    top5_idx = np.argsort(act_counts)[-5:][::-1]
    print("\n  Top-5 most-frequent actions:")
    for idx in top5_idx:
        atype = ("assign" if idx < N_ASSIGN
                 else "charge" if idx < N_ASSIGN + N_CHARGE else "noop")
        print("    action {:3d} ({:<6})  count={:,}  ({:.2f}%)".format(
              idx, atype, act_counts[idx], 100 * act_counts[idx] / N))

    rare = (act_counts > 0) & (act_counts < 10)
    print("\n  Rare actions (1-9 occurrences): {}".format(rare.sum()))

    print("\n  Per-drone CHARGE counts:")
    for di in range(N_DRONES):
        print("    drone {}: {:,}".format(di, charge_counts[di]))

    print("\n  Per-drone ASSIGN counts (summed over order slots):")
    for di in range(N_DRONES):
        c = assign_counts[di * K_MAX:(di + 1) * K_MAX].sum()
        print("    drone {}: {:,}".format(di, c))

    slot_counts = np.array([assign_counts[s::K_MAX].sum() for s in range(K_MAX)])
    print("\n  Per-slot ASSIGN counts (summed over drones):")
    for s in range(K_MAX):
        bar = "#" * int(20 * slot_counts[s] / slot_counts.max())
        print("    slot {:2d}: {:>8,}  {}".format(s, slot_counts[s], bar))

    # ------------------------------------------------------------------
    sep("4. OBSERVATION-SPACE STATISTICS")
    socs = obs[:, SOC_INDICES]
    print("  Drone SoC (state-of-charge) across all steps:")
    print("    mean={:.3f}  std={:.3f}  min={:.3f}  max={:.3f}".format(
          socs.mean(), socs.std(), socs.min(), socs.max()))
    print("    Fraction of (drone,step) pairs with SoC < 0.30: {:.2f}%".format(
          100 * (socs < 0.30).mean()))
    print("    Fraction of (drone,step) pairs with SoC < 0.10: {:.2f}%".format(
          100 * (socs < 0.10).mean()))

    alives = obs[:, ALIVE_INDICES]
    print("\n  Drone alive bits across all steps:")
    print("    Mean alive fraction per step: {:.3f}".format(alives.mean()))
    print("    Fraction of steps with >=1 dead drone: {:.2f}%".format(
          100 * (alives.min(axis=1) < 1).mean()))

    time_vals = obs[:, -1]
    print("\n  Normalised episode time:")
    print("    min={:.3f}  max={:.3f}  mean={:.3f}".format(
          time_vals.min(), time_vals.max(), time_vals.mean()))

    order_obs     = obs[:, OBS_DRONE_END:OBS_ORDER_END].reshape(-1, K_MAX, N_FEAT_ORDER)
    slot_occupied = (order_obs[:, :, 0] != 0) | (order_obs[:, :, 1] != 0)
    mean_orders   = slot_occupied.sum(axis=1).mean()
    print("\n  Mean active orders per step: {:.2f} / {}".format(mean_orders, K_MAX))
    print("  Fraction of steps with full queue (={}): {:.2f}%".format(
          K_MAX, 100 * (slot_occupied.sum(axis=1) == K_MAX).mean()))
    print("  Fraction of steps with empty queue: {:.2f}%".format(
          100 * (slot_occupied.sum(axis=1) == 0).mean()))

    # ------------------------------------------------------------------
    sep("5. COVERAGE-GAP PROXY: OBS FEATURE VARIANCES")
    feat_std      = obs.std(axis=0)
    low_var_thresh = 0.01
    n_low_var     = (feat_std < low_var_thresh).sum()
    print("  Features with std < {}: {} / {}".format(
          low_var_thresh, n_low_var, obs.shape[1]))
    print("  (Low variance means data rarely visits that part of state space)")
    top5_var_idx = np.argsort(feat_std)[-5:][::-1]
    print("  Top-5 highest-variance features (idx, std):")
    for idx in top5_var_idx:
        region = ("drone" if idx < OBS_DRONE_END
                  else "order" if idx < OBS_ORDER_END else "time")
        print("    feat[{:3d}] ({})  std={:.4f}".format(idx, region, feat_std[idx]))

    # ------------------------------------------------------------------
    sep("6. BEHAVIOR-POLICY ENTROPY ESTIMATE")
    probs     = act_counts / act_counts.sum()
    probs_nz  = probs[probs > 0]
    H_emp     = -np.sum(probs_nz * np.log(probs_nz))
    H_uniform = np.log(N_ACTIONS)
    print("  Empirical action entropy : {:.4f} nats".format(H_emp))
    print("  Uniform policy entropy   : {:.4f} nats".format(H_uniform))
    print("  Relative entropy         : {:.1f}%".format(100 * H_emp / H_uniform))
    print("  (100% = fully uniform; lower = more peaked / less coverage)")

    # ------------------------------------------------------------------
    sep("7. OPTIONAL: PCA PLOT")
    try:
        from sklearn.decomposition import PCA
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng        = np.random.default_rng(0)
        idx_sample = rng.choice(N, size=min(5000, N), replace=False)
        obs_sample = obs[idx_sample]
        rew_sample = d["rewards"][idx_sample]

        pca     = PCA(n_components=2)
        z       = pca.fit_transform(obs_sample)
        var_exp = pca.explained_variance_ratio_

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sc = axes[0].scatter(z[:, 0], z[:, 1], c=rew_sample,
                             cmap="RdYlGn", s=4, alpha=0.5)
        plt.colorbar(sc, ax=axes[0], label="step reward")
        axes[0].set_title("PCA of obs (N=5k sample)\ncolor = step reward")
        axes[0].set_xlabel("PC1 ({:.1f}% var)".format(100 * var_exp[0]))
        axes[0].set_ylabel("PC2 ({:.1f}% var)".format(100 * var_exp[1]))

        act_type = np.where(actions[idx_sample] < N_ASSIGN, 0,
                   np.where(actions[idx_sample] < N_ASSIGN + N_CHARGE, 1, 2))
        for t, col, lbl in zip([0, 1, 2], ["steelblue", "orange", "red"],
                                ["assign", "charge", "noop"]):
            m = act_type == t
            axes[1].scatter(z[m, 0], z[m, 1], c=col, s=4, alpha=0.4, label=lbl)
        axes[1].set_title("PCA of obs\ncolor = action type")
        axes[1].set_xlabel("PC1 ({:.1f}% var)".format(100 * var_exp[0]))
        axes[1].set_ylabel("PC2 ({:.1f}% var)".format(100 * var_exp[1]))
        axes[1].legend(markerscale=3)

        out_path = os.path.join(PROJECT_ROOT, "logs", "dlogs_pca.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        print("  PCA scatter saved -> " + out_path)
        print("  PC1 explains {:.1f}%,  PC2 {:.1f}% of variance.".format(
              100 * var_exp[0], 100 * var_exp[1]))
    except ImportError as e:
        print("  Skipped (missing dependency): {}".format(e))
        print("  Install scikit-learn + matplotlib to enable PCA plot.")

    sep()
    print("Done.\n")


if __name__ == "__main__":
    main()
