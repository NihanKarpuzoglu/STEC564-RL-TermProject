"""Visualization and replay (Spec Section 8).

- render_frame(env)            -> rgb_array top-down grid frame
- Recorder                     -> captures full state sequence for deterministic replay
- Replayer                     -> play/pause/step-fwd/step-back/scrub (matplotlib widgets)
                                  + to_gif / to_mp4 export, optional overlays
- compare(policy_a, policy_b)  -> side-by-side on the same seed
- metrics_dashboard(results)   -> post-hoc metric plots

Overlays read from data the env already exposes (reward terms, action mask,
assignments via info) plus Q/V/pi the agent supplies through the frozen
agent_interface.Introspectable hook.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe default; Replayer switches if a display exists
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .config import (FREE, NOFLY, HUB, CHARGER,
                     IDLE, TO_PICKUP, TO_DROPOFF, TO_CHARGER, CHARGING)

GRID_CMAP = ListedColormap(["#f4f4f4", "#3a3a3a", "#cfe8ff", "#9bd0ff"])
STATUS_COLOR = {IDLE: "#888888", TO_PICKUP: "#2a9d8f", TO_DROPOFF: "#e76f51",
                TO_CHARGER: "#f4a261", CHARGING: "#264653"}


# ---------- snapshot ----------
def snapshot(env) -> dict:
    """Minimal serializable state for deterministic replay."""
    return {
        "t": env.t,
        "grid": env.grid,
        "drones": [(d.x, d.y, d.soc, d.status, d.lost) for d in env.drones],
        "orders": [(o.ox, o.oy, o.dx, o.dy, o.deadline - env.t, o.picked)
                   for o in env.pending],
        "reward_terms": dict(getattr(env, "_last_reward_terms", {})),
        "assignments": list(getattr(env, "_last_assignments", [])),
    }


def _draw(ax, snap, trails=None):
    grid = snap["grid"]
    H, W = grid.shape
    ax.clear()
    ax.imshow(grid.T, origin="lower", cmap=GRID_CMAP, vmin=0, vmax=3,
              extent=[0, H, 0, W])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, H); ax.set_ylim(0, W)

    # orders, colored by deadline urgency (time-to-deadline)
    for (ox, oy, dx, dy, ttl, picked) in snap["orders"]:
        urg = max(0.0, min(1.0, ttl / 60.0))
        col = (1.0, urg, 0.0)  # red (urgent) -> yellow/green (slack)
        ax.scatter(ox + 0.5, oy + 0.5, marker="s", s=40, color=col,
                   edgecolors="black", linewidths=0.4, zorder=3)
        ax.plot([ox + 0.5, dx + 0.5], [oy + 0.5, dy + 0.5],
                color=col, lw=0.5, ls=":", alpha=0.6, zorder=2)
        ax.scatter(dx + 0.5, dy + 0.5, marker="x", s=25, color=col, zorder=3)

    # drones with battery bar and trail
    for i, (x, y, soc, status, lost) in enumerate(snap["drones"]):
        if trails and i < len(trails) and len(trails[i]) > 1:
            tr = np.array(trails[i])
            ax.plot(tr[:, 0] + 0.5, tr[:, 1] + 0.5,
                    color=STATUS_COLOR.get(status, "#888"), lw=1.0, alpha=0.4, zorder=2)
        col = "#000000" if lost else STATUS_COLOR.get(status, "#888")
        ax.scatter(x + 0.5, y + 0.5, s=70, color=col, edgecolors="white",
                   linewidths=0.6, zorder=4)
        # battery bar
        ax.add_patch(plt.Rectangle((x + 0.1, y + 0.9), 0.8 * soc, 0.12,
                                   color="#2a9d8f" if soc > 0.3 else "#e63946",
                                   zorder=5))
    ax.set_title(f"t={snap['t']}", fontsize=9)


def render_frame(env):
    """Return an rgb_array of the current env state."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    _draw(ax, snapshot(env))
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def render_control_frame(env):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    grid = env.grid
    H, W = grid.shape
    ax.imshow(grid.T, origin="lower", cmap=GRID_CMAP, vmin=0, vmax=3,
              extent=[0, H, 0, W])
    ax.scatter(*(env.pos + 0.0), s=80, color="#2a9d8f", zorder=4)
    ax.scatter(*(env.target + 0.0), marker="*", s=120, color="#e76f51", zorder=4)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"t={env.t} soc={env.soc:.2f}")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


# ---------- recording ----------
class Recorder:
    """Captures the full state sequence of an episode for deterministic replay."""

    def __init__(self):
        self.frames: list[dict] = []
        self.trails: list[list] = None

    def capture(self, env):
        snap = snapshot(env)
        if self.trails is None:
            self.trails = [[] for _ in snap["drones"]]
        for i, (x, y, *_rest) in enumerate(snap["drones"]):
            self.trails[i].append((x, y))
            self.trails[i] = self.trails[i][-12:]  # short trail
        snap["_trails"] = [list(t) for t in self.trails]
        self.frames.append(snap)

    def __len__(self):
        return len(self.frames)


def record_episode(policy, env, seed: int) -> Recorder:
    from .evaluate import run_episode
    rec = Recorder()
    run_episode(policy, env, seed, recorder=rec)
    return rec


# ---------- replay (interactive + export) ----------
class Replayer:
    """Replays a recorded episode. Interactive controls when a GUI backend is
    available; always supports headless to_gif / to_mp4 export."""

    def __init__(self, recorder: Recorder, overlays: Optional[dict] = None):
        self.frames = recorder.frames
        self.overlays = overlays or {}  # optional per-frame dicts (reward stream etc.)

    def _render_index(self, ax, i):
        snap = self.frames[i]
        _draw(ax, snap, trails=snap.get("_trails"))

    def to_gif(self, path: str, fps: int = 5):
        import imageio
        imgs = []
        for i in range(len(self.frames)):
            fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
            self._render_index(ax, i)
            fig.tight_layout(pad=0.2)
            fig.canvas.draw()
            imgs.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            plt.close(fig)
        imageio.mimsave(path, imgs, fps=fps)
        return path

    def to_mp4(self, path: str, fps: int = 10):
        import imageio
        with imageio.get_writer(path, fps=fps) as w:
            for i in range(len(self.frames)):
                fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
                self._render_index(ax, i)
                fig.tight_layout(pad=0.2)
                fig.canvas.draw()
                w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
                plt.close(fig)
        return path

    def play(self):
        """Interactive player: play/pause, step-forward, step-back, scrub.
        Requires an interactive matplotlib backend."""
        import matplotlib.pyplot as ipt
        from matplotlib.widgets import Slider, Button
        ipt.switch_backend(matplotlib.rcParamsDefault["backend"])

        fig, ax = ipt.subplots(figsize=(5, 5.6))
        ipt.subplots_adjust(bottom=0.2)
        state = {"i": 0, "playing": False}

        def show(i):
            state["i"] = int(i) % len(self.frames)
            self._render_index(ax, state["i"])
            slider.eventson = False
            slider.set_val(state["i"])
            slider.eventson = True
            fig.canvas.draw_idle()

        ax_slider = ipt.axes([0.15, 0.10, 0.7, 0.03])
        slider = Slider(ax_slider, "step", 0, len(self.frames) - 1, valinit=0, valstep=1)
        slider.on_changed(lambda v: show(v))

        def mk_button(x, label, cb):
            b = Button(ipt.axes([x, 0.03, 0.12, 0.05]), label)
            b.on_clicked(cb)
            return b

        b_prev = mk_button(0.15, "◀ step", lambda e: show(state["i"] - 1))
        b_play = mk_button(0.30, "play", None)
        b_next = mk_button(0.45, "step ▶", lambda e: show(state["i"] + 1))

        timer = fig.canvas.new_timer(interval=200)

        def tick():
            if state["playing"]:
                show(state["i"] + 1)

        def toggle(e):
            state["playing"] = not state["playing"]
            b_play.label.set_text("pause" if state["playing"] else "play")
        b_play.on_clicked(toggle)
        timer.add_callback(tick)
        timer.start()
        show(0)
        ipt.show()


# ---------- comparison ----------
def compare(policy_a, policy_b, seed: int, config=None, save: Optional[str] = None,
            labels=("A", "B")):
    """Render two policies on the same seed side by side as a GIF (or rgb arrays)."""
    from .env_dispatch import DroneDispatchEnv
    cfg = config
    ra = record_episode(policy_a, DroneDispatchEnv(cfg) if cfg else DroneDispatchEnv(), seed)
    rb = record_episode(policy_b, DroneDispatchEnv(cfg) if cfg else DroneDispatchEnv(), seed)
    n = max(len(ra), len(rb))

    def frame(k):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=100)
        for ax, rec, lab in ((axes[0], ra, labels[0]), (axes[1], rb, labels[1])):
            idx = min(k, len(rec.frames) - 1)
            snap = rec.frames[idx]
            _draw(ax, snap, trails=snap.get("_trails"))
            ax.set_title(f"{lab}  t={snap['t']}", fontsize=9)
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img

    if save:
        import imageio
        imageio.mimsave(save, [frame(k) for k in range(n)], fps=5)
        return save
    return [frame(k) for k in range(n)]


# ---------- overlays (8.2) ----------
def reward_stream_figure(recorder: Recorder):
    """Stacked reward-term contributions per recorded frame (reward-gaming view)."""
    terms = ["delivered", "ontime", "late", "dropped", "energy", "depletion", "noop"]
    series = {k: [f["reward_terms"].get(k, 0.0) for f in recorder.frames] for k in terms}
    fig, ax = plt.subplots(figsize=(7, 3))
    bottom_pos = np.zeros(len(recorder.frames))
    for k in terms:
        ax.bar(range(len(recorder.frames)), series[k], bottom=None, label=k, alpha=0.7)
    ax.legend(fontsize=7, ncol=4)
    ax.set_xlabel("frame"); ax.set_ylabel("reward term")
    fig.tight_layout()
    return fig


def overlay_action_values(ax, agent, obs, cfg):
    """Per-action Q overlay using the Introspectable hook (value-based)."""
    from agent_interface import gather_overlays
    ov = gather_overlays(agent, obs)
    q = ov.get("q")
    if q is None:
        return None
    ax.bar(range(len(q)), np.nan_to_num(q, nan=np.nanmin(q)))
    ax.set_title("per-action Q"); ax.set_xlabel("action")
    return q


def state_value_heatmap(ax, agent, obs):
    """V(s) grid heatmap from the Introspectable hook."""
    from agent_interface import gather_overlays
    v = gather_overlays(agent, obs).get("v")
    if v is None:
        return None
    ax.imshow(v.T, origin="lower", cmap="viridis")
    ax.set_title("state value V(s)")
    return v


def metrics_dashboard(results: dict, save: Optional[str] = None):
    """Post-hoc plots of the headline metrics across seeds."""
    per = results["per_seed"]
    keys = ["cost_per_order", "success_rate", "ontime_rate",
            "energy_per_order", "depletion_events"]
    fig, axes = plt.subplots(1, len(keys), figsize=(3 * len(keys), 3))
    for ax, k in zip(axes, keys):
        ax.bar(range(len(per)), [m[k] for m in per])
        ax.set_title(k, fontsize=8); ax.set_xlabel("seed")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=100); plt.close(fig); return save
    return fig
