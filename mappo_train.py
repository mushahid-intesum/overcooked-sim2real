"""
mappo_train.py  — MAPPO for Overcooked-AI partial-obs sim2real
──────────────────────────────────────────────────────────────────────────────
Fixes vs previous version
    1. Advantage normalisation over full rollout, not per-minibatch
       (per-minibatch collapses to zero with sparse cooperative rewards)
    2. Reward shaping: sparse soup delivery + dense shaped reward from
       Overcooked's built-in potential-based shaping
    3. Separate critic lr (lower) and linear lr decay schedulers
    4. GRU hidden detached at update time — stored hidden is input state,
       not recomputed — prevents stale-state log-prob ratio corruption
    5. Critic and actor loss display with enough precision to see small values
    6. Advantages normalised globally before shuffle, not inside minibatch
    7. Live training dashboard via matplotlib (optional, graceful fallback)
    8. Corrected env API: reset() returns obs directly, step() returns 4-tuple
    9. Single team reward correctly handled (not per-agent dict)

Architecture
────────────
  Actor  (on-device, TFLM):   MobileNet-V1 DWS CNN + scalar MLP → GRU → logits
  Critic (training only):      MLP on full global state → V(s)
  CTDE:  critic sees full state, actor sees partial noisy FoV obs only

Actions (6) — matches Action.INDEX_TO_ACTION order in Overcooked-AI:
    0: NORTH   1: SOUTH   2: EAST   3: WEST   4: STAY   5: INTERACT
"""

from __future__ import annotations

import argparse
import collections
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from overcooked_partial_obs_wrapper import (
    NUM_CHANNELS,
    NoiseCfg,
    PartialObsWrapper,
)

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Cfg:
    layout:            str   = "cramped_room"
    fov_radius:        int   = 5
    num_envs:          int   = 8
    episode_steps:     int   = 400

    # Network
    hidden_dim:        int   = 128
    action_dim:        int   = 6
    width_mult:        float = 0.25

    # PPO — actor
    actor_lr:          float = 3e-4
    # Critic lr lower than actor — prevents value fn from over-fitting fast
    # and making GAE targets unstable before the actor has learned anything
    critic_lr:         float = 1e-4
    gamma:             float = 0.99
    gae_lambda:        float = 0.95
    clip_eps:          float = 0.2
    vf_coef:           float = 0.5
    ent_coef:          float = 0.02     # higher than default for sparse coop task
    max_grad_norm:     float = 0.5
    ppo_epochs:        int   = 4
    minibatch_size:    int   = 256

    # Reward shaping
    sparse_factor:         float = 5.0   # scale on soup delivery reward
    reward_shaping_factor: float = 1.0   # scale on Overcooked shaped reward

    # Rollout
    rollout_steps:     int   = 128
    total_steps:       int   = 10_000_000

    # Logging / saving / eval
    log_interval:      int   = 10
    save_interval:     int   = 100
    eval_interval:     int   = 50    # run one eval episode every N updates
    save_dir:          str   = "checkpoints"
    dashboard:         bool  = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# MobileNet-V1 depthwise-separable backbone
# ─────────────────────────────────────────────────────────────────────────────

class DWSConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw    = nn.Conv2d(in_ch, in_ch,  3, stride, 1, groups=in_ch, bias=False)
        self.pw    = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.bn_pw = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn_pw(self.pw(F.relu(self.bn_dw(self.dw(x))))))


class MobileNetEncoder(nn.Module):
    """MobileNet-V1 at width_mult=0.25 for symbolic (C,D,D) grid tensors."""

    def __init__(self, in_channels=NUM_CHANNELS, embed_dim=128,
                 width_mult=0.25, input_size=11):
        super().__init__()
        def ch(c): return max(1, int(c * width_mult))

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ch(32), 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch(32)),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DWSConvBlock(ch(32),  ch(64),  1),
            DWSConvBlock(ch(64),  ch(128), 2),   # 11→6
            DWSConvBlock(ch(128), ch(128), 1),
            DWSConvBlock(ch(128), ch(256), 2),   # 6→3
            DWSConvBlock(ch(256), ch(256), 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(ch(256), embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(self.pool(self.blocks(self.stem(x))).flatten(1))


# ─────────────────────────────────────────────────────────────────────────────
# Actor
# ─────────────────────────────────────────────────────────────────────────────

SCALAR_DIM = 7   # [delta_row, delta_col, held_onehot x5]

class Actor(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, fov_size=11,
                 hidden_dim=128, action_dim=6, width_mult=0.25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cnn = MobileNetEncoder(in_channels, hidden_dim, width_mult, fov_size)
        self.scalar_enc = nn.Sequential(
            nn.Linear(SCALAR_DIM, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 32),         nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gru  = nn.GRUCell(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, grid, scalars, hidden):
        # dim=1 not dim=-1 — negative axes survive into ONNX Concat nodes
        # and break esp-ppq's layout pattern resolver (axis=-1 not in perm list)
        fused      = self.fusion(
            torch.cat([self.cnn(grid), self.scalar_enc(scalars)], dim=1)
        )
        new_hidden = self.gru(fused, hidden)
        return self.head(new_hidden), new_hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def get_action_logprob_entropy(self, grid, scalars, hidden, action=None):
        logits, new_hidden = self.forward(grid, scalars, hidden)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), new_hidden


# ─────────────────────────────────────────────────────────────────────────────
# Centralised critic (training only)
# ─────────────────────────────────────────────────────────────────────────────

class CentralCritic(nn.Module):
    def __init__(self, global_grid_channels=NUM_CHANNELS,
                 global_grid_h=5, global_grid_w=4,
                 num_agents=2, hidden_dim=256):
        super().__init__()
        flat = global_grid_channels * global_grid_h * global_grid_w
        self.net = nn.Sequential(
            nn.Linear(flat + num_agents * SCALAR_DIM, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, global_grid, scalars):
        flat = torch.cat([global_grid.flatten(1), scalars], dim=1)
        return self.net(flat).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout buffer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    grids:          List[Dict[str, torch.Tensor]] = field(default_factory=list)
    scalars:        List[Dict[str, torch.Tensor]] = field(default_factory=list)
    hiddens:        List[Dict[str, torch.Tensor]] = field(default_factory=list)
    actions:        List[Dict[str, torch.Tensor]] = field(default_factory=list)
    log_probs:      List[Dict[str, torch.Tensor]] = field(default_factory=list)
    # Single scalar team reward per step per env
    rewards:        List[torch.Tensor]            = field(default_factory=list)
    shaped_rewards: List[torch.Tensor]            = field(default_factory=list)
    dones:          List[torch.Tensor]            = field(default_factory=list)
    global_grids:   List[torch.Tensor]            = field(default_factory=list)
    global_scalars: List[torch.Tensor]            = field(default_factory=list)

    def clear(self):
        for f in self.__dataclass_fields__:
            setattr(self, f, [])

    def __len__(self):
        return len(self.actions)


# ─────────────────────────────────────────────────────────────────────────────
# GAE
# ─────────────────────────────────────────────────────────────────────────────

def compute_gae(rewards, values, dones, gamma, lam):
    """
    rewards : (T, N)
    values  : (T+1, N)
    dones   : (T, N)
    Returns unnormalised advantages (T, N) and returns (T, N).
    Normalisation happens OUTSIDE over the full flattened (T*N,) vector.
    """
    T, N = rewards.shape
    adv  = torch.zeros(T, N)
    gae  = torch.zeros(N)
    for t in reversed(range(T)):
        delta  = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae    = delta + gamma * lam * (1 - dones[t]) * gae
        adv[t] = gae
    return adv, adv + values[:T]


# ─────────────────────────────────────────────────────────────────────────────
# Live training dashboard
# ─────────────────────────────────────────────────────────────────────────────

class Dashboard:
    """
    Matplotlib live dashboard.

    Layout (3 columns × 2 rows + right column spanning both rows for game render):

        ┌─────────────────────┬──────────┬──────────────────────┐
        │  episode return     │ actor    │                      │
        │  (100-ep window)    │ loss     │   GAME RENDER        │
        ├──────────┬──────────┤──────────┤   (eval rollout,     │
        │  critic  │ entropy  │  steps/s │    no noise)         │
        │  loss    │          │          │                      │
        └──────────┴──────────┴──────────┴──────────────────────┘

    The game render panel shows a single frame from an eval rollout captured
    via EvalRenderer. It updates every time dashboard.update_render() is called.

    Falls back gracefully to headless mode (no crash, just no window).
    Pass enabled=False or use --no_dashboard to skip entirely.
    """

    def __init__(self, enabled: bool = True):
        self.enabled  = enabled
        self.fig      = None
        self._im      = None          # imshow handle for game render
        self._ax_game = None
        self._data: Dict[str, List] = {
            k: [] for k in ["xs", "ret", "aloss", "closs", "ent", "sps"]
        }
        if not enabled:
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")   # swap to Qt5Agg if preferred
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            self._plt = plt
            self.fig  = plt.figure(figsize=(17, 7))
            self.fig.suptitle(
                "MAPPO — Overcooked-AI partial obs sim2real",
                fontsize=12, fontweight="bold",
            )

            # Outer: metrics left (3/5 width) + game render right (2/5 width)
            outer = gridspec.GridSpec(
                1, 2, figure=self.fig,
                width_ratios=[3, 2], hspace=0.05, wspace=0.30,
            )
            # Metrics: 2×3 grid in left column
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 3, subplot_spec=outer[0],
                hspace=0.55, wspace=0.40,
            )

            axes = {
                "ret":   self.fig.add_subplot(inner[0, :2]),   # wide top-left
                "aloss": self.fig.add_subplot(inner[0, 2]),
                "closs": self.fig.add_subplot(inner[1, 0]),
                "ent":   self.fig.add_subplot(inner[1, 1]),
                "sps":   self.fig.add_subplot(inner[1, 2]),
            }
            titles = {
                "ret":   ("Mean episode return (100-ep window)", "return", "b"),
                "aloss": ("Actor loss",                          "loss",   "r"),
                "closs": ("Critic loss",                         "loss",   "g"),
                "ent":   ("Policy entropy",                      "nats",   "m"),
                "sps":   ("Steps / second",                      "sps",    "c"),
            }
            self._lines = {}
            for key, (title, ylabel, color) in titles.items():
                ax = axes[key]
                ax.set_title(title, fontsize=9)
                ax.set_ylabel(ylabel, fontsize=8)
                ax.set_xlabel("update", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.25)
                self._lines[key], = ax.plot([], [], color=color, lw=1.5)
            self._axes = axes

            # Game render panel — right column, spans both rows
            self._ax_game = self.fig.add_subplot(outer[1])
            self._ax_game.set_title("Eval rollout (no noise)", fontsize=10)
            self._ax_game.axis("off")
            # Placeholder grey image so the panel fills immediately
            placeholder = np.full((300, 400, 3), 40, dtype=np.uint8)
            self._im = self._ax_game.imshow(placeholder, aspect="auto")
            self._ax_game.text(
                0.5, 0.5, "waiting for first eval…",
                transform=self._ax_game.transAxes,
                ha="center", va="center",
                fontsize=9, color="white", alpha=0.7,
            )

            plt.ion()
            plt.show(block=False)

        except Exception as exc:
            print(f"[Dashboard] disabled ({exc})")
            self.fig      = None
            self._ax_game = None
            self.enabled  = False

    def update(self, update_idx, mean_ret, actor_loss, critic_loss, entropy, sps):
        """Update metric plots. Call every log_interval updates."""
        if not self.enabled or self.fig is None:
            return
        d = self._data
        d["xs"].append(update_idx)
        d["ret"].append(mean_ret)
        d["aloss"].append(actor_loss)
        d["closs"].append(critic_loss)
        d["ent"].append(entropy)
        d["sps"].append(sps)
        try:
            for key in ["ret", "aloss", "closs", "ent", "sps"]:
                self._lines[key].set_xdata(d["xs"])
                self._lines[key].set_ydata(d[key])
                self._axes[key].relim()
                self._axes[key].autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def update_render(self, frame: np.ndarray, step: int, ep_return: float):
        """
        Push a new game frame (H×W×3 uint8 RGB) into the render panel.
        Call from EvalRenderer after each step of the eval rollout.

        Parameters
        ----------
        frame     : np.ndarray (H, W, 3) uint8  — RGB game frame
        step      : int                          — current eval step number
        ep_return : float                        — accumulated return so far
        """
        if not self.enabled or self._ax_game is None or self._im is None:
            return
        try:
            self._im.set_data(frame)
            self._im.set_extent([0, frame.shape[1], frame.shape[0], 0])
            self._ax_game.set_title(
                f"Eval — step {step}   ret {ep_return:.1f}",
                fontsize=9,
            )
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def save(self, path: str):
        if self.fig is not None:
            try:
                self.fig.savefig(path, dpi=150, bbox_inches="tight")
                print(f"  Dashboard saved: {path}")
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Eval renderer
# ─────────────────────────────────────────────────────────────────────────────

class EvalRenderer:
    """
    Runs a single full episode with the current policy (no noise, no FoV mask)
    and pushes each frame to the Dashboard's game render panel.

    Uses Overcooked-AI's StateVisualizer to render proper game graphics via
    pygame → numpy array → matplotlib imshow (no pygame window needed).

    Parameters
    ----------
    layout     : Overcooked layout name
    actors     : dict {agent_id: Actor}  — current policy (eval mode)
    dashboard  : Dashboard instance
    device     : torch.device
    tile_size  : pixel size of each grid tile (default 80)
    fps_delay  : seconds between frame pushes (default 0.05 → ~20fps visual)
    """

    def __init__(
        self,
        layout:    str,
        actors:    Dict[str, "Actor"],
        dashboard: Dashboard,
        device:    torch.device,
        tile_size: int   = 80,
        fps_delay: float = 0.05,
    ):
        self.actors    = actors
        self.dashboard = dashboard
        self.device    = device
        self.fps_delay = fps_delay
        self._ready    = False

        # Build a clean eval env — no noise wrapper, direct OvercookedEnv
        try:
            from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
            from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
            from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
            import pygame

            mdp      = OvercookedGridworld.from_layout_name(layout)
            self.env = OvercookedEnv.from_mdp(mdp, horizon=400)
            self.mdp = mdp

            # StateVisualizer renders to a pygame Surface without a window
            self.viz = StateVisualizer(
                tile_size=tile_size,
                is_rendering_hud=True,
                is_rendering_cooking_timer=True,
                is_rendering_action_probs=False,
            )
            self.grid = mdp.terrain_mtx

            # Init pygame display in offscreen mode so surfarray works
            # without opening a visible window
            import os
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            pygame.display.init()
            pygame.display.set_mode((1, 1))   # minimal surface required

            self._pygame  = pygame
            self._ready   = True

        except Exception as exc:
            print(f"[EvalRenderer] disabled — {exc}")

    def _state_to_frame(self, state) -> np.ndarray:
        """
        Render an OvercookedState → (H, W, 3) uint8 RGB numpy array.

        StateVisualizer.render_state() returns a pygame Surface.
        pygame.surfarray.array3d() gives (W, H, 3) in column-major order,
        so we transpose axes 0 and 1 to get standard (H, W, 3) row-major.
        """
        surface = self.viz.render_state(
            state  = state,
            grid   = self.grid,
            hud_data = {
                "time_left": self.env.horizon - state.timestep,
                "score":     int(self.env.game_stats.get(
                    "cumulative_sparse_rewards_by_agent",
                    [0, 0]
                )[0]) if hasattr(self.env, "game_stats") else 0,
            },
        )
        arr = self._pygame.surfarray.array3d(surface)  # (W, H, 3) uint8
        return arr.transpose(1, 0, 2)                  # → (H, W, 3)

    def _obs_from_state(self, agent_idx: int) -> Dict[str, np.ndarray]:
        """
        Build a clean (no-noise, full-FoV) obs dict for one agent from the
        current env state. Used so the eval rollout uses the same obs format
        the actor was trained on, just without noise applied.

        Since EvalRenderer bypasses the PartialObsWrapper, we reconstruct
        the minimal obs the actor needs:
            grid    : (NUM_CHANNELS, D, D)  — full global grid cropped to
                      a large square around the agent (simulates full visibility)
            scalars : (7,)                  — [delta_row, delta_col, held×5]
        """
        from overcooked_ai_py.mdp.actions import Direction

        state   = self.env.state
        player  = state.players[agent_idx]
        x, y    = player.position             # (col, row) in Overcooked coords
        pos     = (y, x)                      # → (row, col) for grid indexing

        # Build full grid channels
        terrain = self.mdp.terrain_mtx
        H, W    = len(terrain), len(terrain[0])
        C       = NUM_CHANNELS
        full_grid = np.zeros((C, H, W), dtype=np.float32)

        # Terrain
        for r, row in enumerate(terrain):
            for c, cell in enumerate(row):
                if cell != ' ':
                    full_grid[0, r, c] = 1.0  # wall
                else:
                    full_grid[1, r, c] = 1.0  # floor

        # Agents
        for i, p in enumerate(state.players):
            px, py = p.position
            ch = 2 if i == agent_idx else 3   # self / teammate
            full_grid[ch, py, px] = 1.0

        # Objects
        _CH = {"onion": 4, "tomato": 5, "dish": 6, "soup": 7}
        for (ox, oy), obj in state.objects.items():
            ch = _CH.get(obj.name)
            if ch is not None:
                full_grid[ch, oy, ox] = 1.0

        # Static locations
        for px, py in self.mdp.get_pot_locations():
            full_grid[8, py, px] = 1.0
        for px, py in self.mdp.get_serving_locations():
            full_grid[9, py, px] = 1.0

        # Centre a large crop on the agent — use H and W as radius so the
        # full grid fits; actor trained with fov_radius=5 gets an 11×11 crop.
        # For eval we use the same 11×11 crop (fov_radius=5, D=11).
        R   = 5
        D   = 2 * R + 1
        pad = R
        padded = np.zeros((C, H + 2*pad, W + 2*pad), dtype=np.float32)
        padded[0, :, :] = 1.0                   # default: wall
        padded[:, pad:pad+H, pad:pad+W] = full_grid
        pr, pc = pos[0] + pad, pos[1] + pad
        crop   = padded[:, pr-R:pr+R+1, pc-R:pc+R+1].copy()

        # Prev pos for velocity — use zeros on first step (no prev available)
        scalars = np.zeros(7, dtype=np.float32)
        # held object
        obj = player.held_object
        if obj is not None:
            held_map = {"onion": 1, "tomato": 2, "dish": 3, "soup": 4}
            scalars[2 + held_map.get(obj.name, 0)] = 1.0
        else:
            scalars[2] = 1.0   # HELD_NONE

        return {"grid": crop, "scalars": scalars}

    def run_episode(self):
        """
        Run one full eval episode. For each step:
          1. Render state → push frame to dashboard
          2. Query actors for actions (greedy argmax, no sampling)
          3. Step the env

        Blocks until the episode is done (horizon=400 steps max).
        All actors are set to eval() before and restored to train() after.
        """
        if not self._ready:
            return

        # Switch actors to eval mode
        for a in self.actors.values():
            a.eval()

        self.env.reset()
        hiddens = {
            a: self.actors[a].init_hidden(1, self.device)
            for a in self.actors
        }
        ep_return = 0.0
        step      = 0

        from overcooked_ai_py.mdp.actions import Action

        try:
            while not self.env.is_done():
                # Render current state → push to dashboard
                frame = self._state_to_frame(self.env.state)
                self.dashboard.update_render(frame, step, ep_return)

                # Get actions from each actor (greedy — argmax, not sample)
                joint = []
                for i, agent_id in enumerate(["player_0", "player_1"]):
                    obs    = self._obs_from_state(i)
                    g      = torch.from_numpy(obs["grid"]).unsqueeze(0).to(self.device)
                    s      = torch.from_numpy(obs["scalars"]).unsqueeze(0).to(self.device)
                    h      = hiddens[agent_id]

                    with torch.no_grad():
                        logits, new_h = self.actors[agent_id](g, s, h)
                    hiddens[agent_id] = new_h

                    action_idx = logits.argmax(dim=-1).item()
                    joint.append(Action.INDEX_TO_ACTION[action_idx])

                _, reward, done, _ = self.env.step(tuple(joint))
                ep_return += float(reward)
                step      += 1

                if self.fps_delay > 0:
                    time.sleep(self.fps_delay)

        except Exception as exc:
            print(f"[EvalRenderer] episode error: {exc}")

        finally:
            # Always restore train mode
            for a in self.actors.values():
                a.train()

        # Push final frame
        try:
            frame = self._state_to_frame(self.env.state)
            self.dashboard.update_render(frame, step, ep_return)
        except Exception:
            pass

        print(f"  [Eval] episode return={ep_return:.1f}  steps={step}")


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(layout: str, fov_radius: int, noise_cfg: NoiseCfg) -> PartialObsWrapper:
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    mdp  = OvercookedGridworld.from_layout_name(layout)
    base = OvercookedEnv.from_mdp(mdp, horizon=400)
    return PartialObsWrapper(base, fov_radius=fov_radius, noise_cfg=noise_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# MAPPO Trainer
# ─────────────────────────────────────────────────────────────────────────────

class MAPPOTrainer:

    def __init__(self, cfg: Cfg):
        self.cfg    = cfg
        self.device = torch.device(cfg.device)
        self.agents = ["player_0", "player_1"]

        noise_cfg = NoiseCfg(domain_randomise=True)
        self.envs = [make_env(cfg.layout, cfg.fov_radius, noise_cfg)
                     for _ in range(cfg.num_envs)]

        # Probe for global state dims
        _obs = self.envs[0].reset()
        global_grid, global_scalars = self.envs[0].get_global_state()
        C, H, W = global_grid.shape
        D = 2 * cfg.fov_radius + 1

        # Networks
        self.actors = {
            a: Actor(NUM_CHANNELS, D, cfg.hidden_dim,
                     cfg.action_dim, cfg.width_mult).to(self.device)
            for a in self.agents
        }
        self.critic = CentralCritic(C, H, W, len(self.agents), 256).to(self.device)

        # Separate optimisers with different learning rates
        actor_params = [p for a in self.actors.values() for p in a.parameters()]
        self.actor_opt  = torch.optim.Adam(actor_params,              lr=cfg.actor_lr,  eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),  lr=cfg.critic_lr, eps=1e-5)

        # Linear lr decay over training
        total_updates = cfg.total_steps // (cfg.rollout_steps * cfg.num_envs)
        self.actor_sched  = torch.optim.lr_scheduler.LinearLR(
            self.actor_opt,  start_factor=1.0, end_factor=0.05,
            total_iters=total_updates)
        self.critic_sched = torch.optim.lr_scheduler.LinearLR(
            self.critic_opt, start_factor=1.0, end_factor=0.05,
            total_iters=total_updates)

        self.buffer       = RolloutBuffer()
        self.ep_returns   = collections.deque(maxlen=100)
        self.global_step  = 0
        self.update_count = 0
        self.dashboard    = Dashboard(enabled=cfg.dashboard)
        self.eval_renderer = EvalRenderer(
            layout    = cfg.layout,
            actors    = self.actors,
            dashboard = self.dashboard,
            device    = self.device,
        )

        os.makedirs(cfg.save_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────── training loop ──

    def train(self):
        cfg    = self.cfg
        device = self.device

        obs_list = [env.reset() for env in self.envs]
        hidden   = {a: self.actors[a].init_hidden(cfg.num_envs, device)
                    for a in self.agents}
        ep_rews  = np.zeros(cfg.num_envs)
        t_start  = time.time()

        while self.global_step < cfg.total_steps:
            self.buffer.clear()

            # ── Rollout collection ────────────────────────────────────────
            with torch.no_grad():
                for _ in range(cfg.rollout_steps):
                    step_grids, step_scalars = {}, {}
                    step_hid, step_actions, step_lp = {}, {}, {}

                    for agent in self.agents:
                        g = torch.from_numpy(
                            np.stack([o[agent]["grid"]    for o in obs_list])
                        ).to(device)
                        s = torch.from_numpy(
                            np.stack([o[agent]["scalars"] for o in obs_list])
                        ).to(device)
                        h = hidden[agent]

                        act, lp, _, new_h = \
                            self.actors[agent].get_action_logprob_entropy(g, s, h)

                        step_grids[agent]   = g.cpu()
                        step_scalars[agent] = s.cpu()
                        step_hid[agent]     = h.cpu()   # store INPUT hidden (not new_h)
                        step_actions[agent] = act.cpu()
                        step_lp[agent]      = lp.cpu()
                        hidden[agent]       = new_h      # carry OUTPUT forward

                    # Global state for critic (no noise, no masking)
                    gg_list, gs_list = [], []
                    for env in self.envs:
                        gg, gs = env.get_global_state()
                        gg_list.append(gg); gs_list.append(gs)
                    global_g = torch.from_numpy(np.stack(gg_list)).float()
                    global_s = torch.from_numpy(np.stack(gs_list)).float()

                    # Step all envs
                    new_obs_list = []
                    step_rews   = np.zeros(cfg.num_envs)
                    step_shaped = np.zeros(cfg.num_envs)
                    step_dones  = np.zeros(cfg.num_envs)

                    for i, env in enumerate(self.envs):
                        act_d = {a: step_actions[a][i].item() for a in self.agents}
                        obs_n, reward, done, info = env.step(act_d)

                        # Sparse delivery reward (scale up since it's rare)
                        sparse = float(reward) * cfg.sparse_factor

                        # Dense shaped reward from Overcooked's info dict
                        # Key names differ slightly by version — check all known keys
                        shaped = 0.0
                        if isinstance(info, dict):
                            if "shaped_r_by_agent" in info:
                                shaped = sum(info["shaped_r_by_agent"])
                            elif "shaped_r" in info:
                                shaped = float(info["shaped_r"])
                        shaped *= cfg.reward_shaping_factor

                        step_rews[i]   = sparse
                        step_shaped[i] = shaped
                        step_dones[i]  = float(done)
                        ep_rews[i]    += sparse + shaped

                        if done:
                            self.ep_returns.append(float(ep_rews[i]))
                            ep_rews[i] = 0.0
                            obs_n = env.reset()
                            # Zero out hidden for this env index only
                            for agent in self.agents:
                                hidden[agent][i].zero_()

                        new_obs_list.append(obs_n)

                    obs_list = new_obs_list

                    self.buffer.grids.append(step_grids)
                    self.buffer.scalars.append(step_scalars)
                    self.buffer.hiddens.append(step_hid)
                    self.buffer.actions.append(step_actions)
                    self.buffer.log_probs.append(step_lp)
                    self.buffer.rewards.append(
                        torch.from_numpy(step_rews).float())
                    self.buffer.shaped_rewards.append(
                        torch.from_numpy(step_shaped).float())
                    self.buffer.dones.append(
                        torch.from_numpy(step_dones).float())
                    self.buffer.global_grids.append(global_g)
                    self.buffer.global_scalars.append(global_s)
                    self.global_step += cfg.num_envs

            # ── PPO update ────────────────────────────────────────────────
            metrics = self._ppo_update()
            self.actor_sched.step()
            self.critic_sched.step()
            self.update_count += 1

            # ── Logging ───────────────────────────────────────────────────
            if self.update_count % cfg.log_interval == 0:
                mean_ret = (np.mean(self.ep_returns)
                            if self.ep_returns else float("nan"))
                sps      = self.global_step / (time.time() - t_start)
                actor_lr = self.actor_opt.param_groups[0]["lr"]

                print(
                    f"step={self.global_step:>10,}  "
                    f"upd={self.update_count:>5}  "
                    f"ret={mean_ret:>8.2f}  "
                    f"aloss={metrics['actor_loss']:>+9.5f}  "
                    f"closs={metrics['critic_loss']:>9.6f}  "
                    f"ent={metrics['entropy']:>6.4f}  "
                    f"lr={actor_lr:.2e}  "
                    f"sps={sps:>6.0f}"
                )
                self.dashboard.update(
                    self.update_count, mean_ret,
                    metrics["actor_loss"], metrics["critic_loss"],
                    metrics["entropy"], sps,
                )

            if self.update_count % cfg.save_interval == 0:
                self._save_checkpoint()
                self.dashboard.save(
                    os.path.join(cfg.save_dir,
                                 f"dashboard_{self.update_count:06d}.png"))

            if self.update_count % cfg.eval_interval == 0:
                print(f"  [Eval] running episode at update {self.update_count}…")
                self.eval_renderer.run_episode()

        self._save_checkpoint(final=True)
        self.dashboard.save(os.path.join(cfg.save_dir, "dashboard_final.png"))
        print("Training complete.")

    # ────────────────────────────────────────────────────────── PPO update ──

    def _ppo_update(self) -> Dict[str, float]:
        cfg    = self.cfg
        device = self.device
        T      = len(self.buffer)
        N      = cfg.num_envs

        # Combined reward: sparse delivery + dense shaping
        rews  = torch.stack([
            self.buffer.rewards[t] + self.buffer.shaped_rewards[t]
            for t in range(T)
        ])                                         # (T, N)
        dones = torch.stack(self.buffer.dones)     # (T, N)

        # Value estimates over rollout + bootstrap
        with torch.no_grad():
            vals = []
            for t in range(T):
                v = self.critic(
                    self.buffer.global_grids[t].to(device),
                    self.buffer.global_scalars[t].to(device),
                ).cpu()
                vals.append(v)
            # Bootstrap from last observed global state
            v_boot = self.critic(
                self.buffer.global_grids[-1].to(device),
                self.buffer.global_scalars[-1].to(device),
            ).cpu()
            vals.append(v_boot)
        values_t = torch.stack(vals)               # (T+1, N)

        # GAE
        adv, returns = compute_gae(rews, values_t, dones, cfg.gamma, cfg.gae_lambda)

        # Normalise advantages over the FULL rollout before any minibatch split.
        # This is the key fix: with sparse coop rewards most returns are 0,
        # so per-minibatch std is near-zero → divides by 1e-8 → collapsed signal.
        adv_flat = adv.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        ret_flat = returns.reshape(-1)

        flat = self._flatten_buffer(T, N)

        total_aloss = total_closs = total_ent = 0.0
        n_upd = 0
        indices = np.arange(T * N)

        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, T * N, cfg.minibatch_size):
                mb  = indices[start:start + cfg.minibatch_size]
                mb_adv = adv_flat[mb].to(device)
                mb_ret = ret_flat[mb].to(device)

                # Actor losses — averaged over agents
                aloss  = torch.tensor(0.0, device=device)
                mb_ent = torch.tensor(0.0, device=device)

                for agent in self.agents:
                    g      = flat["grids"][agent][mb].to(device)
                    s      = flat["scalars"][agent][mb].to(device)
                    # Detach: hidden is a fixed initial condition for this step,
                    # not recomputed through time — avoids stale-state log-prob issue
                    h      = flat["hiddens"][agent][mb].to(device).detach()
                    act    = flat["actions"][agent][mb].to(device)
                    lp_old = flat["log_probs"][agent][mb].to(device)

                    _, lp_new, ent, _ = \
                        self.actors[agent].get_action_logprob_entropy(g, s, h, act)

                    ratio = (lp_new - lp_old).exp()
                    surr1 = ratio * mb_adv
                    surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
                    pg    = -torch.min(surr1, surr2).mean()

                    aloss  += pg - cfg.ent_coef * ent.mean()
                    mb_ent += ent.mean()

                aloss  /= len(self.agents)
                mb_ent /= len(self.agents)

                # Critic loss
                v_pred = self.critic(
                    flat["global_grids"][mb].to(device),
                    flat["global_scalars"][mb].to(device),
                )
                closs = cfg.vf_coef * F.mse_loss(v_pred, mb_ret)

                # Optimise actors
                self.actor_opt.zero_grad()
                aloss.backward()
                nn.utils.clip_grad_norm_(
                    [p for a in self.actors.values() for p in a.parameters()],
                    cfg.max_grad_norm)
                self.actor_opt.step()

                # Optimise critic
                self.critic_opt.zero_grad()
                closs.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), cfg.max_grad_norm)
                self.critic_opt.step()

                total_aloss += aloss.item()
                total_closs += closs.item()
                total_ent   += mb_ent.item()
                n_upd += 1

        return {
            "actor_loss":  total_aloss / n_upd,
            "critic_loss": total_closs / n_upd,
            "entropy":     total_ent   / n_upd,
        }

    # ──────────────────────────────────────────────────────────── helpers ──

    def _flatten_buffer(self, T, N):
        flat = {
            "grids":          {a: [] for a in self.agents},
            "scalars":        {a: [] for a in self.agents},
            "hiddens":        {a: [] for a in self.agents},
            "actions":        {a: [] for a in self.agents},
            "log_probs":      {a: [] for a in self.agents},
            "global_grids":   [],
            "global_scalars": [],
        }
        for t in range(T):
            for a in self.agents:
                flat["grids"][a].append(self.buffer.grids[t][a])
                flat["scalars"][a].append(self.buffer.scalars[t][a])
                flat["hiddens"][a].append(self.buffer.hiddens[t][a])
                flat["actions"][a].append(self.buffer.actions[t][a])
                flat["log_probs"][a].append(self.buffer.log_probs[t][a])
            flat["global_grids"].append(self.buffer.global_grids[t])
            flat["global_scalars"].append(self.buffer.global_scalars[t])

        for a in self.agents:
            flat["grids"][a]     = torch.cat(flat["grids"][a])
            flat["scalars"][a]   = torch.cat(flat["scalars"][a])
            flat["hiddens"][a]   = torch.cat(flat["hiddens"][a])
            flat["actions"][a]   = torch.cat(flat["actions"][a])
            flat["log_probs"][a] = torch.cat(flat["log_probs"][a])
        flat["global_grids"]   = torch.cat(flat["global_grids"])
        flat["global_scalars"] = torch.cat(flat["global_scalars"])
        return flat

    def _save_checkpoint(self, final=False):
        tag  = "final" if final else f"step{self.global_step:010d}"
        path = os.path.join(self.cfg.save_dir, f"mappo_{tag}.pt")
        torch.save({
            "global_step":  self.global_step,
            "update_count": self.update_count,
            "cfg":          self.cfg,
            "actors":       {a: m.state_dict() for a, m in self.actors.items()},
            "critic":       self.critic.state_dict(),
            "actor_opt":    self.actor_opt.state_dict(),
            "critic_opt":   self.critic_opt.state_dict(),
        }, path)
        print(f"  Checkpoint saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Export actor for QAT / TFLM pipeline
# ─────────────────────────────────────────────────────────────────────────────

def export_actor(checkpoint_path, agent="player_0", output_path="actor_export.pt"):
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    cfg   = ckpt["cfg"]
    D     = 2 * cfg.fov_radius + 1
    actor = Actor(NUM_CHANNELS, D, cfg.hidden_dim, cfg.action_dim, cfg.width_mult)
    actor.load_state_dict(ckpt["actors"][agent])
    actor.eval()
    torch.save({"model_state_dict": actor.state_dict(), "cfg": cfg}, output_path)
    print(f"Actor exported to {output_path}")
    return actor


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout",          default="cramped_room")
    parser.add_argument("--fov_radius",      type=int,   default=5)
    parser.add_argument("--num_envs",        type=int,   default=8)
    parser.add_argument("--total_steps",     type=int,   default=10_000_000)
    parser.add_argument("--actor_lr",        type=float, default=3e-4)
    parser.add_argument("--critic_lr",       type=float, default=1e-4)
    parser.add_argument("--rollout_steps",   type=int,   default=128)
    parser.add_argument("--hidden_dim",      type=int,   default=128)
    parser.add_argument("--ent_coef",        type=float, default=0.02)
    parser.add_argument("--reward_shaping",  type=float, default=1.0)
    parser.add_argument("--save_dir",        default="checkpoints")
    parser.add_argument("--eval_interval",   type=int,   default=50)
    parser.add_argument("--no_dashboard",    action="store_true")
    parser.add_argument("--export",          default=None,
                        help="Path to checkpoint to export instead of training")
    args = parser.parse_args()

    if args.export:
        export_actor(args.export)
    else:
        cfg = Cfg(
            layout               = args.layout,
            fov_radius           = args.fov_radius,
            num_envs             = args.num_envs,
            total_steps          = args.total_steps,
            actor_lr             = args.actor_lr,
            critic_lr            = args.critic_lr,
            rollout_steps        = args.rollout_steps,
            hidden_dim           = args.hidden_dim,
            ent_coef             = args.ent_coef,
            reward_shaping_factor= args.reward_shaping,
            eval_interval        = args.eval_interval,
            save_dir             = args.save_dir,
            dashboard            = not args.no_dashboard,
        )
        MAPPOTrainer(cfg).train()