"""
overcooked_partial_obs_wrapper.py
──────────────────────────────────────────────────────────────────────────────
Partial-observability wrapper for Overcooked-AI (PettingZoo parallel API).

Converts the fully-observable Overcooked state into a per-agent
multi-channel grid tensor that mimics what an on-board camera + IMU
would produce on a real tank-car robot.

Observation tensor shape:  (C, FOV_H, FOV_W)
    C channels (one-hot object layers):
        0  — walls / static terrain
        1  — empty passable floor
        2  — self agent position
        3  — teammate agent position (visible only if in FoV)
        4  — onion (raw ingredient)
        5  — tomato (raw ingredient)
        6  — dish (empty plate)
        7  — soup (cooked, on counter)
        8  — pot (cooking station)
        9  — serving location
        10 — delivery zone
    + scalar feature vector appended after CNN encoding:
        [delta_x, delta_y,           ← ego-velocity (IMU analog)
         held_object_onehot (5-dim)] ← what agent carries

Forward-cone FoV:
    The agent always faces one of 4 cardinal directions.
    - Front hemisphere (±90° from heading): full radius R cells visible
    - Cells are ego-centric: agent is always at the bottom-centre of the
      crop, facing "up" in the tensor.
    - Behind the agent: zeroed out (blind region).
    - The cone is wider in front: cells within ±45° get full R,
      cells between ±45°–90° get floor(R * 0.6) (side periphery).

Noise pipeline (applied in order, each independently configurable):
    1. Gaussian noise  — on scalar IMU features
    2. Obs delay       — 1-2 step ring buffer lag on full tensor
    3. Cell dropout    — random grid cells zeroed (occlusion)
    4. Object miss     — non-self object channels stochastically zeroed

Usage:
    from overcooked_partial_obs_wrapper import PartialObsWrapper, NoiseCfg

    base_env = OvercookedEnv(...)          # your existing PettingZoo env
    noise_cfg = NoiseCfg()                 # all defaults
    env = PartialObsWrapper(base_env, fov_radius=5, noise_cfg=noise_cfg)

    obs, infos = env.reset()
    # obs["player_0"].shape == (C, FOV_H, FOV_W)
    # infos["player_0"]["scalars"].shape == (7,)
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any, Dict, List, Optional, Tuple
from overcooked_ai_py.mdp.actions import Action

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Grid channel map
# ─────────────────────────────────────────────────────────────────────────────

CHANNELS = {
    "wall":     0,
    "floor":    1,
    "self":     2,
    "teammate": 3,
    "onion":    4,
    "tomato":   5,
    "dish":     6,
    "soup":     7,
    "pot":      8,
    "serve":    9,
    "delivery": 10,
}
NUM_CHANNELS = len(CHANNELS)

# Held-object one-hot indices (for scalar feature)
HELD_NONE   = 0
HELD_ONION  = 1
HELD_TOMATO = 2
HELD_DISH   = 3
HELD_SOUP   = 4
NUM_HELD    = 5

# Cardinal direction → (row_delta, col_delta) in grid coords
# NORTH = agent moves toward row 0
DIRECTION_VEC = {
    0: (-1,  0),  # NORTH
    1: ( 0,  1),  # EAST
    2: ( 1,  0),  # SOUTH
    3: ( 0, -1),  # WEST
}

# ─────────────────────────────────────────────────────────────────────────────
# Noise configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class NoiseCfg:
    # ── Gaussian noise on scalar IMU features ──────────────────────────────
    imu_noise_enabled: bool  = True
    imu_noise_std:     float = 0.05   # std in normalised units (~0–1 range)

    # ── Observation delay ───────────────────────────────────────────────────
    delay_enabled:     bool  = True
    delay_min_steps:   int   = 1      # minimum lag steps
    delay_max_steps:   int   = 2      # maximum lag steps (sampled each episode)

    # ── Cell dropout (occlusion / motion blur) ──────────────────────────────
    dropout_enabled:   bool  = True
    dropout_p:         float = 0.08   # probability any cell is zeroed

    # ── Object detection miss (false negative) ──────────────────────────────
    obj_miss_enabled:  bool  = True
    obj_miss_p:        float = 0.05   # prob any non-self, non-terrain object
                                       # channel is zeroed at its cell

    # ── Domain randomisation ranges (sampled per episode reset) ─────────────
    dr_imu_std_range:      Tuple[float, float] = (0.02, 0.10)
    dr_dropout_p_range:    Tuple[float, float] = (0.03, 0.15)
    dr_obj_miss_p_range:   Tuple[float, float] = (0.02, 0.10)
    domain_randomise:      bool = True   # if False, use fixed values above


# ─────────────────────────────────────────────────────────────────────────────
# Utility: build the forward-cone visibility mask for a given FoV radius
# ─────────────────────────────────────────────────────────────────────────────

def _build_cone_mask(fov_radius: int) -> np.ndarray:
    """
    Pre-compute a boolean visibility mask for the ego-centric crop.

    The crop is always (2*R+1) × (2*R+1) with the agent at the centre.
    In ego-centric frame the agent always "faces up" (toward row 0).

    Cone rules (relative to centre row=R, col=R):
        row < R   (in front):  |col - R| / (R - row) ≤ tan(90°) → always visible
        row == R  (same row):  col within R*0.6 of centre → periphery visible
        row > R   (behind):    invisible
    Side periphery: columns within ±floor(R*0.6) at agent's own row.
    """
    D = 2 * fov_radius + 1
    mask = np.zeros((D, D), dtype=bool)
    side_reach = max(1, int(fov_radius * 0.6))

    for r in range(D):
        for c in range(D):
            dr = r - fov_radius   # negative = in front (north)
            dc = abs(c - fov_radius)

            if dr < 0:
                # Forward hemisphere: full cone, ±90°
                # tan(45°) = 1 → within 45° when dc ≤ |dr|
                # allow up to 90°: dc ≤ fov_radius always true for any dr<0
                # We tighten: full R in ±45°, side_reach beyond that
                if dc <= abs(dr):
                    mask[r, c] = True          # within ±45° cone
                elif dc <= side_reach:
                    mask[r, c] = True          # side periphery
            elif dr == 0:
                if dc <= side_reach:
                    mask[r, c] = True          # lateral peripheral strip
            # dr > 0 → behind agent → stays False

    return mask   # shape (D, D)


def _rotate_crop_to_ego(crop: np.ndarray, direction: int) -> np.ndarray:
    """
    Rotate the world-frame crop so the agent always faces "up" (north) in
    the returned tensor.

        direction 0 (NORTH): no rotation needed
        direction 1 (EAST):  rotate 90° CCW
        direction 2 (SOUTH): rotate 180°
        direction 3 (WEST):  rotate 90° CW
    """
    k = (4 - direction) % 4   # number of 90° CCW rotations
    if k == 0:
        return crop
    return np.rot90(crop, k=k, axes=(1, 2))   # rotate spatial dims only


# ─────────────────────────────────────────────────────────────────────────────
# Core wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PartialObsWrapper:
    """
    Wraps a PettingZoo-style Overcooked parallel environment and converts
    the fully-observable state into per-agent partial observations.

    The wrapper exposes the same parallel API:
        reset() → (obs_dict, info_dict)
        step(action_dict) → (obs_dict, rew_dict, done_dict, trunc_dict, info_dict)

    Each obs_dict value is a dict:
        {
          "grid":    np.ndarray shape (NUM_CHANNELS, D, D)  float32 [0,1]
          "scalars": np.ndarray shape (7,)                  float32
                     [delta_row, delta_col, held_0..4]
        }

    The critic (centralised) should use env.get_global_state() which returns
    the full grid + all agent scalars concatenated.

    Parameters
    ----------
    env          : Overcooked PettingZoo parallel env (already constructed)
    fov_radius   : int, cells visible in each direction (default 5)
    noise_cfg    : NoiseCfg, sensor noise parameters
    """

    def __init__(
        self,
        env,
        fov_radius: int = 5,
        noise_cfg: Optional[NoiseCfg] = None,
    ):
        self.env       = env
        self.R         = fov_radius
        self.D         = 2 * fov_radius + 1
        self.noise_cfg = noise_cfg or NoiseCfg()

        # Pre-compute ego-centric cone mask (same for all agents/directions
        # because we rotate the crop before applying it)
        self._cone_mask = _build_cone_mask(fov_radius)  # (D, D)

        # Per-episode state
        self._prev_pos: Dict[str, Tuple[int, int]] = {}
        self._delay_buffers: Dict[str, collections.deque] = {}
        self._delay_steps: Dict[str, int] = {}
        self._active_noise_cfg: NoiseCfg = self.noise_cfg

        # Cache grid dimensions (set on first reset)
        self._grid_h: int = 0
        self._grid_w: int = 0

    # ─────────────────────────────────────────────── public API ──────────────

    @property
    def agents(self) -> List[str]:
        return [f"player_{i}" for i in range(self.env.mdp.num_players)]

    def reset(self):
        self.env.reset()                         # mutates env.state, returns nothing
        self._active_noise_cfg = self._sample_noise_cfg()
        state = self._get_state()
        self._grid_h, self._grid_w = state["H"], state["W"]
        for agent in self.agents:
            self._prev_pos[agent] = self._agent_pos(state, agent)
            delay = (np.random.randint(
                self._active_noise_cfg.delay_min_steps,
                self._active_noise_cfg.delay_max_steps + 1,
            ) if self._active_noise_cfg.delay_enabled else 0)
            self._delay_steps[agent] = delay
            zero_obs = {
                "grid":    np.zeros((NUM_CHANNELS, self.D, self.D), dtype=np.float32),
                "scalars": np.zeros(7, dtype=np.float32),
            }
            self._delay_buffers[agent] = collections.deque(
                [zero_obs] * (delay + 1), maxlen=delay + 1
            )

        return self._build_all_partial_obs(state)

    def step(self, action_dict: Dict[str, int]):
        joint_action = tuple(
            Action.INDEX_TO_ACTION[action_dict[f"player_{i}"]]
            for i in range(self.env.mdp.num_players)
        )
        _, reward, done, info = self.env.step(joint_action)
        state = self._get_state()
        for agent in self.agents:
            self._prev_pos[agent] = self._agent_pos(state, agent)
        partial_obs = self._build_all_partial_obs(state)
        return partial_obs, reward, done, info

    def get_global_state(self) -> np.ndarray:
        """
        Returns the full observable state for the centralised critic.
        Shape: (NUM_CHANNELS, grid_h, grid_w) flattened + all agent scalars.
        The critic never goes through the noise pipeline.
        """
        state  = self._get_state()
        global_grid = self._build_global_grid(state)   # (C, H, W)
        scalar_parts = []
        for agent in self.agents:
            pos  = self._agent_pos(state, agent)
            prev = self._prev_pos.get(agent, pos)
            held = self._held_onehot(state, agent)
            vel  = np.array([pos[0] - prev[0], pos[1] - prev[1]], dtype=np.float32)
            scalar_parts.append(np.concatenate([vel, held]))
        scalars = np.concatenate(scalar_parts)
        return global_grid.astype(np.float32), scalars

    def close(self):
        self.env.close()

    # ──────────────────────────────────────────── internal helpers ────────────

    def _get_state(self):
        raw = self.env.state      # OvercookedEnv.state — NOT self.env.env.state
        mdp = self.env.mdp

        terrain_mtx = mdp.terrain_mtx
        H = len(terrain_mtx)
        W = len(terrain_mtx[0])
        terrain = np.array(
            [[0 if cell == ' ' else 1 for cell in row] for row in terrain_mtx],
            dtype=np.int32,
        )

        agent_pos, agent_dir = {}, {}
        for i, player in enumerate(raw.players):
            x, y = player.position          # Overcooked: x=col, y=row
            agent_pos[f"player_{i}"] = (y, x)                    # → (row, col)
            # agent_dir[f"player_{i}"] = _DIR_TUPLE_TO_IDX.get(
            #     tuple(player.orientation), 0
            # )

        objects = []
        for (x, y), obj in raw.objects.items():
            objects.append({"type": obj.name, "pos": (y, x)})    # (x,y)→(row,col)

        # pot/serve locs are also (x,y) — swap to (row,col)
        pot_locs   = [(y, x) for x, y in mdp.get_pot_locations()]
        serve_locs = [(y, x) for x, y in mdp.get_serving_locations()]
        # serving locations double as delivery zones in Overcooked

        return {
            "terrain": terrain, "agent_pos": agent_pos, "agent_dir": agent_dir,
            "objects": objects, "pot_locs": pot_locs, "serve_locs": serve_locs,
            "delivery_locs": [],   # not a separate concept in Overcooked-AI
            "H": H, "W": W,
        }

    def _build_global_grid(self, state: Dict) -> np.ndarray:
        """Build the full (C, H, W) symbolic grid (no masking, no noise)."""
        H, W = state["H"], state["W"]
        grid = np.zeros((NUM_CHANNELS, H, W), dtype=np.float32)

        # Terrain
        grid[CHANNELS["wall"]]  = (state["terrain"] == 1).astype(np.float32)
        grid[CHANNELS["floor"]] = (state["terrain"] == 0).astype(np.float32)

        # Agents
        for i, agent in enumerate(self.agents):
            pos = state["agent_pos"].get(agent)
            if pos is not None:
                ch = CHANNELS["self"] if i == 0 else CHANNELS["teammate"]
                grid[ch, pos[0], pos[1]] = 1.0

        # Dynamic objects
        _OBJ_CH = {
            "onion":  CHANNELS["onion"],
            "tomato": CHANNELS["tomato"],
            "dish":   CHANNELS["dish"],
            "soup":   CHANNELS["soup"],
        }
        for obj in state["objects"]:
            ch = _OBJ_CH.get(obj["type"])
            if ch is not None:
                r, c = obj["pos"]
                if 0 <= r < H and 0 <= c < W:
                    grid[ch, r, c] = 1.0

        # Static locations
        for r, c in state["pot_locs"]:
            grid[CHANNELS["pot"], r, c] = 1.0
        for r, c in state["serve_locs"]:
            grid[CHANNELS["serve"], r, c] = 1.0
        for r, c in state["delivery_locs"]:
            grid[CHANNELS["delivery"], r, c] = 1.0

        return grid

    def _build_all_partial_obs(
        self, state: Dict
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Build partial obs for every active agent."""
        global_grid = self._build_global_grid(state)
        result = {}

        for i, agent in enumerate(self.agents):
            pos    = state["agent_pos"].get(agent)
            dir_   = state["agent_dir"].get(agent, 0)
            prev   = self._prev_pos.get(agent, pos)

            if pos is None:
                # Agent done — return zeros
                result[agent] = {
                    "grid":    np.zeros((NUM_CHANNELS, self.D, self.D), dtype=np.float32),
                    "scalars": np.zeros(7, dtype=np.float32),
                }
                continue

            # ── 1. Crop world-frame grid around agent ─────────────────────
            crop = self._crop_global(global_grid, pos, state)

            # ── 2. Rotate crop to ego-centric frame ───────────────────────
            ego_crop = _rotate_crop_to_ego(crop, dir_)

            # ── 3. Apply forward-cone mask ────────────────────────────────
            masked = ego_crop * self._cone_mask[np.newaxis, :, :]

            # ── 4. Noise: cell dropout ────────────────────────────────────
            cfg = self._active_noise_cfg
            if cfg.dropout_enabled:
                masked = self._apply_dropout(masked, cfg.dropout_p)

            # ── 5. Noise: object detection miss ───────────────────────────
            if cfg.obj_miss_enabled:
                masked = self._apply_obj_miss(masked, cfg.obj_miss_p)

            # ── 6. Build scalar features (IMU analog) ─────────────────────
            vel  = np.array(
                [pos[0] - prev[0], pos[1] - prev[1]], dtype=np.float32
            )
            if cfg.imu_noise_enabled:
                vel = vel + np.random.normal(0, cfg.imu_noise_std, size=vel.shape)
            held = self._held_onehot(state, agent)
            scalars = np.concatenate([vel, held]).astype(np.float32)

            raw_obs = {"grid": masked, "scalars": scalars}

            # ── 7. Noise: observation delay ───────────────────────────────
            if cfg.delay_enabled and self._delay_steps[agent] > 0:
                buf = self._delay_buffers[agent]
                buf.append(raw_obs)
                delayed_obs = buf[0]   # oldest entry = delayed output
            else:
                delayed_obs = raw_obs

            result[agent] = delayed_obs

        return result

    def _crop_global(
        self,
        global_grid: np.ndarray,
        agent_pos: Tuple[int, int],
        state: Dict,
    ) -> np.ndarray:
        """
        Extract a (C, D, D) world-frame crop centred on the agent.
        Pads with the wall channel at boundaries.
        """
        C, H, W = global_grid.shape
        R = self.R
        D = self.D
        r0, c0 = agent_pos

        # Pad the global grid with walls on all sides
        pad = R
        padded = np.zeros((C, H + 2 * pad, W + 2 * pad), dtype=np.float32)
        padded[CHANNELS["wall"], :, :] = 1.0    # default: wall
        padded[:, pad:pad + H, pad:pad + W] = global_grid

        # Crop
        pr, pc = r0 + pad, c0 + pad
        crop = padded[:, pr - R:pr + R + 1, pc - R:pc + R + 1]
        assert crop.shape == (C, D, D), f"Crop shape mismatch: {crop.shape}"
        return crop.copy()

    def _apply_dropout(self, grid: np.ndarray, p: float) -> np.ndarray:
        """Zero random cells across all channels simultaneously."""
        C, H, W = grid.shape
        mask = np.random.random((H, W)) > p         # True = keep
        return grid * mask[np.newaxis, :, :]

    def _apply_obj_miss(self, grid: np.ndarray, p: float) -> np.ndarray:
        """
        Stochastically zero object channels at occupied cells.
        Only affects non-self, non-terrain channels.
        """
        obj_channels = [
            CHANNELS["teammate"],
            CHANNELS["onion"],
            CHANNELS["tomato"],
            CHANNELS["dish"],
            CHANNELS["soup"],
        ]
        out = grid.copy()
        for ch in obj_channels:
            layer = out[ch]
            miss_mask = np.random.random(layer.shape) < p
            out[ch] = layer * (~miss_mask | (layer == 0))
        return out

    def _agent_pos(self, state: Dict, agent: str) -> Tuple[int, int]:
        return state["agent_pos"].get(agent, (0, 0))

    def _held_onehot(self, state: Dict, agent: str) -> np.ndarray:
        """
        Returns a 5-dim one-hot of what the agent is currently holding.
        Reads from the underlying Overcooked state.
        """
        vec = np.zeros(NUM_HELD, dtype=np.float32)
        try:
            i = int(agent.split("_")[1])
            player = self.env.state.players[i]
            obj = player.held_object
            if obj is None:
                vec[HELD_NONE] = 1.0
            else:
                name_map = {
                    "onion":  HELD_ONION,
                    "tomato": HELD_TOMATO,
                    "dish":   HELD_DISH,
                    "soup":   HELD_SOUP,
                }
                idx = name_map.get(obj.name, HELD_NONE)
                vec[idx] = 1.0
        except (IndexError, AttributeError):
            vec[HELD_NONE] = 1.0
        return vec

    def _sample_noise_cfg(self) -> NoiseCfg:
        """Sample domain-randomised noise parameters for one episode."""
        cfg = dataclasses.replace(self.noise_cfg)  # shallow copy
        if not cfg.domain_randomise:
            return cfg
        rng = np.random.default_rng()
        cfg.imu_noise_std = float(
            rng.uniform(*cfg.dr_imu_std_range)
        )
        cfg.dropout_p = float(
            rng.uniform(*cfg.dr_dropout_p_range)
        )
        cfg.obj_miss_p = float(
            rng.uniform(*cfg.dr_obj_miss_p_range)
        )
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (run directly: python overcooked_partial_obs_wrapper.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("Running cone mask smoke test...")
    mask = _build_cone_mask(fov_radius=5)
    D = 11
    print(f"Cone mask shape: {mask.shape}")
    print(f"Visible cells: {mask.sum()} / {D*D}")
    print("Mask (1=visible, 0=blind):")
    for row in mask:
        print("  " + "".join("█" if v else "░" for v in row))

    print("\nCone mask at radius=3:")
    m3 = _build_cone_mask(3)
    for row in m3:
        print("  " + "".join("█" if v else "░" for v in row))

    print("\nNoiseCfg defaults:")
    cfg = NoiseCfg()
    for f in dataclasses.fields(cfg):
        print(f"  {f.name}: {getattr(cfg, f.name)}")

    print("\nSmoke test passed. Import and wrap your Overcooked env to use.")
