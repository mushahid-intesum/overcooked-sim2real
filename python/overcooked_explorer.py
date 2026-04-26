"""
overcooked_explorer.py
======================
Mirrors the physical robot's explore-and-move loop inside the Overcooked
simulation using the PartialObsWrapper.

The agent receives raw numpy arrays (the symbolic grid tensors) — NOT rendered
images. These arrays are the training targets for a perception model that maps
real camera images -> symbolic grid tensors.

Saved data structure
────────────────────
  <save_dir>/
    <layout>/                    e.g. cramped_room/
      step_000/
        N/
          shot_00.npz            contains: grid (C,D,D), scalars (7,)
          shot_01.npz
          ...
        E/  S/  W/  ...
      step_001/
        ...
      metadata.json              layout, fov_radius, noise_cfg, agent_id

Each .npz file:
  np.load("shot_00.npz")
    -> arr["grid"]     shape (C, D, D)  float32   binary channel tensor
    -> arr["scalars"]  shape (7,)       float32   velocity + held-object
    -> arr["dir"]      scalar str       facing direction N/E/S/W
    -> arr["shot"]     scalar int       shot index within direction
    -> arr["step"]     scalar int       move-step index

FOV note
────────
A 165-degree fisheye lens covers almost the full hemisphere in front of the
camera. To match this in the grid world we use a large fov_radius (default 10)
so the crop window is 21x21 cells — large enough to contain any standard
Overcooked layout from any position. The cone mask is also widened to
~165 degrees (nearly full circle) via a wide_cone_mask override.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from _dataclasses import NoiseCfg
from partial_obs_wrapper import make_env
from utils import NUM_CHANNELS, CHANNELS
from overcooked_ai_py.mdp.actions import Action


# ── Direction helpers ─────────────────────────────────────────────────────────

DIRECTION_NAMES = ["N", "E", "S", "W"]

DIR_TO_ORIENTATION = {
    "N": (0, -1),
    "E": (1,  0),
    "S": (0,  1),
    "W": (-1, 0),
}

_MOVE_ACTION = {
    "N": Action.ACTION_TO_INDEX[(0, -1)],
    "E": Action.ACTION_TO_INDEX[(1,  0)],
    "S": Action.ACTION_TO_INDEX[(0,  1)],
    "W": Action.ACTION_TO_INDEX[(-1, 0)],
}


# ── Wide fisheye cone mask ────────────────────────────────────────────────────

def build_fisheye_mask(fov_radius: int, fov_degrees: float = 165.0) -> np.ndarray:
    """
    Build a boolean cone mask for a wide-angle (fisheye) lens.

    Unlike the default forward-only cone, this covers `fov_degrees` centred
    on the forward direction. 165 degrees leaves only a small blind spot
    directly behind the agent.

    Returns
    -------
    mask : np.ndarray  shape (D, D)  bool
        True = visible, False = masked out.
        D = 2 * fov_radius + 1
    """
    D = 2 * fov_radius + 1
    centre = fov_radius                 # agent sits at (centre, centre)
    half_angle = np.radians(fov_degrees / 2.0)

    # Forward direction in ego-frame is "up" = negative row = angle -pi/2
    forward_angle = -np.pi / 2.0

    mask = np.zeros((D, D), dtype=bool)
    for r in range(D):
        for c in range(D):
            if r == centre and c == centre:
                mask[r, c] = True       # agent's own cell always visible
                continue
            dr = r - centre             # row offset  (+down)
            dc = c - centre             # col offset  (+right)
            angle = np.arctan2(dr, dc)  # angle from centre
            diff  = abs(np.arctan2(
                np.sin(angle - forward_angle),
                np.cos(angle - forward_angle)
            ))
            if diff <= half_angle:
                mask[r, c] = True

    return mask


# ─────────────────────────────────────────────────────────────────────────────
class OvercookedExplorer:
    """
    Runs the robot-style scan-and-move loop and saves raw numpy arrays.

    Parameters
    ----------
    layout         : Overcooked layout name, e.g. "cramped_room"
    fov_radius     : half-size of the crop window (default 10 for fisheye)
    fov_degrees    : camera horizontal FOV in degrees (default 165 fisheye)
    noise_cfg      : NoiseCfg instance
    agent_id       : which player the explorer controls
    images_per_dir : obs frames captured per facing direction
    max_steps      : safety limit on total move steps
    render         : print ASCII floor-map to terminal each step
    save_dir       : root folder; arrays saved under save_dir/<layout>/
                     set None to disable saving
    """

    SCAN_DIRS = ["N", "E", "S", "W"]

    def __init__(
        self,
        layout: str                = "cramped_room",
        fov_radius: int            = 10,
        fov_degrees: float         = 165.0,
        noise_cfg: NoiseCfg | None = None,
        agent_id: str              = "player_0",
        images_per_dir: int        = 5,
        max_steps: int             = 50,
        render: bool               = False,
        save_dir: str | None       = "dataset",
    ):
        self.layout         = layout
        self.fov_radius     = fov_radius
        self.fov_degrees    = fov_degrees
        self.noise_cfg      = noise_cfg or NoiseCfg()
        self.agent_id       = agent_id
        self.images_per_dir = images_per_dir
        self.max_steps      = max_steps
        self.render         = render

        # Override the env's cone mask with the fisheye version
        self.env = make_env(layout, fov_radius, noise_cfg)
        self.env.cone_mask = build_fisheye_mask(fov_radius, fov_degrees)
        print(f"[Explorer] Fisheye mask {fov_degrees}°: "
              f"{self.env.cone_mask.sum()} / {(2*fov_radius+1)**2} cells visible")

        # Save path: save_dir / layout_name /
        if save_dir:
            self.layout_dir = Path(save_dir) / layout
            self.layout_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Explorer] Arrays -> {self.layout_dir.resolve()}")
        else:
            self.layout_dir = None

        self.step_count  = 0
        self.obs_history: list[dict] = []

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self):
        obs_dict     = self.env.reset()
        done         = False
        total_reward = 0.0

        # Save metadata once per run
        if self.layout_dir:
            self._save_metadata()

        print(f"[Explorer] Starting layout={self.layout}  "
              f"fov_radius={self.fov_radius}  fov={self.fov_degrees}°  "
              f"max_steps={self.max_steps}")

        while not done and self.step_count < self.max_steps:
            print(f"\n[Explorer] ── Step {self.step_count} ──")

            scan_bundle = self._scan_all_directions(obs_dict)
            self.obs_history.append(scan_bundle)

            if self.layout_dir:
                self._save_step_arrays(scan_bundle, self.step_count)

            if self.render:
                self._render_scan(scan_bundle, self.step_count)

            move_dir = self.decide_move(scan_bundle, self.step_count)
            print(f"[Explorer] Chosen move: {move_dir}")

            obs_dict, reward, done, info = self._move(move_dir)
            total_reward += reward
            self.step_count += 1

            print(f"[Explorer] Reward: {reward:.2f}  |  Total: {total_reward:.2f}")

        print(f"\n[Explorer] Done. Steps={self.step_count}  "
              f"Total reward={total_reward:.2f}")
        self.env.reset()
        return self.obs_history

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _save_metadata(self):
        """Write a metadata.json next to the step folders."""
        import dataclasses
        meta = {
            "layout":         self.layout,
            "fov_radius":     self.fov_radius,
            "fov_degrees":    self.fov_degrees,
            "grid_diameter":  2 * self.fov_radius + 1,
            "num_channels":   NUM_CHANNELS,
            "channels":       CHANNELS,
            "agent_id":       self.agent_id,
            "images_per_dir": self.images_per_dir,
            "directions":     self.SCAN_DIRS,
            "noise_cfg":      dataclasses.asdict(self.noise_cfg),
            "array_format": {
                "grid":    f"float32  shape ({NUM_CHANNELS}, D, D)  binary 0/1",
                "scalars": "float32  shape (7,)  [vel_r, vel_c, held_none, held_onion, held_tomato, held_dish, held_soup]",
                "dir":     "str  facing direction when shot was taken",
                "shot":    "int  shot index within direction (0..images_per_dir-1)",
                "step":    "int  move-step index",
            },
        }
        path = self.layout_dir / "metadata.json"
        path.write_text(json.dumps(meta, indent=2))
        print(f"[Explorer] Metadata -> {path}")

    # ── Array saving ──────────────────────────────────────────────────────────

    def _save_step_arrays(
        self,
        bundle: dict[str, list[dict]],
        step: int,
    ) -> None:
        """
        Save all frames from one scan step as .npz files.

        Layout:
            dataset/
              cramped_room/
                metadata.json
                step_000/
                  N/
                    shot_00.npz   <- np.load() gives grid, scalars, dir, shot, step
                    shot_01.npz
                    ...
                  E/  S/  W/
                step_001/
                  ...
        """
        step_dir = self.layout_dir / f"step_{step:03d}"

        for direction, frames in bundle.items():
            dir_folder = step_dir / direction
            dir_folder.mkdir(parents=True, exist_ok=True)

            for frame in frames:
                npz_path = dir_folder / f"shot_{frame['shot']:02d}.npz"
                np.savez_compressed(
                    npz_path,
                    grid    = frame["grid"],       # (C, D, D) float32
                    scalars = frame["scalars"],    # (7,)      float32
                    dir     = np.array(frame["dir"]),
                    shot    = np.array(frame["shot"]),
                    step    = np.array(step),
                )

        n_files = sum(len(v) for v in bundle.values())
        print(f"[Save] step_{step:03d}: {n_files} arrays -> {step_dir}")

    # ── Scanning ──────────────────────────────────────────────────────────────

    def _scan_all_directions(self, current_obs: dict) -> dict[str, list[dict]]:
        bundle: dict[str, list[dict]] = {d: [] for d in self.SCAN_DIRS}

        player_idx   = int(self.agent_id.split("_")[1])
        raw_env      = self.env.env
        original_ori = raw_env.state.players[player_idx].orientation

        for direction in self.SCAN_DIRS:
            raw_env.state.players[player_idx].orientation = DIR_TO_ORIENTATION[direction]

            for shot in range(self.images_per_dir):
                state   = self.env._get_state()
                partial = self.env._build_all_partial_obs(state)
                obs     = partial[self.agent_id]
                bundle[direction].append({
                    "grid":    obs["grid"].copy(),      # (C, D, D) float32
                    "scalars": obs["scalars"].copy(),   # (7,)      float32
                    "shot":    shot,
                    "dir":     direction,
                })

            print(f"[Scan]   {direction}: {self.images_per_dir} frames")

        raw_env.state.players[player_idx].orientation = original_ori
        return bundle

    # ── Moving ────────────────────────────────────────────────────────────────

    def _move(self, direction: str):
        num_players = self.env.env.mdp.num_players
        action_dict = {
            f"player_{i}": (
                _MOVE_ACTION[direction]
                if f"player_{i}" == self.agent_id
                else Action.ACTION_TO_INDEX[Action.STAY]
            )
            for i in range(num_players)
        }
        return self.env.step(action_dict)

    # ── Decision function ─────────────────────────────────────────────────────

    def decide_move(self, scan_bundle: dict[str, list[dict]], step: int) -> str:
        """
        Default: move toward direction with most visible floor cells.
        Override in a subclass to plug in RL / planning logic.
        """
        floor_ch = CHANNELS["floor"]
        scores = {
            d: float(sum(f["grid"][floor_ch].sum() for f in frames) / max(len(frames), 1))
            for d, frames in scan_bundle.items()
        }
        print("[Decide] Floor scores: "
              + "  ".join(f"{d}={v:.1f}" for d, v in scores.items()))
        return max(scores, key=scores.get)

    # ── ASCII render ──────────────────────────────────────────────────────────

    def _render_scan(self, bundle: dict[str, list[dict]], step: int):
        floor_ch = CHANNELS["floor"]
        print(f"\n  [Render] Step {step}:")
        for direction, frames in bundle.items():
            avg = np.mean([f["grid"][floor_ch] for f in frames], axis=0)
            print(f"    {direction}:")
            for row in avg:
                print("      " + "".join("█" if v > 0.5 else "░" for v in row))


# ─────────────────────────────────────────────────────────────────────────────
#  How to load the saved arrays
# ─────────────────────────────────────────────────────────────────────────────
#
#  import numpy as np
#  data = np.load("dataset/cramped_room/step_000/N/shot_00.npz")
#
#  grid    = data["grid"]      # shape (C, D, D)  float32  -- training TARGET
#  scalars = data["scalars"]   # shape (7,)       float32  -- optional target
#  dir     = str(data["dir"])  # "N"
#  shot    = int(data["shot"]) # 0
#  step    = int(data["step"]) # 0
#
#  Training setup:
#    Input  : real camera image (JPEG/PNG from the physical robot)
#    Target : grid + scalars arrays saved here
#    Model  : CNN encoder -> multi-head decoder, one output per channel
# ─────────────────────────────────────────────────────────────────────────────


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    noise = NoiseCfg(
        dropout_enabled  = False,   # keep arrays clean for training targets
        obj_miss_enabled = False,
        delay_enabled    = False,
        domain_randomise = False,
    )

    explorer = OvercookedExplorer(
        layout         = "cramped_room",
        fov_radius     = 10,          # 21x21 crop -- covers full layout
        fov_degrees    = 165.0,       # fisheye FOV
        noise_cfg      = noise,
        agent_id       = "player_0",
        images_per_dir = 5,
        max_steps      = 20,
        render         = True,
        save_dir       = "dataset",   # arrays -> dataset/cramped_room/
    )

    history = explorer.run()

    total_frames = sum(
        len(b[d]) for b in history for d in ["N", "E", "S", "W"]
    )
    print(f"\n[Main] {len(history)} steps, {total_frames} arrays saved.")
    print(f"[Main] Load with: np.load('dataset/cramped_room/step_000/N/shot_00.npz')")