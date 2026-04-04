"""
quantize_actor.py
──────────────────────────────────────────────────────────────────────────────
Quantize the trained MAPPO Actor network to INT8 .espdl format for ESP32-S3
deployment using esp-ppq.

Key differences from the reference MobileNetV2 script
──────────────────────────────────────────────────────
1. Three inputs instead of one:
       grid       : (1, C, D, D)  float32  — ego-crop symbolic grid tensor
       scalars    : (1, 7)        float32  — [Δrow, Δcol, held_onehot×5]
       hidden_in  : (1, 128)      float32  — GRU hidden state

   Two outputs:
       logits     : (1, 6)        float32  — action logits (argmax → action)
       hidden_out : (1, 128)      float32  — GRU hidden state for next step

2. Calibration data is generated from the Overcooked-AI environment, NOT
   ImageNet. Using ImageNet distributions on a symbolic grid tensor would
   produce completely wrong quantization scales.

3. No ReLU6 → ReLU conversion needed — Actor uses ReLU throughout.

4. Layerwise equalization is used (same as reference) to handle the
   DWS conv weight scale imbalance that causes INT8 error spikes.

5. The GRU cell decomposes to Split/Sigmoid/Tanh/Mul/Add in ONNX — all
   supported by esp-ppq. No special handling needed.

Usage
──────
  # Quantize from a checkpoint
  python quantize_actor.py --checkpoint checkpoints/mappo_final.pt

  # Quantize a specific agent (default: player_0)
  python quantize_actor.py --checkpoint checkpoints/mappo_final.pt --agent player_1

  # Quick test with random calibration data (no Overcooked install needed)
  python quantize_actor.py --checkpoint checkpoints/mappo_final.pt --random_calib

  # Both agents in one run
  python quantize_actor.py --checkpoint checkpoints/mappo_final.pt --both_agents

Output
──────
  models/player_0_actor.espdl   — quantized model for ESP32-S3
  models/player_0_actor.onnx    — intermediate ONNX (kept for inspection)
  models/player_0_actor_quant_error_report.txt  — per-layer quantization error

On-device inference loop (C pseudocode for reference)
──────────────────────────────────────────────────────
  float hidden[128] = {0};          // zero-init once per episode

  while (!done) {
      float grid[11*11*11];         // fill from camera perception pipeline
      float scalars[7];             // fill from IMU + held-object state

      // feed grid, scalars, hidden → get logits, new_hidden
      esp_dl_run(model, inputs={grid, scalars, hidden}, ...);

      int action = argmax(logits, 6);
      hidden = new_hidden;          // carry GRU state forward

      execute_action(action);
  }
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mappo_train import Actor, Cfg
from overcooked_partial_obs_wrapper import NUM_CHANNELS, NoiseCfg, PartialObsWrapper

# ── esp-ppq ──────────────────────────────────────────────────────────────────
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_torch, get_target_platform


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TARGET       = "esp32s3"
NUM_OF_BITS  = 8
DEVICE       = "cpu"
HIDDEN_DIM   = 128
FOV_RADIUS   = 5
D            = 2 * FOV_RADIUS + 1   # 11
C            = NUM_CHANNELS          # 11

# Input shapes — ORDER must match Actor.forward(grid, scalars, hidden)
INPUT_SHAPES = [
    [1, C, D, D],   # grid
    [1, 7],          # scalars
    [1, HIDDEN_DIM], # hidden_in
]


# ─────────────────────────────────────────────────────────────────────────────
# Calibration dataset
# ─────────────────────────────────────────────────────────────────────────────

class OvercookedCalibDataset(torch.utils.data.Dataset):
    """
    Runs the Overcooked-AI environment with random actions and collects
    (grid, scalars, hidden) tuples as calibration samples.

    This is the correct calibration distribution for the Actor — it matches
    the real input statistics the quantized model will see at inference time.

    Parameters
    ----------
    layout        : Overcooked layout name
    num_samples   : total calibration samples to collect
    fov_radius    : must match training config
    noise_enabled : whether to apply sensor noise during collection
                    (True = matches noisy training distribution,
                     False = clean eval distribution — either is defensible)
    """

    def __init__(
        self,
        layout:        str   = "cramped_room",
        num_samples:   int   = 1024,
        fov_radius:    int   = FOV_RADIUS,
        noise_enabled: bool  = True,
    ):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
        from overcooked_ai_py.mdp.actions import Action

        noise_cfg = NoiseCfg(
            imu_noise_enabled = noise_enabled,
            dropout_enabled   = noise_enabled,
            obj_miss_enabled  = noise_enabled,
            delay_enabled     = False,          # delay complicates calib batching
            domain_randomise  = noise_enabled,
        )
        mdp  = OvercookedGridworld.from_layout_name(layout)
        base = OvercookedEnv.from_mdp(mdp, horizon=400)
        env  = PartialObsWrapper(base, fov_radius=fov_radius, noise_cfg=noise_cfg)

        collected = 0
        hidden    = np.zeros((1, HIDDEN_DIM), dtype=np.float32)

        print(f"Collecting {num_samples} calibration samples from Overcooked ({layout})…")
        obs = env.reset()

        while collected < num_samples:
            # Collect both agents' observations each step — doubles data rate
            for agent_id in ["player_0", "player_1"]:
                if collected >= num_samples:
                    break
                grid_np    = obs[agent_id]["grid"]     # (C, D, D)
                scalar_np  = obs[agent_id]["scalars"]  # (7,)

                self.samples.append((
                    torch.from_numpy(grid_np).unsqueeze(0),       # (1, C, D, D)
                    torch.from_numpy(scalar_np).unsqueeze(0),     # (1, 7)
                    torch.from_numpy(hidden).clone(),             # (1, H)
                ))
                collected += 1

            # Random action step
            act_d = {
                a: np.random.randint(0, 6)
                for a in ["player_0", "player_1"]
            }
            obs, _, done, _ = env.step(act_d)

            if done:
                obs    = env.reset()
                hidden = np.zeros((1, HIDDEN_DIM), dtype=np.float32)

        print(f"  Collected {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class RandomCalibDataset(torch.utils.data.Dataset):
    """
    Fallback calibration dataset using random tensors in the expected
    input ranges. Use only for debugging the quantization pipeline — real
    calibration should always use OvercookedCalibDataset.

    Grid values are 0/1 (binary, one-hot channels).
    Scalars are in [-1, 1] for velocity, 0/1 for held-object.
    Hidden state is in [-1, 1] (typical GRU range).
    """

    def __init__(self, num_samples: int = 1024):
        self.num_samples = num_samples
        print(f"[WARNING] Using RANDOM calibration data. "
              f"Quantization scales will be inaccurate. "
              f"Use OvercookedCalibDataset for real deployment.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Grid: binary 0/1 (symbolic, one-hot channels)
        grid    = (torch.rand(1, C, D, D) > 0.85).float()
        # Scalars: velocity ~N(0, 0.5), held one-hot
        scalars = torch.zeros(1, 7)
        scalars[0, 0] = torch.randn(1).item() * 0.5   # delta_row
        scalars[0, 1] = torch.randn(1).item() * 0.5   # delta_col
        held_idx = torch.randint(0, 5, (1,)).item()
        scalars[0, 2 + held_idx] = 1.0                # held one-hot
        # Hidden: tanh range
        hidden  = torch.tanh(torch.randn(1, HIDDEN_DIM))
        return grid, scalars, hidden


# ─────────────────────────────────────────────────────────────────────────────
# Collate functions
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn_stack(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stack a batch of (grid, scalars, hidden) tuples into
    (B, C, D, D), (B, 7), (B, H) tensors.
    """
    grids   = torch.cat([s[0] for s in batch], dim=0)
    scalars = torch.cat([s[1] for s in batch], dim=0)
    hiddens = torch.cat([s[2] for s in batch], dim=0)
    return grids, scalars, hiddens


def collate_fn_to_device(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> List[torch.Tensor]:
    """
    Move a batch to device and return as a list matching
    Actor.forward(grid, scalars, hidden) argument order.

    esp-ppq's collate_fn receives the OUTPUT of the DataLoader's
    collate_fn (which is collate_fn_stack above) and must return
    something that can be passed directly to the model as *args.
    """
    grids, scalars, hiddens = batch
    return [
        grids.to(DEVICE),
        scalars.to(DEVICE),
        hiddens.to(DEVICE),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Quantization setting
# ─────────────────────────────────────────────────────────────────────────────

def build_quant_setting(model: nn.Module) -> tuple:
    """
    Build quantization setting for the Actor using layerwise equalization.

    Equalization corrects for weight scale imbalances across the DWS conv
    layers — common in MobileNet-style architectures. The Actor uses ReLU
    (not ReLU6) so no activation replacement is needed.

    Returns (setting, model) — model is unchanged since no ReLU6 replacement
    is needed, but we follow the reference signature for consistency.
    """
    setting = QuantizationSettingFactory.espdl_setting()

    # Layerwise equalization — handles DWS conv weight scale disparities
    setting.equalization                           = True
    setting.equalization_setting.iterations       = 4
    setting.equalization_setting.value_threshold  = 0.4
    setting.equalization_setting.opt_level        = 2
    setting.equalization_setting.interested_layers = None  # all layers

    return setting, model


# ─────────────────────────────────────────────────────────────────────────────
# Load actor from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_actor(checkpoint_path: str, agent: str = "player_0") -> Actor:
    """
    Load Actor weights from a MAPPO training checkpoint.
    Falls back to a randomly initialised Actor if checkpoint not found
    (for pipeline testing only — do not deploy random weights).
    """
    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        print("          Using randomly initialised weights — for pipeline testing only.")
        actor = Actor(
            in_channels = C,
            fov_size    = D,
            hidden_dim  = HIDDEN_DIM,
            action_dim  = 6,
            width_mult  = 0.25,
        )
        return actor

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg: Cfg = ckpt["cfg"]

    actor = Actor(
        in_channels = C,
        fov_size    = D,
        hidden_dim  = cfg.hidden_dim,
        action_dim  = cfg.action_dim,
        width_mult  = cfg.width_mult,
    )
    actor.load_state_dict(ckpt["actors"][agent])
    print(f"Loaded {agent} from {checkpoint_path}  "
          f"(step={ckpt['global_step']:,}  updates={ckpt['update_count']})")
    return actor


# ─────────────────────────────────────────────────────────────────────────────
# Main quantization function
# ─────────────────────────────────────────────────────────────────────────────

def _check_no_negative_axes(actor: nn.Module, device: str = "cpu"):
    """
    Export to ONNX in-memory and assert no Concat/Gather/Unsqueeze node
    has a negative axis attribute. esp-ppq's layout_patterns.py calls
    var_perm.index(int(axis)) which raises ValueError for negative values.

    Raises AssertionError with the offending node name if found.
    """
    import io, onnx as _onnx
    buf = io.BytesIO()
    dummy = (
        torch.zeros(1, C, D, D, device=device),
        torch.zeros(1, 7,      device=device),
        torch.zeros(1, HIDDEN_DIM, device=device),
    )
    torch.onnx.export(
        model=actor, args=dummy, f=buf,
        opset_version=13,
        input_names=["grid", "scalars", "hidden_in"],
        output_names=["logits", "hidden_out"],
        do_constant_folding=True, dynamo=False,
    )
    buf.seek(0)
    model = _onnx.load_model_from_string(buf.read())
    axis_ops = {"Concat", "Gather", "Unsqueeze", "Squeeze", "Flatten"}
    bad = []
    for node in model.graph.node:
        if node.op_type in axis_ops:
            for attr in node.attribute:
                if attr.name == "axis" and attr.i < 0:
                    bad.append(f"{node.op_type} node '{node.name}' axis={attr.i}")
    if bad:
        raise AssertionError(
            "Negative axis values found in ONNX graph — will crash esp-ppq "
            "layout resolver.\nFix: replace dim=-1 with explicit positive dim "
            "in Actor.forward().\nOffending nodes:\n  " + "\n  ".join(bad)
        )
    print("  Pre-export axis check: OK (no negative axes)")


def quantize_actor(
    checkpoint_path: str,
    agent:           str  = "player_0",
    output_dir:      str  = "models",
    layout:          str  = "cramped_room",
    calib_samples:   int  = 1024,
    calib_batch:     int  = 32,
    calib_steps:     int  = 32,
    random_calib:    bool = False,
    error_report:    bool = True,
):
    """
    Full pipeline: load → calibrate → quantize → export .espdl

    Parameters
    ----------
    checkpoint_path : path to mappo_*.pt checkpoint
    agent           : "player_0" or "player_1"
    output_dir      : directory for output files
    layout          : Overcooked layout for calibration data
    calib_samples   : number of calibration samples to collect
    calib_batch     : batch size for calibration dataloader
    calib_steps     : number of calibration steps for PPQ (≤ len(dataloader))
    random_calib    : use random data instead of Overcooked environment
    error_report    : print per-layer quantization error report
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    actor = load_actor(checkpoint_path, agent)
    actor.eval()

    # ── 1b. Pre-flight: assert no negative axes survive into ONNX ─────────────
    # esp-ppq's layout_patterns.py does var_perm.index(int(axis)) which raises
    # ValueError for negative values. Catch this before entering the ppq pipeline.
    print("Running pre-export axis check…")
    _check_no_negative_axes(actor, device=DEVICE)

    # ── 2. Apply quant setting ────────────────────────────────────────────────
    quant_setting, actor = build_quant_setting(actor)

    # ── 3. Build calibration dataloader ───────────────────────────────────────
    print()
    if random_calib:
        dataset = RandomCalibDataset(num_samples=calib_samples)
    else:
        try:
            dataset = OvercookedCalibDataset(
                layout        = layout,
                num_samples   = calib_samples,
                fov_radius    = FOV_RADIUS,
                noise_enabled = True,
            )
        except Exception as exc:
            print(f"[WARNING] Overcooked calibration failed ({exc}), "
                  f"falling back to random calibration.")
            dataset = RandomCalibDataset(num_samples=calib_samples)

    dataloader = DataLoader(
        dataset     = dataset,
        batch_size  = calib_batch,
        shuffle     = True,           # shuffle = better coverage of all states
        num_workers = 0,              # 0 avoids forking issues with Overcooked
        collate_fn  = collate_fn_stack,
    )

    # ── 4. Output paths ───────────────────────────────────────────────────────
    espdl_path = os.path.join(output_dir, f"{agent}_actor.espdl")

    # ── 5. Quantize ───────────────────────────────────────────────────────────
    print(f"\nQuantizing {agent} → {espdl_path}")
    print(f"  Target   : {TARGET}  ({NUM_OF_BITS}-bit)")
    print(f"  Inputs   : grid{INPUT_SHAPES[0]}  scalars{INPUT_SHAPES[1]}  "
          f"hidden{INPUT_SHAPES[2]}")
    print(f"  Calib    : {len(dataset)} samples, {calib_steps} steps, "
          f"batch={calib_batch}")
    print()

    quant_graph = espdl_quantize_torch(
        model              = actor,
        espdl_export_file  = espdl_path,
        calib_dataloader   = dataloader,
        calib_steps        = calib_steps,
        # input_shape is a list-of-shapes for multi-input models.
        # ORDER matches Actor.forward(grid, scalars, hidden) positional args.
        input_shape        = INPUT_SHAPES,
        target             = TARGET,
        num_of_bits        = NUM_OF_BITS,
        # collate_fn_to_device moves the DataLoader batch to device and
        # returns a list — esp-ppq unpacks it as *args to the model.
        collate_fn         = collate_fn_to_device,
        setting            = quant_setting,
        device             = DEVICE,
        error_report       = error_report,
        skip_export        = False,
        export_test_values = False,
        verbose            = 1,
        opset_version      = 13,     # 13+ normalises negative axes; required for esp-ppq layout resolver
    )

    print(f"\nExport complete.")
    print(f"  .espdl : {espdl_path}")
    print(f"  .onnx  : {os.path.splitext(espdl_path)[0]}.onnx  (keep for inspection)")

    return quant_graph


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantize MAPPO Actor to INT8 .espdl for ESP32-S3"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/mappo_final.pt",
        help="Path to mappo_*.pt training checkpoint",
    )
    parser.add_argument(
        "--agent", default="player_0", choices=["player_0", "player_1"],
        help="Which agent's actor to export",
    )
    parser.add_argument(
        "--both_agents", action="store_true",
        help="Export both player_0 and player_1 in one run",
    )
    parser.add_argument(
        "--output_dir", default="models",
        help="Output directory for .espdl and .onnx files",
    )
    parser.add_argument(
        "--layout", default="cramped_room",
        help="Overcooked layout for calibration data collection",
    )
    parser.add_argument(
        "--calib_samples", type=int, default=1024,
        help="Number of calibration samples to collect from the environment",
    )
    parser.add_argument(
        "--calib_batch", type=int, default=32,
        help="Calibration dataloader batch size",
    )
    parser.add_argument(
        "--calib_steps", type=int, default=32,
        help="Number of PPQ calibration steps",
    )
    parser.add_argument(
        "--random_calib", action="store_true",
        help="Use random calibration data (debug only — do not deploy)",
    )
    parser.add_argument(
        "--no_error_report", action="store_true",
        help="Skip per-layer quantization error report",
    )
    args = parser.parse_args()

    agents = ["player_0", "player_1"] if args.both_agents else [args.agent]

    for agent in agents:
        quantize_actor(
            checkpoint_path = args.checkpoint,
            agent           = agent,
            output_dir      = args.output_dir,
            layout          = args.layout,
            calib_samples   = args.calib_samples,
            calib_batch     = args.calib_batch,
            calib_steps     = args.calib_steps,
            random_calib    = args.random_calib,
            error_report    = not args.no_error_report,
        )
        print()