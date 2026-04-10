from __future__ import annotations
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from networks import Actor
from _dataclasses import Cfg, NoiseCfg
from utils import *
from partial_obs_wrapper import PartialObsWrapper

from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_torch

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

import io
import onnx as _onnx

TARGET = "esp32s3"
NUM_OF_BITS = 8
DEVICE = "cpu"
HIDDEN_DIM = 128
FOV_RADIUS = 5
D = 2 * FOV_RADIUS + 1   
C = NUM_CHANNELS      

INPUT_SHAPES = [
    [1, C, D, D],  
    [1, 7],        
    [1, HIDDEN_DIM], 
]

class CalibDataset(Dataset):
    def __init__(self, layout='cramped_room', num_samples=1024, fov_radius=FOV_RADIUS, noise_enabled=True):
        self.samples = []

        noise_cfg = NoiseCfg()
        mdp = OvercookedGridworld.from_layout_name(layout)
        base = OvercookedEnv.from_mdp(mdp, horizon=400)
        env = PartialObsWrapper(base, fov_radius=fov_radius, noise_cfg=noise_cfg)

        collected = 0
        hidden = np.zeros((1, HIDDEN_DIM), dtype=np.float32)

        obs = env.reset()

        while collected < num_samples:
            for agent_id in ["player_0", "player_1"]:
                if collected >= num_samples:
                    break

                grid_np = obs[agent_id]["grid"]
                scalar_np = obs[agent_id]["scalars"]

                self.samples.append((
                    torch.from_numpy(grid_np).unsqueeze(0),  
                    torch.from_numpy(scalar_np).unsqueeze(0),   
                    torch.from_numpy(hidden).clone(),      
                ))

                collected += 1

            act_d = {
                a: np.random.randint(0, 6)
                for a in ["player_0", "player_1"]
            }
            obs, _, done, _ = env.step(act_d)

            if done:
                obs = env.reset()
                hidden = np.zeros((1, HIDDEN_DIM), dtype=np.float32)

        print(f"  Collected {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn_stack(batch):
    grids = torch.cat([s[0] for s in batch], dim=0)
    scalars = torch.cat([s[1] for s in batch], dim=0)
    hiddens = torch.cat([s[2] for s in batch], dim=0)

    return grids, scalars, hiddens


def collate_fn_to_device(batch):
    grids, scalars, hiddens = batch

    return [
        grids.to(DEVICE),
        scalars.to(DEVICE),
        hiddens.to(DEVICE),
    ]

def build_quant_setting(model):
    setting = QuantizationSettingFactory.espdl_setting()

    setting.equalization = True
    setting.equalization_setting.iterations = 4
    setting.equalization_setting.value_threshold = 0.4
    setting.equalization_setting.opt_level = 2
    setting.equalization_setting.interested_layers = None 

    return setting, model

def load_actor(checkpoint_path, agent = "player_0"):
    if not os.path.exists(checkpoint_path):
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
        hidden_dim = cfg.hidden_dim,
        action_dim = cfg.action_dim,
        width_mult = cfg.width_mult,
    )
    actor.load_state_dict(ckpt["actors"][agent])
 
    return actor

def check_no_negative_axes(actor, device):
    buf = io.BytesIO()
    dummy = (
        torch.zeros(1, C, D, D, device=device),
        torch.zeros(1, 7, device=device),
        torch.zeros(1, HIDDEN_DIM, device=device),
    )

    torch.onnx.export(
        model=actor, args=dummy, f=buf, opset_version=13,
        input_names=["grid", "scalars", "hidden_in"],
        output_names=["logits", "hidden_out"],
        do_constant_folding=True, dynamo=False
    )

    buf.seek(0)
    model = _onnx.load_model_from_string(buf.read())
    axis_ops = {"Concat", "Gather", "Unsqueeze", "Squeeze", "Flatten"}
    bad = []

    for node in model.graph.node:
        if node.op_type in axis_ops:
            for attr in node.attribute:
                if attr.name == 'axis' and attr.i < 0:
                    bad.append(f"{node.op_type} node '{node.name}' axis={attr.i}")

    if bad:
        raise AssertionError(
            "Negative axis values found in ONNX graph — will crash esp-ppq "
            "layout resolver.\nFix: replace dim=-1 with explicit positive dim "
            "in Actor.forward().\nOffending nodes:\n  " + "\n  ".join(bad)
        )
    print("  Pre-export axis check: OK (no negative axes)")

def quantize_actor(ckpt_path, agent='player_0', out_dir='models', layout='cramped_room', calib_samples=1024,
                   calib_batch=32, calib_steps=32, error_report=True):
    
    os.makedirs(out_dir, exist_ok=True)

    actor = load_actor(ckpt_path, agent)
    actor.eval()

    check_no_negative_axes(actor, device=DEVICE)
    quant_setting, actor = build_quant_setting(actor)

    dataset = CalibDataset(layout = layout, num_samples = calib_samples, fov_radius = FOV_RADIUS, noise_enabled = True)

    dataloader = DataLoader(dataset=dataset, batch_size=calib_batch, shuffle=True, num_workers=0, collate_fn=collate_fn_stack)

    espdl_path = os.path.join(out_dir, f"{agent}_actor.espdl")

    quant_graph = espdl_quantize_torch(
        model = actor,
        espdl_export_file = espdl_path,
        calib_dataloader = dataloader,
        calib_steps = calib_steps,
        input_shape = INPUT_SHAPES,
        target = TARGET,
        num_of_bits = NUM_OF_BITS,
        collate_fn = collate_fn_to_device,
        setting = quant_setting,
        device = DEVICE,
        error_report = error_report,
        skip_export = False,
        export_test_values = False,
        verbose = 1,
        opset_version = 13
    )

    return quant_graph


if __name__ == "__main__":
    checkpoint = "checkpoints/mappo_step0000000000.pt"
    agent = "player_0"
    store_both_agents = True
    output_dir = "models"
    layout = "cramped_room"
    calib_samples = 1024
    calib_batch =32
    calib_steps =32
    no_error_report = False

    agents = ["player_0", "player_1"] if store_both_agents else [agent]

    for agent in agents:
        quantize_actor(
            ckpt_path = checkpoint,
            agent = agent,
            out_dir = output_dir,
            layout = layout,
            calib_samples = calib_samples,
            calib_batch = calib_batch,
            calib_steps = calib_steps,
            error_report = not no_error_report,
        )