"""
mappo_train.py
──────────────────────────────────────────────────────────────────────────────
Full MAPPO (Multi-Agent PPO with Centralised Training, Decentralised Execution)
training pipeline for the Overcooked-AI partial-obs sim2real project.

Architecture
────────────
  Actor  (deployed on ESP32-S3 via TFLM):
      PartialObsEncoder  (MobileNet-V1 depthwise-separable CNN)
          input : (C, D, D) grid tensor  +  scalar features (7,)
          output: 128-dim embedding
      GRU cell           hidden_dim = 128
      Linear head        → action logits (6 discrete actions)

  Critic (centralised, training-only):
      GlobalStateEncoder (shallow MLP on flattened full grid)
      Linear head        → scalar value estimate V(s)

  The actor observes only the partial, noisy, FoV-masked obs.
  The critic observes the full global state (no noise, no masking).
  This is the standard CTDE setup for MAPPO.

Actions (6)
────────────
  0: STAY       1: NORTH      2: SOUTH
  3: EAST       4: WEST       5: INTERACT

Training loop
────────────
  Rollout collection → GAE advantage estimation →
  Multiple PPO epochs on minibatches → log metrics

Usage
────────────
  python mappo_train.py                        # run with defaults
  python mappo_train.py --layout cramped_room  # specify layout
  python mappo_train.py --help

Dependencies:
  pip install torch numpy gymnasium pettingzoo
  pip install git+https://github.com/HumanCompatibleAI/overcooked_ai.git
"""

from __future__ import annotations

import argparse
import collections
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from torch.utils.tensorboard import SummaryWriter

# Local
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
    # Environment
    layout:        str   = "cramped_room"
    fov_radius:    int   = 5
    num_envs:      int   = 8          # parallel env instances
    episode_steps: int   = 400        # max steps per episode

    # Network
    hidden_dim:    int   = 128
    gru_layers:    int   = 1
    action_dim:    int   = 6

    # PPO
    lr:            float = 3e-4
    gamma:         float = 0.99
    gae_lambda:    float = 0.95
    clip_eps:      float = 0.2
    vf_coef:       float = 0.5
    ent_coef:      float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs:    int   = 4
    minibatch_size:int   = 256

    # Rollout
    rollout_steps: int   = 128       # steps collected before each update
    total_steps:   int   = 10_000_000

    # Logging
    log_interval:  int   = 10        # log every N updates
    save_interval: int   = 100       # checkpoint every N updates
    save_dir:      str   = "checkpoints"

    # Device
    device:        str   = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# MobileNet-V1 depthwise-separable block
# ─────────────────────────────────────────────────────────────────────────────

class DWSConvBlock(nn.Module):
    """Depthwise-separable conv + BN + ReLU (MobileNet-V1 style)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride,
            padding=1, groups=in_ch, bias=False,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.bn_pw = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_dw(self.dw(x)))
        x = F.relu(self.bn_pw(self.pw(x)))
        return x


class MobileNetEncoder(nn.Module):
    """
    Lightweight MobileNet-V1 backbone adapted for symbolic grid tensors.

    Input:  (B, C, D, D)  where C = NUM_CHANNELS, D = 2*fov_radius+1
    Output: (B, embed_dim)

    Width multiplier α = 0.25 keeps parameter count tiny for ESP32-S3.
    The full channel sequence mirrors MobileNet-V1 but scaled down.
    """

    def __init__(
        self,
        in_channels:    int = NUM_CHANNELS,
        embed_dim:      int = 128,
        width_mult:     float = 0.25,
        input_size:     int = 11,       # D = 2*R+1 for R=5
    ):
        super().__init__()
        def ch(c: int) -> int:
            return max(1, int(c * width_mult))

        # Initial standard conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ch(32), kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(ch(32)),
            nn.ReLU(inplace=True),
        )

        # DWS blocks — stride 1 throughout (input is already small: 11×11)
        self.blocks = nn.Sequential(
            DWSConvBlock(ch(32),  ch(64),  stride=1),
            DWSConvBlock(ch(64),  ch(128), stride=2),   # → 6×6 for D=11
            DWSConvBlock(ch(128), ch(128), stride=1),
            DWSConvBlock(ch(128), ch(256), stride=2),   # → 3×3 for D=11
            DWSConvBlock(ch(256), ch(256), stride=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Compute flattened dim after pool
        cnn_out_dim = ch(256)

        self.proj = nn.Sequential(
            nn.Linear(cnn_out_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, D)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)     # (B, cnn_out_dim)
        return self.proj(x)             # (B, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Actor (decentralised — deployed on-device)
# ─────────────────────────────────────────────────────────────────────────────

SCALAR_DIM = 7   # [delta_row, delta_col, held_onehot×5]

class Actor(nn.Module):
    """
    Partial-obs actor: MobileNet CNN encoder + scalar fusion + GRU + head.

    Forward takes a single time step (no sequence dim) and the previous
    GRU hidden state, returning action logits and the new hidden state.
    This matches the on-device inference loop on the ESP32-S3.

    Parameters
    ──────────
    in_channels : number of grid channels (NUM_CHANNELS)
    fov_size    : D = 2*R+1, spatial size of the ego-crop
    hidden_dim  : GRU hidden size
    action_dim  : number of discrete actions
    """

    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        fov_size:    int = 11,
        hidden_dim:  int = 128,
        action_dim:  int = 6,
        width_mult:  float = 0.25,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.cnn = MobileNetEncoder(
            in_channels=in_channels,
            embed_dim=hidden_dim,
            width_mult=width_mult,
            input_size=fov_size,
        )

        # Scalar feature MLP
        self.scalar_enc = nn.Sequential(
            nn.Linear(SCALAR_DIM, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # Fusion projection
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # GRU (single step)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Action head
        self.head = nn.Linear(hidden_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        grid:    torch.Tensor,   # (B, C, D, D)  float32
        scalars: torch.Tensor,   # (B, 7)         float32
        hidden:  torch.Tensor,   # (B, hidden_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        ───────
        logits : (B, action_dim)
        hidden : (B, hidden_dim)   new GRU state
        """
        cnn_feat    = self.cnn(grid)                          # (B, H)
        scalar_feat = self.scalar_enc(scalars)                # (B, 32)
        fused       = self.fusion(
            torch.cat([cnn_feat, scalar_feat], dim=-1)
        )                                                     # (B, H)
        hidden_new  = self.gru(fused, hidden)                 # (B, H)
        logits      = self.head(hidden_new)                   # (B, A)
        return logits, hidden_new

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def get_action_and_value_inputs(
        self,
        grid:    torch.Tensor,
        scalars: torch.Tensor,
        hidden:  torch.Tensor,
        action:  Optional[torch.Tensor] = None,
    ):
        """Convenience method for PPO update — returns action, log_prob, entropy, new_hidden."""
        logits, new_hidden = self.forward(grid, scalars, hidden)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, new_hidden


# ─────────────────────────────────────────────────────────────────────────────
# Centralised Critic (training-only)
# ─────────────────────────────────────────────────────────────────────────────

class CentralCritic(nn.Module):
    """
    Centralised value function.
    Input: flattened global state (full grid + all agent scalars).
    This never runs on-device — only used during training.
    """

    def __init__(
        self,
        global_grid_channels: int = NUM_CHANNELS,
        global_grid_h:        int = 5,    # set from env at runtime
        global_grid_w:        int = 4,
        num_agents:           int = 2,
        hidden_dim:           int = 256,
    ):
        super().__init__()
        flat_grid  = global_grid_channels * global_grid_h * global_grid_w
        scalar_dim = num_agents * SCALAR_DIM

        self.net = nn.Sequential(
            nn.Linear(flat_grid + scalar_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(
        self,
        global_grid: torch.Tensor,   # (B, C, H, W)
        scalars:     torch.Tensor,   # (B, num_agents * 7)
    ) -> torch.Tensor:
        flat = torch.cat([global_grid.flatten(1), scalars], dim=-1)
        return self.net(flat).squeeze(-1)   # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout buffer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    """
    Stores one rollout chunk for all agents across all parallel envs.
    All tensors are on CPU during collection; moved to device at update time.
    """
    # Per-step, per-agent, per-env tensors
    grids:      List[Dict[str, torch.Tensor]] = field(default_factory=list)
    scalars:    List[Dict[str, torch.Tensor]] = field(default_factory=list)
    hiddens:    List[Dict[str, torch.Tensor]] = field(default_factory=list)
    actions:    List[Dict[str, torch.Tensor]] = field(default_factory=list)
    log_probs:  List[Dict[str, torch.Tensor]] = field(default_factory=list)
    rewards:    List[Dict[str, torch.Tensor]] = field(default_factory=list)
    dones:      List[Dict[str, torch.Tensor]] = field(default_factory=list)
    # Global state for critic
    global_grids:   List[torch.Tensor] = field(default_factory=list)
    global_scalars: List[torch.Tensor] = field(default_factory=list)

    def clear(self):
        for f in self.__dataclass_fields__:
            setattr(self, f, [])

    def __len__(self):
        return len(self.actions)


# ─────────────────────────────────────────────────────────────────────────────
# GAE computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards:    torch.Tensor,   # (T, N)  N = num_envs
    values:     torch.Tensor,   # (T+1, N)
    dones:      torch.Tensor,   # (T, N)
    gamma:      float,
    lam:        float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns advantages (T, N) and returns (T, N).
    Vectorised over N environments.
    """
    T, N = rewards.shape
    advantages = torch.zeros(T, N)
    gae = torch.zeros(N)

    for t in reversed(range(T)):
        next_val    = values[t + 1]
        delta       = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae         = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:T]
    return advantages, returns


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(layout, fov_radius, noise_cfg):
    mdp = OvercookedGridworld.from_layout_name(layout)
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

        # Build envs
        noise_cfg = NoiseCfg(domain_randomise=True)
        self.envs  = [
            make_env(cfg.layout, cfg.fov_radius, noise_cfg)
            for _ in range(cfg.num_envs)
        ]

        # Determine global grid dims from one env reset
        self.envs[0].reset()
        global_grid, global_scalars = self.envs[0].get_global_state()
        C, H, W = global_grid.shape
        D = 2 * cfg.fov_radius + 1

        self.writer = SummaryWriter(
            log_dir=os.path.join("runs", f"mappo_{int(time.time())}")
        )

        # Networks
        self.actors: Dict[str, Actor] = {
            agent: Actor(
                in_channels=NUM_CHANNELS,
                fov_size=D,
                hidden_dim=cfg.hidden_dim,
                action_dim=cfg.action_dim,
            ).to(self.device)
            for agent in self.agents
        }

        self.critic = CentralCritic(
            global_grid_channels=C,
            global_grid_h=H,
            global_grid_w=W,
            num_agents=len(self.agents),
            hidden_dim=256,
        ).to(self.device)

        # Optimisers — shared across actors + separate for critic
        actor_params = []
        for actor in self.actors.values():
            actor_params += list(actor.parameters())
        self.actor_opt  = torch.optim.Adam(actor_params, lr=cfg.lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr, eps=1e-5)

        self.buffer = RolloutBuffer()

        # Running stats
        self.ep_returns:   collections.deque = collections.deque(maxlen=100)
        self.global_step:  int = 0
        self.update_count: int = 0

        os.makedirs(cfg.save_dir, exist_ok=True)

    # ──────────────────────────────────────────────────── training loop ──────

    def train(self):
        cfg    = self.cfg
        device = self.device

        # Initial reset
        obs_list   = [env.reset() for env in self.envs]
        hidden     = {
            agent: self.actors[agent].init_hidden(cfg.num_envs, device)
            for agent in self.agents
        }
        ep_rews = np.zeros((cfg.num_envs, len(self.agents)))

        t_start = time.time()

        while self.global_step < cfg.total_steps:

            self.buffer.clear()

            # ── Rollout collection ────────────────────────────────────────
            with torch.no_grad():
                for _ in range(cfg.rollout_steps):
                    step_grids   = {}
                    step_scalars = {}
                    step_actions = {}
                    step_lp      = {}
                    step_hid     = {}

                    for idx, agent in enumerate(self.agents):
                        grids_np   = np.stack([
                            o[agent]["grid"] for o in obs_list
                        ])                                               # (N, C, D, D)
                        scalars_np = np.stack([
                            o[agent]["scalars"] for o in obs_list
                        ])                                               # (N, 7)

                        g  = torch.from_numpy(grids_np).to(device)
                        s  = torch.from_numpy(scalars_np).to(device)
                        h  = hidden[agent]

                        action, log_prob, _, new_h = \
                            self.actors[agent].get_action_and_value_inputs(g, s, h)

                        step_grids[agent]   = g.cpu()
                        step_scalars[agent] = s.cpu()
                        step_actions[agent] = action.cpu()
                        step_lp[agent]      = log_prob.cpu()
                        step_hid[agent]     = h.cpu()
                        hidden[agent]       = new_h

                    # Global state for critic
                    gg_list, gs_list = [], []
                    for env in self.envs:
                        gg, gs = env.get_global_state()
                        gg_list.append(gg)
                        gs_list.append(gs)
                    global_g = torch.from_numpy(np.stack(gg_list)).to(device)
                    global_s = torch.from_numpy(np.stack(gs_list)).to(device)

                    # Step all envs
                    action_dicts = [
                        {agent: step_actions[agent][i].item()
                         for agent in self.agents}
                        for i in range(cfg.num_envs)
                    ]
                    new_obs_list, rews_list, done_list = [], [], []
                    step_rews  = {a: np.zeros(cfg.num_envs) for a in self.agents}
                    step_dones = {a: np.zeros(cfg.num_envs) for a in self.agents}

                    for i, (env, act_d) in enumerate(zip(self.envs, action_dicts)):
                        obs_n, rew_n, done_n, info_n = env.step(act_d)
                        new_obs_list.append(obs_n)
                        for agent in self.agents:
                            step_rews[agent][i]  = rew_n
                            step_dones[agent][i] = float(done_n)
                            ep_rews[i, self.agents.index(agent)] += rew_n

                        # Reset done envs
                        if done_n:
                            self.ep_returns.append(ep_rews[i].sum())
                            ep_rews[i] = 0.0
                            
                            new_obs_list[-1] = env.reset()
                            for agent in self.agents:
                                hidden[agent][i] = 0.0

                    obs_list = new_obs_list

                    # Store in buffer
                    self.buffer.grids.append(step_grids)
                    self.buffer.scalars.append(step_scalars)
                    self.buffer.hiddens.append(step_hid)
                    self.buffer.actions.append(step_actions)
                    self.buffer.log_probs.append(step_lp)
                    self.buffer.rewards.append(
                        {a: torch.from_numpy(step_rews[a]).float()
                         for a in self.agents}
                    )
                    self.buffer.dones.append(
                        {a: torch.from_numpy(step_dones[a]).float()
                         for a in self.agents}
                    )
                    self.buffer.global_grids.append(global_g.cpu())
                    self.buffer.global_scalars.append(global_s.cpu())

                    self.global_step += cfg.num_envs

            next_gg_list, next_gs_list = [], []
            for env in self.envs:
                gg, gs = env.get_global_state()
                next_gg_list.append(gg)
                next_gs_list.append(gs)

            self.buffer.global_grids.append(
                torch.from_numpy(np.stack(next_gg_list)).cpu()
            )
            self.buffer.global_scalars.append(
                torch.from_numpy(np.stack(next_gs_list)).cpu()
            )

            # ── PPO update ────────────────────────────────────────────────
            metrics = self._ppo_update()
            self.update_count += 1

            # ── Logging ───────────────────────────────────────────────────
            if self.update_count % cfg.log_interval == 0:
                mean_ret = (np.mean(self.ep_returns)
                            if self.ep_returns else float("nan"))
                sps      = self.global_step / (time.time() - t_start)
                print(
                    f"step={self.global_step:>10,}  "
                    f"updates={self.update_count:>6}  "
                    f"mean_ep_ret={mean_ret:>8.2f}  "
                    f"actor_loss={metrics['actor_loss']:.6f}  "
                    f"critic_loss={metrics['critic_loss']:.6f}  "
                    f"entropy={metrics['entropy']:.4f}  "
                    f"kl={metrics['approx_kl']:.5f}  "
                    f"clip={metrics['clipfrac']:.3f}  "
                    f"ev={metrics['explained_var']:.3f}  "
                    f"sps={sps:>6.0f}"
                )
                self.writer.add_scalar("loss/actor", metrics["actor_loss"], self.update_count)
                self.writer.add_scalar("loss/critic", metrics["critic_loss"], self.update_count)
                self.writer.add_scalar("policy/entropy", metrics["entropy"], self.update_count)
                self.writer.add_scalar("policy/approx_kl", metrics["approx_kl"], self.update_count)
                self.writer.add_scalar("policy/clipfrac", metrics["clipfrac"], self.update_count)
                self.writer.add_scalar("critic/explained_variance", metrics["explained_var"], self.update_count)
                self.writer.add_scalar("env/mean_return", mean_ret, self.update_count)
                self.writer.add_scalar("train/sps", sps, self.update_count)

                try:
                    print(self.envs[0].render())
                except Exception as e:
                    print(e)
                    print('Error Rendering')

            if self.update_count % cfg.save_interval == 0:
                self._save_checkpoint()

        self._save_checkpoint(final=True)
        print("Training complete.")

    # ──────────────────────────────────────────────────── PPO update ─────────

    def _ppo_update(self) -> Dict[str, float]:
        cfg    = self.cfg
        device = self.device
        T      = len(self.buffer)      # rollout steps
        N      = cfg.num_envs
        approx_kl = 0.0
        clipfrac  = 0.0
 
        # ── Compute values for all timesteps + bootstrap ──────────────────
        with torch.no_grad():
            all_values = []
            for t in range(T+1):  
                gg = self.buffer.global_grids[t].to(device)
                gs = self.buffer.global_scalars[t].to(device)
                v  = self.critic(gg, gs)
                all_values.append(v.cpu())

            values_t = torch.stack(all_values, dim=0)   # (T+1, N)
 
        # ── Per-agent GAE ─────────────────────────────────────────────────
        agent_adv  = {}
        agent_ret  = {}
        for agent in self.agents:
            rews  = torch.stack([self.buffer.rewards[t][agent]  for t in range(T)])
            dones = torch.stack([self.buffer.dones[t][agent]    for t in range(T)])
            adv, ret = compute_gae(rews, values_t, dones, cfg.gamma, cfg.gae_lambda)
            agent_adv[agent] = adv.reshape(-1)   # (T*N,)
            agent_ret[agent] = ret.reshape(-1)
 
        # Flatten all buffer fields
        flat = self._flatten_buffer(T, N)
 
        # ── Multiple PPO epochs ───────────────────────────────────────────
        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        total_entropy     = 0.0
        n_updates         = 0
 
        indices = np.arange(T * N)
        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, T * N, cfg.minibatch_size):
                mb_idx = indices[start:start + cfg.minibatch_size]
 
                # ── Actor losses (summed over agents) ────────────────────
                actor_loss = torch.tensor(0.0, device=device)
                mb_entropy = torch.tensor(0.0, device=device)
 
                for agent in self.agents:
                    mb_grid   = flat["grids"][agent][mb_idx].to(device)
                    mb_scal   = flat["scalars"][agent][mb_idx].to(device)
                    mb_hid    = flat["hiddens"][agent][mb_idx].to(device)
                    mb_act    = flat["actions"][agent][mb_idx].to(device)
                    mb_lp_old = flat["log_probs"][agent][mb_idx].to(device)
                    mb_adv    = agent_adv[agent][mb_idx].to(device)
 
                    # Normalise advantages per minibatch
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
 
                    _, mb_lp_new, mb_ent, _ = \
                        self.actors[agent].get_action_and_value_inputs(
                            mb_grid, mb_scal, mb_hid, mb_act
                        )
 
                    ratio     = (mb_lp_new - mb_lp_old).exp()
                    surr1     = ratio * mb_adv
                    surr2     = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
                    pg_loss   = -torch.min(surr1, surr2).mean()
                    ent_loss  = mb_ent.mean()
 
                    actor_loss += pg_loss - cfg.ent_coef * ent_loss
                    mb_entropy += ent_loss

                    approx_kl += (mb_lp_old - mb_lp_new).mean().item()
                    clipfrac  += ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
 
                actor_loss /= len(self.agents)
                mb_entropy /= len(self.agents)
 
                # ── Critic loss ──────────────────────────────────────────
                mb_gg  = flat["global_grids"][mb_idx].to(device)
                mb_gs  = flat["global_scalars"][mb_idx].to(device)
                # Average returns over agents as the critic target
                mb_ret = torch.stack(
                    [agent_ret[a][mb_idx].to(device) for a in self.agents]
                ).mean(0)
 
                v_pred     = self.critic(mb_gg, mb_gs)
                critic_loss = cfg.vf_coef * F.mse_loss(v_pred, mb_ret)
 
                # ── Optimise ─────────────────────────────────────────────
                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for a in self.actors.values() for p in a.parameters()],
                    cfg.max_grad_norm,
                )
                self.actor_opt.step()
 
                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_opt.step()
 
                total_actor_loss  += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy     += mb_entropy.item()
                n_updates         += 1

        with torch.no_grad():
            v_pred_all = []
            v_true_all = []

            for t in range(T):
                gg = self.buffer.global_grids[t].to(device)
                gs = self.buffer.global_scalars[t].to(device)
                v_pred_all.append(self.critic(gg, gs).cpu())

            v_pred_all = torch.cat(v_pred_all)
            v_true_all = torch.cat([
                torch.stack([agent_ret[a] for a in self.agents]).mean(0)
            ])

            var_y = torch.var(v_true_all)
            explained_var = 1 - torch.var(v_true_all - v_pred_all) / (var_y + 1e-8)
 
        return {
            "actor_loss":  total_actor_loss  / n_updates,
            "critic_loss": total_critic_loss / n_updates,
            "entropy":     total_entropy     / n_updates,
            "approx_kl":   approx_kl         / n_updates,
            "clipfrac":    clipfrac          / n_updates,
            "explained_var": explained_var.item(),
        }

    # ──────────────────────────────────────────────────── helpers ────────────

    def _flatten_buffer(self, T: int, N: int) -> Dict:
        """Concatenate rollout lists into flat (T*N, ...) tensors."""
        flat: Dict[str, Any] = {
            "grids":          {a: [] for a in self.agents},
            "scalars":        {a: [] for a in self.agents},
            "hiddens":        {a: [] for a in self.agents},
            "actions":        {a: [] for a in self.agents},
            "log_probs":      {a: [] for a in self.agents},
            "global_grids":   [],
            "global_scalars": [],
        }
        for t in range(T):
            for agent in self.agents:
                flat["grids"][agent].append(self.buffer.grids[t][agent])
                flat["scalars"][agent].append(self.buffer.scalars[t][agent])
                flat["hiddens"][agent].append(self.buffer.hiddens[t][agent])
                flat["actions"][agent].append(self.buffer.actions[t][agent])
                flat["log_probs"][agent].append(self.buffer.log_probs[t][agent])
            flat["global_grids"].append(self.buffer.global_grids[t])
            flat["global_scalars"].append(self.buffer.global_scalars[t])

        for agent in self.agents:
            flat["grids"][agent]    = torch.cat(flat["grids"][agent])
            flat["scalars"][agent]  = torch.cat(flat["scalars"][agent])
            flat["hiddens"][agent]  = torch.cat(flat["hiddens"][agent])
            flat["actions"][agent]  = torch.cat(flat["actions"][agent])
            flat["log_probs"][agent]= torch.cat(flat["log_probs"][agent])
        flat["global_grids"]   = torch.cat(flat["global_grids"])
        flat["global_scalars"] = torch.cat(flat["global_scalars"])
        return flat

    def _save_checkpoint(self, final: bool = False):
        tag  = "final" if final else f"step{self.global_step:010d}"
        path = os.path.join(self.cfg.save_dir, f"mappo_{tag}.pt")
        state = {
            "global_step":  self.global_step,
            "update_count": self.update_count,
            "cfg":          self.cfg,
            "actors":       {a: m.state_dict() for a, m in self.actors.items()},
            "critic":       self.critic.state_dict(),
            "actor_opt":    self.actor_opt.state_dict(),
            "critic_opt":   self.critic_opt.state_dict(),
        }
        torch.save(state, path)
        print(f"  Checkpoint saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Export actor for QAT / TFLM pipeline
# ─────────────────────────────────────────────────────────────────────────────

def export_actor(
    checkpoint_path: str,
    agent: str = "player_0",
    output_path: str = "actor_export.pt",
):
    """
    Load a checkpoint and export the actor's state dict for your existing
    QAT / TFLM pipeline.

    The exported actor takes:
        grid    : (1, C, D, D)
        scalars : (1, 7)
        hidden  : (1, hidden_dim)
    and returns:
        logits  : (1, 6)
        hidden  : (1, hidden_dim)
    """
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    cfg   = ckpt["cfg"]
    D     = 2 * cfg.fov_radius + 1
    actor = Actor(
        in_channels=NUM_CHANNELS,
        fov_size=D,
        hidden_dim=cfg.hidden_dim,
        action_dim=cfg.action_dim,
    )
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
    parser.add_argument("--layout",       default="cramped_room")
    parser.add_argument("--fov_radius",   type=int,   default=5)
    parser.add_argument("--num_envs",     type=int,   default=8)
    parser.add_argument("--total_steps",  type=int,   default=10_000_000)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--rollout_steps",type=int,   default=128)
    parser.add_argument("--hidden_dim",   type=int,   default=128)
    parser.add_argument("--save_dir",     default="checkpoints")
    parser.add_argument("--export",       default=None,
                        help="Path to checkpoint to export (skips training)")
    args = parser.parse_args()

    if args.export:
        export_actor(args.export)
    else:
        cfg = Cfg(
            layout        = args.layout,
            fov_radius    = args.fov_radius,
            num_envs      = args.num_envs,
            total_steps   = args.total_steps,
            lr            = args.lr,
            rollout_steps = args.rollout_steps,
            hidden_dim    = args.hidden_dim,
            save_dir      = args.save_dir,
        )
        trainer = MAPPOTrainer(cfg)
        trainer.train()
