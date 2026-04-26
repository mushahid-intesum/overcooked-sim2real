from dataclasses import dataclass, field
from typing import Dict, List
import torch

@dataclass
class Cfg:
    layout: str = "cramped_room"
    fov_radius: int = 5
    num_envs: int = 8
    episode_steps: int = 400
    hidden_dim: int = 128
    action_dim: int = 6
    width_mult: float = 0.25
    actor_lr: float = 3e-4
    critic_lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.02
    max_grad_norm: float = 0.5
    ppo_epochs: int = 1
    minibatch_size: int = 256
    sparse_factor: float = 1.0
    reward_shaping_factor: float = 1.0
    rollout_steps: int = 128
    total_steps: int = 10_000_000
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50
    save_dir: str = "checkpoints"
    dashboard: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class NoiseCfg:
    delay_enabled: bool = True
    delay_min_steps: int = 1
    delay_max_steps: int = 2
    dropout_enabled: bool = True
    dropout_p: float = 0.08
    obj_miss_enabled: bool = True
    obj_miss_p: float = 0.05
    dr_dropout_p_range: tuple = (0.03, 0.15)
    dr_obj_miss_p_range: tuple = (0.02, 0.10)
    domain_randomise: bool = True


@dataclass
class RolloutBuffer:
    grids:          List[Dict[str, torch.Tensor]] = field(default_factory=list)
    scalars:        List[Dict[str, torch.Tensor]] = field(default_factory=list)
    hiddens:        List[Dict[str, torch.Tensor]] = field(default_factory=list)
    actions:        List[Dict[str, torch.Tensor]] = field(default_factory=list)
    log_probs:      List[Dict[str, torch.Tensor]] = field(default_factory=list)
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
