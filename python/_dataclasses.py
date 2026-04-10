from dataclasses import dataclass, field
from typing import Dict, List
import torch

@dataclass
class Cfg:
    layout = "cramped_room"
    fov_radius = 5
    num_envs = 8
    episode_steps = 400

    hidden_dim = 128
    action_dim = 6
    width_mult = 0.25

    actor_lr = 3e-4

    critic_lr = 1e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    vf_coef = 0.5
    ent_coef = 0.02   
    max_grad_norm = 0.5
    ppo_epochs = 4
    minibatch_size = 256

    sparse_factor = 5.0  
    reward_shaping_factor = 1.0

    rollout_steps = 128
    total_steps = 10_000_000

    log_interval = 10
    save_interval = 100
    eval_interval = 50  
    save_dir = "checkpoints"
    dashboard = True

    device = "cuda" if torch.cuda.is_available() else "cpu"


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
    
@dataclass
class NoiseCfg:
    delay_enabled = True
    delay_min_steps = 1  
    delay_max_steps = 2      

    dropout_enabled  = True
    dropout_p = 0.08  

    obj_miss_enabled = True
    obj_miss_p = 0.05   

    dr_dropout_p_range = (0.03, 0.15)
    dr_obj_miss_p_range = (0.02, 0.10)
    domain_randomise = True  