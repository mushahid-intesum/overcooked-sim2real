import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from _dataclasses import *
from networks import Actor, Critic
from partial_obs_wrapper import *

from utils import NUM_CHANNELS, compute_gae
import os
from dashboard import Dashboard
from renderer import EvalRenderer
import time

layout = "cramped_room"
fov_radius = 5
num_envs = 8
total_steps = 10_000_000
actor_lr = 3e-4
critic_lr = 1e-4
rollout_steps = 128
hidden_dim = 128
ent_coef =  0.02
reward_shaping_factor = 1.0
save_dir = "checkpoints"
eval_interval = 50
no_dashboard = "store_true"
export = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
noise_cfg = NoiseCfg()

agents = ["player_0", "player_1"]

cfg = Cfg()

cfg.layout = layout
cfg.fov_radius = fov_radius
cfg.num_envs = num_envs
cfg.total_steps = total_steps
cfg.actor_lr = actor_lr
cfg.critic_lr = critic_lr
cfg.rollout_steps = rollout_steps
cfg.hidden_dim = hidden_dim
cfg.ent_coef = ent_coef
cfg.reward_shaping_factor = reward_shaping_factor
cfg.save_dir = save_dir
cfg.eval_interval = eval_interval
cfg.dashboard = no_dashboard

envs = [make_env(cfg.layout, cfg.fov_radius, noise_cfg) for _ in range(cfg.num_envs)]

_obs = envs[0].reset()
global_grid, global_scalars = envs[0].get_global_state()
C, H, W = global_grid.shape
D = 2 * cfg.fov_radius + 1

global_step = 0

actors = {
    a: Actor(NUM_CHANNELS, cfg.hidden_dim,
        cfg.action_dim, cfg.width_mult).to(device)
        for a in agents
    }
critic = Critic(C, len(agents), 256, cfg.width_mult).to(device)

actor_params = [p for a in actors.values() for p in a.parameters()]
actor_optimizer = torch.optim.Adam(actor_params, lr=cfg.actor_lr,  eps=1e-5)
critic_optimizer = torch.optim.Adam(critic.parameters(),  lr=cfg.critic_lr, eps=1e-5)

total_updates = cfg.total_steps // (cfg.rollout_steps * cfg.num_envs)
actor_scheduler = torch.optim.lr_scheduler.LinearLR(
    actor_optimizer, start_factor=1.0, end_factor=0.05, total_iters=total_updates
)

critic_scheduler = torch.optim.lr_scheduler.LinearLR(
    critic_optimizer, start_factor=1.0, end_factor=0.05, total_iters=total_updates
)

buffer = RolloutBuffer()
episode_returns = collections.deque(maxlen=100)

update_count = 0
dashboard = Dashboard(enabled=cfg.dashboard)
eval_renderer = EvalRenderer(
    layout = cfg.layout,
    actors = actors,
    dashboard = dashboard,
    device = device,
)

os.makedirs(cfg.save_dir, exist_ok=True)

def export_actor(checkpoint_path, agent="player_0", output_path="actor_export.pt"):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["cfg"]
    D = 2 * cfg.fov_radius + 1
    actor = Actor(NUM_CHANNELS, D, cfg.hidden_dim, cfg.action_dim, cfg.width_mult)

    actor.load_state_dict(ckpt["actors"][agent])
    actor.eval()

    torch.save({"model_state_dict": actor.state_dict(), "cfg": cfg}, output_path)
    print(f"Actor exported to {output_path}")

    return actor

def train():
    observation_list = [env.reset() for env in envs]
    hidden = {a: actors[a].init_hidden(cfg.num_envs, device) for a in agents}
    episode_rewards  = np.zeros(cfg.num_envs)

    update_count = 0

    t_start  = time.time()

    global_step = 0

    while global_step < total_steps:
        buffer.clear()

        with torch.no_grad():
            for _ in range(cfg.rollout_steps):
                step_grids, step_scalars = {}, {}
                step_hiddens, step_actions, step_log_prob = {}, {}, {}

                for agent in agents:
                    grid = torch.from_numpy(
                            np.stack([o[agent]["grid"]    for o in observation_list])
                        ).to(device)
                    
                    scalar = torch.from_numpy(
                            np.stack([o[agent]["scalars"] for o in observation_list])
                        ).to(device)
                    
                    _hidden = hidden[agent]

                    action, log_prob, _, new_hidden = actors[agent].get_action_logprob_entropy(grid, scalar, _hidden)

                    step_grids[agent] = grid.cpu()
                    step_scalars[agent] = scalar.cpu()
                    step_hiddens[agent] = _hidden.cpu()  
                    step_actions[agent] = action.cpu()
                    step_log_prob[agent] = log_prob.cpu()
                    hidden[agent] = new_hidden

                global_grid_list, global_scalar_list = [], []

                for env in envs:
                    _global_grid, _global_scalar = env.get_global_state()

                    global_grid_list.append(_global_grid)
                    global_scalar_list.append(_global_scalar)

                global_grid = torch.from_numpy(np.stack(global_grid_list)).float()
                global_scalar = torch.from_numpy(np.stack(global_scalar_list)).float()

                new_observation_list = []
                step_rewards = np.zeros(cfg.num_envs)
                step_shaped = np.zeros(cfg.num_envs)
                step_dones = np.zeros(cfg.num_envs)

                for i, env in enumerate(envs):
                    action_dict = {a: step_actions[a][i].item() for a in agents}
                    observation, reward, done, info = env.step(action_dict)

                    sparse_reward = float(reward) * cfg.sparse_factor

                    shaped = 0.0
                    if isinstance(info, dict):
                        if "shaped_r_by_agent" in info:
                            shaped = sum(info["shaped_r_by_agent"])
                        elif "shaped_r" in info:
                            shaped = float(info["shaped_r"])
                    shaped *= cfg.reward_shaping_factor

                    step_rewards[i] = sparse_reward
                    step_shaped[i] = shaped
                    step_dones[i]  = float(done)
                    episode_rewards[i] += sparse_reward + shaped

                    if done:
                        episode_returns.append(float(episode_rewards[i]))
                        episode_rewards[i] = 0.0
                        observation = env.reset()
                            
                        for agent in agents:
                            hidden[agent][i].zero_()

                    new_observation_list.append(observation)

                observation_list = new_observation_list

                buffer.grids.append(step_grids)
                buffer.scalars.append(step_scalars)
                buffer.hiddens.append(step_hiddens)
                buffer.actions.append(step_actions)
                buffer.log_probs.append(step_log_prob)
                buffer.rewards.append(torch.from_numpy(step_rewards).float())
                buffer.shaped_rewards.append(torch.from_numpy(step_shaped).float())
                buffer.dones.append(torch.from_numpy(step_dones).float())
                buffer.global_grids.append(global_grid)
                buffer.global_scalars.append(global_scalar)
                global_step += cfg.num_envs

        metrics = _ppo_update()
        actor_scheduler.step()
        critic_scheduler.step()
        update_count += 1

        if update_count % cfg.log_interval == 0:
            mean_ret = (np.mean(episode_returns)
                        if episode_returns else float("nan"))
            sps = global_step / (time.time() - t_start)
            actor_lr = actor_optimizer.param_groups[0]["lr"]

            print(
                f"step={global_step:>10,}  "
                f"upd={update_count:>5}  "
                f"ret={mean_ret:>8.2f}  "
                f"aloss={metrics['actor_loss']:>+9.5f}  "
                f"closs={metrics['critic_loss']:>9.6f}  "
                f"ent={metrics['entropy']:>6.4f}  "
                f"lr={actor_lr:.2e}  "
                f"sps={sps:>6.0f}"
            )
            dashboard.update(
                update_count, mean_ret,
                metrics["actor_loss"], metrics["critic_loss"],
                metrics["entropy"], sps,
            )

        if update_count % cfg.save_interval == 0:
            _save_checkpoint()
            dashboard.save(
                os.path.join(cfg.save_dir,
                            f"dashboard_{update_count:06d}.png"))
        if update_count % cfg.eval_interval == 0:
            print(f"  [Eval] running episode at update {update_count}…")
            eval_renderer.run_episode()

    _save_checkpoint(final=True)
    dashboard.save(os.path.join(cfg.save_dir, "dashboard_final.png"))
    print("Training complete.")

def _flatten_buffer(buffer_len):
    flat = {
        "grids": {a: [] for a in agents},
        "scalars": {a: [] for a in agents},
        "hiddens": {a: [] for a in agents},
        "actions": {a: [] for a in agents},
        "log_probs": {a: [] for a in agents},
        "global_grids": [],
        "global_scalars": [],
    }
    
    for t in range(buffer_len):
        for a in agents:
            flat["grids"][a].append(buffer.grids[t][a])
            flat["scalars"][a].append(buffer.scalars[t][a])
            flat["hiddens"][a].append(buffer.hiddens[t][a])
            flat["actions"][a].append(buffer.actions[t][a])
            flat["log_probs"][a].append(buffer.log_probs[t][a])

        flat["global_grids"].append(buffer.global_grids[t])
        flat["global_scalars"].append(buffer.global_scalars[t])

    for a in agents:
        flat["grids"][a] = torch.cat(flat["grids"][a])
        flat["scalars"][a] = torch.cat(flat["scalars"][a])
        flat["hiddens"][a] = torch.cat(flat["hiddens"][a])
        flat["actions"][a] = torch.cat(flat["actions"][a])
        flat["log_probs"][a] = torch.cat(flat["log_probs"][a])

    flat["global_grids"] = torch.cat(flat["global_grids"])
    flat["global_scalars"] = torch.cat(flat["global_scalars"])
    
    return flat

def _ppo_update():
    buffer_len = len(buffer)

    rewards = torch.stack([buffer.rewards[t] + buffer.shaped_rewards[t] for t in range(buffer_len)])
    dones = torch.stack(buffer.dones)

    with torch.no_grad():
        values = []
        for t in range(buffer_len):
            value = critic(buffer.global_grids[t].to(device), buffer.global_scalars[t].to(device)).cpu()

            values.append(value)
        
        value_bootstrap = critic(buffer.global_grids[-1].to(device), buffer.global_scalars[-1].to(device)).cpu()
        values.append(value_bootstrap)

    values_t = torch.stack(values)

    advantage, returns = compute_gae(rewards, values_t, dones, cfg.gamma, cfg.gae_lambda)

    advantage_flat = advantage.reshape(-1)
    advantage_flat = (advantage_flat - advantage_flat.mean()) / (advantage_flat.std() + 1e-8)
    return_flat = returns.reshape(-1)

    flat = _flatten_buffer(buffer_len)

    total_actor_loss = total_critic_loss = total_entropy = 0.0
    n_update = 0
    indices = np.arange(buffer_len * num_envs)

    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(indices)

        for start in range(0, buffer_len * num_envs, cfg.minibatch_size):
            minibatch = indices[start:start + cfg.minibatch_size]
            minibatch_advantage = advantage_flat[minibatch].to(device)
            minibatch_return = return_flat[minibatch].to(device)

            actor_loss = torch.tensor(0.0, device=device)
            minibatch_entropy = torch.tensor(0.0, device=device)

            for agent in agents:
                grid = flat['grids'][agent][minibatch].to(device)
                scalar = flat['scalars'][agent][minibatch].to(device)
                hidden = flat["hiddens"][agent][minibatch].to(device).detach()
                action = flat["actions"][agent][minibatch].to(device)
                log_prob_old = flat["log_probs"][agent][minibatch].to(device)

                _, log_prob_new, entropy, _ = actors[agent].get_action_logprob_entropy(grid, scalar, hidden, action)

                ratio = (log_prob_new - log_prob_old).exp()
                surrogate1 = ratio * minibatch_advantage
                surrogate2 = ratio.clamp(1 - cfg.clip_eps)

                pg = -torch.min(surrogate1, surrogate2).mean()

                actor_loss += pg - cfg.ent_coef * entropy.mean()
                minibatch_entropy += entropy.mean()

            actor_loss /= len(agents)
            minibatch_entropy /= len(agents)

            value_pred = critic(flat["global_grids"][minibatch].to(device), flat["global_scalars"][minibatch].to(device))

            critic_loss = cfg.vf_coef * F.mse_loss(value_pred, minibatch_return)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_([p for a in actors.values() for p in a.parameters()], cfg.max_grad_norm)
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += minibatch_entropy.item()
            n_update += 1

    return {
        "actor_loss":  total_actor_loss / n_update,
        "critic_loss": total_critic_loss / n_update,
        "entropy":     total_entropy / n_update,
    }

def _save_checkpoint(final=False):
    tag  = "final" if final else f"step{global_step:010d}"
    path = os.path.join(cfg.save_dir, f"mappo_{tag}.pt")
    
    torch.save({
        "global_step": global_step,
        "update_count": update_count,
        "cfg": cfg,
        "actors": {a: m.state_dict() for a, m in actors.items()},
        "critic": critic.state_dict(),
        "actor_opt": actor_optimizer.state_dict(),
        "critic_opt": critic_optimizer.state_dict(),
    }, path)

    print(f"  Checkpoint saved: {path}")

if __name__ == "__main__":
    train()