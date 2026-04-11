import numpy as np
import torch
from utils import NUM_CHANNELS
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.actions import Action
import time

import pygame

class EvalRenderer:
    def __init__(
        self,
        layout,
        actors,
        dashboard,
        device,
        tile_size = 80,
        fps_delay = 0.05,
    ):
        self.actors = actors
        self.dashboard = dashboard
        self.device = device
        self.fps_delay = fps_delay
        self._ready = False

        try:
            mdp = OvercookedGridworld.from_layout_name(layout)
            self.env = OvercookedEnv.from_mdp(mdp, horizon=400)
            self.mdp = mdp

            self.viz = StateVisualizer(
                tile_size=tile_size,
                is_rendering_hud=True,
                is_rendering_cooking_timer=True,
                is_rendering_action_probs=False,
            )
            self.grid = mdp.terrain_mtx

            import os
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            pygame.display.init()
            pygame.display.set_mode((1, 1))  

            self._pygame  = pygame
            self._ready   = True

        except Exception as exc:
            print(f"[EvalRenderer] disabled — {exc}")

    def _state_to_frame(self, state):
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
        arr = self._pygame.surfarray.array3d(surface) 
        return arr.transpose(1, 0, 2)                

    def _obs_from_state(self, agent_idx: int):
        state   = self.env.state
        player  = state.players[agent_idx]
        x, y    = player.position   
        pos     = (y, x)                   

        terrain = self.mdp.terrain_mtx
        H, W = len(terrain), len(terrain[0])
        C = NUM_CHANNELS
        full_grid = np.zeros((C, H, W), dtype=np.float32)

        for r, row in enumerate(terrain):
            for c, cell in enumerate(row):
                if cell != ' ':
                    full_grid[0, r, c] = 1.0 
                else:
                    full_grid[1, r, c] = 1.0  

        for i, p in enumerate(state.players):
            px, py = p.position
            ch = 2 if i == agent_idx else 3  
            full_grid[ch, py, px] = 1.0

        _CH = {"onion": 4, "tomato": 5, "dish": 6, "soup": 7}
        for (ox, oy), obj in state.objects.items():
            ch = _CH.get(obj.name)
            if ch is not None:
                full_grid[ch, oy, ox] = 1.0

        for px, py in self.mdp.get_pot_locations():
            full_grid[8, py, px] = 1.0
        for px, py in self.mdp.get_serving_locations():
            full_grid[9, py, px] = 1.0

        R = 5
        D = 2 * R + 1
        pad = R
        padded = np.zeros((C, H + 2*pad, W + 2*pad), dtype=np.float32)
        padded[0, :, :] = 1.0              
        padded[:, pad:pad+H, pad:pad+W] = full_grid
        pr, pc = pos[0] + pad, pos[1] + pad
        crop   = padded[:, pr-R:pr+R+1, pc-R:pc+R+1].copy()

        scalars = np.zeros(7, dtype=np.float32)
        # held object
        obj = player.held_object
        if obj is not None:
            held_map = {"onion": 1, "tomato": 2, "dish": 3, "soup": 4}
            scalars[2 + held_map.get(obj.name, 0)] = 1.0
        else:
            scalars[2] = 1.0  

        return {"grid": crop, "scalars": scalars}

    def run_episode(self):
        if not self._ready:
            return

        for a in self.actors.values():
            a.eval()

        self.env.reset()
        hiddens = {
            a: self.actors[a].init_hidden(1, self.device)
            for a in self.actors
        }
        ep_return = 0.0
        step = 0

        try:
            while not self.env.is_done():
                frame = self._state_to_frame(self.env.state)
                self.dashboard.update_render(frame, step, ep_return)

                joint = []
                for i, agent_id in enumerate(["player_0", "player_1"]):
                    obs = self._obs_from_state(i)
                    g = torch.from_numpy(obs["grid"]).unsqueeze(0).to(self.device)
                    s = torch.from_numpy(obs["scalars"]).unsqueeze(0).to(self.device)
                    h = hiddens[agent_id]

                    with torch.no_grad():
                        logits, new_h = self.actors[agent_id](g, s, h)
                    hiddens[agent_id] = new_h

                    action_idx = logits.argmax(dim=-1).item()
                    joint.append(Action.INDEX_TO_ACTION[action_idx])

                _, reward, _, _ = self.env.step(tuple(joint))
                ep_return += float(reward)
                step      += 1

                if self.fps_delay > 0:
                    time.sleep(self.fps_delay)

        except Exception as exc:
            print(f"[EvalRenderer] episode error: {exc}")

        finally:
            for a in self.actors.values():
                a.train()

        try:
            frame = self._state_to_frame(self.env.state)
            self.dashboard.update_render(frame, step, ep_return)
        except Exception:
            pass

        print(f"  [Eval] episode return={ep_return:.1f}  steps={step}")
