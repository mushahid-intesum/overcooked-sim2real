from utils import *
from _dataclasses import NoiseCfg
import numpy as np
import collections
import dataclasses
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

def make_env(layout, fov_radius, noise_cfg):
    mdp  = OvercookedGridworld.from_layout_name(layout)
    base = OvercookedEnv.from_mdp(mdp, horizon=400)

    return PartialObsWrapper(base, fov_radius=fov_radius, noise_cfg=noise_cfg)

class PartialObsWrapper:
    def __init__(self, env, fov_radius, noise_cfg):
        self.env = env
        self.radius = fov_radius
        self.diameter = 2 * fov_radius + 1

        self.noise_cfg = noise_cfg or NoiseCfg()

        self.cone_mask = build_cone_mask(fov_radius)

        self.prev_pos = {}
        self.delay_buffers = {}
        self.delay_steps = {}
        self.active_noise_cfg = self.noise_cfg

        self.grid_h = 0
        self.grid_w = 0

        self.agents = [f"player_{i}" for i in range(self.env.mdp.num_players)]    

    def reset(self):
        self.env.reset()                        
        self.active_noise_cfg = self._sample_noise_cfg()
        state = self._get_state()
        self.grid_h, self.grid_w = state["H"], state["W"]

        for agent in self.agents:
            self.prev_pos[agent] = self._agent_pos(state, agent)
            delay = (np.random.randint(
                self.active_noise_cfg.delay_min_steps,
                self.active_noise_cfg.delay_max_steps + 1,
            ) if self.active_noise_cfg.delay_enabled else 0)

            self.delay_steps[agent] = delay

            zero_obs = {
                "grid":    np.zeros((NUM_CHANNELS, self.diameter, self.diameter), dtype=np.float32),
                "scalars": np.zeros(7, dtype=np.float32),
            }

            self.delay_buffers[agent] = collections.deque(
                [zero_obs] * (delay + 1), maxlen=delay + 1
            )

        return self._build_all_partial_obs(state)
    

    def step(self, action_dict):
        joint_action = tuple(
            Action.INDEX_TO_ACTION[action_dict[f"player_{i}"]]
            for i in range(self.env.mdp.num_players)
        )

        _, reward, done, info = self.env.step(joint_action)
        state = self._get_state()

        for agent in self.agents:
            self.prev_pos[agent] = self._agent_pos(state, agent)

        partial_obs = self._build_all_partial_obs(state)

        return partial_obs, reward, done, info
    

    def close(self):
        self.env.close()


    def get_global_state(self):
        state  = self._get_state()
        global_grid = self._build_global_grid(state)   # (C, H, W)
        scalar_parts = []

        for agent in self.agents:
            pos = self._agent_pos(state, agent)
            prev = self.prev_pos.get(agent, pos)
            held = self._held_onehot(state, agent)
            vel = np.array([pos[0] - prev[0], pos[1] - prev[1]], dtype=np.float32)
            scalar_parts.append(np.concatenate([vel, held]))

        scalars = np.concatenate(scalar_parts)

        return global_grid.astype(np.float32), scalars
    

    def _get_state(self):
        raw_state = self.env.state
        mdp = self.env.mdp

        terrain_mtx = mdp.terrain_mtx

        H = len(terrain_mtx)
        W = len(terrain_mtx[0])

        terrain = np.array(
            [[0 if cell == ' ' else 1 for cell in row] for row in terrain_mtx],
            dtype=np.int32,
        )

        agent_pos, agent_dir = {}, {}
        for i, player in enumerate(raw_state.players):
            x, y = player.position   
            agent_pos[f"player_{i}"] = (y, x) 

        objects = []
        for (x, y), obj in raw_state.objects.items():
            objects.append({"type": obj.name, "pos": (y, x)})  

        pot_locs = [(y, x) for x, y in mdp.get_pot_locations()]
        serve_locs = [(y, x) for x, y in mdp.get_serving_locations()]

        return {
            "terrain": terrain, "agent_pos": agent_pos, "agent_dir": agent_dir,
            "objects": objects, "pot_locs": pot_locs, "serve_locs": serve_locs,
            "delivery_locs": [], 
            "H": H, "W": W,
        }


    def _build_global_grid(self, state):
        """Build the full (C, H, W) symbolic grid (no masking, no noise)."""
        H, W = state["H"], state["W"]
        grid = np.zeros((NUM_CHANNELS, H, W), dtype=np.float32)

        grid[CHANNELS["wall"]]  = (state["terrain"] == 1).astype(np.float32)
        grid[CHANNELS["floor"]] = (state["terrain"] == 0).astype(np.float32)

        for i, agent in enumerate(self.agents):
            pos = state["agent_pos"].get(agent)
            if pos is not None:
                ch = CHANNELS["self"] if i == 0 else CHANNELS["teammate"]
                grid[ch, pos[0], pos[1]] = 1.0

        # Dynamic objects
        _OBJ_CH = {
            "onion": CHANNELS["onion"],
            "tomato": CHANNELS["tomato"],
            "dish": CHANNELS["dish"],
            "soup": CHANNELS["soup"],
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


    def _build_all_partial_obs(self, state):
        global_grid = self._build_global_grid(state)
        result = {}

        for i, agent in enumerate(self.agents):
            pos = state["agent_pos"].get(agent)
            dir_ = state["agent_dir"].get(agent, 0)
            prev = self.prev_pos.get(agent, pos)

            if pos is None:
                result[agent] = {
                    "grid": np.zeros((NUM_CHANNELS, self.diameter, self.diameter), dtype=np.float32),
                    "scalars": np.zeros(7, dtype=np.float32),
                }
                continue

            crop = self._crop_global(global_grid, pos, state)

            ego_crop = rotate_crop_to_ego(crop, dir_)

            masked = ego_crop * self.cone_mask[np.newaxis, :, :]

            cfg = self.active_noise_cfg
            if cfg.dropout_enabled:
                masked = self._apply_dropout(masked, cfg.dropout_p)

            if cfg.obj_miss_enabled:
                masked = self._apply_obj_miss(masked, cfg.obj_miss_p)

            vel  = np.array(
                [pos[0] - prev[0], pos[1] - prev[1]], dtype=np.float32
            )

            held = self._held_onehot(state, agent)
            scalars = np.concatenate([vel, held]).astype(np.float32)

            raw_obs = {"grid": masked, "scalars": scalars}

            if cfg.delay_enabled and self.delay_steps[agent] > 0:
                buf = self.delay_buffers[agent]
                buf.append(raw_obs)
                delayed_obs = buf[0]  
            else:
                delayed_obs = raw_obs

            result[agent] = delayed_obs

        return result
    

    def _crop_global(self, global_grid, agent_pos, state):
        C, H, W = global_grid.shape
        R = self.radius
        D = self.diameter
        r0, c0 = agent_pos

        pad = R
        padded = np.zeros((C, H + 2 * pad, W + 2 * pad), dtype=np.float32)
        padded[CHANNELS["wall"], :, :] = 1.0 
        padded[:, pad:pad + H, pad:pad + W] = global_grid

        pr, pc = r0 + pad, c0 + pad
        crop = padded[:, pr - R:pr + R + 1, pc - R:pc + R + 1]

        assert crop.shape == (C, D, D), f"Crop shape mismatch: {crop.shape}"

        return crop.copy()

    def _apply_dropout(self, grid, p):
        C, H, W = grid.shape
        mask = np.random.random((H, W)) > p

        return grid * mask[np.newaxis, :, :]

    def _apply_obj_miss(self, grid, p):
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

    def _agent_pos(self, state, agent):
        return state["agent_pos"].get(agent, (0, 0))

    def _held_onehot(self, state, agent):
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

    def _sample_noise_cfg(self):
        cfg = dataclasses.replace(self.noise_cfg)

        if not cfg.domain_randomise:
            return cfg
        
        rng = np.random.default_rng()

        cfg.dropout_p = float(
            rng.uniform(*cfg.dr_dropout_p_range)
        )
        cfg.obj_miss_p = float(
            rng.uniform(*cfg.dr_obj_miss_p_range)
        )

        return cfg

if __name__ == "__main__":
    print("Running cone mask smoke test...")
    mask = build_cone_mask(fov_radius=5)
    D = 11
    print(f"Cone mask shape: {mask.shape}")
    print(f"Visible cells: {mask.sum()} / {D*D}")
    print("Mask (1=visible, 0=blind):")
    for row in mask:
        print("  " + "".join("█" if v else "░" for v in row))

    print("\nCone mask at radius=3:")
    m3 = build_cone_mask(3)
    for row in m3:
        print("  " + "".join("█" if v else "░" for v in row))

    print("\nNoiseCfg defaults:")
    cfg = NoiseCfg()
    for f in dataclasses.fields(cfg):
        print(f"  {f.name}: {getattr(cfg, f.name)}")

    print("\nSmoke test passed. Import and wrap your Overcooked env to use.")
