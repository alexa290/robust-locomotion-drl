"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.stats import truncnorm

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class CustomHopper(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
        "render_fps": 125,
    }

    def __init__(
        self,
        xml_file: str = "hopper.xml",
        frame_skip: int = 4,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.7, float("inf")),
        healthy_angle_range: Tuple[float, float] = (-0.2, 0.2),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        domain: Optional[str] = None,

        # Task 4: UDR options
        enable_udr: bool = False,
        # udr_range can be:
        # - float p      -> +-p relative range (e.g., 0.10 means [0.9*m, 1.1*m])
        # - (low, high)  -> multipliers (e.g., (0.8, 1.2))
        # - (p1, p2, p3) -> per-link +-p for the 3 non-torso links
        udr_range: Union[float, Tuple[float, float], Tuple[float, float, float], List[float]] = 0.10,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            domain,
            enable_udr,
            udr_range,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if xml_file == "hopper.xml":
             xml_file = os.path.join(os.path.dirname(__file__), "assets/hopper.xml")

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

        # Default link masses (excluding world). This includes torso + 3 remaining links.
        self.original_masses = np.copy(self.model.body_mass[1:])

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.model.body_mass[1] -= 1.0

        # Task 4: store UDR config
        self.enable_udr = enable_udr
        self.udr_range = udr_range


    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = np.clip(self.data.qvel.flatten(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": x_position_after,
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        # Task 4: UDR each episode
        if self.enable_udr:
            self.set_random_parameters()

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }

    def set_random_parameters(self):
        """Set random masses (UDR, Task 4)."""
        new_masses = self.sample_parameters()
        self.set_parameters(new_masses)

    def sample_parameters(self):
        """Sample masses according to a uniform domain randomization distribution.
        N.B. You can't change the mass of the torso (first link in body_mass[1:]).
        We randomize the remaining 3 link masses uniformly.
        """
        # Current masses of links (excluding world): [torso, link2, link3, link4]
        current = np.array(self.model.body_mass[1:], dtype=np.float64)

        # Keep torso fixed (already shifted in SOURCE if domain == 'source')
        torso_mass = current[0]

        # Nominal masses for the other 3 links (use original_masses as baseline)
        base = np.array(self.original_masses, dtype=np.float64)
        base_other = base[1:]  # 3 elements

        u = self.udr_range

        # Build bounds for uniform sampling
        if isinstance(u, (list, tuple, np.ndarray)) and len(u) == 2 and all(isinstance(x, (int, float)) for x in u):
            # (low_mult, high_mult)
            low_mult, high_mult = float(u[0]), float(u[1])
            low = base_other * low_mult
            high = base_other * high_mult

        elif isinstance(u, (list, tuple, np.ndarray)) and len(u) == 3:
            # per-link +-p
            p = np.array(u, dtype=np.float64)
            low = base_other * (1.0 - p)
            high = base_other * (1.0 + p)

        else:
            # single float +-p
            p = float(u)
            low = base_other * (1.0 - p)
            high = base_other * (1.0 + p)

        sampled_other = self.np_random.uniform(low=low, high=high)

        # Recompose full vector [torso, other3]
        new_masses = np.concatenate([[torso_mass], sampled_other]).astype(np.float64)
        return new_masses

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array(self.model.body_mass[1:])
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.model.body_mass[1:] = task

""" Registered environments """
gym.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)