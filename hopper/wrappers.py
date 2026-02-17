# project_extension/extension/wrappers.py
"""
This module defines a collection of modular Gymnasium wrappers used to
augment MuJoCo-based locomotion environments with robustness, safety,
and goal-conditioned task structure.

The wrappers implement three complementary axes of the project design:

1. Robustness and degradation modeling
   - ActuatorStrengthWrapper applies train-time actuator domain
     randomization by globally scaling control actions.
   - FaultInjectionWrapper injects evaluation-time hard actuator faults
     affecting selected joints after a configurable time step.
   - GroundFrictionDRWrapper performs domain randomization on ground
     friction parameters at episode reset.

2. Goal-conditioned task formulation
   - GoalConditionedWrapper augments observations with goal information,
     optionally overrides the native environment reward with a shaped
     goal-reaching objective, and supports early termination on success.
   - The wrapper is designed to preserve a consistent observation space
     across training and evaluation modes.

3. Safety-aware control
   - SafetyConstraintWrapper enforces soft or hard safety constraints
     based on proxy signals (e.g., torso pitch and vertical speed),
     applying penalties and optional termination on violation.
   - SafetyActionL2PenaltyWrapper provides an additional train-time
     regularization term to discourage excessive control effort.

Together, these wrappers enable systematic study of fault tolerance,
safety constraints, and goal-directed behavior in continuous-control
locomotion tasks, while maintaining modularity, reproducibility, and
clear separation between training-time randomization and evaluation-time
perturbations.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from .configs import FaultSpec, FrictionDRSpec, GoalSpec, RewardSpec, SafetySpec


def _get_qpos_qvel(env: gym.Env):
    """
    Helper to access MuJoCo state. Works with Gymnasium MuJoCo envs that expose env.unwrapped.data.
    """
    u = env.unwrapped
    data = getattr(u, "data", None)
    if data is None:
        raise RuntimeError("MuJoCo data not found. Ensure env is a MuJoCo-based environment.")
    qpos = np.array(data.qpos, dtype=np.float32).copy()
    qvel = np.array(data.qvel, dtype=np.float32).copy()
    return qpos, qvel


class ActuatorStrengthWrapper(gym.Wrapper):
    """
    Train-time actuator Domain Randomization (GLOBAL gain).

    Samples a single gain g per episode and scales the whole action vector:
        a_applied = g * a_policy

    If dr_range = 0.30:
        g ~ Uniform(0.7, 1.3)
    """

    def __init__(self, env: gym.Env, dr_range: float, seed: Optional[int] = None):
        super().__init__(env)
        self.dr_range = float(dr_range)
        self._rng = np.random.default_rng(seed)
        self._gain = 1.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.dr_range > 0.0:
            self._gain = float(self._rng.uniform(1.0 - self.dr_range, 1.0 + self.dr_range))
        else:
            self._gain = 1.0
        info = dict(info)
        info["actuator_dr_gain"] = float(self._gain)
        info["actuator_dr_range"] = float(self.dr_range)
        return obs, info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32) * float(self._gain)
        return self.env.step(a)


class FaultInjectionWrapper(gym.Wrapper):
    """
    Eval-time hard fault injection on selected action indices after start_step.

    strength scales specified action components:
        a[idx] <- strength * a[idx]
    """

    def __init__(
        self,
        env: gym.Env,
        strength: float,
        actuator_indices: Tuple[int, ...],
        start_step: Optional[int] = 0,
    ):
        super().__init__(env)
        self.strength = float(strength)
        self.actuator_indices = tuple(int(i) for i in actuator_indices)
        self.start_step = None if start_step is None else int(start_step)
        self._t = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._t = 0
        info = dict(info)
        info["fault_strength"] = self.strength
        info["fault_indices"] = self.actuator_indices
        info["fault_start_step"] = self.start_step
        return obs, info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).copy()
        if self.start_step is None or self._t >= self.start_step:
            for idx in self.actuator_indices:
                if 0 <= idx < a.shape[0]:
                    a[idx] *= self.strength
        obs, reward, terminated, truncated, info = self.env.step(a)
        self._t += 1
        info = dict(info)
        info["fault_active"] = bool(self.start_step is None or (self._t - 1) >= self.start_step)
        return obs, reward, terminated, truncated, info


class GroundFrictionDRWrapper(gym.Wrapper):
    """
    Domain Randomization on floor friction (MuJoCo geom friction).
    Scales the friction parameters for specified geom_names at each reset.
    """

    def __init__(self, env: gym.Env, spec: FrictionDRSpec, seed: Optional[int] = None):
        super().__init__(env)
        self.dr_spec = spec
        self._rng = np.random.default_rng(seed)
        self._base_friction = None

    def _get_geom_ids(self):
        u = self.env.unwrapped
        model = getattr(u, "model", None)
        if model is None:
            raise RuntimeError("MuJoCo model not found.")
        geom_ids = []
        for name in self.dr_spec.geom_names:
            try:
                gid = model.geom_name2id(name)
                geom_ids.append(gid)
            except Exception:
                pass
        return geom_ids

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        u = self.env.unwrapped
        model = getattr(u, "model", None)
        if model is None:
            return obs, info

        if self._base_friction is None:
            self._base_friction = np.array(model.geom_friction, dtype=np.float32).copy()

        mult = float(self._rng.uniform(1.0 - self.dr_spec.range, 1.0 + self.dr_spec.range))
        geom_ids = self._get_geom_ids()
        for gid in geom_ids:
            model.geom_friction[gid, :] = self._base_friction[gid, :] * mult

        info = dict(info)
        info["friction_dr_mult"] = mult
        info["friction_dr_range"] = float(self.dr_spec.range)
        info["friction_dr_geom_names"] = tuple(self.dr_spec.geom_names)
        return obs, info


class GoalConditionedWrapper(gym.Wrapper):
    """
    Goal-conditioned wrapper for 2D targets.

    - Augments observation with: [goal_x, goal_y, dist_to_goal]
    - Optionally overrides the environment reward with a shaped goal-reaching reward
    - Optionally terminates the episode when the goal is reached
    """

    def __init__(
        self,
        env: gym.Env,
        goals: Optional[Tuple[GoalSpec, ...]],
        goal_x_min: float,
        goal_x_max: float,
        goal_y: float,
        goal_tolerance: float,
        reward_spec: RewardSpec,
        seed: Optional[int] = None,
        *,
        override_reward: bool = True,
        terminate_on_success: bool = True,
    ):
        super().__init__(env)
        self.goals = goals
        self.goal_x_min = float(goal_x_min)
        self.goal_x_max = float(goal_x_max)
        self.goal_y = float(goal_y)
        self.goal_tolerance = float(goal_tolerance)
        self.rspec = reward_spec
        self._rng = np.random.default_rng(seed)

        self.override_reward = bool(override_reward)
        self.terminate_on_success = bool(terminate_on_success)

        assert isinstance(env.observation_space, gym.spaces.Box), "This wrapper expects Box observations."
        low = np.concatenate([env.observation_space.low, np.array([-np.inf, -np.inf, 0.0], dtype=np.float32)])
        high = np.concatenate([env.observation_space.high, np.array([np.inf, np.inf, np.inf], dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32)

        self._goal = np.zeros(2, dtype=np.float32)
        self._prev_dist: Optional[float] = None
        self._prev_action: Optional[np.ndarray] = None

    def _sample_goal(self) -> np.ndarray:
        if self.goals is not None and len(self.goals) > 0:
            g = self.goals[int(self._rng.integers(0, len(self.goals)))]
            return np.array([float(g.x), float(g.y)], dtype=np.float32)
        x = float(self._rng.uniform(self.goal_x_min, self.goal_x_max))
        y = float(self.goal_y)
        return np.array([x, y], dtype=np.float32)

    def _agent_xy(self) -> np.ndarray:
        qpos, _ = _get_qpos_qvel(self.env)
        return np.array([float(qpos[0]), 0.0], dtype=np.float32)

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        dist = float(np.linalg.norm(self._agent_xy() - self._goal))
        return np.concatenate([np.asarray(obs, dtype=np.float32), self._goal, np.array([dist], dtype=np.float32)])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._goal = self._sample_goal()
        self._prev_dist = float(np.linalg.norm(self._agent_xy() - self._goal))
        self._prev_action = None

        info = dict(info)
        info["goal_x"] = float(self._goal[0])
        info["goal_y"] = float(self._goal[1])
        info["goal_tol"] = float(self.goal_tolerance)
        info["goal_override_reward"] = bool(self.override_reward)
        info["goal_terminate_on_success"] = bool(self.terminate_on_success)

        obs_aug = self._augment_obs(obs)
        # hard assert: NEVER return un-augmented obs
        assert obs_aug.shape == self.observation_space.shape, (obs_aug.shape, self.observation_space.shape)
        return obs_aug, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        # metrics
        agent_xy = self._agent_xy()
        dist = float(np.linalg.norm(agent_xy - self._goal))

        if self._prev_dist is None:
            self._prev_dist = dist
        progress = float(self._prev_dist - dist)
        self._prev_dist = dist

        a = np.asarray(action, dtype=np.float32).reshape(-1)
        a_l2 = float(np.sum(a * a))

        achange_l2 = 0.0
        if self._prev_action is not None:
            da = a - self._prev_action
            achange_l2 = float(np.sum(da * da))
        self._prev_action = a.copy()

        is_success = bool(dist <= self.goal_tolerance)

        info = dict(info)
        info["goal_dist"] = dist
        info["goal_progress"] = progress
        info["penalty_action_l2"] = float(self.rspec.w_action_l2 * a_l2)
        info["penalty_action_change_l2"] = float(self.rspec.w_action_change_l2 * achange_l2)
        info["is_success"] = is_success

        # choose reward (but NEVER change obs shape)
        reward_out = float(env_reward)

        if self.override_reward:
            # distance clipping
            dist_cap = 3.5
            dist_eff = min(dist, dist_cap)

            # near-goal dense bonus
            near_radius = 2.0
            near_coeff = 3.0
            near_bonus = 0.0
            if dist < near_radius:
                near_bonus = near_coeff * (1.0 - dist / near_radius)

            shaped = (
                float(self.rspec.alive_bonus)
                + float(self.rspec.w_progress) * progress
                - float(self.rspec.w_distance) * dist_eff
                + float(near_bonus)
                - float(self.rspec.w_action_l2) * a_l2
                - float(self.rspec.w_action_change_l2) * achange_l2
            )

            # fall penalty only on TERMINATED (not timeout)
            if terminated and (not is_success):
                shaped -= 10.0

            if is_success:
                shaped += float(self.rspec.success_bonus)
                info["terminated_on_success"] = bool(self.terminate_on_success)
                if self.terminate_on_success:
                    terminated = True
            else:
                info["terminated_on_success"] = False

            info["dist_eff"] = float(dist_eff)
            info["near_bonus"] = float(near_bonus)

            reward_out = float(shaped)
        else:
            info["terminated_on_success"] = False

        obs_aug = self._augment_obs(obs)
        # hard assert: NEVER return un-augmented obs
        assert obs_aug.shape == self.observation_space.shape, (obs_aug.shape, self.observation_space.shape)

        return obs_aug, reward_out, bool(terminated), bool(truncated), info


class SafetyConstraintWrapper(gym.Wrapper):
    """
    Applies safety penalties (and optional termination) based on simple proxy signals.

    For Hopper we use:
      - torso_pitch ~ qpos[2]
      - vertical_speed ~ abs(qvel[1])
    """

    def __init__(self, env: gym.Env, spec: SafetySpec):
        super().__init__(env)
        self.safety_spec = spec

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        qpos, qvel = _get_qpos_qvel(self.env)
        torso_pitch = float(qpos[2]) if qpos.shape[0] > 2 else 0.0
        vertical_speed = float(abs(qvel[1])) if qvel.shape[0] > 1 else 0.0

        pitch_limit = float(self.safety_spec.max_torso_pitch_rad)
        vspeed_limit = float(self.safety_spec.max_vertical_speed)

        pitch_excess = max(0.0, abs(torso_pitch) - pitch_limit)
        vspeed_excess = max(0.0, abs(vertical_speed) - vspeed_limit)

        v_pitch = pitch_excess > 0.0
        v_vspeed = vspeed_excess > 0.0
        violated = bool(v_pitch or v_vspeed)

        penalty = 0.0
        if v_pitch:
            penalty += float(self.safety_spec.penalty_torso) * float(pitch_excess)
        if v_vspeed:
            penalty += float(self.safety_spec.penalty_vertical_speed) * float(vspeed_excess)

        reward = float(reward) - float(penalty)

        pitch_margin = float(pitch_limit - abs(torso_pitch))
        vspeed_margin = float(vspeed_limit - abs(vertical_speed))

        info = dict(info)
        info["torso_pitch"] = torso_pitch
        info["vertical_speed"] = vertical_speed
        info["safety_violation"] = violated
        info["violate_torso_pitch"] = bool(v_pitch)
        info["violate_vertical_speed"] = bool(v_vspeed)
        info["safety_penalty"] = float(penalty)
        info["safety_spec"] = asdict(self.safety_spec)

        # margins + termination reason
        info["pitch_margin"] = pitch_margin
        info["vspeed_margin"] = vspeed_margin

        terminated_on_violation = bool(violated and self.safety_spec.terminate_on_violation)
        info["terminated_on_violation"] = terminated_on_violation

        if terminated_on_violation:
            terminated = True

        return obs, float(reward), bool(terminated), bool(truncated), info


class SafetyActionL2PenaltyWrapper(gym.Wrapper):
    """
    Optional action L2 penalty wrapper (train only).
    """

    def __init__(self, env: gym.Env, lam: float):
        super().__init__(env)
        self.lam = float(lam)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        penalty = self.lam * float(np.sum(a * a))
        reward = float(reward) - float(penalty)
        info = dict(info)
        info["action_l2_penalty"] = float(penalty)
        info["action_l2_lambda"] = float(self.lam)
        return obs, float(reward), bool(terminated), bool(truncated), info
