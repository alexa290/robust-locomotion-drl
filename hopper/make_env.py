# project_extension/extension/make_env.py
"""
This module defines a unified environment factory used across training and
evaluation phases of the project.

Given an experiment configuration, a domain (source or target), and an
execution mode, it constructs a Gymnasium environment by composing a
sequence of task, robustness, and safety wrappers in a controlled and
reproducible manner.

The factory supports multiple execution modes, including training, nominal
evaluation, faulted evaluation, goal-conditioned evaluation, and
goal-conditioned evaluation under actuator faults. Each mode selectively
enables or disables domain randomization, safety constraints, reward
overrides, termination conditions, and fault injection.

Key design principles include:
- Consistent observation space across all modes by always applying the
  goal-conditioned wrapper.
- Strict separation between training-time randomization and evaluation-time
  perturbations.
- Explicit control over safety constraints and reward shaping.
- Deterministic seeding applied after all wrappers to avoid observation
  inconsistencies.

This centralized construction logic ensures experimental consistency,
prevents configuration drift across phases, and enables systematic
benchmarking of robustness and safety in sim-to-real and fault-tolerant
locomotion scenarios.
"""
from __future__ import annotations

import gymnasium as gym

# IMPORTANT: registers CustomHopper-source-v0 and CustomHopper-target-v0
from env.custom_hopper import *  # noqa: F401,F403

from .configs import ExperimentConfig
from .wrappers import (
    ActuatorStrengthWrapper,
    FaultInjectionWrapper,
    SafetyActionL2PenaltyWrapper,
    GroundFrictionDRWrapper,
    GoalConditionedWrapper,
    SafetyConstraintWrapper,
)


def make_env(cfg: ExperimentConfig, domain: str, mode: str, seed: int):
    """
    domain: "source" or "target"
    mode:   "train" | "eval" | "eval_fault" | "eval_goal" | "eval_fault_goal"
    """
    assert domain in ("source", "target")
    assert mode in ("train", "eval", "eval_fault", "eval_goal", "eval_fault_goal")

    env_id = cfg.source_env_id if domain == "source" else cfg.target_env_id
    env = gym.make(env_id)

    # optional time limit override
    if cfg.max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=int(cfg.max_episode_steps))

    # Evaluation-mode switches
    if mode in ("eval", "eval_fault"):
        use_friction_dr = False
        use_safety = False
        goal_override_reward = False
        goal_terminate_on_success = False

    elif mode in ("eval_goal", "eval_fault_goal"):
        use_friction_dr = False
        use_safety = False
        goal_override_reward = True
        goal_terminate_on_success = True

    else:  # mode == "train"
        use_friction_dr = cfg.friction_dr is not None and getattr(cfg.friction_dr, "range", 0.0) > 0.0
        use_safety = True
        goal_override_reward = True
        goal_terminate_on_success = True

    # P2: friction DR
    if use_friction_dr:
        env = GroundFrictionDRWrapper(env, spec=cfg.friction_dr, seed=seed + 11)

    # P2: goal-conditioned wrapper (ALWAYS ON)
    env = GoalConditionedWrapper(
        env,
        goals=cfg.goals,
        goal_x_min=cfg.goal_x_min,
        goal_x_max=cfg.goal_x_max,
        goal_y=cfg.goal_y,
        goal_tolerance=cfg.goal_tolerance,
        reward_spec=cfg.reward,
        seed=seed + 22,
        override_reward=goal_override_reward,
        terminate_on_success=goal_terminate_on_success,
    )

    # P2: safety constraints
    if use_safety:
        env = SafetyConstraintWrapper(env, spec=cfg.safety)

    # P1: actuator DR (TRAIN ONLY)
    if mode == "train" and cfg.actuator_dr_range > 0.0:
        env = ActuatorStrengthWrapper(env, dr_range=cfg.actuator_dr_range, seed=seed + 33)

    # optional action L2 penalty (train only)
    if mode == "train" and cfg.safety_lambda_action_l2 > 0.0:
        env = SafetyActionL2PenaltyWrapper(env, lam=cfg.safety_lambda_action_l2)

    # P1: fault injection (EVAL ONLY)
    if mode in ("eval_fault", "eval_fault_goal"):
        if cfg.fault is not None:
            env = FaultInjectionWrapper(
                env,
                strength=cfg.fault.strength,
                actuator_indices=cfg.fault.actuator_indices,
                start_step=cfg.fault.start_step,
            )

    # IMPORTANT: seed/reset OUTERMOST env (after wrappers)
    # This avoids obs-shape inconsistencies across wrappers.
    obs, _ = env.reset(seed=seed)

    # Sanity check: obs must match the declared observation_space shape
    if hasattr(env.observation_space, "shape") and env.observation_space.shape is not None:
        assert obs.shape == env.observation_space.shape, (
            f"[make_env] obs shape mismatch in mode={mode}, domain={domain}: "
            f"obs={obs.shape} vs space={env.observation_space.shape}"
        )

    return env
