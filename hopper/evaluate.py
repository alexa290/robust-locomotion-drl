# project_extension/extension/evaluate.py
"""
This module implements the evaluation pipeline for trained reinforcement
learning agents under both nominal and degraded operating conditions.

Given a trained SAC policy and an experiment configuration, the module
systematically evaluates performance in the target domain across four
distinct scenarios:

  1. Nominal evaluation (eval):
     - Target domain
     - No hard actuator faults
     - Native environment reward (goal task inactive)

  2. Fault evaluation (eval_fault):
     - Target domain
     - Injected hard actuator faults (multiple predefined fault cases)
     - Native environment reward

  3. Goal-conditioned evaluation (eval_goal):
     - Target domain
     - No hard actuator faults
     - Goal-conditioned navigation task with success termination

  4. Faulted goal-conditioned evaluation (eval_fault_goal):
     - Target domain
     - Injected hard actuator faults
     - Goal-conditioned navigation with increased task difficulty

For each configuration, the agent is rolled out for a fixed number of
episodes and a comprehensive set of metrics is collected, including
returns, episode length, success and failure rates, safety violations,
control energy, and safety margin statistics.

All results are aggregated and appended to a centralized CSV file to
enable reproducible analysis and comparison across experiments, random
seeds, fault scenarios, and evaluation modes.

The evaluation logic is intentionally decoupled from training, allowing
consistent benchmarking of robustness, safety, and task performance
under controlled fault and goal variations.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from stable_baselines3 import SAC

from .configs import (
    ExperimentConfig,
    FaultSpec,
    EVAL_FAULT_SPECS,
)
RUNS_DIR = Path("runs")
from .make_env import make_env


def _model_path(cfg: ExperimentConfig, seed: int) -> Path:
    return RUNS_DIR / cfg.name / f"model_seed{seed}.zip"


def _append_row(row: Dict) -> None:
    out = RUNS_DIR / "results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "exp", "seed", "domain", "mode", "fault_tag",
        "n_episodes",
        "mean_return", "std_return",
        "success_rate", "violation_rate",
        "mean_ep_len", "fail_rate",
        "mean_energy",

        "total_timesteps", "actuator_dr_range",

        "goal_x_min", "goal_x_max", "goal_tolerance",
        "fault_goal_x_min", "fault_goal_x_max", "fault_goal_tolerance",

        "friction_range",
        "safety_max_pitch", "safety_max_vspeed", "safety_terminate",

        "fault_strength", "fault_indices", "fault_start_step",
    ]
    write_header = not out.exists()

    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for k in fieldnames:
            row.setdefault(k, None)
        w.writerow(row)


def _run_episodes(
    model: SAC,
    env,
    n_episodes: int,
    deterministic: bool,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Rollout helper.

    Returns:
      mean_return, std_return, success_rate, violation_rate,
      mean_ep_len, fail_rate, mean_energy.
    """
    returns = []
    lengths = []
    successes = []
    violations = []
    energies = []

    for _ in range(int(n_episodes)):
        obs, info = env.reset()

        ep_ret = 0.0
        ep_len = 0
        ep_success = False
        ep_violation = False
        ep_energy = 0.0

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            ep_ret += float(reward)
            ep_len += 1

            a = np.asarray(action, dtype=np.float32)
            ep_energy += float(np.sum(a * a))

            if isinstance(info, dict):
                if info.get("is_success", False):
                    ep_success = True
                if info.get("safety_violation", False):
                    ep_violation = True

        returns.append(ep_ret)
        lengths.append(ep_len)
        successes.append(1.0 if ep_success else 0.0)
        violations.append(1.0 if ep_violation else 0.0)
        energies.append(ep_energy)

    mean_r = float(np.mean(returns)) if returns else np.nan
    std_r = float(np.std(returns)) if returns else np.nan
    succ = float(np.mean(successes)) if successes else np.nan
    viol = float(np.mean(violations)) if violations else np.nan
    mean_len = float(np.mean(lengths)) if lengths else np.nan
    fail_rate = float(1.0 - succ) if np.isfinite(succ) else np.nan
    mean_energy = float(np.mean(energies)) if energies else np.nan

    return mean_r, std_r, succ, viol, mean_len, fail_rate, mean_energy, 


def _with_fault(cfg: ExperimentConfig, fault: FaultSpec) -> ExperimentConfig:
    return ExperimentConfig(
        name=cfg.name,
        algo=cfg.algo,

        total_timesteps=cfg.total_timesteps,
        n_eval_episodes=cfg.n_eval_episodes,
        source_env_id=cfg.source_env_id,
        target_env_id=cfg.target_env_id,

        actuator_dr_range=cfg.actuator_dr_range,
        safety_lambda_action_l2=cfg.safety_lambda_action_l2,
        fault=fault,

        goals=cfg.goals,
        goal_x_min=cfg.goal_x_min,
        goal_x_max=cfg.goal_x_max,
        goal_y=cfg.goal_y,
        goal_tolerance=cfg.goal_tolerance,

        fault_goal_x_min=cfg.fault_goal_x_min,
        fault_goal_x_max=cfg.fault_goal_x_max,
        fault_goal_tolerance=cfg.fault_goal_tolerance,

        max_episode_steps=cfg.max_episode_steps,
        reward=cfg.reward,
        safety=cfg.safety,
        friction_dr=cfg.friction_dr,
    )


def _with_fault_goal(cfg: ExperimentConfig, fault: FaultSpec) -> ExperimentConfig:
    """
    Same as eval_goal, but with an injected fault.
    Goal distribution is identical to eval_goal (random per episode).
    """
    return ExperimentConfig(
        name=cfg.name,
        algo=cfg.algo,

        total_timesteps=cfg.total_timesteps,
        n_eval_episodes=cfg.n_eval_episodes,
        source_env_id=cfg.source_env_id,
        target_env_id=cfg.target_env_id,

        actuator_dr_range=cfg.actuator_dr_range,
        safety_lambda_action_l2=cfg.safety_lambda_action_l2,
        fault=fault,

        # goal IDENTICO a eval_goal
        goals=cfg.goals,
        goal_x_min=cfg.goal_x_min,
        goal_x_max=cfg.goal_x_max,
        goal_y=cfg.goal_y,
        goal_tolerance=cfg.goal_tolerance,

        max_episode_steps=cfg.max_episode_steps,
        reward=cfg.reward,
        safety=cfg.safety,
        friction_dr=cfg.friction_dr,
    )

def evaluate_one(cfg: ExperimentConfig, seed: int, deterministic: bool = True) -> None:
    """
    Runs evaluation for ONE (experiment, seed) pair.

    We separate evaluation into four modes:

      1. eval
         - target domain
         - no hard faults
         - GOAL TASK INACTIVE (native env reward is kept; goal is only part of the observation)

      2. eval_fault
         - target domain
         - hard actuator fault injected (27 cases)
         - GOAL TASK INACTIVE (native env reward)

      3. eval_goal
         - target domain
         - no hard faults
         - GOAL TASK ACTIVE (goal-shaped reward + terminate on success)

      4. eval_fault_goal
         - target domain
         - hard actuator fault injected (27 cases)
         - GOAL TASK ACTIVE with harder goals (cfg overridden via _with_fault_goal)

    IMPORTANT:
      Training uses goal task ACTIVE, so we keep GoalConditionedWrapper always applied
      to preserve observation shape. The difference across modes is whether we override reward.
    """
    model_file = _model_path(cfg, seed)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    model = SAC.load(str(model_file))

    # EVAL: nominal target performance (native env reward)
    env_eval = make_env(cfg, domain="target", mode="eval", seed=seed + 12345)
    (mean_r, std_r, _succ_ignored, viol,
     mean_len, _fail_ignored, mean_energy) = _run_episodes(model, env_eval, cfg.n_eval_episodes, deterministic)
    env_eval.close()

    _append_row({
        "exp": cfg.name,
        "seed": seed,
        "domain": "target",
        "mode": "eval",
        "fault_tag": None,
        "n_episodes": cfg.n_eval_episodes,
        "mean_return": mean_r,
        "std_return": std_r,
        "success_rate": None,   # not meaningful (goal task inactive)
        "violation_rate": viol,
        "mean_ep_len": mean_len,
        "fail_rate": None,      # not meaningful
        "mean_energy": mean_energy,

        "total_timesteps": cfg.total_timesteps,
        "actuator_dr_range": cfg.actuator_dr_range,

        "goal_x_min": cfg.goal_x_min,
        "goal_x_max": cfg.goal_x_max,
        "goal_tolerance": cfg.goal_tolerance,
        "fault_goal_x_min": cfg.fault_goal_x_min,
        "fault_goal_x_max": cfg.fault_goal_x_max,
        "fault_goal_tolerance": cfg.fault_goal_tolerance,

        "friction_range": cfg.friction_dr.range,
        "safety_max_pitch": cfg.safety.max_torso_pitch_rad,
        "safety_max_vspeed": cfg.safety.max_vertical_speed,
        "safety_terminate": cfg.safety.terminate_on_violation,

        "fault_strength": None,
        "fault_indices": None,
        "fault_start_step": None,
    })

    # EVAL_FAULT: target + hard faults (native env reward)
    for fault in EVAL_FAULT_SPECS:
        cfg_fault = _with_fault(cfg, fault)
        env_fault = make_env(cfg_fault, domain="target", mode="eval_fault", seed=seed + 54321)

        (mean_f, std_f, _succ_ignored_f, viol_f,
         mean_len_f, _fail_ignored_f, mean_energy_f) = _run_episodes(model, env_fault, cfg_fault.n_eval_episodes, deterministic)
        env_fault.close()

        _append_row({
            "exp": cfg.name,
            "seed": seed,
            "domain": "target",
            "mode": "eval_fault",
            "fault_tag": fault.tag,
            "n_episodes": cfg_fault.n_eval_episodes,
            "mean_return": mean_f,
            "std_return": std_f,
            "success_rate": None,   # not meaningful (goal task inactive)
            "violation_rate": viol_f,
            "mean_ep_len": mean_len_f,
            "fail_rate": None,      # not meaningful
            "mean_energy": mean_energy_f,

            "total_timesteps": cfg.total_timesteps,
            "actuator_dr_range": cfg.actuator_dr_range,

            "goal_x_min": cfg_fault.goal_x_min,
            "goal_x_max": cfg_fault.goal_x_max,
            "goal_tolerance": cfg_fault.goal_tolerance,
            "fault_goal_x_min": cfg_fault.fault_goal_x_min,
            "fault_goal_x_max": cfg_fault.fault_goal_x_max,
            "fault_goal_tolerance": cfg_fault.fault_goal_tolerance,

            "friction_range": cfg.friction_dr.range,
            "safety_max_pitch": cfg.safety.max_torso_pitch_rad,
            "safety_max_vspeed": cfg.safety.max_vertical_speed,
            "safety_terminate": cfg.safety.terminate_on_violation,

            "fault_strength": fault.strength,
            "fault_indices": str(fault.actuator_indices),
            "fault_start_step": fault.start_step,
        })

    # EVAL_GOAL: goal-conditioned navigation (no faults)
    env_goal = make_env(cfg, domain="target", mode="eval_goal", seed=seed + 22222)
    (mean_g, std_g, succ_g, viol_g,
     mean_len_g, fail_g, mean_energy_g) = _run_episodes(model, env_goal, cfg.n_eval_episodes, deterministic)
    env_goal.close()

    _append_row({
        "exp": cfg.name,
        "seed": seed,
        "domain": "target",
        "mode": "eval_goal",
        "fault_tag": None,
        "n_episodes": cfg.n_eval_episodes,
        "mean_return": mean_g,
        "std_return": std_g,
        "success_rate": succ_g,
        "violation_rate": viol_g,
        "mean_ep_len": mean_len_g,
        "fail_rate": fail_g,
        "mean_energy": mean_energy_g,

        "total_timesteps": cfg.total_timesteps,
        "actuator_dr_range": cfg.actuator_dr_range,

        "goal_x_min": cfg.goal_x_min,
        "goal_x_max": cfg.goal_x_max,
        "goal_tolerance": cfg.goal_tolerance,
        "fault_goal_x_min": cfg.fault_goal_x_min,
        "fault_goal_x_max": cfg.fault_goal_x_max,
        "fault_goal_tolerance": cfg.fault_goal_tolerance,

        "friction_range": cfg.friction_dr.range,
        "safety_max_pitch": cfg.safety.max_torso_pitch_rad,
        "safety_max_vspeed": cfg.safety.max_vertical_speed,
        "safety_terminate": cfg.safety.terminate_on_violation,

        "fault_strength": None,
        "fault_indices": None,
        "fault_start_step": None,
    })

    # EVAL_FAULT_GOAL: goal-conditioned navigation under faults
    for fault in EVAL_FAULT_SPECS:
        cfg_fault_goal = _with_fault_goal(cfg, fault)
        env_fault_goal = make_env(cfg_fault_goal, domain="target", mode="eval_fault_goal", seed=seed + 77777)

        (mean_fg, std_fg, succ_fg, viol_fg,
         mean_len_fg, fail_rate_fg, mean_energy_fg) = _run_episodes(model, env_fault_goal, cfg_fault_goal.n_eval_episodes, deterministic)
        env_fault_goal.close()

        _append_row({
            "exp": cfg.name,
            "seed": seed,
            "domain": "target",
            "mode": "eval_fault_goal",
            "fault_tag": fault.tag,
            "n_episodes": cfg_fault_goal.n_eval_episodes,
            "mean_return": mean_fg,
            "std_return": std_fg,
            "success_rate": succ_fg,
            "violation_rate": viol_fg,
            "mean_ep_len": mean_len_fg,
            "fail_rate": fail_rate_fg,
            "mean_energy": mean_energy_fg,

            "total_timesteps": cfg.total_timesteps,
            "actuator_dr_range": cfg.actuator_dr_range,

            "goal_x_min": cfg_fault_goal.goal_x_min,
            "goal_x_max": cfg_fault_goal.goal_x_max,
            "goal_tolerance": cfg_fault_goal.goal_tolerance,
            "fault_goal_x_min": cfg_fault_goal.fault_goal_x_min,
            "fault_goal_x_max": cfg_fault_goal.fault_goal_x_max,
            "fault_goal_tolerance": cfg_fault_goal.fault_goal_tolerance,

            "friction_range": cfg.friction_dr.range,
            "safety_max_pitch": cfg.safety.max_torso_pitch_rad,
            "safety_max_vspeed": cfg.safety.max_vertical_speed,
            "safety_terminate": cfg.safety.terminate_on_violation,

            "fault_strength": fault.strength,
            "fault_indices": str(fault.actuator_indices),
            "fault_start_step": fault.start_step,
        })
