# project_extension/extension/train.py
"""
This module defines the training pipeline for a single reinforcement learning
experiment configuration and random seed.

For a given experiment setup, it creates a vectorized source-domain
environment, initializes a Soft Actor-Critic (SAC) agent with stable and
well-tested hyperparameters, and performs on-policy interaction to learn
a control policy.

Training supports both single-environment and parallel multi-environment
execution using DummyVecEnv or SubprocVecEnv, enabling scalable data
collection while preserving reproducibility through controlled seeding.
Each environment instance is wrapped with a Monitor to record episode-level
statistics.

Trained models are saved to a structured directory under `runs/<experiment_name>/`,
allowing systematic comparison across experiments and random seeds.

This module is intentionally isolated from evaluation logic to ensure a
clear separation between policy learning and performance assessment under
nominal and faulted conditions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from .configs import ExperimentConfig
from .make_env import make_env


def _exp_dir(cfg: ExperimentConfig) -> Path:
    return Path("runs") / cfg.name


def _model_path(cfg: ExperimentConfig, seed: int) -> Path:
    return _exp_dir(cfg) / f"model_seed{seed}.zip"


def train_one(cfg: ExperimentConfig, seed: int, n_envs: int = 4, use_subproc: bool = True) -> None:
    """
    Train one experiment config (cfg) with one seed.
    Saves the model to runs/<cfg.name>/model_seed<seed>.zip
    """
    out_dir = _exp_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_file = _model_path(cfg, seed)

    def _make(rank: int):
        # different seed per worker
        s = seed + 1000 * rank
        return lambda: Monitor(make_env(cfg, domain="source", mode="train", seed=s))

    # Vec env
    if n_envs <= 1:
        env = DummyVecEnv([_make(0)])
    else:
        env = SubprocVecEnv([_make(i) for i in range(n_envs)]) if use_subproc else DummyVecEnv([_make(i) for i in range(n_envs)])

    # Create SAC
    model = SAC(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        verbose=1,
        tensorboard_log=str(out_dir / "tb"),

        # Core hyperparameters (stability first)
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,

        # Update schedule
        train_freq=1,          # update every env step
        gradient_steps=1,      # one gradient step per env step
        learning_starts=10_000,

        # Entropy / exploration
        ent_coef="auto",       # SB3 will tune alpha automatically

        # Network size
        policy_kwargs=dict(net_arch=[256, 256]),

        # Practical settings
        device="auto",
    )


    model.learn(total_timesteps=int(cfg.total_timesteps))
    model.save(str(model_file))

    env.close()
    print(f"[TRAIN] saved: {model_file}")
