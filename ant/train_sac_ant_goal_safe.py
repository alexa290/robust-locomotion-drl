# train_sac_ant_goal_safe.py
"""
Goal-conditioned locomotion with safety shaping + friction curriculum (terrain DR).

Experiment intent
-----------------
Train SAC on a SOURCE Ant domain with a goal-reaching task (GoalSafeWrapper). During training,
apply a curriculum over ground friction randomization to gradually increase dynamics variability.
Periodically evaluate the same policy on:
  - SOURCE (in-domain)
  - TARGET (domain-shifted dynamics)
under identical task definitions and shared observation normalization.

Key components
--------------
- make_env(...) factory (env_id selects dynamics; wrappers select task/protocol):
    * TimeLimit -> ActionClip -> GoalSafeWrapper -> Monitor
- GoalSafeWrapper (task layer):
    * samples a new goal each episode (goal_mode="xy": random direction + radius)
    * shaped reward: progress + near-goal shaping + terminal success bonus
    * safety: continuous tilt penalty + fall penalty, logs metrics in info
- Friction curriculum (dynamics layer):
    * updates env.unwrapped friction randomization range over time
- Logging:
    * SB3 EvalCallback: reward curves for SOURCE and TARGET
    * GoalSafeEpisodeLogger: rolling success/violation/fall stats + per-episode CSV
    * CheckpointCallback: periodic checkpoints + VecNormalize stats
"""

import os
import csv
from collections import deque

import numpy as np
import gymnasium as gym

# Registers custom env IDs (kept unchanged)
from env.custom_ant import *  # noqa: F401,F403

# Unified env constructor:
#   env_id -> selects dynamics domain
#   wrappers -> select task (GoalSafe) and other protocols (ActionClip/Monitor/TimeLimit)
from utils.env_factory import make_env

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback, CheckpointCallback


# =============================================================================
# GLOBAL CONFIG
# =============================================================================
TOTAL_TIMESTEPS = 2_000_000
N_ENVS = 6
SEED = 0
DEVICE = "cpu"

MAX_EPISODE_STEPS = 1000
ACTION_CLIP = 0.8

EVAL_FREQ = 40_000
N_EVAL_EPISODES = 20


# =============================================================================
# GOAL-SAFE TASK CONFIG (primary trade-off knobs)
# =============================================================================
# These parameters are forwarded to GoalSafeWrapper via make_env(...).
# They control: goal sampling distribution, success definition, reward shaping,
# and safety penalties (tilt/fall).
GOALSAFE_KWARGS = dict(
    # Goal sampling (goal_mode="xy": uniform radius + uniform angle)
    goal_radius_range=(2.0, 6.0),
    goal_threshold=0.5,
    goal_terminate_threshold=0.45,
    success_hold_steps=5,  # robust success: K consecutive inside-threshold steps

    # Reward shaping (goal progress + terminal bonus)
    progress_coef=5.0,
    success_bonus=8.0,
    near_goal_radius=1.0,
    near_goal_coef=0.2,

    # Action energy regularization (proxy for torque/energy usage)
    energy_coef=1e-3,

    # Safety metrics and penalties (tilt acts like a constraint signal)
    tilt_violation_thresh=0.7,
    tilt_fall_thresh=1.2,
    min_torso_z=0.20,
    violation_penalty=0.5,  # main safety trade-off knob
    fall_penalty=10.0,

    # Continuous penalty yields smoother optimization than binary penalties
    continuous_safety=True,

    # 2D navigation in the horizontal plane
    goal_mode="xy",
)


# =============================================================================
# FRICTION CURRICULUM (terrain variability)
# =============================================================================
# friction_dr_range = r means: friction_scale ~ Uniform(1-r, 1+r) per resample
FRICTION_SCHEDULE = [
    (0,         0.00),  # nominal friction at start
    (300_000,   0.20),
    (700_000,   0.40),
    (1_200_000, 0.60),  # strongest friction variability later in training
]
FRICTION_RESAMPLE_EVERY = 2  # resample friction every N episodes (implemented in env)


# =============================================================================
# HELPERS
# =============================================================================
def next_run_name(base_name: str, log_root="logs"):
    """
    Creates unique run folders to avoid overwriting:
      logs/<base_name>_run1, logs/<base_name>_run2, ...
    """
    i = 1
    while True:
        name = f"{base_name}_run{i}"
        if not os.path.exists(os.path.join(log_root, name)):
            return name
        i += 1


def make_train_env(env_id: str):
    """
    Vectorized training environment with VecNormalize(obs).

    Notes:
      - reward normalization is OFF because shaped rewards are already scaled
        and we want interpretability across runs.
      - the task is enabled via make_env(..., use_goal_safe=True).
    """
    env = make_vec_env(
        lambda: make_env(
            env_id,
            max_episode_steps=MAX_EPISODE_STEPS,
            action_clip=ACTION_CLIP,
            use_goal_safe=True,
            goalsafe_kwargs=GOALSAFE_KWARGS,
        ),
        n_envs=N_ENVS,
        seed=SEED,
        vec_env_cls=DummyVecEnv,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return env


def make_eval_env(env_id: str, seed: int):
    """
    Single-env evaluation wrapper with VecNormalize in inference mode.
    obs_rms is later overwritten with training stats to ensure identical scaling.
    """
    env = make_vec_env(
        lambda: make_env(
            env_id,
            max_episode_steps=MAX_EPISODE_STEPS,
            action_clip=ACTION_CLIP,
            use_goal_safe=True,
            goalsafe_kwargs=GOALSAFE_KWARGS,
        ),
        n_envs=1,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env.training = False
    env.norm_reward = False
    return env


# =============================================================================
# CALLBACKS
# =============================================================================
class FrictionCurriculumCallback(BaseCallback):
    """
    Dynamics curriculum: increases friction DR range over training.

    Mechanism:
      - uses VecEnv.set_attr(...) to write attributes on the base CustomAntEnv
      - the task wrapper (GoalSafeWrapper) is unaffected; only dynamics change

    Expected env attributes (on env.unwrapped / CustomAnt):
      - friction_dr_enabled : bool
      - friction_dr_range   : float
      - friction_resample_every : int
    """
    def __init__(self, train_env, schedule, resample_every=2, verbose=1):
        super().__init__(verbose)
        self.train_env = train_env
        self.schedule = list(schedule)
        self.resample_every = int(resample_every)
        self._stage = -1

    @staticmethod
    def _stage_from_schedule(t, schedule):
        stage = 0
        for i, (ts, _) in enumerate(schedule):
            if t >= ts:
                stage = i
        return stage

    def _apply(self, r: float):
        # Enable/disable DR depending on r, and set the symmetric Uniform(1-r,1+r) range
        self.train_env.set_attr("friction_dr_enabled", float(r) > 0.0)
        self.train_env.set_attr("friction_dr_range", float(r))
        self.train_env.set_attr("friction_resample_every", int(self.resample_every))

    def _on_step(self) -> bool:
        t = self.num_timesteps
        s = self._stage_from_schedule(t, self.schedule)
        if s != self._stage:
            self._stage = s
            r = float(self.schedule[s][1])
            self._apply(r)
            if self.verbose:
                print(f"[FRICTION-DR] timesteps={t} -> friction_dr_range={r}")
        return True


class GoalSafeEpisodeLogger(BaseCallback):
    """
    Task-level episode logger based on GoalSafeWrapper info fields.

    Collects at episode boundaries:
      - is_success      : robust success indicator (K-step confirmation)
      - time_to_goal    : steps until success (success-only)
      - violations      : count of tilt violations in the episode
      - fell            : fall indicator

    Outputs:
      - TensorBoard rolling-window stats under gs/*
      - goalsafe_episode_metrics.csv in log_dir for report tables/plots
    """
    def __init__(self, out_dir: str, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.out_dir = out_dir
        self.window = int(window)

        self.success_hist = deque(maxlen=self.window)
        self.ttg_hist = deque(maxlen=self.window)
        self.viol_hist = deque(maxlen=self.window)
        self.fall_hist = deque(maxlen=self.window)

        # Raw per-episode rows for CSV export
        self.rows = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)
        if infos is None or dones is None:
            return True

        for i, done in enumerate(dones):
            if not done:
                continue

            info = infos[i]
            is_success = bool(info.get("is_success", False))
            ttg = int(info.get("time_to_goal", -1))
            viol = int(info.get("violations", 0))
            fell = bool(info.get("fell", False))

            # Update rolling buffers
            self.success_hist.append(1 if is_success else 0)
            if is_success and ttg > 0:
                self.ttg_hist.append(ttg)
            self.viol_hist.append(viol)
            self.fall_hist.append(1 if fell else 0)

            # Save raw episode row (timesteps is global SB3 counter)
            self.rows.append([self.num_timesteps, int(is_success), ttg, viol, int(fell)])

        # TensorBoard rolling-window metrics (stable trend visualization)
        if len(self.success_hist) > 0:
            self.logger.record("gs/success_rate_window", float(np.mean(self.success_hist)))
            self.logger.record("gs/violations_mean_window", float(np.mean(self.viol_hist)))
            self.logger.record("gs/fall_rate_window", float(np.mean(self.fall_hist)))
            if len(self.ttg_hist) > 0:
                self.logger.record("gs/time_to_goal_mean_window", float(np.mean(self.ttg_hist)))

        return True

    def _on_training_end(self) -> None:
        # Export per-episode metrics for report plotting/tables
        os.makedirs(self.out_dir, exist_ok=True)
        out_csv = os.path.join(self.out_dir, "goalsafe_episode_metrics.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timesteps", "is_success", "time_to_goal", "violations", "fell"])
            w.writerows(self.rows)
        print(f"[CSV] saved: {out_csv}")


# =============================================================================
# TRAIN
# =============================================================================
def train_one(base_run_name: str, source_id: str, target_id: str):
    """
    One experiment run:
      - trains on SOURCE dynamics with goal-safe task
      - evaluates periodically on SOURCE and TARGET (same task + shared obs normalization)
      - applies friction curriculum only on the training env
      - saves checkpoints + final model + VecNormalize stats
    """
    run_name = next_run_name(base_run_name)

    log_dir = os.path.join("logs", run_name)
    model_dir = os.path.join("models", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Training env: SOURCE dynamics + GoalSafeWrapper via make_env(...)
    train_env = make_train_env(source_id)

    # Evaluation envs: SOURCE and TARGET with identical wrapper configuration
    eval_source = make_eval_env(source_id, SEED + 1000)
    eval_target = make_eval_env(target_id, SEED + 2000)

    # Share VecNormalize statistics (standard SB3 practice for consistent scaling)
    eval_source.obs_rms = train_env.obs_rms
    eval_target.obs_rms = train_env.obs_rms

    # Reward-based evaluation callbacks (for learning curves + best model checkpoints)
    eval_cb_source = EvalCallback(
        eval_source,
        best_model_save_path=os.path.join(model_dir, "best_on_source"),
        log_path=os.path.join(log_dir, "eval_source"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    eval_cb_target = EvalCallback(
        eval_target,
        best_model_save_path=os.path.join(model_dir, "best_on_target"),
        log_path=os.path.join(log_dir, "eval_target"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    # Dynamics curriculum (training env only)
    friction_cb = FrictionCurriculumCallback(
        train_env=train_env,
        schedule=FRICTION_SCHEDULE,
        resample_every=FRICTION_RESAMPLE_EVERY,
        verbose=1,
    )

    # Task-level CSV + TensorBoard rolling metrics (success/violations/falls/ttg)
    gs_logger = GoalSafeEpisodeLogger(out_dir=log_dir, window=100)

    # Periodic checkpoints + VecNormalize persistence
    checkpoint_cb = CheckpointCallback(
        save_freq=200_000,
        save_path=model_dir,
        name_prefix="ckpt",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    callbacks = CallbackList([eval_cb_source, eval_cb_target, friction_cb, gs_logger, checkpoint_cb])

    # SAC policy (kept explicit for reproducibility)
    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=log_dir,

        learning_rate=1e-4,
        buffer_size=1_000_000,
        learning_starts=50_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,

        ent_coef=0.005,
        use_sde=True,
        sde_sample_freq=4,

        policy_kwargs=dict(net_arch=[256, 256]),
        seed=SEED,
        device=DEVICE,
    )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    finally:
        # Persist artifacts even if training is interrupted
        try:
            model.save(os.path.join(model_dir, "final_model"))
        except Exception:
            pass
        try:
            train_env.save(os.path.join(model_dir, "vecnormalize.pkl"))
        except Exception:
            pass
        try:
            train_env.close()
            eval_source.close()
            eval_target.close()
        except Exception:
            pass

        print("Saved:")
        print(f"  Logs:   {log_dir}")
        print(f"  Models: {model_dir}")
        print("TensorBoard: tensorboard --logdir logs")


def main():
    """
    Entry point: keeps env IDs aligned with the rest of your project
    (so runs remain comparable and TensorBoard tags stay consistent).
    """
    source_id = "CustomAntBase-source-v0"
    target_id = "CustomAntBase-target-v0"

    base_name = f"SAC_CustomAnt_GoalSafe_FrictionUDR_seed{SEED}"
    train_one(base_run_name=base_name, source_id=source_id, target_id=target_id)


if __name__ == "__main__":
    main()
