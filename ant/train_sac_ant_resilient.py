# train_sac_ant_resilient.py
"""
Actuator-robust SAC training for Ant with TWO explicit motor-UDR formulations.

Experiment intent
-----------------
Train a SAC policy on a SOURCE domain while injecting actuator uncertainty during training
(Motor UDR). Evaluate transfer to a TARGET domain and quantify robustness under two
post-training stress tests.

Motor UDR is implemented in the *environment dynamics* (selected via env_id), not via wrappers:
  (A) per_actuator_down:
      - independent per-actuator weakening: s_i ~ Uniform(1-r, 1)
      - env ids: CustomAntUDRMotor-source-v0 / CustomAntUDRMotor-target-v0

  (B) global_symmetric:
      - one shared gain for all actuators: g ~ Uniform(1-r, 1+r)
      - env ids: CustomAntUDRMotorGlobalSym-source-v0 / CustomAntUDRMotorGlobalSym-target-v0

During training
---------------
- vectorized SAC on SOURCE
- motor_dr_range follows a curriculum (MOTOR_UDR_SCHEDULE)
- periodic evaluation on:
    * SOURCE (nominal)
    * TARGET (nominal)
    * TARGET_FAULTY (optionally follows current motor UDR schedule; see CurriculumDomainCallback)

Final robustness tests (select via EVAL_FAULT_MODE)
---------------------------------------------------
- "udr"   : sweep motor_dr_range on TARGET (stochastic robustness curve), save CSV + PNG
- "fixed" : deterministic within-episode faults (actuator index, remaining gain, start step),
            save CSV + aggregated PNG
"""

import os
import csv
from collections import defaultdict

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Optional: silence MuJoCo warnings (keeps logs readable)
try:
    import mujoco
    mujoco.set_mju_user_warning(lambda msg: None)
except Exception:
    pass

# Registers all CustomAnt* env IDs (source/target + DR/UDR variants)
from env.custom_ant import *  # noqa: F401,F403

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# =============================================================================
# GLOBAL CONFIG
# =============================================================================
ACTION_CLIP = 0.8
TOTAL_TIMESTEPS = 2_000_000
N_ENVS = 6
SEED = 0
DEVICE = "cpu"
MAX_EPISODE_STEPS = 1000

EVAL_FREQ = 40_000
N_EVAL_EPISODES = 10
N_FINAL_EVAL_EPISODES = 50

# Optional mass curriculum (kept for extensibility; disabled below via enable_mass=False)
MASS_UDR_SCHEDULE = [
    (0,         0.00),
    (250_000,   0.05),
    (500_000,   0.10),
    (750_000,   0.15),
    (1_000_000, 0.20),
]

# Motor curriculum: increases motor_dr_range over training (harder dynamics later)
MOTOR_UDR_SCHEDULE = [
    (0,         0.00),
    (400_000,   0.10),
    (800_000,   0.20),
    (1_200_000, 0.30),
    (1_600_000, 0.40),
    (1_800_000, 0.50),
]

RESAMPLE_EVERY_MASS = 2
RESAMPLE_EVERY_MOTOR = 2

# Sweep values for the post-training stochastic robustness curve
MOTOR_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_SWEEP_EVAL_EPISODES = 20

# -------------------------------------------------------------------------
# Motor-UDR mode selector (defines which env IDs are used)
# -------------------------------------------------------------------------
MOTOR_UDR_MODE = "global_symmetric"  # "per_actuator_down" or "global_symmetric"

# -------------------------------------------------------------------------
# Fault evaluation mode selector (defines the final robustness test)
# -------------------------------------------------------------------------
EVAL_FAULT_MODE = "fixed"  # "udr" or "fixed"

# Deterministic fault grid: (fault_indices, remaining_strength, fault_start_step)
# NOTE: This grid assumes single-actuator faults (indices are (0,), (1,), (2,))
FAULT_GRID = [
    # actuator 0
    ((0,), 0.7, 0),   ((0,), 0.7, 50),  ((0,), 0.7, 150),
    ((0,), 0.5, 0),   ((0,), 0.5, 50),  ((0,), 0.5, 150),
    ((0,), 0.3, 0),   ((0,), 0.3, 50),  ((0,), 0.3, 150),

    # actuator 1
    ((1,), 0.7, 0),   ((1,), 0.7, 50),  ((1,), 0.7, 150),
    ((1,), 0.5, 0),   ((1,), 0.5, 50),  ((1,), 0.5, 150),
    ((1,), 0.3, 0),   ((1,), 0.3, 50),  ((1,), 0.3, 150),

    # actuator 2
    ((2,), 0.7, 0),   ((2,), 0.7, 50),  ((2,), 0.7, 150),
    ((2,), 0.5, 0),   ((2,), 0.5, 50),  ((2,), 0.5, 150),
    ((2,), 0.3, 0),   ((2,), 0.3, 50),  ((2,), 0.3, 150),
]
N_FIXED_EVAL_EPISODES = 20


# =============================================================================
# WRAPPERS (protocol-level, not DR/UDR definitions)
# =============================================================================
class ActionClipWrapper(gym.Wrapper):
    """
    Pre-step action saturation to stabilize training when actuator gains vary.
    This reduces rare high-magnitude actions that can destabilize MuJoCo dynamics.
    """
    def __init__(self, env, clip: float = 0.8):
        super().__init__(env)
        self.clip = float(clip)

    def step(self, action):
        return self.env.step(np.clip(action, -self.clip, self.clip))


class MotorFaultWrapper(gym.Wrapper):
    """
    Deterministic within-episode actuator fault injection.

    At t == fault_start_step, multiply actuator_gainprm[idx,0] by fault_strength
    for the selected fault_indices, and keep it for the rest of the episode.

    This is used only for EVAL (robustness grid), not for training DR.
    """
    def __init__(self, env, fault_indices=(0,), fault_strength=0.7, fault_start_step=0):
        super().__init__(env)
        self.fault_indices = tuple(int(i) for i in fault_indices)
        self.fault_strength = float(fault_strength)
        self.fault_start_step = int(fault_start_step)
        self._t = 0
        self._applied = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._t = 0
        self._applied = False
        if self.fault_start_step == 0:
            self._apply_fault()
        return obs, info

    def _apply_fault(self):
        if self._applied:
            return
        uw = self.env.unwrapped
        gainprm = uw.model.actuator_gainprm
        n_act = gainprm.shape[0]
        idx = [i for i in self.fault_indices if 0 <= i < n_act]
        if idx:
            gainprm[idx, 0] *= self.fault_strength
        self._applied = True

    def step(self, action):
        if (not self._applied) and (self._t == self.fault_start_step):
            self._apply_fault()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        return obs, reward, terminated, truncated, info


# =============================================================================
# HELPERS
# =============================================================================
def next_run_name(base_name: str, log_root="logs"):
    """
    Ensures experiments do not overwrite previous runs:
      logs/<base>_run1, logs/<base>_run2, ...
    """
    i = 1
    while True:
        name = f"{base_name}_run{i}"
        if not os.path.exists(os.path.join(log_root, name)):
            return name
        i += 1


def make_monitored_env(env_id: str,
                       max_episode_steps: int = 1000,
                       action_clip: float = 0.8,
                       fault_cfg=None):
    """
    Builds one environment instance with a fixed evaluation/training protocol:

      gym.make(env_id)
        -> TimeLimit (consistent horizon)
        -> ActionClipWrapper (stability)
        -> [optional] MotorFaultWrapper (deterministic eval faults)
        -> Monitor (episode returns/lengths)

    IMPORTANT:
    The *motor UDR mode* is fully determined by env_id (CustomAntUDRMotor* vs CustomAntUDRMotorGlobalSym*).
    """
    env = gym.make(env_id)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = ActionClipWrapper(env, clip=action_clip)

    if fault_cfg is not None:
        env = MotorFaultWrapper(
            env,
            fault_indices=fault_cfg["fault_indices"],
            fault_strength=fault_cfg["fault_strength"],
            fault_start_step=fault_cfg["fault_start_step"],
        )

    env = Monitor(env)
    return env


def make_train_env(env_id: str, n_envs: int, seed: int, max_episode_steps: int, action_clip: float):
    """
    Vectorized training env + observation normalization (VecNormalize).

    Reward normalization is OFF to keep reward scale comparable across DR settings.
    VecNormalize stats are saved and reused for all evaluations.
    """
    env = make_vec_env(
        lambda: make_monitored_env(
            env_id,
            max_episode_steps=max_episode_steps,
            action_clip=action_clip,
        ),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return env


def make_eval_env(env_id: str, seed: int, max_episode_steps: int, action_clip: float, fault_cfg=None):
    """
    Single-env evaluation env with the same wrappers and VecNormalize in inference mode.
    """
    env = make_vec_env(
        lambda: make_monitored_env(
            env_id,
            max_episode_steps=max_episode_steps,
            action_clip=action_clip,
            fault_cfg=fault_cfg,
        ),
        n_envs=1,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env.training = False
    env.norm_reward = False
    return env


def load_eval_env_with_vecnorm(env_id: str, vecnorm_path: str, seed: int, max_episode_steps: int, action_clip: float, fault_cfg=None):
    """
    Recreates an eval env and loads saved VecNormalize statistics from training.
    """
    env = make_vec_env(
        lambda: make_monitored_env(
            env_id,
            max_episode_steps=max_episode_steps,
            action_clip=action_clip,
            fault_cfg=fault_cfg,
        ),
        n_envs=1,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    return env


def safe_load_model(model_cls, path: str, device: str = "cpu"):
    """
    Best-effort load: avoids crashing if a model artifact is missing/corrupted.
    """
    if not os.path.exists(path):
        return None
    try:
        return model_cls.load(path, device=device)
    except Exception:
        return None


def set_nominal_eval(env):
    """
    Forces evaluation dynamics to be nominal (no mass DR and no motor DR),
    independent of training-time schedules.
    """
    env.set_attr("dr_enabled", False)
    env.set_attr("dr_range", 0.0)
    env.set_attr("motor_dr_enabled", False)
    env.set_attr("motor_dr_range", 0.0)


# =============================================================================
# CALLBACKS
# =============================================================================
class EvalCallbackWithVecNorm(EvalCallback):
    """
    Extension of SB3 EvalCallback:
    when a new best model is found, also saves VecNormalize statistics alongside it.

    This avoids a common pitfall: evaluating a "best_model.zip" with mismatched obs normalization.
    """
    def __init__(self, *args, vecnorm_save_name="best_vecnormalize.pkl", **kwargs):
        super().__init__(*args, **kwargs)
        self.vecnorm_save_name = vecnorm_save_name
        self._last_saved_best = None

    def _on_step(self) -> bool:
        res = super()._on_step()
        try:
            if self._last_saved_best is None:
                self._last_saved_best = self.best_mean_reward

            if self.best_mean_reward is not None and self.best_mean_reward > self._last_saved_best:
                self._last_saved_best = self.best_mean_reward
                venv = self.model.get_vec_normalize_env()
                if venv is not None and self.best_model_save_path is not None:
                    os.makedirs(self.best_model_save_path, exist_ok=True)
                    vn_path = os.path.join(self.best_model_save_path, self.vecnorm_save_name)
                    venv.save(vn_path)
        except Exception:
            pass
        return res


class TensorboardEvalLogger(BaseCallback):
    """
    Mirrors SB3 EvalCallback results into explicit TensorBoard scalars:
      eval/source_mean_reward, eval/target_mean_reward, eval/target_faulty_mean_reward
    """
    def __init__(self, eval_source: EvalCallback, eval_target: EvalCallback, eval_target_faulty: EvalCallback):
        super().__init__()
        self.eval_source = eval_source
        self.eval_target = eval_target
        self.eval_target_faulty = eval_target_faulty
        self._last_src = 0
        self._last_tgt = 0
        self._last_ft = 0

    def _log_eval(self, eval_cb, prefix, last_attr):
        results = getattr(eval_cb, "evaluations_results", None)
        if results is None or len(results) == 0:
            return getattr(self, last_attr)
        if len(results) == getattr(self, last_attr):
            return getattr(self, last_attr)

        rewards = results[-1]
        self.logger.record(f"eval/{prefix}_mean_reward", float(np.mean(rewards)))
        self.logger.record(f"eval/{prefix}_std_reward", float(np.std(rewards)))
        return len(results)

    def _on_step(self) -> bool:
        self._last_src = self._log_eval(self.eval_source, "source", "_last_src")
        self._last_tgt = self._log_eval(self.eval_target, "target", "_last_tgt")
        self._last_ft  = self._log_eval(self.eval_target_faulty, "target_faulty", "_last_ft")
        return True


class CurriculumDomainCallback(BaseCallback):
    """
    Training-time curriculum over DR parameters (written via VecEnv.set_attr).

    In this script:
      - enable_motor=True  => motor_dr_range follows MOTOR_UDR_SCHEDULE
      - enable_mass=False  => mass DR is disabled (schedule kept for extensibility)

    Important detail:
      - train_env receives the curriculum always
      - eval_faulty_env can optionally mirror the current motor DR range, if you want
        "TARGET_FAULTY" curves to track the same DR severity used during training.
    """
    def __init__(self, train_env, eval_faulty_env,
                 enable_mass: bool, enable_motor: bool,
                 mass_schedule, motor_schedule,
                 resample_every_mass: int, resample_every_motor: int,
                 verbose=1):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_faulty_env = eval_faulty_env

        self.enable_mass = bool(enable_mass)
        self.enable_motor = bool(enable_motor)

        self.mass_schedule = list(mass_schedule)
        self.motor_schedule = list(motor_schedule)

        self.resample_every_mass = int(resample_every_mass)
        self.resample_every_motor = int(resample_every_motor)

        self._mass_stage = -1
        self._motor_stage = -1

    @staticmethod
    def _stage_from_schedule(t, schedule):
        stage = 0
        for i, (ts, _) in enumerate(schedule):
            if t >= ts:
                stage = i
        return stage

    def _apply_mass(self, dr_range: float):
        self.train_env.set_attr("dr_enabled", dr_range > 0.0)
        self.train_env.set_attr("dr_range", float(dr_range))
        self.train_env.set_attr("resample_every", int(self.resample_every_mass))

    def _apply_motor(self, motor_range: float):
        # Training env: motor UDR severity
        self.train_env.set_attr("motor_dr_enabled", motor_range > 0.0)
        self.train_env.set_attr("motor_dr_range", float(motor_range))
        self.train_env.set_attr("motor_resample_every", int(self.resample_every_motor))

        # Optional: reflect the same motor UDR on TARGET_FAULTY evaluation
        self.eval_faulty_env.set_attr("motor_dr_enabled", motor_range > 0.0)
        self.eval_faulty_env.set_attr("motor_dr_range", float(motor_range))
        self.eval_faulty_env.set_attr("motor_resample_every", 1)

    def _on_step(self) -> bool:
        t = self.num_timesteps

        if self.enable_mass:
            s = self._stage_from_schedule(t, self.mass_schedule)
            if s != self._mass_stage:
                self._mass_stage = s
                dr_range = float(self.mass_schedule[s][1])
                self._apply_mass(dr_range)
                if self.verbose:
                    print(f"[MASS-UDR] timesteps={t} -> dr_range={dr_range}")

        if self.enable_motor:
            s = self._stage_from_schedule(t, self.motor_schedule)
            if s != self._motor_stage:
                self._motor_stage = s
                motor_range = float(self.motor_schedule[s][1])
                self._apply_motor(motor_range)
                if self.verbose:
                    print(f"[MOTOR-UDR] timesteps={t} -> motor_dr_range={motor_range}")

        return True


# =============================================================================
# FINAL TESTS
# =============================================================================
def motor_sweep_test_csv(model_path: str, vecnorm_path: str,
                         target_id: str, out_csv_path: str,
                         seed: int, max_episode_steps: int, n_episodes: int,
                         motor_ranges, action_clip: float,
                         device: str = "cpu"):
    """
    Post-training stochastic robustness test:
    sweep motor_dr_range on TARGET and measure mean/std return (deterministic policy).
    """
    m = safe_load_model(SAC, model_path, device=device)
    if m is None:
        print(f"[SWEEP] failed to load model: {model_path}")
        return

    rows = [["motor_dr_range", "mean_reward", "std_reward"]]

    for i, r in enumerate(motor_ranges):
        e = load_eval_env_with_vecnorm(
            target_id, vecnorm_path, seed + 9000 + i,
            max_episode_steps, action_clip
        )

        # Keep mass nominal; only vary motor DR for this sweep
        e.set_attr("dr_enabled", False)
        e.set_attr("dr_range", 0.0)
        e.set_attr("motor_dr_enabled", float(r) > 0.0)
        e.set_attr("motor_dr_range", float(r))
        e.set_attr("motor_resample_every", 1)

        mean_r, std_r = evaluate_policy(m, e, n_eval_episodes=n_episodes, deterministic=True)
        e.close()

        rows.append([f"{r:.2f}", f"{mean_r:.6f}", f"{std_r:.6f}"])
        print(f"[SWEEP] motor_dr_range={r:.2f} -> {mean_r:.2f} +/- {std_r:.2f}")

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"[SWEEP] saved CSV: {out_csv_path}")


def plot_motor_sweep(csv_path: str, out_png_path: str, title: str):
    """
    Visualization of the sweep CSV:
      - mean return curve + ±1 std band
    """
    xs, ys, ystd = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for row in reader:
            xs.append(float(row[0]))
            ys.append(float(row[1]))
            ystd.append(float(row[2]))

    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.figure()

    plt.plot(xs, ys, marker="o", label="Mean return", color="tab:blue")
    plt.fill_between(xs, np.array(ys) - np.array(ystd), np.array(ys) + np.array(ystd),
                     alpha=0.2, label="±1 std", color="tab:blue")

    plt.xlabel("motor_dr_range")
    plt.ylabel("mean reward (deterministic)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", frameon=True)

    plt.savefig(out_png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] saved: {out_png_path}")


def fixed_fault_grid_test(model_path: str, vecnorm_path: str,
                          target_id: str, out_csv_path: str,
                          seed: int, max_episode_steps: int, n_episodes: int,
                          fault_grid, action_clip: float,
                          device: str = "cpu"):
    """
    Post-training deterministic robustness test:
    run a grid of time-triggered actuator faults on TARGET and record mean/std return.

    Each grid point is defined by:
      - fault_indices     : which actuator(s) are affected
      - fault_strength    : remaining gain multiplier (e.g., 0.5 = -50%)
      - fault_start_step  : within-episode time when the fault activates
    """
    m = safe_load_model(SAC, model_path, device=device)
    if m is None:
        print(f"[FIXED] failed to load model: {model_path}")
        return

    rows = [["fault_indices", "fault_strength", "fault_start_step", "mean_reward", "std_reward"]]

    for k, (fault_indices, fault_strength, fault_start_step) in enumerate(fault_grid):
        fault_cfg = {
            "fault_indices": fault_indices,
            "fault_strength": float(fault_strength),
            "fault_start_step": int(fault_start_step),
        }

        e = load_eval_env_with_vecnorm(
            target_id, vecnorm_path, seed + 12000 + k,
            max_episode_steps, action_clip,
            fault_cfg=fault_cfg
        )

        # Deterministic faults => keep domain randomization disabled
        set_nominal_eval(e)

        mean_r, std_r = evaluate_policy(m, e, n_eval_episodes=n_episodes, deterministic=True)
        e.close()

        rows.append([str(tuple(fault_indices)), f"{fault_strength:.3f}", str(int(fault_start_step)),
                     f"{mean_r:.6f}", f"{std_r:.6f}"])

        loss_pct = (1.0 - float(fault_strength)) * 100.0
        print(f"[FIXED] idx={fault_indices} strength={fault_strength:.2f} (-{loss_pct:.0f}%) start={fault_start_step}"
              f" -> {mean_r:.2f} +/- {std_r:.2f}")

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"[FIXED] saved CSV: {out_csv_path}")


def plot_fixed_fault_grid(csv_path: str, out_png_path: str, title: str):
    """
    Aggregated visualization for the fault grid.

    For each actuator index and fault_strength:
      - average mean return over different fault_start_step values
    This yields one curve per actuator showing sensitivity to that actuator failure.
    """
    agg = defaultdict(list)  # key=(idx, strength) -> list[mean_return]

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for row in reader:
            idxs_str = row[0]          # e.g. "(0,)"
            strength = float(row[1])
            mean_r = float(row[3])

            # Parse "(0,)" -> 0 (assumes single-actuator faults as defined in FAULT_GRID)
            cleaned = idxs_str.strip().replace("(", "").replace(")", "").replace(" ", "")
            cleaned = cleaned.replace(",", "")
            idx = int(cleaned)

            agg[(idx, strength)].append(mean_r)

    # Series per actuator: strengths -> mean over start steps
    series = {0: ([], []), 1: ([], []), 2: ([], [])}
    for (idx, strength), means_list in agg.items():
        series[idx][0].append(strength)
        series[idx][1].append(float(np.mean(means_list)))

    # Sort each curve by strength
    for idx in series:
        strengths = np.array(series[idx][0], dtype=float)
        means = np.array(series[idx][1], dtype=float)
        if len(strengths) == 0:
            continue
        order = np.argsort(strengths)
        series[idx] = (strengths[order], means[order])

    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.figure()

    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    for idx in [0, 1, 2]:
        xs, ys = series[idx]
        if len(xs) == 0:
            continue
        plt.plot(xs, ys, marker="o", color=colors[idx], label=f"Actuator {idx}")

    plt.xlabel("fault_strength (remaining gain)")
    plt.ylabel("mean reward (avg over start_step)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", frameon=True)

    plt.savefig(out_png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] saved: {out_png_path}")


# =============================================================================
# TRAINING RUNNER
# =============================================================================
def train_one(base_run_name: str, source_id: str, target_id: str):
    """
    One run:
      - train SAC on SOURCE with motor UDR curriculum enabled
      - evaluate periodically on SOURCE/TARGET/TARGET_FAULTY (nominal by default)
      - save checkpoints + best models + VecNormalize stats
      - run final robustness test ("udr" sweep or "fixed" fault grid)
    """
    run_name = next_run_name(base_run_name)

    log_dir = os.path.join("logs", run_name)
    model_dir = os.path.join("models", run_name)
    results_dir = os.path.join("results", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Training env is SOURCE; motor UDR behavior depends on selected env_id
    train_env = make_train_env(
        source_id, n_envs=N_ENVS, seed=SEED,
        max_episode_steps=MAX_EPISODE_STEPS, action_clip=ACTION_CLIP
    )

    # Eval envs (same wrappers, different domains)
    eval_source = make_eval_env(source_id, SEED + 1000, MAX_EPISODE_STEPS, ACTION_CLIP)
    eval_target = make_eval_env(target_id, SEED + 2000, MAX_EPISODE_STEPS, ACTION_CLIP)
    eval_target_faulty = make_eval_env(target_id, SEED + 3000, MAX_EPISODE_STEPS, ACTION_CLIP)

    # Share obs normalization statistics to match training scaling
    eval_source.obs_rms = train_env.obs_rms
    eval_target.obs_rms = train_env.obs_rms
    eval_target_faulty.obs_rms = train_env.obs_rms

    # Keep periodic evaluations nominal unless explicitly changed elsewhere
    set_nominal_eval(eval_source)
    set_nominal_eval(eval_target)
    set_nominal_eval(eval_target_faulty)

    # Initialize motor DR state; CurriculumDomainCallback will ramp it up
    train_env.set_attr("motor_dr_enabled", float(MOTOR_UDR_SCHEDULE[0][1]) > 0.0)
    train_env.set_attr("motor_dr_range", float(MOTOR_UDR_SCHEDULE[0][1]))
    train_env.set_attr("motor_resample_every", int(RESAMPLE_EVERY_MOTOR))

    # Save best models on each evaluation domain + store vecnormalize with them
    eval_cb_source = EvalCallbackWithVecNorm(
        eval_source,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        vecnorm_save_name="best_vecnormalize.pkl",
    )
    eval_cb_target = EvalCallbackWithVecNorm(
        eval_target,
        best_model_save_path=os.path.join(model_dir, "best_on_target"),
        log_path=os.path.join(log_dir, "target_eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        vecnorm_save_name="best_vecnormalize.pkl",
    )
    eval_cb_target_faulty = EvalCallbackWithVecNorm(
        eval_target_faulty,
        best_model_save_path=os.path.join(model_dir, "best_on_target_faulty"),
        log_path=os.path.join(log_dir, "target_faulty_eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        vecnorm_save_name="best_vecnormalize.pkl",
    )

    # Mirror eval results into explicit TensorBoard scalars
    tb_logger = TensorboardEvalLogger(eval_cb_source, eval_cb_target, eval_cb_target_faulty)

    # Periodic checkpoints (plus VecNormalize) for recovery/ablation
    checkpoint_cb = CheckpointCallback(
        save_freq=200_000,
        save_path=model_dir,
        name_prefix="ckpt",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # DR curriculum: only motor enabled in this experiment
    curriculum_cb = CurriculumDomainCallback(
        train_env=train_env,
        eval_faulty_env=eval_target_faulty,
        enable_mass=False,
        enable_motor=True,
        mass_schedule=MASS_UDR_SCHEDULE,
        motor_schedule=MOTOR_UDR_SCHEDULE,
        resample_every_mass=RESAMPLE_EVERY_MASS,
        resample_every_motor=RESAMPLE_EVERY_MOTOR,
        verbose=1,
    )

    callbacks = CallbackList([
        eval_cb_source, eval_cb_target, eval_cb_target_faulty,
        tb_logger, checkpoint_cb, curriculum_cb
    ])

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
            eval_target_faulty.close()
        except Exception:
            pass

        final_model_zip = os.path.join(model_dir, "final_model.zip")
        vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")

        # ----------------------------
        # Final robustness evaluation
        # ----------------------------
        if EVAL_FAULT_MODE == "udr":
            sweep_csv = os.path.join(results_dir, "motor_sweep.csv")
            motor_sweep_test_csv(
                model_path=final_model_zip,
                vecnorm_path=vecnorm_path,
                target_id=target_id,
                out_csv_path=sweep_csv,
                seed=SEED,
                max_episode_steps=MAX_EPISODE_STEPS,
                n_episodes=N_SWEEP_EVAL_EPISODES,
                motor_ranges=MOTOR_SWEEP,
                action_clip=ACTION_CLIP,
                device=DEVICE,
            )
            plot_path = os.path.join("plots", f"{run_name}_motor_sweep.png")
            plot_motor_sweep(
                csv_path=sweep_csv,
                out_png_path=plot_path,
                title=f"{run_name} - motor robustness sweep",
            )

        elif EVAL_FAULT_MODE == "fixed":
            fixed_csv = os.path.join(results_dir, "fixed_fault_grid.csv")
            fixed_fault_grid_test(
                model_path=final_model_zip,
                vecnorm_path=vecnorm_path,
                target_id=target_id,
                out_csv_path=fixed_csv,
                seed=SEED,
                max_episode_steps=MAX_EPISODE_STEPS,
                n_episodes=N_FIXED_EVAL_EPISODES,
                fault_grid=FAULT_GRID,
                action_clip=ACTION_CLIP,
                device=DEVICE,
            )
            plot_path = os.path.join("plots", f"{run_name}_fixed_fault_grid.png")
            plot_fixed_fault_grid(
                csv_path=fixed_csv,
                out_png_path=plot_path,
                title=f"{run_name} - fixed fault grid",
            )

        print("Saved:")
        print(f"  Logs:    {log_dir}")
        print(f"  Models:  {model_dir}")
        print(f"  Results: {results_dir}")
        print("TensorBoard: tensorboard --logdir logs")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """
    Selects env IDs purely from MOTOR_UDR_MODE.

    This guarantees that the two training modes are not accidentally identical:
      - per_actuator_down  -> CustomAntUDRMotor-*
      - global_symmetric   -> CustomAntUDRMotorGlobalSym-*
    """
    base_name = f"SAC_CustomAnt_Resilient_{MOTOR_UDR_MODE}_seed{SEED}"

    if MOTOR_UDR_MODE == "per_actuator_down":
        source_id = "CustomAntUDRMotor-source-v0"
        target_id = "CustomAntUDRMotor-target-v0"

    elif MOTOR_UDR_MODE == "global_symmetric":
        source_id = "CustomAntUDRMotorGlobalSym-source-v0"
        target_id = "CustomAntUDRMotorGlobalSym-target-v0"

    else:
        raise ValueError(f"Unknown MOTOR_UDR_MODE: {MOTOR_UDR_MODE}")

    train_one(
        base_run_name=base_name,
        source_id=source_id,
        target_id=target_id,
    )


if __name__ == "__main__":
    main()
