# train_sac_ant_goal_safe_curriculum.py
"""
Goal-conditioned locomotion with explicit safety shaping and a simple goal curriculum.

Experiment intent
-----------------
Train SAC on a SOURCE Ant domain with a goal-reaching task (X-only goals). During training,
the goal distance distribution is progressively hardened (easy -> hard) to stabilize early
learning and improve generalization to longer goals.

Key components
--------------
- GoalSafeWrapper (task layer):
    * samples a new goal every episode
    * augments observations with goal-relative features
    * shaped reward: progress + terminal bonus + near-goal shaping
    * safety signal: continuous penalty beyond a tilt threshold
    * robust success: requires K consecutive inside-goal steps
- Curriculum:
    * changes the goal_x_range at a fixed timestep via env_method(...)
- Evaluation:
    * periodic eval on SOURCE and TARGET domains
    * logs both reward-based metrics (SB3 EvalCallback) and task-level goal metrics
- Artifacts:
    * logs/  (TensorBoard)
    * models/ (best checkpoints + vecnormalize)
    * results/ (config, CSV metrics, plots, final_results.txt)
"""

import os
import json
import gymnasium as gym
import numpy as np

# Register custom env IDs (source/target dynamics)
from env.custom_ant import *  # noqa: F401,F403

# Task wrapper: goal-reaching + safety shaping + metrics in info dict
from env.wrappers import GoalSafeWrapper

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# ======================================================
# CONFIG
# ======================================================
CONFIG = dict(
    # Dynamics domains (task is applied via wrapper below)
    source_id="CustomAntBase-source-v0",
    target_id="CustomAntBase-target-v0",

    # Task: goal specification (X-only goals sampled per episode)
    goal_mode="x",

    # Curriculum (easy -> hard): modifies goal distribution during training
    curriculum_enabled=True,
    curriculum_switch_timestep=500_000,
    goal_x_range_easy=(0.5, 2.5),
    goal_x_range_hard=(1.0, 4.0),

    # Initial goal distribution at training start
    goal_x_range=(0.5, 2.5),

    # Success definition (robust to noise)
    goal_threshold=0.5,
    goal_terminate_threshold=0.4,
    success_hold_steps=5,

    # Shaping / safety (implemented inside GoalSafeWrapper)
    progress_coef=5.0,
    success_bonus=8.0,
    near_goal_radius=1.0,
    near_goal_coef=0.2,
    continuous_safety=True,

    # Training setup
    n_envs=6,
    total_timesteps=1_000_000,
    seed=0,
    max_episode_steps=1000,

    # Evaluation cadence and sample size
    eval_freq=40_000,
    n_eval_episodes=20,
    n_final_eval_episodes=50,

    # SAC hyperparameters (kept consistent across experiments for fair comparison)
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
    net_arch=[256, 256],

    # Device selection (SB3 resolves "auto" to cuda if available)
    device="auto",
)

# Run naming prefix (used to create non-overwriting log/model/result folders)
RUN_PREFIX = "SAC_CustomAnt_GOAL_SAFE_CURRICULUM"


# ======================================================
# Environment builders
# ======================================================
def make_monitored_env(env_id: str, max_episode_steps: int, goal_kwargs: dict):
    """
    Builds one environment instance.

    Stack order:
      1) gym.make(env_id)         -> selects dynamics domain (source/target)
      2) GoalSafeWrapper          -> defines task, shaped reward, safety metrics
      3) TimeLimit                -> enforces consistent horizon
      4) Monitor                  -> records episode return/length
    """
    env = gym.make(env_id)

    # Task layer: goal reaching + safety shaping + success metrics in info
    env = GoalSafeWrapper(
        env,
        goal_mode=goal_kwargs["goal_mode"],
        goal_x_range=goal_kwargs["goal_x_range"],
        goal_threshold=goal_kwargs["goal_threshold"],
        goal_terminate_threshold=goal_kwargs["goal_terminate_threshold"],
        success_hold_steps=goal_kwargs["success_hold_steps"],

        # Shaping/safety weights (kept explicit for reproducibility)
        progress_coef=goal_kwargs["progress_coef"],
        success_bonus=goal_kwargs["success_bonus"],
        near_goal_radius=goal_kwargs["near_goal_radius"],
        near_goal_coef=goal_kwargs["near_goal_coef"],
        continuous_safety=goal_kwargs["continuous_safety"],
    )

    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env)
    return env


def next_run_name(base_name: str, log_root="logs"):
    """
    Returns a unique run name to avoid overwriting previous experiments:
      logs/<base_name>_run1, logs/<base_name>_run2, ...
    """
    i = 1
    while True:
        name = f"{base_name}_run{i}"
        if not os.path.exists(os.path.join(log_root, name)):
            return name
        i += 1


def make_train_env(env_id: str, n_envs: int, seed: int, max_episode_steps: int, goal_kwargs: dict):
    """
    Vectorized training environment (DummyVecEnv) with observation normalization.
    Reward normalization is disabled to keep shaped rewards interpretable/comparable.
    """
    env = make_vec_env(
        lambda: make_monitored_env(env_id, max_episode_steps=max_episode_steps, goal_kwargs=goal_kwargs),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return env


def make_eval_env(env_id: str, seed: int, max_episode_steps: int, goal_kwargs: dict):
    """
    Single-env evaluation environment (VecNormalize in inference mode).
    The obs_rms will be overwritten with the training env statistics for consistency.
    """
    env = make_vec_env(
        lambda: make_monitored_env(env_id, max_episode_steps=max_episode_steps, goal_kwargs=goal_kwargs),
        n_envs=1,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env.training = False
    env.norm_reward = False
    return env


def load_eval_env_with_vecnorm(env_id: str, vecnorm_path: str, seed: int, max_episode_steps: int, goal_kwargs: dict):
    """
    Loads VecNormalize statistics from training and attaches them to a fresh eval env.
    This guarantees identical observation scaling in final evaluations.
    """
    env = make_vec_env(
        lambda: make_monitored_env(env_id, max_episode_steps=max_episode_steps, goal_kwargs=goal_kwargs),
        n_envs=1,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    return env


# ======================================================
# Curriculum callback
# ======================================================
class GoalCurriculumCallback(BaseCallback):
    """
    Hardens the goal distribution during training.

    Mechanism:
      - At training start, sets an "easy" goal_x_range.
      - After switch_timestep, updates to a "hard" goal_x_range.
    Implementation:
      - Uses VecEnv.env_method(...) to call set_goal_x_range inside GoalSafeWrapper.
    """
    def __init__(self, train_env, switch_timestep: int, easy_range, hard_range, verbose: int = 0):
        super().__init__(verbose)
        self.train_env = train_env
        self.switch_timestep = int(switch_timestep)
        self.easy_range = tuple(easy_range)
        self.hard_range = tuple(hard_range)
        self._switched = False

    def _on_training_start(self) -> None:
        # Start with easier goals to stabilize early learning
        self.train_env.env_method("set_goal_x_range", self.easy_range)

    def _on_step(self) -> bool:
        # One-time switch to the harder goal distribution
        if (not self._switched) and (self.num_timesteps >= self.switch_timestep):
            self.train_env.env_method("set_goal_x_range", self.hard_range)
            self._switched = True
        return True


# ======================================================
# Goal metrics eval (task-level)
# ======================================================
def rollout_goal_metrics(model: SAC, env, n_episodes: int, deterministic: bool = True):
    """
    Runs rollouts on VecEnv(n_envs=1) and collects task-level metrics produced by GoalSafeWrapper.

    Note:
      This is separate from SB3 reward evaluation; it measures goal success, distance,
      safety violations, time-to-goal, and fall events.
    """
    ep_returns, ep_lens = [], []
    ep_success, ep_final_dist, ep_time_to_goal, ep_violations, ep_abs_dx0, ep_fell = [], [], [], [], [], []

    obs = env.reset()
    cur_ret, cur_len = 0.0, 0
    finished = 0

    while finished < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, infos = env.step(action)

        cur_ret += float(reward[0])
        cur_len += 1

        if bool(done[0]):
            info = infos[0] if isinstance(infos, (list, tuple)) else infos

            ep_returns.append(cur_ret)
            ep_lens.append(cur_len)

            ep_success.append(int(info.get("is_success", False)))
            ep_final_dist.append(float(info.get("dist_to_goal", np.nan)))
            ep_time_to_goal.append(int(info.get("time_to_goal", -1)))
            ep_violations.append(int(info.get("violations", 0)))
            ep_fell.append(int(info.get("fell", False)))

            # Episode goal offset sampled at reset (more meaningful than instantaneous dx)
            dx0 = info.get("goal_dx0", np.nan)
            try:
                ep_abs_dx0.append(float(abs(dx0)))
            except Exception:
                ep_abs_dx0.append(np.nan)

            cur_ret, cur_len = 0.0, 0
            finished += 1

    return dict(
        returns=np.asarray(ep_returns, dtype=np.float32),
        lens=np.asarray(ep_lens, dtype=np.int32),
        success=np.asarray(ep_success, dtype=np.int32),
        final_dist=np.asarray(ep_final_dist, dtype=np.float32),
        time_to_goal=np.asarray(ep_time_to_goal, dtype=np.int32),
        violations=np.asarray(ep_violations, dtype=np.int32),
        abs_dx0=np.asarray(ep_abs_dx0, dtype=np.float32),
        fell=np.asarray(ep_fell, dtype=np.int32),
    )


def summarize_goal_metrics(ep):
    """
    Aggregates per-episode arrays into scalar metrics for logging/reporting.
    """
    success_rate = float(np.mean(ep["success"])) if len(ep["success"]) else 0.0
    mean_final_dist = float(np.nanmean(ep["final_dist"])) if len(ep["final_dist"]) else float("nan")

    ttg = ep["time_to_goal"][ep["success"] == 1]
    mean_time_to_goal = float(np.mean(ttg)) if len(ttg) else float("nan")

    vio = ep["violations"]
    violation_rate = float(np.mean(vio > 0)) if len(vio) else 0.0
    mean_violations = float(np.mean(vio)) if len(vio) else 0.0

    fell = ep["fell"]
    fall_rate = float(np.mean(fell == 1)) if len(fell) else 0.0

    mean_return = float(np.mean(ep["returns"])) if len(ep["returns"]) else float("nan")
    mean_len = float(np.mean(ep["lens"])) if len(ep["lens"]) else float("nan")

    return dict(
        success_rate=success_rate,
        mean_final_dist=mean_final_dist,
        mean_time_to_goal=mean_time_to_goal,
        violation_rate=violation_rate,
        mean_violations=mean_violations,
        fall_rate=fall_rate,
        mean_return=mean_return,
        mean_len=mean_len,
    )


def moving_average(x: np.ndarray, w: int = 3):
    """
    Lightweight smoothing for plotting (reduces high-variance eval noise).
    """
    if w <= 1 or len(x) < w:
        return x
    y = np.convolve(x, np.ones(w, dtype=np.float32) / float(w), mode="valid")
    pad = np.full((w - 1,), y[0], dtype=np.float32)
    return np.concatenate([pad, y], axis=0)


class GoalEvalMetricsCallback(BaseCallback):
    """
    Periodic task-level evaluation on SOURCE and TARGET.

    Outputs:
      - TensorBoard scalars under goal/<domain>_*
      - CSV table goal_metrics_eval.csv for post-processing/report tables
      - In-memory history to generate summary plots at the end of training
    """
    def __init__(self, eval_env_source, eval_env_target, eval_freq: int, n_eval_episodes: int, results_dir: str):
        super().__init__()
        self.eval_env_source = eval_env_source
        self.eval_env_target = eval_env_target
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.results_dir = results_dir
        self._last_eval_step = 0

        self.history = {"timesteps": [], "source": [], "target": [], "source_ep": [], "target_ep": []}

        self.metrics_csv = os.path.join(results_dir, "goal_metrics_eval.csv")
        if not os.path.exists(self.metrics_csv):
            with open(self.metrics_csv, "w", encoding="utf-8") as f:
                f.write(
                    "timesteps,domain,success_rate,mean_final_dist,mean_time_to_goal,violation_rate,mean_violations,fall_rate,mean_return,mean_len\n"
                )

    def _eval_one(self, env, domain: str):
        ep = rollout_goal_metrics(self.model, env, self.n_eval_episodes, deterministic=True)
        sm = summarize_goal_metrics(ep)

        # TensorBoard
        self.logger.record(f"goal/{domain}_success_rate", sm["success_rate"])
        self.logger.record(f"goal/{domain}_mean_final_dist", sm["mean_final_dist"])
        self.logger.record(f"goal/{domain}_mean_time_to_goal", sm["mean_time_to_goal"])
        self.logger.record(f"goal/{domain}_violation_rate", sm["violation_rate"])
        self.logger.record(f"goal/{domain}_mean_violations", sm["mean_violations"])
        self.logger.record(f"goal/{domain}_fall_rate", sm["fall_rate"])

        # CSV (for reproducible report numbers)
        with open(self.metrics_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{self.num_timesteps},{domain},{sm['success_rate']},{sm['mean_final_dist']},{sm['mean_time_to_goal']},"
                f"{sm['violation_rate']},{sm['mean_violations']},{sm['fall_rate']},{sm['mean_return']},{sm['mean_len']}\n"
            )

        return sm, ep

    def _on_step(self) -> bool:
        # Evaluate only every eval_freq steps (keeps overhead controlled)
        if (self.num_timesteps - self._last_eval_step) < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        src_sm, src_ep = self._eval_one(self.eval_env_source, "source")
        tgt_sm, tgt_ep = self._eval_one(self.eval_env_target, "target")

        self.history["timesteps"].append(int(self.num_timesteps))
        self.history["source"].append(src_sm)
        self.history["target"].append(tgt_sm)
        self.history["source_ep"].append(src_ep)
        self.history["target_ep"].append(tgt_ep)
        return True

    def save_plots(self, plots_dir: str, goal_abs_dx_max: float, smooth_w: int = 3):
        """
        Generates report-ready plots (PNG):
          - learning curves (source vs target) for key task metrics
          - final success vs goal distance bins
          - histograms for final distance and time-to-goal
        """
        os.makedirs(plots_dir, exist_ok=True)
        import matplotlib.pyplot as plt

        ts = np.asarray(self.history["timesteps"], dtype=np.int64)
        if len(ts) == 0:
            return

        def series(domain_list, key):
            return np.asarray([d[key] for d in domain_list], dtype=np.float32)

        keys = ["success_rate", "mean_final_dist", "mean_time_to_goal", "violation_rate", "mean_violations", "fall_rate"]
        for key in keys:
            y_src = moving_average(series(self.history["source"], key), w=smooth_w)
            y_tgt = moving_average(series(self.history["target"], key), w=smooth_w)

            plt.figure()
            plt.plot(ts, y_src, label="source")
            plt.plot(ts, y_tgt, label="target")
            plt.xlabel("timesteps")
            plt.ylabel(key)
            plt.title(f"{key} (moving avg w={smooth_w})")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best", frameon=True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"curve_{key}.png"), dpi=200)
            plt.close()

        def plot_success_vs_dx0(ep, domain: str):
            abs_dx0 = ep["abs_dx0"]
            success = ep["success"]
            m = np.isfinite(abs_dx0)
            abs_dx0 = abs_dx0[m]
            success = success[m]
            if len(abs_dx0) == 0:
                return

            bins = np.linspace(0.0, float(goal_abs_dx_max), 6)  # 5 bins
            bin_ids = np.digitize(abs_dx0, bins) - 1

            rates, centers = [], []
            for b in range(len(bins) - 1):
                mask = bin_ids == b
                rates.append(float(np.mean(success[mask])) if np.sum(mask) else np.nan)
                centers.append(0.5 * (bins[b] + bins[b + 1]))

            plt.figure()
            plt.plot(centers, rates, marker="o")
            plt.xlabel("|goal_dx0| (episode goal offset)")
            plt.ylabel("success_rate")
            plt.ylim(-0.05, 1.05)
            plt.title(f"Final success vs goal distance ({domain})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"final_success_vs_abs_dx0_{domain}.png"), dpi=200)
            plt.close()

        last_src = self.history["source_ep"][-1]
        last_tgt = self.history["target_ep"][-1]
        plot_success_vs_dx0(last_src, "source")
        plot_success_vs_dx0(last_tgt, "target")

        def plot_hist(arr, title, fname):
            plt.figure()
            plt.hist(arr, bins=20)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, fname), dpi=200)
            plt.close()

        plot_hist(last_src["final_dist"], "Final dist to goal (source)", "final_dist_hist_source.png")
        plot_hist(last_tgt["final_dist"], "Final dist to goal (target)", "final_dist_hist_target.png")

        ttg_src = last_src["time_to_goal"][last_src["success"] == 1]
        ttg_tgt = last_tgt["time_to_goal"][last_tgt["success"] == 1]
        if len(ttg_src):
            plot_hist(ttg_src, "Time to goal (success only, source)", "time_to_goal_hist_source.png")
        if len(ttg_tgt):
            plot_hist(ttg_tgt, "Time to goal (success only, target)", "time_to_goal_hist_target.png")


# ======================================================
# Reward eval logger (standard SB3 eval results)
# ======================================================
class TensorboardEvalLogger(BaseCallback):
    """
    Mirrors SB3 EvalCallback results into TensorBoard under eval/<domain>_*.

    This produces clean reward-based learning curves for:
      - source
      - target
    """
    def __init__(self, eval_source: EvalCallback, eval_target: EvalCallback):
        super().__init__()
        self.eval_source = eval_source
        self.eval_target = eval_target
        self._last_src = 0
        self._last_tgt = 0

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
        return True


# ======================================================
# Final results report (reward-based)
# ======================================================
def safe_load_model(model_cls, path: str, device: str):
    """
    Safe model loader used by the final evaluation stage.
    """
    try:
        return model_cls.load(path, device=device)
    except Exception:
        return None


def eval_one(tag: str, model_path: str, eval_s_s, eval_s_t, n_episodes: int, device: str):
    """
    Evaluates one checkpoint on:
      - source->source (in-domain)
      - source->target (transfer)
    """
    m = safe_load_model(SAC, model_path, device=device)
    if m is None:
        return [f"[{tag}] NOT FOUND ({model_path})"]

    mean_s_s, std_s_s = evaluate_policy(m, eval_s_s, n_eval_episodes=n_episodes, deterministic=True)
    mean_s_t, std_s_t = evaluate_policy(m, eval_s_t, n_eval_episodes=n_episodes, deterministic=True)

    return [
        f"[{tag}] {os.path.basename(model_path)}",
        f"  source -> source : {mean_s_s:.2f} +/- {std_s_s:.2f}",
        f"  source -> target : {mean_s_t:.2f} +/- {std_s_t:.2f}",
    ]


def write_final_results(run_name, source_id, target_id, model_dir, results_dir, vecnorm_path,
                        seed, max_episode_steps, n_final_eval_episodes, goal_kwargs, device: str):
    """
    Produces a compact final report (TXT) comparing:
      - best-on-source checkpoint
      - best-on-target checkpoint
      - final checkpoint
    under consistent VecNormalize statistics.
    """
    eval_s_s = load_eval_env_with_vecnorm(source_id, vecnorm_path, seed + 3000, max_episode_steps, goal_kwargs)
    eval_s_t = load_eval_env_with_vecnorm(target_id, vecnorm_path, seed + 4000, max_episode_steps, goal_kwargs)

    lines = []
    lines.append("=" * 60)
    lines.append(f"RUN: {run_name}")
    lines.append(f"source env: {source_id}")
    lines.append(f"target env: {target_id}")
    lines.append("=" * 60)
    lines.append("")

    candidates = [
        ("BEST_ON_SOURCE", os.path.join(model_dir, "best_model.zip")),
        ("BEST_ON_TARGET", os.path.join(model_dir, "best_on_target", "best_model.zip")),
        ("FINAL", os.path.join(model_dir, "final_model.zip")),
    ]
    for tag, p in candidates:
        lines.extend(eval_one(tag, p, eval_s_s, eval_s_t, n_final_eval_episodes, device=device))
        lines.append("")

    with open(os.path.join(results_dir, "final_results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    try:
        eval_s_s.close()
        eval_s_t.close()
    except Exception:
        pass


# ======================================================
# MAIN
# ======================================================
def main():
    """
    Training entry point:
      - builds train env (SOURCE dynamics + GoalSafe task)
      - builds eval envs (SOURCE + TARGET) with shared normalization stats
      - trains SAC with curriculum + periodic evaluations
      - saves final artifacts and generates report plots
    """
    goal_kwargs = dict(
        goal_mode=CONFIG["goal_mode"],
        goal_x_range=CONFIG["goal_x_range"],
        goal_threshold=CONFIG["goal_threshold"],
        goal_terminate_threshold=CONFIG["goal_terminate_threshold"],
        success_hold_steps=CONFIG["success_hold_steps"],

        progress_coef=CONFIG["progress_coef"],
        success_bonus=CONFIG["success_bonus"],
        near_goal_radius=CONFIG["near_goal_radius"],
        near_goal_coef=CONFIG["near_goal_coef"],
        continuous_safety=CONFIG["continuous_safety"],
    )

    base_run_name = f"{RUN_PREFIX}_seed{CONFIG['seed']}"
    run_name = next_run_name(base_run_name)

    log_dir = os.path.join("logs", run_name)
    model_dir = os.path.join("models", run_name)
    results_dir = os.path.join("results", run_name)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Persist config for reproducibility
    with open(os.path.join(results_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    # Training env (SOURCE dynamics) with VecNormalize(obs)
    train_env = make_train_env(
        CONFIG["source_id"],
        n_envs=CONFIG["n_envs"],
        seed=CONFIG["seed"],
        max_episode_steps=CONFIG["max_episode_steps"],
        goal_kwargs=goal_kwargs,
    )

    # Eval envs (SOURCE + TARGET) with identical task wrapper
    eval_env_source = make_eval_env(
        CONFIG["source_id"],
        CONFIG["seed"] + 1000,
        CONFIG["max_episode_steps"],
        goal_kwargs,
    )
    eval_env_target = make_eval_env(
        CONFIG["target_id"],
        CONFIG["seed"] + 2000,
        CONFIG["max_episode_steps"],
        goal_kwargs,
    )

    # Share observation normalization statistics (train -> eval)
    eval_env_source.obs_rms = train_env.obs_rms
    eval_env_target.obs_rms = train_env.obs_rms

    # Reward-based evaluation (SB3) for checkpoint selection
    eval_cb_source = EvalCallback(
        eval_env_source,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=CONFIG["eval_freq"],
        n_eval_episodes=CONFIG["n_eval_episodes"],
        deterministic=True,
        render=False,
    )
    eval_cb_target = EvalCallback(
        eval_env_target,
        best_model_save_path=os.path.join(model_dir, "best_on_target"),
        log_path=os.path.join(log_dir, "target_eval"),
        eval_freq=CONFIG["eval_freq"],
        n_eval_episodes=CONFIG["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    # Mirrors reward eval into TensorBoard with clean tag names
    tb_reward_logger = TensorboardEvalLogger(eval_cb_source, eval_cb_target)

    # Task-level evaluation (success/violations/time-to-goal) + CSV + plots
    goal_metrics_cb = GoalEvalMetricsCallback(
        eval_env_source=eval_env_source,
        eval_env_target=eval_env_target,
        eval_freq=CONFIG["eval_freq"],
        n_eval_episodes=CONFIG["n_eval_episodes"],
        results_dir=results_dir,
    )

    # Periodic checkpoints for safety (and vecnormalize persistence)
    checkpoint_cb = CheckpointCallback(
        save_freq=200_000,
        save_path=model_dir,
        name_prefix="ckpt",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Curriculum callback (optional): easy -> hard goal distribution
    curriculum_cb = None
    if CONFIG.get("curriculum_enabled", False):
        curriculum_cb = GoalCurriculumCallback(
            train_env=train_env,
            switch_timestep=CONFIG["curriculum_switch_timestep"],
            easy_range=CONFIG["goal_x_range_easy"],
            hard_range=CONFIG["goal_x_range_hard"],
        )

    cb_items = [eval_cb_source, eval_cb_target, tb_reward_logger, goal_metrics_cb, checkpoint_cb]
    if curriculum_cb is not None:
        cb_items.insert(0, curriculum_cb)
    callbacks = CallbackList(cb_items)

    # SAC policy and training configuration (kept explicit for reproducibility)
    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=log_dir,

        learning_rate=CONFIG["learning_rate"],
        buffer_size=CONFIG["buffer_size"],
        learning_starts=CONFIG["learning_starts"],
        batch_size=CONFIG["batch_size"],
        tau=CONFIG["tau"],
        gamma=CONFIG["gamma"],
        train_freq=CONFIG["train_freq"],
        gradient_steps=CONFIG["gradient_steps"],

        ent_coef=CONFIG["ent_coef"],
        use_sde=CONFIG["use_sde"],
        sde_sample_freq=CONFIG["sde_sample_freq"],

        policy_kwargs=dict(net_arch=CONFIG["net_arch"]),
        seed=CONFIG["seed"],
        device=CONFIG["device"],
    )

    try:
        model.learn(total_timesteps=CONFIG["total_timesteps"], callback=callbacks)
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
            eval_env_source.close()
            eval_env_target.close()
        except Exception:
            pass

        # Final deterministic report using saved VecNormalize statistics
        vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")
        if os.path.exists(vecnorm_path):
            write_final_results(
                run_name, CONFIG["source_id"], CONFIG["target_id"],
                model_dir, results_dir,
                vecnorm_path, CONFIG["seed"], CONFIG["max_episode_steps"],
                CONFIG["n_final_eval_episodes"], goal_kwargs,
                device=CONFIG["device"],
            )

        # Generate report-ready plots from task-level evaluation history
        try:
            goal_metrics_cb.save_plots(
                plots_dir,
                goal_abs_dx_max=float(abs(CONFIG["goal_x_range_hard"][1])),
                smooth_w=3,
            )
        except Exception:
            pass

        print("Saved:")
        print(f"  Logs:    {log_dir}")
        print(f"  Models:  {model_dir}")
        print(f"  Results: {os.path.join(results_dir, 'final_results.txt')}")
        print(f"  Plots:   {plots_dir}")
        print("TensorBoard: tensorboard --logdir logs")


if __name__ == "__main__":
    main()
