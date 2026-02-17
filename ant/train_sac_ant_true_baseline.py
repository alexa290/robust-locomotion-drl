# train_sac_ant_true_baseline_improved.py
"""
Baseline SAC for goal-conditioned locomotion (X-only goal per episode).

Improvements vs previous version
--------------------------------
1) Robust success definition (handled inside GoalSafeWrapper):
   - success must hold for K consecutive steps (success_hold_steps)
   - optional tighter termination radius (goal_terminate_threshold < goal_threshold)

2) More reliable evaluation:
   - n_eval_episodes increased (reduces noisy curves)
   - optional moving-average smoothing for plots

3) Cleaner run/config management:
   - single CONFIG block
   - saves config.json alongside results for reproducibility

This file contains:
- NO mass/motor domain randomization
- NO fault injection
"""

import os
import json
import gymnasium as gym
import numpy as np

# Registers CustomAntBase-source-v0 / CustomAntBase-target-v0 (dynamics only)
from env.custom_ant import *  # noqa: F401,F403

# Task wrapper: goal-conditioned + success metrics in info dict
from env.wrappers import GoalSafeWrapper

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# ======================================================
# CONFIG (single source of truth for the experiment)
# ======================================================
CONFIG = dict(
    # Dynamics domains (no DR): source = nominal, target = shifted dynamics (e.g., heavier legs)
    source_id="CustomAntBase-source-v0",
    target_id="CustomAntBase-target-v0",

    # Goal-conditioned task (forward-only goals along x)
    goal_mode="x",
    goal_x_range=(1.0, 4.0),         # episode goal offset sampled at reset
    goal_threshold=0.5,              # success radius (enter region)
    goal_terminate_threshold=0.4,    # stricter terminal radius (forces margin at termination)
    success_hold_steps=5,            # success requires K consecutive steps inside threshold

    # Training budget
    n_envs=6,
    total_timesteps=1_000_000,
    seed=0,
    max_episode_steps=1000,

    # Evaluation cadence and robustness of reported statistics
    eval_freq=40_000,
    n_eval_episodes=20,              # higher => less noisy curves
    n_final_eval_episodes=50,        # final numbers for tables/report

    # SAC hyperparameters (kept standard / SB3-friendly)
    learning_rate=1e-4,
    buffer_size=1_000_000,
    learning_starts=50_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef=0.005,
    use_sde=True,                    # state-dependent exploration (often helps MuJoCo)
    sde_sample_freq=4,
    net_arch=[256, 256],

    # Runtime device selection
    device="auto",                   # "auto" lets SB3 pick CUDA if available
)

RUN_PREFIX = "SAC_CustomAnt_GOAL_BASELINE_X_IMPROVED"


# ======================================================
# Environment builders (dynamics selected by env_id, task added via wrapper)
# ======================================================
def make_monitored_env(env_id: str, max_episode_steps: int, goal_kwargs: dict):
    """
    Builds ONE environment instance with:
      1) gym.make(env_id)        -> selects dynamics (source/target)
      2) GoalSafeWrapper         -> goal-conditioned task + success metrics
      3) TimeLimit               -> fixed horizon for stable train/eval
      4) Monitor                 -> episode return/length logging (SB3 compatible)
    """
    env = gym.make(env_id)

    # Task wrapper injects goal + computes task-level metrics in info dict
    env = GoalSafeWrapper(
        env,
        goal_mode=goal_kwargs["goal_mode"],
        goal_x_range=goal_kwargs["goal_x_range"],
        goal_threshold=goal_kwargs["goal_threshold"],
        goal_terminate_threshold=goal_kwargs["goal_terminate_threshold"],
        success_hold_steps=goal_kwargs["success_hold_steps"],
    )

    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env)
    return env


def next_run_name(base_name: str, log_root="logs"):
    """Avoids overwriting previous experiments by creating logs/<name>_runK."""
    i = 1
    while True:
        name = f"{base_name}_run{i}"
        if not os.path.exists(os.path.join(log_root, name)):
            return name
        i += 1


def make_train_env(env_id: str, n_envs: int, seed: int, max_episode_steps: int, goal_kwargs: dict):
    """
    Vectorized training env (DummyVecEnv) + observation normalization (VecNormalize).
    Reward is NOT normalized to keep shaped returns interpretable/comparable.
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
    Single-env evaluation wrapper + VecNormalize in eval mode.
    Note: we later copy obs_rms from train_env to ensure identical scaling.
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
    Creates a fresh eval env and loads VecNormalize statistics from training.
    This enables consistent offline evaluation of saved checkpoints/models.
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
# Task-level evaluation utilities (uses metrics emitted by GoalSafeWrapper)
# ======================================================
def rollout_goal_metrics(model: SAC, env, n_episodes: int, deterministic: bool = True):
    """
    Runs n_episodes on VecEnv(n_envs=1) and aggregates task metrics from info:
      - is_success, dist_to_goal, time_to_goal, violations, goal_dx0
    Used to produce learning curves beyond raw episodic reward.
    """
    ep_returns, ep_lens = [], []
    ep_success, ep_final_dist, ep_time_to_goal, ep_violations, ep_abs_dx0 = [], [], [], [], []

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

            # goal_dx0 is constant per episode (sampled at reset): better for binning by difficulty
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
    )


def summarize_goal_metrics(ep):
    """Converts episode arrays into scalar summaries (for TensorBoard + CSV)."""
    success_rate = float(np.mean(ep["success"])) if len(ep["success"]) else 0.0
    mean_final_dist = float(np.nanmean(ep["final_dist"])) if len(ep["final_dist"]) else float("nan")

    ttg = ep["time_to_goal"][ep["success"] == 1]  # time-to-goal only for successful episodes
    mean_time_to_goal = float(np.mean(ttg)) if len(ttg) else float("nan")

    vio = ep["violations"]
    violation_rate = float(np.mean(vio > 0)) if len(vio) else 0.0
    mean_violations = float(np.mean(vio)) if len(vio) else 0.0

    mean_return = float(np.mean(ep["returns"])) if len(ep["returns"]) else float("nan")
    mean_len = float(np.mean(ep["lens"])) if len(ep["lens"]) else float("nan")

    return dict(
        success_rate=success_rate,
        mean_final_dist=mean_final_dist,
        mean_time_to_goal=mean_time_to_goal,
        violation_rate=violation_rate,
        mean_violations=mean_violations,
        mean_return=mean_return,
        mean_len=mean_len,
    )


def moving_average(x: np.ndarray, w: int = 3):
    """Simple causal-ish smoothing for plotting (pads with first valid value)."""
    if w <= 1 or len(x) < w:
        return x
    y = np.convolve(x, np.ones(w, dtype=np.float32) / float(w), mode="valid")
    pad = np.full((w - 1,), y[0], dtype=np.float32)
    return np.concatenate([pad, y], axis=0)


class GoalEvalMetricsCallback(BaseCallback):
    """
    Periodically evaluates the CURRENT policy on:
      - source dynamics
      - target dynamics
    and logs task-level metrics to TensorBoard + CSV.
    """
    def __init__(self, eval_env_source, eval_env_target, eval_freq: int, n_eval_episodes: int, results_dir: str):
        super().__init__()
        self.eval_env_source = eval_env_source
        self.eval_env_target = eval_env_target
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.results_dir = results_dir
        self._last_eval_step = 0

        # In-memory history used to generate PNG plots at the end
        self.history = {"timesteps": [], "source": [], "target": [], "source_ep": [], "target_ep": []}

        # CSV is useful for report tables (mean±std can be computed offline)
        self.metrics_csv = os.path.join(results_dir, "goal_metrics_eval.csv")
        if not os.path.exists(self.metrics_csv):
            with open(self.metrics_csv, "w", encoding="utf-8") as f:
                f.write(
                    "timesteps,domain,success_rate,mean_final_dist,mean_time_to_goal,violation_rate,mean_violations,mean_return,mean_len\n"
                )

    def _eval_one(self, env, domain: str):
        ep = rollout_goal_metrics(self.model, env, self.n_eval_episodes, deterministic=True)
        sm = summarize_goal_metrics(ep)

        # TensorBoard scalars (task-level, more interpretable than raw reward alone)
        self.logger.record(f"goal/{domain}_success_rate", sm["success_rate"])
        self.logger.record(f"goal/{domain}_mean_final_dist", sm["mean_final_dist"])
        self.logger.record(f"goal/{domain}_mean_time_to_goal", sm["mean_time_to_goal"])
        self.logger.record(f"goal/{domain}_violation_rate", sm["violation_rate"])
        self.logger.record(f"goal/{domain}_mean_violations", sm["mean_violations"])

        # CSV row (one row per eval point, per domain)
        with open(self.metrics_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{self.num_timesteps},{domain},{sm['success_rate']},{sm['mean_final_dist']},{sm['mean_time_to_goal']},"
                f"{sm['violation_rate']},{sm['mean_violations']},{sm['mean_return']},{sm['mean_len']}\n"
            )
        return sm, ep

    def _on_step(self) -> bool:
        # Run evaluation only every eval_freq steps (keeps training fast)
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
        Writes PNG plots for the report:
          - learning curves (source vs target) for key task metrics
          - final success vs goal distance bins
          - final histograms (dist-to-goal and time-to-goal)
        """
        os.makedirs(plots_dir, exist_ok=True)
        import matplotlib.pyplot as plt

        ts = np.asarray(self.history["timesteps"], dtype=np.int64)
        if len(ts) == 0:
            return

        def series(domain_list, key):
            return np.asarray([d[key] for d in domain_list], dtype=np.float32)

        # Learning curves (smoothed for readability)
        keys = ["success_rate", "mean_final_dist", "mean_time_to_goal", "violation_rate", "mean_violations"]
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

        # Final: success vs |dx0| bins (shows performance vs goal difficulty)
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

        # Histograms (distribution-level view for the final policy)
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
# Reward eval logger (SB3 EvalCallback -> TensorBoard scalars)
# ======================================================
class TensorboardEvalLogger(BaseCallback):
    """
    Mirrors SB3 EvalCallback results into TensorBoard under:
      eval/source_mean_reward, eval/target_mean_reward (+ std)
    Useful to compare with non-goal-safe baselines that only log reward curves.
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
# Final results report (reward-based, for the report table)
# ======================================================
def safe_load_model(model_cls, path: str, device: str):
    """Loads a saved SB3 model; returns None if missing/corrupted."""
    try:
        return model_cls.load(path, device=device)
    except Exception:
        return None


def eval_one(tag: str, model_path: str, eval_s_s, eval_s_t, n_episodes: int, device: str):
    """Evaluates one model on (source->source) and (source->target) and formats lines."""
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
    Produces a single text file with final mean±std returns for:
      - best on source (EvalCallback)
      - best on target
      - final checkpoint
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
# MAIN (end-to-end training + evaluation + artifacts)
# ======================================================
def main():
    # Minimal set of task parameters passed into GoalSafeWrapper
    goal_kwargs = dict(
        goal_mode=CONFIG["goal_mode"],
        goal_x_range=CONFIG["goal_x_range"],
        goal_threshold=CONFIG["goal_threshold"],
        goal_terminate_threshold=CONFIG["goal_terminate_threshold"],
        success_hold_steps=CONFIG["success_hold_steps"],
    )

    base_run_name = f"{RUN_PREFIX}_seed{CONFIG['seed']}"
    run_name = next_run_name(base_run_name)

    # Standard folder layout: logs/ (TB), models/ (checkpoints), results/ (txt+png+csv)
    log_dir = os.path.join("logs", run_name)
    model_dir = os.path.join("models", run_name)
    results_dir = os.path.join("results", run_name)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Persist config for reproducibility (hyperparams + goal definition)
    with open(os.path.join(results_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    # Train on SOURCE dynamics with vectorization + observation normalization
    train_env = make_train_env(
        CONFIG["source_id"],
        n_envs=CONFIG["n_envs"],
        seed=CONFIG["seed"],
        max_episode_steps=CONFIG["max_episode_steps"],
        goal_kwargs=goal_kwargs,
    )

    # Eval on both SOURCE and TARGET using the same task definition (goal kwargs)
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

    # Share obs normalization statistics (train->eval consistency)
    eval_env_source.obs_rms = train_env.obs_rms
    eval_env_target.obs_rms = train_env.obs_rms

    # SB3 EvalCallbacks select "best_model.zip" based on eval mean reward
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

    # Additional TB curves for reward (to align with other baselines)
    tb_reward_logger = TensorboardEvalLogger(eval_cb_source, eval_cb_target)

    # Task-level metrics curves (success, time-to-goal, violations) + CSV + plots
    goal_metrics_cb = GoalEvalMetricsCallback(
        eval_env_source=eval_env_source,
        eval_env_target=eval_env_target,
        eval_freq=CONFIG["eval_freq"],
        n_eval_episodes=CONFIG["n_eval_episodes"],
        results_dir=results_dir,
    )

    # Periodic checkpoints (useful if training crashes; also stores VecNormalize stats)
    checkpoint_cb = CheckpointCallback(
        save_freq=200_000,
        save_path=model_dir,
        name_prefix="ckpt",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    callbacks = CallbackList([eval_cb_source, eval_cb_target, tb_reward_logger, goal_metrics_cb, checkpoint_cb])

    # SAC policy trained on vectorized env; evaluation uses deterministic actions
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
        # Always persist artifacts even if training is interrupted
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

        # Offline final evaluation (re-load envs + VecNormalize from disk)
        vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")
        if os.path.exists(vecnorm_path):
            write_final_results(
                run_name, CONFIG["source_id"], CONFIG["target_id"],
                model_dir, results_dir,
                vecnorm_path, CONFIG["seed"], CONFIG["max_episode_steps"],
                CONFIG["n_final_eval_episodes"], goal_kwargs,
                device=CONFIG["device"],
            )

        # Produce PNGs for report figures (curves + histograms)
        try:
            goal_metrics_cb.save_plots(
                plots_dir,
                goal_abs_dx_max=float(abs(CONFIG["goal_x_range"][1])),
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
