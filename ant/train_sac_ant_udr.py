import os
import gymnasium as gym
import numpy as np

from env.custom_ant import *  # registers CustomAnt* env IDs (UDR Mass source/target)

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# =============================================================================
# ENV BUILDERS (dynamics only)
# =============================================================================
def make_monitored_env(env_id: str, max_episode_steps: int = 1000):
    """
    Single env instance for locomotion return evaluation.
    Wrapper stack:
      gym.make(env_id) -> TimeLimit -> Monitor

    NOTE: Task wrappers (GoalSafe / Fault injection) are NOT used here.
    This experiment is pure dynamics robustness via MASS-UDR.
    """
    env = gym.make(env_id)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)  # fixed horizon
    env = Monitor(env)  # logs episodic return/length
    return env


def next_run_name(base_name: str, log_root="logs"):
    """
    Avoids overwriting previous runs:
      logs/<base>_run1, logs/<base>_run2, ...
    """
    i = 1
    while True:
        name = f"{base_name}_run{i}"
        if not os.path.exists(os.path.join(log_root, name)):
            return name
        i += 1


def make_train_env(env_id: str, n_envs: int, seed: int, max_episode_steps: int):
    """
    Vectorized training env + VecNormalize (obs only).
    Reward normalization is OFF to keep returns comparable across domains.
    """
    env = make_vec_env(
        lambda: make_monitored_env(env_id, max_episode_steps=max_episode_steps),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return env


def make_eval_env(env_id: str, seed: int, max_episode_steps: int):
    """
    Single-env evaluation wrapper + VecNormalize in eval mode.
    """
    env = make_vec_env(
        lambda: make_monitored_env(env_id, max_episode_steps=max_episode_steps),
        n_envs=1,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env.training = False
    env.norm_reward = False
    return env


def load_eval_env_with_vecnorm(env_id: str, vecnorm_path: str, seed: int, max_episode_steps: int):
    """
    Rebuilds an eval env and attaches training-time observation normalization stats.
    This ensures evaluation is consistent and not biased by domain shift in obs scaling.
    """
    env = make_vec_env(
        lambda: make_monitored_env(env_id, max_episode_steps=max_episode_steps),
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
    Robust model loader: returns None if missing or unreadable.
    """
    if not os.path.exists(path):
        return None
    try:
        return model_cls.load(path, device=device)
    except Exception:
        return None


# =============================================================================
# TENSORBOARD EVAL LOGGER (log source/target eval curves)
# =============================================================================
class TensorboardEvalLogger(BaseCallback):
    """
    SB3 EvalCallback stores results internally (evaluations_results/length).
    This callback mirrors the latest eval outputs into TensorBoard:
      - eval/source_mean_reward, eval/source_std_reward
      - eval/target_mean_reward, eval/target_std_reward
    """
    def __init__(self, eval_source: EvalCallback, eval_target: EvalCallback):
        super().__init__()
        self.eval_source = eval_source
        self.eval_target = eval_target
        self._last_src = 0
        self._last_tgt = 0

    def _log_eval(self, eval_cb, prefix, last_attr):
        results = getattr(eval_cb, "evaluations_results", None)
        lengths = getattr(eval_cb, "evaluations_length", None)

        if results is None or len(results) == 0:
            return getattr(self, last_attr)
        if len(results) == getattr(self, last_attr):
            return getattr(self, last_attr)

        rewards = results[-1]
        self.logger.record(f"eval/{prefix}_mean_reward", float(np.mean(rewards)))
        self.logger.record(f"eval/{prefix}_std_reward", float(np.std(rewards)))

        if lengths is not None and len(lengths) > 0:
            self.logger.record(f"eval/{prefix}_mean_ep_length", float(np.mean(lengths[-1])))

        return len(results)

    def _on_step(self) -> bool:
        self._last_src = self._log_eval(self.eval_source, "source", "_last_src")
        self._last_tgt = self._log_eval(self.eval_target, "target", "_last_tgt")
        return True


# =============================================================================
# MASS-UDR CURRICULUM (dr_range schedule)
# =============================================================================
class CurriculumUDRCallback(BaseCallback):
    """
    Applies a piecewise-constant curriculum on mass UDR range.
    It updates env attributes (VecEnv.set_attr) during training:
      - dr_enabled
      - dr_range
      - resample_every
    """
    def __init__(self, vec_env, schedule, resample_every=2, verbose=1):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.schedule = list(schedule)   # list[(timestep, dr_range)]
        self.resample_every = int(resample_every)
        self._stage = -1

    def _apply(self, dr_range: float):
        self.vec_env.set_attr("dr_enabled", dr_range > 0.0)
        self.vec_env.set_attr("dr_range", float(dr_range))
        self.vec_env.set_attr("resample_every", self.resample_every)

    def _on_step(self) -> bool:
        t = self.num_timesteps

        stage = 0
        for i, (ts, _) in enumerate(self.schedule):
            if t >= ts:
                stage = i

        if stage != self._stage:
            self._stage = stage
            dr_range = float(self.schedule[stage][1])
            self._apply(dr_range)
            if self.verbose:
                print(f"[MASS-UDR] timesteps={t} -> dr_range={dr_range:.3f} | resample_every={self.resample_every}")

        return True


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Dynamics domains (same morphology shift on target; UDR is enabled only via set_attr during training/eval)
    source_id = "CustomAntUDRMass-source-v0"
    target_id = "CustomAntUDRMass-target-v0"

    n_envs = 6
    total_timesteps = 1_000_000
    seed = 1
    max_episode_steps = 1000

    eval_freq = 20_000
    n_eval_episodes = 20         # periodic model selection (lower variance)
    n_final_eval_episodes = 50   # final report numbers

    # Curriculum: ramp mass UDR up to 0.20 (~matches target leg +20% scale)
    udr_schedule = [
        (0,        0.00),
        (100_000,  0.03),
        (250_000,  0.06),
        (450_000,  0.10),
        (650_000,  0.15),
        (850_000,  0.20),
    ]
    resample_every = 2

    base_run_name = "SAC_CustomAnt_UDRmass_seed1"
    run_name = next_run_name(base_run_name)

    log_dir = os.path.join("logs", run_name)
    model_dir = os.path.join("models", run_name)
    results_dir = os.path.join("results", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Train env (vectorized) + obs normalization
    train_env = make_train_env(source_id, n_envs=n_envs, seed=seed, max_episode_steps=max_episode_steps)

    # Initialize UDR state (stage 0)
    train_env.set_attr("dr_enabled", udr_schedule[0][1] > 0.0)
    train_env.set_attr("dr_range", float(udr_schedule[0][1]))
    train_env.set_attr("resample_every", int(resample_every))

    # Eval envs (source + target). Share obs_rms for consistent normalization.
    eval_env_source = make_eval_env(source_id, seed + 1000, max_episode_steps=max_episode_steps)
    eval_env_target = make_eval_env(target_id, seed + 2000, max_episode_steps=max_episode_steps)
    eval_env_source.obs_rms = train_env.obs_rms
    eval_env_target.obs_rms = train_env.obs_rms

    # Periodic evaluation: best on source + best on target (separate folders)
    eval_cb_source = EvalCallback(
        eval_env_source,
        best_model_save_path=model_dir,   # writes best_model.zip (best on SOURCE)
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    eval_cb_target = EvalCallback(
        eval_env_target,
        best_model_save_path=os.path.join(model_dir, "best_on_target"),  # best_model.zip (best on TARGET)
        log_path=os.path.join(log_dir, "target_eval"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    tb_eval_logger = TensorboardEvalLogger(eval_cb_source, eval_cb_target)
    udr_cb = CurriculumUDRCallback(train_env, schedule=udr_schedule, resample_every=resample_every, verbose=1)
    callbacks = CallbackList([eval_cb_source, eval_cb_target, tb_eval_logger, udr_cb])

    # SAC configuration (stable/transfer-friendly defaults)
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
        gradient_steps=2,

        ent_coef=0.003,          # slightly more conservative exploration
        use_sde=True,
        sde_sample_freq=4,

        policy_kwargs=dict(net_arch=[256, 256]),
        seed=seed,
        device="cpu",
    )

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # Save final artifacts + VecNormalize stats
    final_model_path = os.path.join(model_dir, "final_model")
    vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")
    model.save(final_model_path)
    train_env.save(vecnorm_path)

    train_env.close()
    eval_env_source.close()
    eval_env_target.close()

    # ----------------------------
    # FINAL EVALUATION (deterministic)
    # ----------------------------
    best_on_source_path = os.path.join(model_dir, "best_model.zip")
    best_on_target_path = os.path.join(model_dir, "best_on_target", "best_model.zip")
    final_model_zip = final_model_path + ".zip"

    eval_s_s = load_eval_env_with_vecnorm(source_id, vecnorm_path, seed + 3000, max_episode_steps=max_episode_steps)
    eval_s_t = load_eval_env_with_vecnorm(target_id, vecnorm_path, seed + 4000, max_episode_steps=max_episode_steps)

    def eval_one(tag: str, model_path: str):
        m = safe_load_model(SAC, model_path, device="cpu")
        if m is None:
            return [f"{tag}: NOT FOUND ({model_path})"]

        mean_s_s, std_s_s = evaluate_policy(m, eval_s_s, n_eval_episodes=n_final_eval_episodes, deterministic=True)
        mean_s_t, std_s_t = evaluate_policy(m, eval_s_t, n_eval_episodes=n_final_eval_episodes, deterministic=True)

        return [
            f"[{tag}] {os.path.basename(model_path)}",
            f"  source -> source : {mean_s_s:.2f} +/- {std_s_s:.2f}",
            f"  source -> target : {mean_s_t:.2f} +/- {std_s_t:.2f}",
        ]

    lines = []
    lines.append("========== FINAL RESULTS (deterministic) ==========")
    lines.append(f"Run name: {run_name}")
    lines.append(f"Episodes: {n_final_eval_episodes}")
    lines.append("")
    lines += eval_one("BEST ON SOURCE", best_on_source_path)
    lines.append("")
    lines += eval_one("BEST ON TARGET", best_on_target_path)
    lines.append("")
    lines += eval_one("FINAL MODEL", final_model_zip)
    lines.append("===================================================")

    eval_s_s.close()
    eval_s_t.close()

    out_path = os.path.join(results_dir, "final_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines) + "\n")
    print(f"Saved results to: {out_path}")
    print("TensorBoard: tensorboard --logdir logs")


if __name__ == "__main__":
    main()
