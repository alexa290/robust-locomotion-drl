# utils/env_factory.py
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from env.wrappers import ActionClipWrapper, MotorFaultWrapper, GoalSafeWrapper


def make_env(
    env_id: str,
    max_episode_steps: int = 1000,
    action_clip: float = 0.8,
    use_goal_safe: bool = False,
    goalsafe_kwargs: dict | None = None,
    fault_cfg: dict | None = None,
):
    """
    Unified env constructor.

    Key idea:
      - env_id selects the *dynamics* (source/target, DR/UDR variants, etc.)
      - wrappers select the *task* (goal-safe) and *evaluation protocol* (fault injection)

    Wrapper order:
      1) TimeLimit          -> consistent episode horizon (prevents eval "freeze")
      2) ActionClipWrapper  -> stabilizes learning
      3) MotorFaultWrapper  -> deterministic within-episode faults (mainly for Proposta 1 eval)
      4) GoalSafeWrapper    -> goal-conditioned + safety reward/metrics (Proposta 2)
      5) Monitor            -> records episode returns + lengths
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

    if use_goal_safe:
        goalsafe_kwargs = goalsafe_kwargs or {}
        env = GoalSafeWrapper(env, **goalsafe_kwargs)

    env = Monitor(env)
    return env
