# env/wrappers.py
import numpy as np
import gymnasium as gym

try:
    import mujoco
except Exception:
    mujoco = None


class ActionClipWrapper(gym.Wrapper):
    """
    Clips actions to [-clip, +clip] before stepping the environment.
    Useful for stabilizing training when dynamics vary (DR) or when
    the agent tends to produce high-torque actions.
    """
    def __init__(self, env, clip: float = 0.8):
        super().__init__(env)
        self.clip = float(clip)

    def step(self, action):
        action = np.clip(action, -self.clip, self.clip)
        return self.env.step(action)


class MotorFaultWrapper(gym.Wrapper):
    """
    Deterministic fault injection during an episode:
      - At a specified step t = fault_start_step, scales actuator_gainprm[idx,0]
        by fault_strength for selected actuator indices.
    This wrapper is meant for robustness EVAL protocols (Proposta 1).
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


class GoalSafeWrapper(gym.Wrapper):
    """
    Goal-conditioned navigation with explicit safety metrics.

    Observation augmentation:
      - Appends [dx, dy, dist] where d = goal_xy - torso_xy

    Reward shaping:
      - progress term: progress_coef * (prev_dist - dist)
      - energy penalty: -energy_coef * ||a||^2
      - safety penalty: based on torso tilt (roll/pitch)
      - success bonus on confirmed goal achievement
      - fall penalty if robot falls
      - near-goal shaping reduces dithering

    Termination:
      - success (inside goal region for K consecutive steps) + terminate radius
      - fall (excess tilt or torso too low)
      - or outer TimeLimit truncation
    """
    def __init__(
        self,
        env,
        goal_radius_range=(2.0, 6.0),
        goal_threshold=0.5,
        goal_terminate_threshold=None,
        success_hold_steps=5,

        # shaping
        progress_coef=5.0,
        success_bonus=8.0,
        near_goal_radius=1.0,
        near_goal_coef=0.2,

        # energy
        energy_coef=1e-3,

        # safety
        tilt_violation_thresh=0.7,
        tilt_fall_thresh=1.2,
        min_torso_z=0.20,
        violation_penalty=0.5,
        fall_penalty=10.0,
        continuous_safety=True,

        # goal sampling
        goal_mode="xy",            # "xy" or "x"
        goal_x_range=None,         # only if goal_mode="x"
    ):
        super().__init__(env)

        self.goal_radius_range = tuple(goal_radius_range)
        self.goal_threshold = float(goal_threshold)
        self.goal_terminate_threshold = float(goal_terminate_threshold) if goal_terminate_threshold is not None else float(goal_threshold)
        self.success_hold_steps = int(success_hold_steps)

        self.progress_coef = float(progress_coef)
        self.success_bonus = float(success_bonus)
        self.near_goal_radius = float(near_goal_radius)
        self.near_goal_coef = float(near_goal_coef)

        self.energy_coef = float(energy_coef)

        self.tilt_violation_thresh = float(tilt_violation_thresh)
        self.tilt_fall_thresh = float(tilt_fall_thresh)
        self.min_torso_z = float(min_torso_z)
        self.violation_penalty = float(violation_penalty)
        self.fall_penalty = float(fall_penalty)
        self.continuous_safety = bool(continuous_safety)

        self.goal_mode = str(goal_mode)
        # If goal_mode == "x" and goal_x_range not provided, fallback to radius range
        self.goal_x_range = tuple(goal_x_range) if goal_x_range is not None else self.goal_radius_range

        self.goal_xy = np.zeros(2, dtype=np.float32)
        self.goal_origin_xy = np.zeros(2, dtype=np.float32)

        self.prev_dist = None
        self.violations = 0
        self.step_count = 0
        self._success_streak = 0
        self._time_to_goal = -1

        # Expand observation space by 3 dims: dx, dy, dist
        orig = env.observation_space
        assert isinstance(orig, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(orig.shape[0] + 3,),
            dtype=np.float32,
        )

        self._torso_id = None

    # ---- curriculum hook ----
    def set_goal_x_range(self, new_range):
        """
        Runtime update used for curriculum learning.
        Call via VecEnv.env_method("set_goal_x_range", (xmin, xmax)).
        """
        self.goal_x_range = tuple(new_range)

    def _uw(self):
        return self.env.unwrapped

    def _get_torso_id(self):
        if self._torso_id is not None:
            return self._torso_id
        uw = self._uw()
        try:
            self._torso_id = uw.model.body("torso").id
        except Exception:
            if mujoco is None:
                raise RuntimeError("mujoco not available, cannot resolve torso id")
            self._torso_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        return self._torso_id

    def _torso_pos(self):
        uw = self._uw()
        tid = self._get_torso_id()
        return np.array(uw.data.xpos[tid], dtype=np.float32)

    def _roll_pitch(self):
        uw = self._uw()
        tid = self._get_torso_id()
        R = np.array(uw.data.xmat[tid], dtype=np.float32).reshape(3, 3)
        v = float(-R[2, 0])
        v = max(-1.0, min(1.0, v))
        pitch = float(np.arcsin(v))
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        return roll, pitch

    def _dist_to_goal(self):
        p = self._torso_pos()
        dxy = self.goal_xy - p[:2]
        return float(np.linalg.norm(dxy))

    def _augment_obs(self, obs):
        p = self._torso_pos()
        dxy = self.goal_xy - p[:2]
        dist = float(np.linalg.norm(dxy))
        extra = np.array([dxy[0], dxy[1], dist], dtype=np.float32)
        return np.concatenate([np.asarray(obs, dtype=np.float32), extra], axis=0)

    def _sample_goal(self):
        p = self._torso_pos()
        self.goal_origin_xy = p[:2].astype(np.float32).copy()

        if self.goal_mode == "x":
            xmin, xmax = self.goal_x_range
            dx0 = float(np.random.uniform(xmin, xmax))
            self.goal_xy = (self.goal_origin_xy + np.array([dx0, 0.0], dtype=np.float32)).astype(np.float32)
            return

        rmin, rmax = self.goal_radius_range
        r = float(np.random.uniform(rmin, rmax))
        ang = float(np.random.uniform(-np.pi, np.pi))
        self.goal_xy = (self.goal_origin_xy + np.array([r * np.cos(ang), r * np.sin(ang)], dtype=np.float32)).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.violations = 0
        self.step_count = 0
        self._success_streak = 0
        self._time_to_goal = -1

        self._sample_goal()
        self.prev_dist = self._dist_to_goal()

        goal_d0 = (self.goal_xy - self.goal_origin_xy).astype(np.float32)

        info = dict(info)
        info.update({
            "dist_to_goal": float(self.prev_dist),
            "is_success": False,
            "violations": 0,
            "time_to_goal": -1,

            "goal_dx0": float(goal_d0[0]),
            "goal_dy0": float(goal_d0[1]),
            "goal_x": float(self.goal_xy[0]),
            "goal_y": float(self.goal_xy[1]),
            "goal_origin_x": float(self.goal_origin_xy[0]),
            "goal_origin_y": float(self.goal_origin_xy[1]),
        })

        return self._augment_obs(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        dist = self._dist_to_goal()
        progress = float(self.prev_dist - dist)
        self.prev_dist = dist

        # Energy regularization
        a = np.asarray(action, dtype=np.float32)
        energy = self.energy_coef * float(np.sum(a * a))

        # Safety: torso tilt cost
        roll, pitch = self._roll_pitch()
        tilt = max(abs(roll), abs(pitch))
        violated = (tilt > self.tilt_violation_thresh)
        if violated:
            self.violations += 1

        if violated:
            safety_pen = self.violation_penalty * (float(tilt - self.tilt_violation_thresh) if self.continuous_safety else 1.0)
        else:
            safety_pen = 0.0

        # Fall detection
        torso_z = float(self._torso_pos()[2])
        fell = (
            (abs(roll) > self.tilt_fall_thresh) or
            (abs(pitch) > self.tilt_fall_thresh) or
            (torso_z < self.min_torso_z)
        )

        # Robust success: require K consecutive steps inside threshold
        inside_success = dist < self.goal_threshold
        inside_terminate = dist < self.goal_terminate_threshold

        if inside_success:
            self._success_streak += 1
            if self._success_streak == self.success_hold_steps and self._time_to_goal < 0:
                self._time_to_goal = int(self.step_count)
        else:
            self._success_streak = 0

        success = (self._time_to_goal >= 0)

        # Termination rules
        if success and inside_terminate:
            terminated = True
        if fell:
            terminated = True

        # Shaped reward
        reward = (self.progress_coef * progress) - energy - safety_pen

        # Near-goal shaping
        if dist < self.near_goal_radius:
            reward += self.near_goal_coef * float(self.near_goal_radius - dist)

        # Terminal bonuses/penalties
        if success and inside_terminate:
            reward += self.success_bonus
        if fell:
            reward -= self.fall_penalty

        goal_d0 = (self.goal_xy - self.goal_origin_xy).astype(np.float32)

        info = dict(info)
        info.update({
            "dist_to_goal": float(dist),
            "is_success": bool(success),
            "violations": int(self.violations),
            "time_to_goal": int(self._time_to_goal) if success else -1,
            "roll": float(roll),
            "pitch": float(pitch),
            "fell": bool(fell),

            "goal_dx0": float(goal_d0[0]),
            "goal_dy0": float(goal_d0[1]),
            "goal_x": float(self.goal_xy[0]),
            "goal_y": float(self.goal_xy[1]),
            "goal_origin_x": float(self.goal_origin_xy[0]),
            "goal_origin_y": float(self.goal_origin_xy[1]),
        })

        return self._augment_obs(obs), float(reward), terminated, truncated, info
