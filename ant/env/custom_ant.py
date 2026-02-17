# env/custom_ant.py
"""
Custom MuJoCo Ant environment with *dynamics-only* domain randomization (DR/UDR).

IMPORTANT DESIGN CHOICE
-----------------------
Wrappers (GoalSafeWrapper, ActionClipWrapper, MotorFaultWrapper) live in env/wrappers.py.
This file MUST stay focused on:
  - Base Ant dynamics (source/target shift)
  - Mass UDR (runtime via set_attr)
  - Motor UDR (runtime via set_attr) with two modes:
      * per-actuator down (independent weakening)
      * global symmetric (shared gain)
  - Friction DR (runtime via set_attr) on floor/ground geoms
  - Env registration IDs used across your training scripts

This prevents duplicated wrapper implementations and import confusion.
"""

import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.ant_v5 import AntEnv

try:
    import mujoco
except Exception:
    mujoco = None


class CustomAntEnv(AntEnv):
    """
    Unified Custom Ant supporting:
      - Source vs Target domain shift (target_leg_scale on leg bodies)
      - Mass UDR (one-sided UP) controlled at runtime:
          dr_enabled, dr_range, resample_every
      - Motor UDR controlled at runtime:
          motor_dr_enabled, motor_dr_range, motor_resample_every
        Two motor modes, selected by env registration flags:
          (A) per-actuator down: s_i ~ Uniform(1-r, 1)
          (B) global symmetric : g   ~ Uniform(1-r, 1+r) shared for all actuators
      - Friction DR on floor/ground geoms controlled at runtime:
          friction_dr_enabled, friction_dr_range, friction_resample_every

    Guarantees:
      - No compounding: parameters reset from stored originals every episode.
      - Safe defaults: all DR switches OFF unless explicitly enabled via set_attr.
    """

    def __init__(
        self,
        domain="source",
        target_leg_scale=1.20,

        # Motor mode selector (set by env ID):
        #   - global symmetric uses shared g ~ U(1-r, 1+r)
        #   - otherwise uses per-actuator down s_i ~ U(1-r, 1)
        motor_udr_global=False,
        motor_udr_symmetric=False,

        # Locks: prevent enabling DR in env variants that shouldn't have it
        allow_mass_udr=True,
        allow_motor_udr=True,

        # Optional: apply motor DR to subset of actuators
        motor_apply="all",           # "all" | "subset"
        motor_actuator_idx=None,     # list[int] if subset

        **kwargs,
    ):
        super().__init__(**kwargs)

        self.domain = str(domain)
        self.target_leg_scale = float(target_leg_scale)

        self.allow_mass_udr = bool(allow_mass_udr)
        self.allow_motor_udr = bool(allow_motor_udr)

        # Store originals (avoid compounding)
        self._mass0 = self.model.body_mass.copy()
        self._gainprm0 = self.model.actuator_gainprm.copy()
        self._biasprm0 = self.model.actuator_biasprm.copy()
        self._geom_friction0 = self.model.geom_friction.copy()

        # Episode counter for resampling schedules
        self._ep_count = 0

        # -----------------------
        # Mass UDR runtime controls
        # -----------------------
        self.dr_enabled = False
        self.dr_range = 0.0              # r => hi = 1+r, scale ~ U(1, hi)
        self.resample_every = 1
        self._last_mass_scales = None

        # -----------------------
        # Motor UDR runtime controls
        # -----------------------
        self.motor_dr_enabled = False
        self.motor_dr_range = 0.0        # r => scales from mode
        self.motor_resample_every = 1
        self._last_motor_scales = None

        self.motor_udr_global = bool(motor_udr_global)
        self.motor_udr_symmetric = bool(motor_udr_symmetric)

        self.motor_apply = str(motor_apply)
        self.motor_actuator_idx = None if motor_actuator_idx is None else list(motor_actuator_idx)

        # -----------------------
        # Friction DR runtime controls
        # -----------------------
        self.friction_dr_enabled = False
        self.friction_dr_range = 0.0     # r => scale ~ U(1-r, 1+r)
        self.friction_resample_every = 1
        self._last_friction_scale = None
        self._floor_geom_idx = self._find_floor_geoms()

    # ---------------------------
    # helpers
    # ---------------------------
    @staticmethod
    def _sample_uniform(lo: float, hi: float, size):
        return np.random.uniform(float(lo), float(hi), size=size)

    @staticmethod
    def _sample_one_sided_up(hi: float, size):
        # U[1, hi]
        return np.random.uniform(1.0, float(hi), size=size)

    @staticmethod
    def _sample_one_sided_down(lo: float, size):
        # U[lo, 1]
        return np.random.uniform(float(lo), 1.0, size=size)

    @staticmethod
    def _sample_friction_scale(r: float) -> float:
        lo = max(0.0, 1.0 - float(r))
        hi = 1.0 + float(r)
        return float(np.random.uniform(lo, hi))

    @staticmethod
    def _leg_body_slice():
        # Convention used in your project: skip first 2 bodies, scale legs only
        return slice(2, None)

    def _find_floor_geoms(self):
        """
        Attempt to detect floor/ground geoms by name.
        If none found, friction DR becomes a no-op (safe default).
        """
        idx = []
        try:
            for j in range(self.model.ngeom):
                name = self.model.geom(j).name
                if name is None:
                    continue
                lname = str(name).lower()
                if ("floor" in lname) or ("ground" in lname):
                    idx.append(j)
        except Exception:
            idx = []
        return idx

    def _motor_indices(self):
        n_act = self.model.actuator_gainprm.shape[0]
        if self.motor_apply == "subset":
            idx = [] if self.motor_actuator_idx is None else list(self.motor_actuator_idx)
            idx = [int(i) for i in idx if 0 <= int(i) < n_act]
            return idx
        return list(range(n_act))

    # ---------------------------
    # MuJoCo internal reset hook
    # ---------------------------
    def reset_model(self):
        self._ep_count += 1

        # 1) Reset to originals (NO compounding)
        masses = self._mass0.copy()
        gainprm = self._gainprm0.copy()
        biasprm = self._biasprm0.copy()
        geom_fric = self._geom_friction0.copy()

        # 2) Source/Target domain shift (legs heavier on target)
        leg_slice = self._leg_body_slice()
        n_legs = len(masses[leg_slice])

        if self.domain == "target":
            masses[leg_slice] *= self.target_leg_scale

        # 3) MASS UDR (source only; one-sided UP)
        if (self.domain != "target") and self.allow_mass_udr and self.dr_enabled and (self.dr_range > 0.0):
            hi = 1.0 + float(self.dr_range)

            if self._last_mass_scales is None:
                self._last_mass_scales = self._sample_one_sided_up(hi, size=n_legs)

            if (self._ep_count - 1) % max(int(self.resample_every), 1) == 0:
                self._last_mass_scales = self._sample_one_sided_up(hi, size=n_legs)

            masses[leg_slice] *= self._last_mass_scales

        # 4) MOTOR UDR (actuator gain scaling)
        idx = self._motor_indices()

        if self.allow_motor_udr and self.motor_dr_enabled and (self.motor_dr_range > 0.0) and len(idx) > 0:
            r = float(self.motor_dr_range)

            # (B) global symmetric: g ~ U(1-r, 1+r), shared for all selected motors
            if self.motor_udr_global and self.motor_udr_symmetric:
                lo = max(0.0, 1.0 - r)
                hi = 1.0 + r

                if self._last_motor_scales is None:
                    self._last_motor_scales = np.array([np.random.uniform(lo, hi)], dtype=np.float32)

                if (self._ep_count - 1) % max(int(self.motor_resample_every), 1) == 0:
                    self._last_motor_scales = np.array([np.random.uniform(lo, hi)], dtype=np.float32)

                g = float(self._last_motor_scales[0])
                gainprm[idx, 0] *= g

            # (A) per-actuator down: s_i ~ U(1-r, 1)
            else:
                lo = max(0.0, 1.0 - r)

                if self._last_motor_scales is None:
                    self._last_motor_scales = self._sample_one_sided_down(lo, size=len(idx))

                if (self._ep_count - 1) % max(int(self.motor_resample_every), 1) == 0:
                    self._last_motor_scales = self._sample_one_sided_down(lo, size=len(idx))

                gainprm[idx, 0] *= self._last_motor_scales

        # 5) FRICTION DR (floor/ground only)
        if self.friction_dr_enabled and (self.friction_dr_range > 0.0) and len(self._floor_geom_idx) > 0:
            if self._last_friction_scale is None:
                self._last_friction_scale = self._sample_friction_scale(self.friction_dr_range)

            if (self._ep_count - 1) % max(int(self.friction_resample_every), 1) == 0:
                self._last_friction_scale = self._sample_friction_scale(self.friction_dr_range)

            s = float(self._last_friction_scale)
            for j in self._floor_geom_idx:
                geom_fric[j, :] *= s

        # 6) Write back to model
        self.model.body_mass[:] = masses
        self.model.actuator_gainprm[:] = gainprm
        self.model.actuator_biasprm[:] = biasprm
        self.model.geom_friction[:] = geom_fric

        return super().reset_model()


# =============================================================================
# ENV REGISTRATION (KEEP IDS STABLE ACROSS EXPERIMENTS)
# =============================================================================

# TRUE BASELINE (no DR)
gym.register(
    id="CustomAntBase-source-v0",
    entry_point="env.custom_ant:CustomAntEnv",
    kwargs=dict(
        domain="source",
        target_leg_scale=1.20,
        allow_mass_udr=False,
        allow_motor_udr=False,
    ),
)
gym.register(
    id="CustomAntBase-target-v0",
    entry_point="env.custom_ant:CustomAntEnv",
    kwargs=dict(
        domain="target",
        target_leg_scale=1.20,
        allow_mass_udr=False,
        allow_motor_udr=False,
    ),
)

# UDR MASSES (runtime via set_attr: dr_enabled/dr_range/resample_every)
gym.register(
    id="CustomAntUDRMass-source-v0",
    entry_point="env.custom_ant:CustomAntEnv",
    kwargs=dict(
        domain="source",
        target_leg_scale=1.20,
        allow_mass_udr=True,
        allow_motor_udr=False,
    ),
)
gym.register(
    id="CustomAntUDRMass-target-v0",
    entry_point="env.custom_ant:CustomAntEnv",
    kwargs=dict(
        domain="target",
        target_leg_scale=1.20,
        allow_mass_udr=True,   # harmless on target; target shift handled separately
        allow_motor_udr=False,
    ),
)

# UDR MOTORS (per-actuator one-sided DOWN) – runtime via set_attr: motor_dr_enabled/motor_dr_range/motor_resample_every
gym.register(
    id="CustomAntUDRMotor-source-v0",
    entry_point="env.custom_ant:CustomAntEnv",
    kwargs=dict(
        domain="source",
        target_leg_scale=1.20,
        allow_mass_udr=False,
        allow_motor_udr=True,
        motor_udr_global=False,
        motor_udr_symmetric=False,
    ),
)
gym.register(
    id="CustomAntUDRMotor-target-v0",
    entry_point="env.custom_ant:CustomAntEnv",
    kwargs=dict(
        domain="target",
        target_leg_scale=1.20,
        allow_mass_udr=False,
        allow_motor_udr=True,
        motor_udr_global=False,
        motor_udr_symmetric=False,
    ),
)

# UDR MOTORS (global + symmetric) – g ~ Uniform(1-r, 1+r)
gym.register(
    id="CustomAntUDRMotorGlobalSym-source-v0",
    entry_point="env.custom_ant:CustomAntEnv",
    kwargs=dict(
        domain="source",
        target_leg_scale=1.20,
        allow_mass_udr=False,
        allow_motor_udr=True,
        motor_udr_global=True,
        motor_udr_symmetric=True,
    ),
)
gym.register(
    id="CustomAntUDRMotorGlobalSym-target-v0",
    entry_point="env.custom_ant:CustomAntEnv",
    kwargs=dict(
        domain="target",
        target_leg_scale=1.20,
        allow_mass_udr=False,
        allow_motor_udr=True,
        motor_udr_global=True,
        motor_udr_symmetric=True,
    ),
)

# Combined MASS + MOTOR UDR (both controlled at runtime)
gym.register(
    id="CustomAntUDRMassMotor-source-v0",
    entry_point="env.custom_ant:CustomAntEnv",
    kwargs=dict(
        domain="source",
        target_leg_scale=1.20,
        allow_mass_udr=True,
        allow_motor_udr=True,
    ),
)
gym.register(
    id="CustomAntUDRMassMotor-target-v0",
    entry_point="env.custom_ant:CustomAntEnv",
    kwargs=dict(
        domain="target",
        target_leg_scale=1.20,
        allow_mass_udr=True,
        allow_motor_udr=True,
    ),
)
