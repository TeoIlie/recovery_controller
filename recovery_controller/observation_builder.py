"""Pure Python observation builder — no ROS2 dependencies, unit-testable.

Constructs the 18-element normalized observation vector expected by the
trained PPO recovery policy.  See SIM_TO_REAL_IMPL.md §Observation Vector
for the full specification.
"""

import numpy as np


def normalize(value: float, lo: float, hi: float) -> float:
    """Normalize to [-1, 1] matching sim's utils.py:normalize_feature."""
    return float(np.clip(2.0 * (value - lo) / (hi - lo) - 1.0, -1.0, 1.0))


# Normalization bounds — must match training config exactly.
# TODO update, or make readable from config file
NORM_BOUNDS: list[tuple[float, float]] = [
    (-5.0, 20.0),  # 0  linear_vel_x
    (-10.0, 10.0),  # 1  linear_vel_y
    (-np.pi, np.pi),  # 2  frenet_u  (heading error)
    (-1.1, 1.1),  # 3  frenet_n  (lateral offset)
    (-5.0, 5.0),  # 4  ang_vel_z
    (-0.5, 0.5),  # 5  delta  (steering angle)
    (-np.pi / 3, np.pi / 3),  # 6  beta  (sideslip)
    (-1.0, 1.0),  # 7  prev_steering_cmd  (raw network output)
    (-5.0, 5.0),  # 8  prev_accl_cmd  (denorm m/s²)
    (0.0, 2612.24),  # 9  prev_avg_wheel_omega
    (-5.0, 20.0),  # 10 curr_vel_cmd
    (-1.95, 1.95),  # 11 lookahead_curvature[0]
    (-1.95, 1.95),  # 12 lookahead_curvature[1]
    (-1.95, 1.95),  # 13 lookahead_curvature[2]
    (-1.95, 1.95),  # 14 lookahead_curvature[3]
    (-1.95, 1.95),  # 15 lookahead_curvature[4]
    (1.2, 2.2),  # 16 lookahead_width[0]  (sparse)
    (1.2, 2.2),  # 17 lookahead_width[1]  (sparse)
]

OBS_DIM = 18


class ObservationBuilder:
    """Builds the 18-element normalized observation vector.

    Tracks internal action state (prev_steering_cmd, prev_accl_cmd,
    curr_vel_cmd) that updates each tick.  Call ``reset()`` when the
    recovery episode begins and ``build()`` every tick thereafter.

    Parameters
    ----------
    zone_width : Full width of the recovery zone (metres).
    a_max      : Max acceleration (m/s²), from deployment config.
    v_min, v_max : Speed clamp bounds (m/s).
    dt         : Control timestep (s).
    """

    def __init__(
        self, zone_width: float, a_max: float, v_min: float, v_max: float, dt: float
    ):
        self.zone_width = zone_width
        self.a_max = a_max
        self.v_min = v_min
        self.v_max = v_max
        self.dt = dt

        # Internal action state — set by reset() and updated by step()
        self.prev_steering_cmd = 0.0
        self.prev_accl_cmd = 0.0
        self.curr_vel_cmd = 0.0

    def reset(self, initial_speed: float) -> None:
        """Call once when recovery activates.

        Parameters
        ----------
        initial_speed : Current forward speed (m/s) from VESC ERPM.
        """
        self.prev_steering_cmd = 0.0
        self.prev_accl_cmd = 0.0
        self.curr_vel_cmd = initial_speed

    def step(self, raw_action: np.ndarray) -> None:
        """Update internal action state after each inference tick.

        Parameters
        ----------
        raw_action : [accl_norm, steer_norm] in [-1, 1] from policy.
        """
        self.prev_accl_cmd = float(raw_action[0]) * self.a_max
        self.prev_steering_cmd = float(raw_action[1])
        self.curr_vel_cmd += self.prev_accl_cmd * self.dt
        self.curr_vel_cmd = float(np.clip(self.curr_vel_cmd, self.v_min, self.v_max))

    def build(
        self,
        vx: float,
        vy: float,
        frenet_u: float,
        frenet_n: float,
        ang_vel_z: float,
        delta: float,
        beta: float,
        wheel_omega: float,
    ) -> np.ndarray:
        """Construct the full 18-element normalized observation.

        Parameters
        ----------
        vx, vy       : Body-frame velocities (m/s).
        frenet_u     : Heading error (rad).
        frenet_n     : Lateral offset (m).
        ang_vel_z    : Yaw rate (rad/s) from IMU.
        delta        : Current steering angle (rad) from VESC servo.
        beta         : Sideslip angle (rad).
        wheel_omega  : Wheel angular velocity (rad/s) from VESC ERPM.

        Returns
        -------
        obs : np.ndarray of shape (18,), each element in [-1, 1].
        """
        raw = np.zeros(OBS_DIM, dtype=np.float32)

        # --- Sensor-derived features (placeholder: 0.0) ---
        raw[0] = vx  # linear_vel_x
        raw[1] = vy  # linear_vel_y
        raw[2] = frenet_u  # frenet_u
        raw[3] = frenet_n  # frenet_n
        raw[4] = ang_vel_z # yaw rate
        raw[5] = delta # delta 
        raw[6] = beta  # beta

        # --- Action history (tracked internally) ---
        raw[7] = self.prev_steering_cmd # prev_steering_cmd
        raw[8] = self.prev_accl_cmd # prev_accl_cmd
        raw[9] = wheel_omega  # prev_avg_wheel_omega
        raw[10] = self.curr_vel_cmd # curr_vel_cmd

        # --- Fixed features (straight-line zone) ---
        raw[11:16] = 0.0  # lookahead curvatures (straight line = 0)
        raw[16] = self.zone_width  # lookahead_width[0]
        raw[17] = self.zone_width  # lookahead_width[1]

        # --- Normalize ---
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for i in range(OBS_DIM):
            lo, hi = NORM_BOUNDS[i]
            obs[i] = normalize(raw[i], lo, hi)

        return obs
