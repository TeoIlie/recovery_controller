"""Pure Python observation builder — no ROS2 dependencies, unit-testable.

Constructs the normalized observation vector expected by the trained PPO
recovery policy.  See SIM_TO_REAL_IMPL.md §Observation Vector for the full
specification.
"""

import numpy as np

# Canonical feature order — defines the mapping from feature names to
# obs-vector indices.  Must match the order the policy was trained with.
FEATURE_ORDER: list[str] = [
    "linear_vel_x",  #  0
    "linear_vel_y",  #  1
    "frenet_u",  #  2
    "frenet_n",  #  3
    "ang_vel_z",  #  4
    "delta",  #  5
    "beta",  #  6
    "prev_steering_cmd",  #  7
    "prev_accl_cmd",  #  8
    "prev_avg_wheel_omega",  #  9
    "curr_vel_cmd",  # 10
    "lookahead_curvatures",  # 11
    "lookahead_curvatures",  # 12
    "lookahead_curvatures",  # 13
    "lookahead_curvatures",  # 14
    "lookahead_curvatures",  # 15
    "lookahead_widths",  # 16
    "lookahead_widths",  # 17
]


def normalize(value: float, lo: float, hi: float) -> float:
    """Normalize to [-1, 1] matching sim's utils.py:normalize_feature."""
    range_val = hi - lo
    if np.isclose(range_val, 0.0, atol=1e-9):
        return 0.0
    return float(np.clip(2.0 * (value - lo) / range_val - 1.0, -1.0, 1.0))


def parse_norm_bounds(raw: dict[str, dict]) -> dict[str, tuple[float, float]]:
    """Parse a norm bounds YAML dict into {name: (min, max)} tuples"""
    bounds = {}
    for name, entry in raw.items():
        bounds[name] = (float(entry["min"]), float(entry["max"]))

    # Validate all required features are present
    required = set(FEATURE_ORDER)
    missing = required - set(bounds.keys())
    if missing:
        raise ValueError(
            f"norm_bounds YAML is missing required features: {sorted(missing)}"
        )

    return bounds


class ObservationBuilder:
    """Builds the normalized observation vector.

    Tracks internal action state (prev_steering_cmd, prev_accl_cmd,
    curr_vel_cmd) that updates each tick.  Call ``reset()`` when the
    recovery episode begins and ``build()`` every tick thereafter.

    Parameters
    ----------
    norm_bounds : Dict mapping feature name to (lo, hi) tuple.
    zone_width  : Full width of the recovery zone (metres).
    dt          : Control timestep (s).
    """

    def __init__(
        self,
        norm_bounds: dict[str, tuple[float, float]],
        zone_width: float,
        dt: float,
    ):
        self.norm_bounds = norm_bounds
        self.obs_dim = len(FEATURE_ORDER)
        self.zone_width = zone_width
        self.dt = dt

        # Derive action bounds from norm_bounds (single source of truth)
        self.v_min, self.v_max = norm_bounds["linear_vel_x"]
        self.a_max = norm_bounds["prev_accl_cmd"][1]

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

    def step(self, raw_accl: float, raw_steer: float) -> None:
        """Update internal action state after each inference tick.

        Parameters
        ----------
        raw_accl  : Normalized acceleration in [-1, 1] from policy.
        raw_steer : Normalized steering in [-1, 1] from policy.
        """
        self.prev_accl_cmd = raw_accl * self.a_max
        self.prev_steering_cmd = raw_steer
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
        """Construct the full normalized observation.

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
        obs : np.ndarray of shape (obs_dim,), each element in [-1, 1].
        """
        raw = {
            "linear_vel_x": vx,
            "linear_vel_y": vy,
            "frenet_u": frenet_u,
            "frenet_n": frenet_n,
            "ang_vel_z": ang_vel_z,
            "delta": delta,
            "beta": beta,
            "prev_steering_cmd": self.prev_steering_cmd,
            "prev_accl_cmd": self.prev_accl_cmd,
            "prev_avg_wheel_omega": wheel_omega,
            "curr_vel_cmd": self.curr_vel_cmd,
            "lookahead_curvatures": 0.0,  # straight-line zone
            "lookahead_widths": self.zone_width,
        }

        obs = np.zeros(self.obs_dim, dtype=np.float32)
        for i, name in enumerate(FEATURE_ORDER):
            lo, hi = self.norm_bounds[name]
            obs[i] = normalize(raw[name], lo, hi)

        return obs
