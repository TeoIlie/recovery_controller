"""Pure Python observation builder — no ROS2 dependencies, unit-testable.

Constructs the normalized observation vector expected by the trained PPO
recovery policy (Gym-Khana ``drift_real`` preset).  See
SIM_TO_REAL_IMPL.md §Observation Vector for the full specification.
"""

import numpy as np

# Canonical feature order — defines the mapping from feature names to
# obs-vector slots.  Each entry is (name, length).  Total obs dim is the sum
# of the lengths.  Must match the order the policy was trained with.
FEATURE_ORDER: list[tuple[str, int]] = [
    ("linear_vel_x", 1),
    ("linear_vel_y", 1),
    ("frenet_u", 1),
    ("frenet_n", 1),
    ("ang_vel_z", 1),
    ("beta", 1),
    ("curr_avg_wheel_omega", 1),
    ("lookahead_curvatures", 5),
    ("lookahead_widths", 2),  # sparse_width_obs=true: first + last only
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

    # Validate all required features are present (one entry per unique name)
    required = {name for name, _ in FEATURE_ORDER}
    missing = required - set(bounds.keys())
    if missing:
        raise ValueError(
            f"norm_bounds YAML is missing required features: {sorted(missing)}"
        )

    return bounds


class ObservationBuilder:
    """Builds the normalized observation vector.

    Parameters
    ----------
    norm_bounds : Dict mapping feature name to (lo, hi) tuple.
    zone_width  : Recovery zone width (m) used as the constant lookahead width.
    dt          : Control timestep (s). Used only by the dead step() path.
    """

    def __init__(
        self,
        norm_bounds: dict[str, tuple[float, float]],
        zone_width: float,
        dt: float,
    ):
        self.norm_bounds = norm_bounds
        self.obs_dim = sum(length for _, length in FEATURE_ORDER)
        self.zone_width = zone_width
        self.dt = dt

        # Dead path: reset()/step() and prev_*/curr_vel_cmd retained for
        # potential revival. Bounds use .get() fallbacks so missing
        # norm_bounds entries don't break the live obs.
        self.v_min = norm_bounds.get("curr_vel_cmd", (0.0, 0.0))[0]
        self.v_max = norm_bounds.get("curr_vel_cmd", (0.0, 0.0))[1]
        self.a_max = norm_bounds.get("prev_accl_cmd", (0.0, 0.0))[1]
        self.prev_steering_cmd = 0.0
        self.prev_accl_cmd = 0.0
        self.curr_vel_cmd = 0.0

    # Dead path below (not called under drift_real).
    def reset(self, initial_speed: float) -> None:
        """[DEAD] Initialize action state on recovery activation."""
        self.prev_steering_cmd = 0.0
        self.prev_accl_cmd = 0.0
        self.curr_vel_cmd = initial_speed

    def step(self, raw_accl: float, raw_steer: float) -> None:
        """[DEAD] Update action state from normalized policy outputs."""
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
        beta: float,
        curr_avg_wheel_omega: float,
    ) -> np.ndarray:
        """Construct the full normalized observation.

        Parameters
        ----------
        vx, vy                : Body-frame velocities (m/s).
        frenet_u              : Heading error (rad).
        frenet_n              : Lateral offset (m).
        ang_vel_z             : Yaw rate (rad/s) from IMU or Vicon.
        beta                  : Sideslip angle (rad).
        curr_avg_wheel_omega  : Current wheel angular velocity (rad/s) from
                                VESC ERPM.

        Returns
        -------
        obs : np.ndarray of shape (obs_dim,), each element in [-1, 1].
        """
        scalars = {
            "linear_vel_x": vx,
            "linear_vel_y": vy,
            "frenet_u": frenet_u,
            "frenet_n": frenet_n,
            "ang_vel_z": ang_vel_z,
            "beta": beta,
            "curr_avg_wheel_omega": curr_avg_wheel_omega,
        }
        # Straight-line recovery zone: curvatures are 0, widths are constant
        arrays = {
            "lookahead_curvatures": np.zeros(5, dtype=np.float32),
            "lookahead_widths": np.full(2, self.zone_width, dtype=np.float32),
        }

        obs = np.zeros(self.obs_dim, dtype=np.float32)
        i = 0
        for name, length in FEATURE_ORDER:
            lo, hi = self.norm_bounds[name]
            if length == 1:
                obs[i] = normalize(scalars[name], lo, hi)
                i += 1
            else:
                for v in arrays[name]:
                    obs[i] = normalize(float(v), lo, hi)
                    i += 1

        return obs
