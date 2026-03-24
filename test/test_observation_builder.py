import math
import numpy as np
import pytest
from recovery_controller.observation_builder import (
    ObservationBuilder,
    normalize,
    parse_norm_bounds,
    FEATURE_ORDER,
)

# Test norm bounds matching the simulator output format
NORM_BOUNDS_RAW = {
    "linear_vel_x": {"min": -5.0, "max": 20.0},
    "linear_vel_y": {"min": -10.0, "max": 10.0},
    "frenet_u": {"min": -3.141592653589793, "max": 3.141592653589793},
    "frenet_n": {"min": -1.1, "max": 1.1},
    "ang_vel_z": {"min": -5.0, "max": 5.0},
    "delta": {"min": -0.5, "max": 0.5},
    "beta": {"min": -1.0471975511965976, "max": 1.0471975511965976},
    "prev_steering_cmd": {"min": -1.0, "max": 1.0},
    "prev_accl_cmd": {"min": -5.0, "max": 5.0},
    "prev_avg_wheel_omega": {"min": 0.0, "max": 2612.24},
    "curr_vel_cmd": {"min": -5.0, "max": 20.0},
    "lookahead_curvatures": {"min": -1.95, "max": 1.95},
    "lookahead_widths": {"min": 1.2, "max": 2.2},
}

NORM_BOUNDS = parse_norm_bounds(NORM_BOUNDS_RAW)
OBS_DIM = len(FEATURE_ORDER)


# --- normalize() ---


def test_normalize_at_min():
    assert normalize(0.0, 0.0, 10.0) == pytest.approx(-1.0)


def test_normalize_at_max():
    assert normalize(10.0, 0.0, 10.0) == pytest.approx(1.0)


def test_normalize_at_midpoint():
    assert normalize(5.0, 0.0, 10.0) == pytest.approx(0.0)


def test_normalize_below_min_clips():
    assert normalize(-100.0, 0.0, 10.0) == pytest.approx(-1.0)


def test_normalize_above_max_clips():
    assert normalize(100.0, 0.0, 10.0) == pytest.approx(1.0)


def test_normalize_quarter():
    """25% of range → -0.5."""
    assert normalize(2.5, 0.0, 10.0) == pytest.approx(-0.5)


def test_normalize_negative_bounds():
    """Symmetric bounds: 0 maps to 0."""
    assert normalize(0.0, -5.0, 5.0) == pytest.approx(0.0)


# --- parse_norm_bounds() ---


def test_parse_norm_bounds_valid():
    bounds = parse_norm_bounds(NORM_BOUNDS_RAW)
    assert bounds["linear_vel_x"] == (-5.0, 20.0)
    assert bounds["delta"] == (-0.5, 0.5)


def test_parse_norm_bounds_missing_feature():
    incomplete = {"linear_vel_x": {"min": -5.0, "max": 20.0}}
    with pytest.raises(ValueError, match="missing required features"):
        parse_norm_bounds(incomplete)


# --- ObservationBuilder shape and range ---


@pytest.fixture
def builder():
    return ObservationBuilder(
        norm_bounds=NORM_BOUNDS,
        zone_width=2.2,
        dt=0.01,
    )


def test_build_returns_correct_shape(builder):
    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_build_all_in_range(builder):
    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.all(obs >= -1.0)
    assert np.all(obs <= 1.0)


def test_derived_action_bounds(builder):
    """a_max, v_min, v_max are derived from norm_bounds."""
    assert builder.a_max == 5.0
    assert builder.v_min == -5.0
    assert builder.v_max == 20.0


# --- Individual feature normalization ---


def test_obs4_ang_vel_z(builder):
    """ang_vel_z=2.5 with bounds [-5, 5] → midpoint+quarter → 0.5."""
    obs = builder.build(
        0.0, 0.0, 0.0, 0.0, ang_vel_z=2.5, delta=0.0, beta=0.0, wheel_omega=0.0
    )
    expected = normalize(2.5, -5.0, 5.0)
    assert obs[4] == pytest.approx(expected, abs=1e-5)


def test_obs5_delta(builder):
    """delta=0.25 rad with bounds [-0.5, 0.5] → 0.5."""
    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, delta=0.25, beta=0.0, wheel_omega=0.0)
    expected = normalize(0.25, -0.5, 0.5)
    assert obs[5] == pytest.approx(expected, abs=1e-5)


def test_obs5_delta_zero(builder):
    """delta=0 → midpoint of [-0.5, 0.5] → 0.0."""
    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, delta=0.0, beta=0.0, wheel_omega=0.0)
    assert obs[5] == pytest.approx(0.0, abs=1e-5)


def test_obs5_delta_clipped(builder):
    """delta beyond bounds clips to ±1."""
    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, delta=1.0, beta=0.0, wheel_omega=0.0)
    assert obs[5] == pytest.approx(1.0, abs=1e-5)


# --- Lookahead curvatures (always 0 → normalized midpoint) ---


def test_obs11_to_15_curvatures_zero(builder):
    """Straight line zone: curvatures = 0, bounds [-1.95, 1.95] → 0.0."""
    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for i in range(11, 16):
        assert obs[i] == pytest.approx(0.0, abs=1e-5), f"obs[{i}] should be 0.0"


# --- Lookahead widths (zone_width) ---


def test_obs16_17_widths(builder):
    """zone_width=2.2 with bounds [1.2, 2.2] → max → 1.0."""
    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    expected = normalize(2.2, 1.2, 2.2)
    assert obs[16] == pytest.approx(expected, abs=1e-5)
    assert obs[17] == pytest.approx(expected, abs=1e-5)


# --- reset() and step() ---


def test_reset_sets_initial_speed(builder):
    builder.reset(7.5)
    assert builder.curr_vel_cmd == pytest.approx(7.5)
    assert builder.prev_steering_cmd == 0.0
    assert builder.prev_accl_cmd == 0.0


def test_step_updates_action_state(builder):
    builder.reset(5.0)
    builder.step(0.5, -0.3)  # accl_norm=0.5, steer_norm=-0.3

    assert builder.prev_accl_cmd == pytest.approx(0.5 * 5.0)  # 2.5 m/s²
    assert builder.prev_steering_cmd == pytest.approx(-0.3)
    # curr_vel_cmd = 5.0 + 2.5 * 0.01 = 5.025
    assert builder.curr_vel_cmd == pytest.approx(5.025)


def test_step_clamps_vel_cmd(builder):
    builder.reset(19.99)
    builder.step(1.0, 0.0)  # full accel
    # 19.99 + 5.0*0.01 = 20.04 → clamped to 20.0
    assert builder.curr_vel_cmd == pytest.approx(20.0)


def test_step_clamps_vel_cmd_negative(builder):
    builder.reset(-4.99)
    builder.step(-1.0, 0.0)  # full brake
    # -4.99 + (-5.0)*0.01 = -5.04 → clamped to -5.0
    assert builder.curr_vel_cmd == pytest.approx(-5.0)


def test_build_uses_action_state_after_step(builder):
    """build() must reflect internal state set by reset() and step()."""
    builder.reset(5.0)
    builder.step(0.5, -0.3)  # accl_norm=0.5, steer_norm=-0.3

    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # prev_steering_cmd = -0.3, bounds [-1, 1]
    assert obs[7] == pytest.approx(normalize(-0.3, -1.0, 1.0), abs=1e-5)
    # prev_accl_cmd = 2.5 m/s², bounds [-5, 5]
    assert obs[8] == pytest.approx(normalize(2.5, -5.0, 5.0), abs=1e-5)
    # curr_vel_cmd = 5.0 + 2.5*0.01 = 5.025, bounds [-5, 20]
    assert obs[10] == pytest.approx(normalize(5.025, -5.0, 20.0), abs=1e-5)


# --- build→step ordering (simulates build→predict→step loop) ---


def _raw_obs(builder, **sensor_kw):
    """Helper: call build() and denormalize obs[7], obs[8], obs[10] back to raw."""
    defaults = dict(
        vx=3.0,
        vy=0.0,
        frenet_u=0.0,
        frenet_n=0.0,
        ang_vel_z=0.0,
        delta=0.0,
        beta=0.0,
        wheel_omega=0.0,
    )
    defaults.update(sensor_kw)
    obs = builder.build(**defaults)

    def denorm(val, lo, hi):
        return (val + 1.0) / 2.0 * (hi - lo) + lo

    steer_raw = denorm(obs[7], *NORM_BOUNDS["prev_steering_cmd"])
    accl_raw = denorm(obs[8], *NORM_BOUNDS["prev_accl_cmd"])
    vel_raw = denorm(obs[10], *NORM_BOUNDS["curr_vel_cmd"])
    return steer_raw, accl_raw, vel_raw


def test_build_step_ordering_first_tick_has_zero_cmds(builder):
    """After reset, the first build() must see prev cmds = 0 (no prior action)."""
    builder.reset(5.0)

    steer, accl, vel = _raw_obs(builder)

    assert steer == pytest.approx(0.0, abs=1e-4)
    assert accl == pytest.approx(0.0, abs=1e-4)
    assert vel == pytest.approx(5.0, abs=1e-4)


def test_build_step_ordering_multi_tick(builder):
    """Simulate build→predict→step for 3 ticks.

    Each tick's build() must contain the *previous* tick's action,
    not the current tick's (which hasn't been chosen yet).
    """
    builder.reset(5.0)

    # --- Tick 0: no prior action ---
    steer, accl, vel = _raw_obs(builder)
    assert steer == pytest.approx(0.0, abs=1e-4), "tick 0: prev_steer should be 0"
    assert accl == pytest.approx(0.0, abs=1e-4), "tick 0: prev_accl should be 0"
    assert vel == pytest.approx(
        5.0, abs=1e-4
    ), "tick 0: vel_cmd should be initial speed"

    # "predict" returns action_0
    action_0 = (0.6, -0.4)  # (raw_accl, raw_steer)
    builder.step(*action_0)

    # --- Tick 1: should see action_0 ---
    steer, accl, vel = _raw_obs(builder)
    assert steer == pytest.approx(
        action_0[1], abs=1e-4
    ), "tick 1: prev_steer should be action_0"
    assert accl == pytest.approx(
        action_0[0] * builder.a_max, abs=1e-4
    ), "tick 1: prev_accl should be action_0 * a_max"
    expected_vel_1 = 5.0 + action_0[0] * builder.a_max * builder.dt
    assert vel == pytest.approx(
        expected_vel_1, abs=1e-4
    ), "tick 1: vel_cmd should integrate action_0"

    # "predict" returns action_1
    action_1 = (-0.2, 0.8)
    builder.step(*action_1)

    # --- Tick 2: should see action_1, NOT action_0 ---
    steer, accl, vel = _raw_obs(builder)
    assert steer == pytest.approx(
        action_1[1], abs=1e-4
    ), "tick 2: prev_steer should be action_1"
    assert accl == pytest.approx(
        action_1[0] * builder.a_max, abs=1e-4
    ), "tick 2: prev_accl should be action_1 * a_max"
    expected_vel_2 = expected_vel_1 + action_1[0] * builder.a_max * builder.dt
    assert vel == pytest.approx(
        expected_vel_2, abs=1e-4
    ), "tick 2: vel_cmd should integrate action_1"
