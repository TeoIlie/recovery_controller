import math
import numpy as np
import pytest
from recovery_controller.observation_builder import (
    ObservationBuilder,
    normalize,
    NORM_BOUNDS,
    OBS_DIM,
)

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


# --- ObservationBuilder shape and range ---


@pytest.fixture
def builder():
    return ObservationBuilder(
        zone_width=2.2, a_max=5.0, v_min=-5.0, v_max=20.0, dt=0.01
    )


def test_build_returns_correct_shape(builder):
    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_build_all_in_range(builder):
    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.all(obs >= -1.0)
    assert np.all(obs <= 1.0)


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
    action = np.array([0.5, -0.3])  # accl_norm=0.5, steer_norm=-0.3
    builder.step(action)

    assert builder.prev_accl_cmd == pytest.approx(0.5 * 5.0)  # 2.5 m/s²
    assert builder.prev_steering_cmd == pytest.approx(-0.3)
    # curr_vel_cmd = 5.0 + 2.5 * 0.01 = 5.025
    assert builder.curr_vel_cmd == pytest.approx(5.025)


def test_step_clamps_vel_cmd(builder):
    builder.reset(19.99)
    action = np.array([1.0, 0.0])  # full accel
    builder.step(action)
    # 19.99 + 5.0*0.01 = 20.04 → clamped to 20.0
    assert builder.curr_vel_cmd == pytest.approx(20.0)


def test_step_clamps_vel_cmd_negative(builder):
    builder.reset(-4.99)
    action = np.array([-1.0, 0.0])  # full brake
    builder.step(action)
    # -4.99 + (-5.0)*0.01 = -5.04 → clamped to -5.0
    assert builder.curr_vel_cmd == pytest.approx(-5.0)


def test_build_uses_action_state_after_step(builder):
    """build() must reflect internal state set by reset() and step()."""
    builder.reset(5.0)
    action = np.array([0.5, -0.3])  # accl_norm=0.5, steer_norm=-0.3
    builder.step(action)

    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # prev_steering_cmd = -0.3, bounds [-1, 1]
    assert obs[7] == pytest.approx(normalize(-0.3, -1.0, 1.0), abs=1e-5)
    # prev_accl_cmd = 2.5 m/s², bounds [-5, 5]
    assert obs[8] == pytest.approx(normalize(2.5, -5.0, 5.0), abs=1e-5)
    # curr_vel_cmd = 5.0 + 2.5*0.01 = 5.025, bounds [-5, 20]
    assert obs[10] == pytest.approx(normalize(5.025, -5.0, 20.0), abs=1e-5)
