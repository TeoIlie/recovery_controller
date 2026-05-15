import numpy as np
import pytest
from recovery_controller.observation_builder import (
    ObservationBuilder,
    normalize,
    parse_norm_bounds,
    FEATURE_ORDER,
)

# Test norm bounds matching the on-car norm_bounds.yaml (drift_real obs)
NORM_BOUNDS_RAW = {
    "linear_vel_x": {"min": -5.0, "max": 20.0},
    "linear_vel_y": {"min": -10.0, "max": 10.0},
    "frenet_u": {"min": -3.141592653589793, "max": 3.141592653589793},
    "frenet_n": {"min": -1.1, "max": 1.1},
    "ang_vel_z": {"min": -5.0, "max": 5.0},
    "beta": {"min": -1.0471975511965976, "max": 1.0471975511965976},
    "curr_avg_wheel_omega": {"min": 0.0, "max": 2949.3087557603685},
    "lookahead_curvatures": {"min": -1.95, "max": 1.95},
    "lookahead_widths": {"min": 1.2, "max": 2.2},
}

NORM_BOUNDS = parse_norm_bounds(NORM_BOUNDS_RAW)
OBS_DIM = sum(length for _, length in FEATURE_ORDER)


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


def test_normalize_asymmetric_bounds():
    """Asymmetric bounds: 0 maps to (0 - lo) scaled."""
    # linear_vel_x: [-5, 20]; v=0 → 2*(5/25) - 1 = -0.6
    assert normalize(0.0, -5.0, 20.0) == pytest.approx(-0.6)


def test_normalize_degenerate_range():
    """lo == hi: avoid div-by-zero, return 0.0."""
    assert normalize(5.0, 3.0, 3.0) == 0.0
    assert normalize(-1.0, 0.0, 0.0) == 0.0


# --- parse_norm_bounds() ---


def test_parse_norm_bounds_valid():
    bounds = parse_norm_bounds(NORM_BOUNDS_RAW)
    assert bounds["linear_vel_x"] == (-5.0, 20.0)
    assert bounds["curr_avg_wheel_omega"] == (0.0, 2949.3087557603685)


def test_parse_norm_bounds_missing_feature():
    incomplete = {"linear_vel_x": {"min": -5.0, "max": 20.0}}
    with pytest.raises(ValueError, match="missing required features"):
        parse_norm_bounds(incomplete)


def test_parse_norm_bounds_validates_unique_names():
    """Each unique name in FEATURE_ORDER must appear once in the YAML."""
    unique_names = {name for name, _ in FEATURE_ORDER}
    assert set(NORM_BOUNDS.keys()) == unique_names


def test_parse_norm_bounds_tolerates_extra_entries():
    """Extra entries (e.g. dead-path bounds) must not raise — revival path."""
    extended = {
        **NORM_BOUNDS_RAW,
        "prev_accl_cmd": {"min": -5.0, "max": 5.0},
        "curr_vel_cmd": {"min": -5.0, "max": 20.0},
        "delta": {"min": -0.5, "max": 0.5},
    }
    bounds = parse_norm_bounds(extended)
    assert bounds["prev_accl_cmd"] == (-5.0, 5.0)
    assert bounds["delta"] == (-0.5, 0.5)


# --- ObservationBuilder shape and range ---


@pytest.fixture
def builder():
    return ObservationBuilder(
        norm_bounds=NORM_BOUNDS,
        zone_width=2.2,
        dt=0.01,
    )


def test_obs_dim_is_fourteen():
    """drift_real + sparse_width_obs: 7 scalars + 5 curvatures + 2 widths = 14."""
    assert OBS_DIM == 14


def test_build_returns_correct_shape(builder):
    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_build_all_in_range(builder):
    """Realistic mid-range inputs all normalize within [-1, 1]."""
    obs = builder.build(
        vx=3.0,
        vy=0.1,
        frenet_u=0.2,
        frenet_n=-0.3,
        ang_vel_z=0.5,
        beta=0.1,
        curr_avg_wheel_omega=1000.0,
    )
    assert np.all(obs >= -1.0)
    assert np.all(obs <= 1.0)


def test_build_clips_extreme_inputs(builder):
    """Out-of-range inputs clip to ±1."""
    obs = builder.build(
        vx=1e6,
        vy=-1e6,
        frenet_u=10.0,
        frenet_n=-10.0,
        ang_vel_z=1e6,
        beta=-1e6,
        curr_avg_wheel_omega=1e6,
    )
    assert np.all(obs >= -1.0)
    assert np.all(obs <= 1.0)
    assert obs[0] == pytest.approx(1.0)   # vx → max
    assert obs[1] == pytest.approx(-1.0)  # vy → min


# --- Individual feature normalization ---


def test_obs0_linear_vel_x(builder):
    """linear_vel_x: asymmetric bounds [-5, 20]; v=0 → -0.6."""
    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert obs[0] == pytest.approx(normalize(0.0, -5.0, 20.0), abs=1e-5)


def test_obs1_linear_vel_y(builder):
    """linear_vel_y: symmetric bounds [-10, 10]; v=2.5 → 0.25."""
    obs = builder.build(0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert obs[1] == pytest.approx(0.25, abs=1e-5)


def test_obs2_frenet_u(builder):
    """frenet_u=π/2 with bounds [-π, π] → 0.5."""
    obs = builder.build(0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0, 0.0)
    assert obs[2] == pytest.approx(0.5, abs=1e-5)


def test_obs3_frenet_n(builder):
    """frenet_n=0.55 with bounds [-1.1, 1.1] → 0.5."""
    obs = builder.build(0.0, 0.0, 0.0, 0.55, 0.0, 0.0, 0.0)
    assert obs[3] == pytest.approx(0.5, abs=1e-5)


def test_obs4_ang_vel_z(builder):
    """ang_vel_z=2.5 with bounds [-5, 5] → 0.5."""
    obs = builder.build(0.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0)
    assert obs[4] == pytest.approx(0.5, abs=1e-5)


def test_obs5_beta(builder):
    """beta=π/6 with bounds [-π/3, π/3] → 0.5."""
    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 6, 0.0)
    assert obs[5] == pytest.approx(0.5, abs=1e-5)


def test_obs6_curr_avg_wheel_omega(builder):
    """curr_avg_wheel_omega=0 with bounds [0, ~2949] → -1.0."""
    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert obs[6] == pytest.approx(-1.0, abs=1e-5)


# --- Lookahead curvatures (always 0 → normalized midpoint) ---


def test_curvatures_zero(builder):
    """Straight-line zone: curvatures = 0, bounds [-1.95, 1.95] → 0.0."""
    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for i in range(7, 12):
        assert obs[i] == pytest.approx(0.0, abs=1e-5), f"obs[{i}] should be 0.0"


# --- Lookahead widths (zone_width replicated) ---


def test_widths_at_upper_bound(builder):
    """zone_width=2.2 with bounds [1.2, 2.2] → max → 1.0 for both sparse slots."""
    obs = builder.build(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for i in range(12, 14):
        assert obs[i] == pytest.approx(1.0, abs=1e-5), f"obs[{i}] should be 1.0"


def test_widths_narrower_zone():
    """zone_width=1.7 with bounds [1.2, 2.2] → midpoint → 0.0."""
    b = ObservationBuilder(NORM_BOUNDS, zone_width=1.7, dt=0.01)
    obs = b.build(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for i in range(12, 14):
        assert obs[i] == pytest.approx(0.0, abs=1e-5)


# --- Dead-code path: reset() and step() ----------------------------------
# These methods are not called by recovery_node under drift_real, but are
# retained for potential reintroduction.  The smoke tests below inject
# extended norm_bounds (with curr_vel_cmd / prev_accl_cmd entries restored)
# to exercise the original logic exactly as it would behave if revived.

DEAD_PATH_NORM_BOUNDS_RAW = {
    **NORM_BOUNDS_RAW,
    "curr_vel_cmd": {"min": -5.0, "max": 20.0},
    "prev_accl_cmd": {"min": -5.0, "max": 5.0},
}
DEAD_PATH_NORM_BOUNDS = {
    name: (float(e["min"]), float(e["max"]))
    for name, e in DEAD_PATH_NORM_BOUNDS_RAW.items()
}


@pytest.fixture
def dead_path_builder():
    """Builder configured with extended bounds so reset/step are active."""
    return ObservationBuilder(
        norm_bounds=DEAD_PATH_NORM_BOUNDS,
        zone_width=2.2,
        dt=0.01,
    )


def test_dead_reset_sets_initial_speed(dead_path_builder):
    dead_path_builder.reset(7.5)
    assert dead_path_builder.curr_vel_cmd == pytest.approx(7.5)
    assert dead_path_builder.prev_steering_cmd == 0.0
    assert dead_path_builder.prev_accl_cmd == 0.0


def test_dead_step_updates_action_state(dead_path_builder):
    dead_path_builder.reset(5.0)
    dead_path_builder.step(0.5, -0.3)  # accl_norm=0.5, steer_norm=-0.3

    assert dead_path_builder.prev_accl_cmd == pytest.approx(0.5 * 5.0)  # 2.5
    assert dead_path_builder.prev_steering_cmd == pytest.approx(-0.3)
    # curr_vel_cmd = 5.0 + 2.5 * 0.01 = 5.025
    assert dead_path_builder.curr_vel_cmd == pytest.approx(5.025)


def test_dead_step_clamps_vel_cmd(dead_path_builder):
    dead_path_builder.reset(19.99)
    dead_path_builder.step(1.0, 0.0)
    assert dead_path_builder.curr_vel_cmd == pytest.approx(20.0)


def test_dead_step_clamps_vel_cmd_negative(dead_path_builder):
    dead_path_builder.reset(-4.99)
    dead_path_builder.step(-1.0, 0.0)
    assert dead_path_builder.curr_vel_cmd == pytest.approx(-5.0)


def test_action_bound_fallbacks_default_to_zero(builder):
    """When norm_bounds lacks dead-path entries, a_max/v_min/v_max are 0.0."""
    assert builder.a_max == 0.0
    assert builder.v_min == 0.0
    assert builder.v_max == 0.0


def test_action_bounds_populated_when_entries_present(dead_path_builder):
    """When norm_bounds has dead-path entries, action bounds are populated."""
    assert dead_path_builder.a_max == 5.0
    assert dead_path_builder.v_min == -5.0
    assert dead_path_builder.v_max == 20.0


def test_build_does_not_mutate_dead_state(builder):
    """Live build() must not touch prev_steering_cmd / prev_accl_cmd / curr_vel_cmd."""
    builder.prev_steering_cmd = 0.7
    builder.prev_accl_cmd = -1.3
    builder.curr_vel_cmd = 4.2
    builder.build(3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0)
    assert builder.prev_steering_cmd == 0.7
    assert builder.prev_accl_cmd == -1.3
    assert builder.curr_vel_cmd == 4.2


def test_dead_path_inactive_with_live_bounds(builder):
    """Without prev_accl_cmd / curr_vel_cmd entries, dead path no-ops safely."""
    # a_max / v_min / v_max fall back to 0.0 → step() can't change vel_cmd
    builder.reset(5.0)
    builder.step(1.0, 0.5)
    assert builder.prev_steering_cmd == pytest.approx(0.5)
    assert builder.prev_accl_cmd == pytest.approx(0.0)  # no a_max
    assert builder.curr_vel_cmd == pytest.approx(0.0)   # clamped to [0, 0]
