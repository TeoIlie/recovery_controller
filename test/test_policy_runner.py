import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# Canonical action bounds for drift_real (matches f1tenth_std vehicle params)
S_MAX = 0.52
V_MIN = -5.0
V_MAX = 20.0
OBS_DIM = 14


def _make_runner(raw_action=(0.0, 0.0), s_max=S_MAX, v_min=V_MIN, v_max=V_MAX):
    """Build a PolicyRunner with a mocked ONNX session.

    Returns the runner; the mocked session is available on `runner.session`.
    """
    with patch("recovery_controller.policy_runner.ort") as mock_ort:
        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[*raw_action]])]
        mock_ort.InferenceSession.return_value = mock_session
        from recovery_controller.policy_runner import PolicyRunner

        return PolicyRunner(
            "/fake/model.onnx", s_max=s_max, v_min=v_min, v_max=v_max
        )


@pytest.fixture
def runner():
    """Default runner: session returns (raw_steer=0.5, raw_speed=-0.3)."""
    return _make_runner(raw_action=(0.5, -0.3))


# --- __init__ ---


def test_session_created_with_path():
    with patch("recovery_controller.policy_runner.ort") as mock_ort:
        mock_ort.InferenceSession.return_value = MagicMock()
        from recovery_controller.policy_runner import PolicyRunner

        PolicyRunner("/some/path/model.onnx", s_max=S_MAX, v_min=V_MIN, v_max=V_MAX)
        mock_ort.InferenceSession.assert_called_once_with(
            "/some/path/model.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )


def test_stores_action_bounds():
    r = _make_runner()
    assert r.s_max == S_MAX
    assert r.v_min == V_MIN
    assert r.v_max == V_MAX


def test_rejects_inverted_speed_bounds():
    with patch("recovery_controller.policy_runner.ort") as mock_ort:
        mock_ort.InferenceSession.return_value = MagicMock()
        from recovery_controller.policy_runner import PolicyRunner

        with pytest.raises(ValueError, match="v_min"):
            PolicyRunner("/fake.onnx", s_max=S_MAX, v_min=20.0, v_max=-5.0)


def test_rejects_zero_speed_range():
    with patch("recovery_controller.policy_runner.ort") as mock_ort:
        mock_ort.InferenceSession.return_value = MagicMock()
        from recovery_controller.policy_runner import PolicyRunner

        with pytest.raises(ValueError, match="v_min"):
            PolicyRunner("/fake.onnx", s_max=S_MAX, v_min=5.0, v_max=5.0)


def test_rejects_nonpositive_s_max():
    with patch("recovery_controller.policy_runner.ort") as mock_ort:
        mock_ort.InferenceSession.return_value = MagicMock()
        from recovery_controller.policy_runner import PolicyRunner

        with pytest.raises(ValueError, match="s_max"):
            PolicyRunner("/fake.onnx", s_max=0.0, v_min=V_MIN, v_max=V_MAX)
        with pytest.raises(ValueError, match="s_max"):
            PolicyRunner("/fake.onnx", s_max=-0.1, v_min=V_MIN, v_max=V_MAX)


# --- predict() ---


def test_predict_returns_tuple(runner):
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    result = runner.predict(obs)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_predict_returns_steer_then_speed(runner):
    """Action vector ordering: index 0 = steering, index 1 = longitudinal."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    raw_steer, raw_speed = runner.predict(obs)
    assert raw_steer == pytest.approx(0.5)
    assert raw_speed == pytest.approx(-0.3)


def test_predict_returns_python_floats(runner):
    """Floats (not numpy scalars) — cleaner downstream typing."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    raw_steer, raw_speed = runner.predict(obs)
    assert type(raw_steer) is float
    assert type(raw_speed) is float


def test_predict_calls_session_run(runner):
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    runner.predict(obs)
    runner.session.run.assert_called_once()
    args, _ = runner.session.run.call_args
    assert args[0] == ["action"]


def test_predict_passes_obs_to_session(runner):
    obs = np.ones(OBS_DIM, dtype=np.float32) * 0.42
    runner.predict(obs)
    args, _ = runner.session.run.call_args
    fed_obs = args[1]["obs"]
    assert fed_obs.shape == (1, OBS_DIM)
    assert fed_obs.dtype == np.float32
    np.testing.assert_array_almost_equal(fed_obs[0], obs)


def test_predict_clips_actions_high():
    r = _make_runner(raw_action=(1.5, 2.0))
    raw_steer, raw_speed = r.predict(np.zeros(OBS_DIM, dtype=np.float32))
    assert raw_steer == pytest.approx(1.0)
    assert raw_speed == pytest.approx(1.0)


def test_predict_clips_actions_low():
    r = _make_runner(raw_action=(-1.5, -2.0))
    raw_steer, raw_speed = r.predict(np.zeros(OBS_DIM, dtype=np.float32))
    assert raw_steer == pytest.approx(-1.0)
    assert raw_speed == pytest.approx(-1.0)


# --- denorm_steering() (symmetric) ---


def test_denorm_steering_full_right(runner):
    assert runner.denorm_steering(1.0) == pytest.approx(S_MAX)


def test_denorm_steering_full_left(runner):
    assert runner.denorm_steering(-1.0) == pytest.approx(-S_MAX)


def test_denorm_steering_center(runner):
    assert runner.denorm_steering(0.0) == pytest.approx(0.0)


def test_denorm_steering_partial(runner):
    assert runner.denorm_steering(0.5) == pytest.approx(0.5 * S_MAX)


def test_denorm_steering_custom_s_max():
    r = _make_runner(s_max=0.4)
    assert r.denorm_steering(1.0) == pytest.approx(0.4)
    assert r.denorm_steering(-0.5) == pytest.approx(-0.2)


# --- denorm_speed() (asymmetric) ---


def test_denorm_speed_at_zero_is_center(runner):
    """v_min=-5, v_max=20 → center = 7.5 (NOT 0)."""
    assert runner.denorm_speed(0.0) == pytest.approx(7.5)


def test_denorm_speed_at_plus_one_is_v_max(runner):
    assert runner.denorm_speed(1.0) == pytest.approx(V_MAX)


def test_denorm_speed_at_minus_one_is_v_min(runner):
    assert runner.denorm_speed(-1.0) == pytest.approx(V_MIN)


def test_denorm_speed_linearity(runner):
    """Half-range scaling: 0.5 → center + half-range = 7.5 + 6.25 = 13.75."""
    assert runner.denorm_speed(0.5) == pytest.approx(13.75)
    assert runner.denorm_speed(-0.5) == pytest.approx(7.5 - 6.25)


def test_denorm_speed_is_inverse_of_obs_normalize():
    """Cross-module invariant: denorm_speed inverts the obs-side normalize for
    linear_vel_x.  Catches any drift between obs/action normalization formulas.
    """
    from recovery_controller.observation_builder import normalize

    r = _make_runner()  # v_min=-5, v_max=20 (matches linear_vel_x bounds)
    for raw_v in [-4.7, -1.0, 0.0, 7.5, 12.3, 19.9]:
        norm = normalize(raw_v, V_MIN, V_MAX)
        round_trip = r.denorm_speed(norm)
        assert round_trip == pytest.approx(raw_v, abs=1e-5)


def test_denorm_speed_symmetric_bounds():
    """If v_min/v_max are symmetric, 0 maps to 0."""
    r = _make_runner(v_min=-10.0, v_max=10.0)
    assert r.denorm_speed(0.0) == pytest.approx(0.0)
    assert r.denorm_speed(1.0) == pytest.approx(10.0)
    assert r.denorm_speed(-1.0) == pytest.approx(-10.0)
