import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def runner():
    """Create a PolicyRunner with a mocked ONNX session."""
    with patch("recovery_controller.policy_runner.ort") as mock_ort:
        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[0.5, -0.3]])]
        mock_ort.InferenceSession.return_value = mock_session
        from recovery_controller.policy_runner import PolicyRunner

        yield PolicyRunner("/fake/model.onnx", s_max=0.5)


# --- predict() ---


def test_predict_returns_tuple(runner):
    obs = np.zeros(18, dtype=np.float32)
    result = runner.predict(obs)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_predict_returns_correct_values(runner):
    obs = np.zeros(18, dtype=np.float32)
    raw_accl, raw_steer = runner.predict(obs)
    assert raw_accl == pytest.approx(0.5)
    assert raw_steer == pytest.approx(-0.3)


def test_predict_calls_session_run(runner):
    obs = np.zeros(18, dtype=np.float32)
    runner.predict(obs)
    runner.session.run.assert_called_once()
    args, _ = runner.session.run.call_args
    assert args[0] == ["action"]


def test_predict_passes_obs_to_session(runner):
    obs = np.ones(18, dtype=np.float32) * 0.42
    runner.predict(obs)
    _, kwargs = runner.session.run.call_args
    fed_obs = (
        kwargs["obs"] if "obs" in kwargs else runner.session.run.call_args[0][1]["obs"]
    )
    assert fed_obs.shape == (1, 18)
    np.testing.assert_array_almost_equal(fed_obs[0], obs)


def test_predict_clips_actions():
    with patch("recovery_controller.policy_runner.ort") as mock_ort:
        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[1.5, -2.0]])]
        mock_ort.InferenceSession.return_value = mock_session
        from recovery_controller.policy_runner import PolicyRunner

        r = PolicyRunner("/fake/model.onnx")
        accl, steer = r.predict(np.zeros(18, dtype=np.float32))
        assert accl == pytest.approx(1.0)
        assert steer == pytest.approx(-1.0)


# --- denorm_steering() ---


def test_denorm_steering_full_right(runner):
    assert runner.denorm_steering(1.0) == pytest.approx(0.5)


def test_denorm_steering_full_left(runner):
    assert runner.denorm_steering(-1.0) == pytest.approx(-0.5)


def test_denorm_steering_center(runner):
    assert runner.denorm_steering(0.0) == pytest.approx(0.0)


def test_denorm_steering_partial(runner):
    assert runner.denorm_steering(0.6) == pytest.approx(0.3)


def test_denorm_steering_custom_s_max():
    with patch("recovery_controller.policy_runner.ort") as mock_ort:
        mock_ort.InferenceSession.return_value = MagicMock()
        from recovery_controller.policy_runner import PolicyRunner

        r = PolicyRunner("/fake/model.onnx", s_max=0.4)
        assert r.denorm_steering(1.0) == pytest.approx(0.4)
        assert r.denorm_steering(-0.5) == pytest.approx(-0.2)


# --- __init__ ---


def test_session_created_with_path():
    with patch("recovery_controller.policy_runner.ort") as mock_ort:
        mock_ort.InferenceSession.return_value = MagicMock()
        from recovery_controller.policy_runner import PolicyRunner

        PolicyRunner("/some/path/model.onnx")
        mock_ort.InferenceSession.assert_called_once_with(
            "/some/path/model.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )


def test_default_s_max():
    with patch("recovery_controller.policy_runner.ort") as mock_ort:
        mock_ort.InferenceSession.return_value = MagicMock()
        from recovery_controller.policy_runner import PolicyRunner

        r = PolicyRunner("/fake/model.onnx")
        assert r.s_max == 0.5
