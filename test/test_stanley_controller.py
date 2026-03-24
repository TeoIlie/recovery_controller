import math
import pytest
from recovery_controller.stanley_controller import StanleyController


@pytest.fixture
def ctrl():
    return StanleyController(k=0.1, k_soft=0.1, k_heading=1.0, target_speed=1.5)


def test_zero_errors(ctrl):
    speed, steer = ctrl.get_action(vx=1.0, frenet_u=0.0, frenet_n=0.0)
    assert steer == pytest.approx(0.0)
    assert speed == pytest.approx(1.5)


def test_positive_heading_error_gives_negative_steering(ctrl):
    _, steer = ctrl.get_action(vx=1.0, frenet_u=0.3, frenet_n=0.0)
    assert steer < 0.0


def test_positive_cross_track_error_gives_negative_steering(ctrl):
    _, steer = ctrl.get_action(vx=1.0, frenet_u=0.0, frenet_n=0.5)
    assert steer < 0.0


def test_speed_matches_target(ctrl):
    speed, _ = ctrl.get_action(vx=2.0, frenet_u=0.1, frenet_n=0.2)
    assert speed == pytest.approx(1.5)


def test_cross_track_saturates_via_arctan():
    ctrl = StanleyController(k=10.0, k_soft=0.01, k_heading=1.0, target_speed=1.0)
    _, steer = ctrl.get_action(vx=0.01, frenet_u=0.0, frenet_n=1000.0)
    # arctan saturates near pi/2; steering = -arctan(...) should be near -pi/2
    assert steer == pytest.approx(-math.pi / 2, abs=0.05)


def test_k_soft_prevents_division_by_zero():
    ctrl = StanleyController(k=0.1, k_soft=0.1, k_heading=1.0, target_speed=1.0)
    # vx=0 should not raise; k_soft keeps denominator positive
    speed, steer = ctrl.get_action(vx=0.0, frenet_u=0.0, frenet_n=0.5)
    assert math.isfinite(steer)
    assert steer < 0.0
