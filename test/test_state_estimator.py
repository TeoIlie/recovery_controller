import math

import numpy as np
import pytest
from recovery_controller.state_estimator import StateEstimator, wrap_angle


# Shared fixture: zone along +x axis, servo params from vesc.yaml
@pytest.fixture
def est():
    return StateEstimator(
        zone_y_min=-1.0,
        zone_y_max=1.0,
        servo_offset=0.512,
        servo_gain=-0.673,
        speed_to_erpm_gain=4600.0,
        wheel_radius=0.049,
    )


# --- wrap_angle ---


def test_wrap_angle_zero():
    assert wrap_angle(0.0) == pytest.approx(0.0)


def test_wrap_angle_pi():
    # pi wraps to -pi (both equivalent); check absolute value
    assert abs(wrap_angle(math.pi)) == pytest.approx(math.pi)


def test_wrap_angle_over_pi():
    assert wrap_angle(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)


def test_wrap_angle_under_neg_pi():
    assert wrap_angle(-math.pi - 0.1) == pytest.approx(math.pi - 0.1)


def test_wrap_angle_two_pi():
    assert wrap_angle(2 * math.pi) == pytest.approx(0.0, abs=1e-10)


# --- steering_angle ---


def test_steering_center(est):
    """Servo at center position → 0 rad."""
    assert est.steering_angle(0.512) == pytest.approx(0.0)


def test_steering_left(est):
    """Higher servo value → negative gain → positive steering (left)."""
    # servo=0.512 + 0.673*0.3 = 0.7139 → should give -0.3 rad
    # (servo - 0.512) / -0.673 = 0.2019 / -0.673 = -0.3
    servo = 0.512 + (-0.673) * 0.3  # = 0.3101
    assert est.steering_angle(servo) == pytest.approx(0.3, abs=1e-6)


def test_steering_right(est):
    servo = 0.512 + (-0.673) * (-0.3)  # = 0.7139
    assert est.steering_angle(servo) == pytest.approx(-0.3, abs=1e-6)


# --- yaw_rate ---


def test_yaw_rate_zero(est):
    assert est.yaw_rate(0.0) == pytest.approx(0.0)


def test_yaw_rate_90dps(est):
    assert est.yaw_rate(90.0) == pytest.approx(math.pi / 2)


def test_yaw_rate_negative(est):
    assert est.yaw_rate(-180.0) == pytest.approx(-math.pi)


# --- wheel_omega ---


def test_wheel_omega_zero(est):
    assert est.wheel_omega(0.0) == pytest.approx(0.0)


def test_wheel_omega_positive(est):
    # ERPM=4600*0.049 = 225.4 → omega = 225.4 / 225.4 = 1.0
    assert est.wheel_omega(225.4) == pytest.approx(1.0, abs=1e-3)


def test_wheel_omega_negative_erpm(est):
    """Negative ERPM (reverse) should return 0 — car never travels backwards during recovery."""
    assert est.wheel_omega(-225.4) == pytest.approx(0.0)


# --- body_frame_velocity ---


def test_body_vel_aligned(est):
    """Car facing +x, moving +x in world → all vx, no vy."""
    vx, vy = est.body_frame_velocity(5.0, 0.0, yaw=0.0)
    assert vx == pytest.approx(5.0)
    assert vy == pytest.approx(0.0)


def test_body_vel_rotated_90(est):
    """Car facing +y (yaw=pi/2), moving +y in world → body vx."""
    vx, vy = est.body_frame_velocity(0.0, 5.0, yaw=math.pi / 2)
    assert vx == pytest.approx(5.0, abs=1e-10)
    assert vy == pytest.approx(0.0, abs=1e-10)


def test_body_vel_sideways(est):
    """Car facing +x, moving +y in world → all vy."""
    vx, vy = est.body_frame_velocity(0.0, 5.0, yaw=0.0)
    assert vx == pytest.approx(0.0, abs=1e-10)
    assert vy == pytest.approx(5.0, abs=1e-10)


# --- frenet_coords (zone centerline along x-axis, center_y = 0.0) ---


def test_frenet_on_centerline_aligned(est):
    """Car on centerline, facing along zone → heading error 0, offset 0."""
    u, n = est.frenet_coords(car_y=0.0, car_yaw=0.0)
    assert u == pytest.approx(0.0)
    assert n == pytest.approx(0.0)


def test_frenet_offset_left(est):
    """Car at y=0.5, zone center at y=0 → offset = +0.5 (left)."""
    u, n = est.frenet_coords(car_y=0.5, car_yaw=0.0)
    assert u == pytest.approx(0.0)
    assert n == pytest.approx(0.5)


def test_frenet_offset_right(est):
    """Car at y=-0.3 → offset = -0.3 (right)."""
    _, n = est.frenet_coords(car_y=-0.3, car_yaw=0.0)
    assert n == pytest.approx(-0.3)


def test_frenet_heading_error(est):
    """Car turned 0.2 rad left of zone heading."""
    u, _ = est.frenet_coords(car_y=0.0, car_yaw=0.2)
    assert u == pytest.approx(0.2)


def test_frenet_heading_wraps(est):
    """Heading error near ±pi wraps correctly."""
    u, _ = est.frenet_coords(car_y=0.0, car_yaw=math.pi - 0.1)
    assert u == pytest.approx(math.pi - 0.1)
    u2, _ = est.frenet_coords(car_y=0.0, car_yaw=-(math.pi - 0.1))
    assert u2 == pytest.approx(-(math.pi - 0.1))


def test_frenet_offset_with_nonzero_center():
    """Zone with center_y != 0: offset is relative to center."""
    est = StateEstimator(
        zone_y_min=1.0,
        zone_y_max=3.0,
        servo_offset=0.512,
        servo_gain=-0.673,
        speed_to_erpm_gain=4600.0,
        wheel_radius=0.049,
    )
    # center_y = 2.0, car at y=2.5 → offset = +0.5
    u, n = est.frenet_coords(car_y=2.5, car_yaw=0.0)
    assert u == pytest.approx(0.0)
    assert n == pytest.approx(0.5)


# --- sideslip ---


def test_sideslip_stopped(est):
    """vx < 0.5 → beta = 0."""
    assert est.sideslip(0.3, 1.0) == 0.0


def test_sideslip_straight(est):
    """Pure forward motion → beta = 0."""
    assert est.sideslip(5.0, 0.0) == pytest.approx(0.0)


def test_sideslip_drifting(est):
    """Known sideslip angle."""
    assert est.sideslip(5.0, 5.0) == pytest.approx(math.pi / 4)


# --- yaw_from_quaternion ---


def _quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """Helper: build a pure-yaw quaternion and return (qx, qy, qz, qw)."""
    qw = math.cos(yaw / 2)
    qz = math.sin(yaw / 2)
    return 0.0, 0.0, qz, qw


def test_yaw_from_quat_zero():
    yaw = StateEstimator.yaw_from_quaternion(*_quat_from_yaw(0.0))
    assert yaw == pytest.approx(0.0)


def test_yaw_from_quat_90():
    yaw = StateEstimator.yaw_from_quaternion(*_quat_from_yaw(math.pi / 2))
    assert yaw == pytest.approx(math.pi / 2)


def test_yaw_from_quat_negative():
    yaw = StateEstimator.yaw_from_quaternion(*_quat_from_yaw(-0.7))
    assert yaw == pytest.approx(-0.7)


def test_yaw_from_quat_near_pi():
    yaw = StateEstimator.yaw_from_quaternion(*_quat_from_yaw(math.pi - 0.01))
    assert yaw == pytest.approx(math.pi - 0.01, abs=1e-6)


# --- body_frame_velocity with quaternion-derived yaw ---


def test_body_vel_via_quaternion_45deg(est):
    """Car facing 45°, moving 5 m/s along that heading → vx=5, vy=0."""
    yaw = math.pi / 4
    world_vx = 5.0 * math.cos(yaw)
    world_vy = 5.0 * math.sin(yaw)
    car_yaw = StateEstimator.yaw_from_quaternion(*_quat_from_yaw(yaw))
    vx, vy = est.body_frame_velocity(world_vx, world_vy, car_yaw)
    assert vx == pytest.approx(5.0, abs=1e-10)
    assert vy == pytest.approx(0.0, abs=1e-10)


def test_body_vel_via_quaternion_lateral(est):
    """Car facing 0°, moving purely in +y → vx=0, vy=5."""
    car_yaw = StateEstimator.yaw_from_quaternion(*_quat_from_yaw(0.0))
    vx, vy = est.body_frame_velocity(0.0, 5.0, car_yaw)
    assert vx == pytest.approx(0.0, abs=1e-10)
    assert vy == pytest.approx(5.0, abs=1e-10)


# --- frenet_coords with quaternion-derived yaw ---


def test_frenet_via_quaternion_aligned(est):
    """Car on centerline facing along zone, yaw from quaternion."""
    car_yaw = StateEstimator.yaw_from_quaternion(*_quat_from_yaw(0.0))
    u, n = est.frenet_coords(car_y=0.0, car_yaw=car_yaw)
    assert u == pytest.approx(0.0)
    assert n == pytest.approx(0.0)


def test_frenet_via_quaternion_heading_error(est):
    """Car turned 0.3 rad from zone heading, yaw from quaternion."""
    car_yaw = StateEstimator.yaw_from_quaternion(*_quat_from_yaw(0.3))
    u, n = est.frenet_coords(car_y=0.0, car_yaw=car_yaw)
    assert u == pytest.approx(0.3, abs=1e-6)
    assert n == pytest.approx(0.0)
