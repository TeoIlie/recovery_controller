"""Pure Python state estimator — no ROS2 dependencies, unit-testable.

Takes raw sensor readings (Vicon pose/twist, VESC, IMU) and produces
derived physical quantities needed by the observation builder.
"""

import math

import numpy as np
from transforms3d.euler import quat2euler


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class StateEstimator:
    """Derives body-frame state from raw sensor data.

    All methods are stateless transforms — call them each tick with the
    latest sensor values.  The class only stores zone geometry (fixed at
    construction time).
    """

    def __init__(
        self,
        zone_start: tuple[float, float],
        zone_end: tuple[float, float],
        servo_offset: float,
        servo_gain: float,
        speed_to_erpm_gain: float,
        wheel_radius: float,
    ):
        """
        Parameters
        ----------
        zone_start          : (x, y) of the entry end of the recovery zone centerline.
        zone_end            : (x, y) of the exit end of the recovery zone centerline.
        servo_offset        : Servo center position (from vesc.yaml steering_angle_to_servo_offset).
        servo_gain          : Servo gain (from vesc.yaml steering_angle_to_servo_gain).
        speed_to_erpm_gain  : ERPM per m/s (from vesc.yaml speed_to_erpm_gain).
        wheel_radius        : Wheel radius in metres.
        """
        dx = zone_end[0] - zone_start[0]
        dy = zone_end[1] - zone_start[1]
        self.zone_heading = math.atan2(dy, dx)
        L = math.hypot(dx, dy)
        # Unit tangent and normal (90° CCW = left)
        self.ux = dx / L
        self.uy = dy / L
        self.nx = -self.uy
        self.ny = self.ux
        self.zone_start = zone_start
        self.servo_offset = servo_offset
        self.servo_gain = servo_gain
        self.speed_to_erpm_gain = speed_to_erpm_gain
        self.wheel_radius = wheel_radius

    @staticmethod
    def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
        """Extract yaw from a quaternion (ROS ordering: x, y, z, w)."""
        # transforms3d expects (w, x, y, z)
        _, _, yaw = quat2euler([qw, qx, qy, qz])
        return yaw

    def body_frame_velocity(
        self, world_vx: float, world_vy: float, yaw: float
    ) -> tuple[float, float]:
        """Rotate Vicon world-frame twist into the car's body frame."""
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        vx = world_vx * cos_y + world_vy * sin_y
        vy = -world_vx * sin_y + world_vy * cos_y
        return vx, vy

    def frenet_coords(
        self, car_x: float, car_y: float, car_yaw: float
    ) -> tuple[float, float]:
        """Compute heading error (frenet_u) and lateral offset (frenet_n)."""
        frenet_u = wrap_angle(car_yaw - self.zone_heading)
        frenet_n = (car_x - self.zone_start[0]) * self.nx + (
            car_y - self.zone_start[1]
        ) * self.ny
        return frenet_u, frenet_n

    def sideslip(self, vx: float, vy: float) -> float:
        """Sideslip angle beta; returns 0 when nearly stopped."""
        if vx < 0.5:
            return 0.0
        return math.atan2(vy, vx)

    def steering_angle(self, servo_position: float) -> float:
        """Invert VESC servo mapping to get steering angle in rad."""
        return (servo_position - self.servo_offset) / self.servo_gain

    def yaw_rate(self, gyro_z_dps: float) -> float:
        """Convert VESC IMU gyro z from deg/s to rad/s."""
        return float(np.deg2rad(gyro_z_dps))

    def wheel_omega(self, erpm: float) -> float:
        """Wheel angular velocity (rad/s) from signed ERPM."""
        # we assume the car is never travelling backwards
        if erpm <= 0:
            return 0.0

        # erpm= speed_to_erpm_gain * speed + speed_to_erpm_offset, but offset is 0
        velocity = erpm / self.speed_to_erpm_gain
        # angular_velocity = velocity / wheel_radius
        return velocity / self.wheel_radius
