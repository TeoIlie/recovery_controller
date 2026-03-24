import numpy as np


class StanleyController:
    def __init__(self, k: float, k_soft: float, k_heading: float, target_speed: float):
        self.k = k
        self.k_soft = k_soft
        self.k_heading = k_heading
        self.target_speed = target_speed

    def set_target_speed(self, target_speed: float) -> None:
        self.target_speed = target_speed

    def compute_steering(
        self, vx: float, heading_error: float, cross_track_error: float
    ) -> float:
        heading_term = self.k_heading * heading_error
        cross_track_term = np.arctan(
            self.k * cross_track_error / (abs(vx) + self.k_soft)
        )
        return -heading_term - cross_track_term

    def get_action(self, vx: float, frenet_u: float, frenet_n: float) -> tuple:
        steering_angle = self.compute_steering(vx, frenet_u, frenet_n)
        return (self.target_speed, steering_angle)
