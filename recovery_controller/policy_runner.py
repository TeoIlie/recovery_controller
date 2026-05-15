"""Lightweight ONNX inference runner for deployed PPO policies.
No dependency on stable-baselines3 or torch. Requires only numpy and onnxruntime.
"""

import numpy as np
import onnxruntime as ort


class PolicyRunner:
    """Run a deterministic PPO policy exported to ONNX.

    Action: index 0 = normalized steering, index 1 = normalized speed
    (matches Gym-Khana CarAction.act for control_input=[speed,steering_angle]).
    Steering denorm is symmetric ([-s_max, s_max]); speed is asymmetric
    ([v_min, v_max]).
    """

    def __init__(self, model_path: str, s_max: float, v_min: float, v_max: float):
        if v_min >= v_max:
            raise ValueError(f"v_min ({v_min}) must be < v_max ({v_max})")
        if s_max <= 0:
            raise ValueError(f"s_max ({s_max}) must be > 0")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.s_max = s_max
        self.v_min = v_min
        self.v_max = v_max
        self._v_center = 0.5 * (v_max + v_min)
        self._v_range = 0.5 * (v_max - v_min)

    def predict(self, obs: np.ndarray) -> tuple:
        """Run one inference step. Returns (raw_steer, raw_speed) in [-1, 1]."""
        obs = obs[np.newaxis, :].astype(np.float32)
        raw_action = self.session.run(["action"], {"obs": obs})[0]
        raw_action = np.clip(raw_action[0], -1.0, 1.0)
        return float(raw_action[0]), float(raw_action[1])

    def denorm_steering(self, raw_steer: float) -> float:
        """Normalized steering [-1, 1] → rad (symmetric)."""
        return raw_steer * self.s_max

    def denorm_speed(self, raw_speed: float) -> float:
        """Normalized speed [-1, 1] → m/s (asymmetric: 0 → v_center)."""
        return raw_speed * self._v_range + self._v_center
