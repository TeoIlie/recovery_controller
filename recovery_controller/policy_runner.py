"""Lightweight ONNX inference runner for deployed PPO policies.
No dependency on stable-baselines3 or torch. Requires only numpy and onnxruntime.
"""

import numpy as np
import onnxruntime as ort


class PolicyRunner:
    """Run a deterministic PPO policy exported to ONNX.

    Parameters
    ----------
    model_path : Path to the ONNX model file.
    s_max      : Maximum steering angle (rad) for denormalization.
    """

    def __init__(self, model_path: str, s_max: float = 0.5):
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.s_max = s_max

    def predict(self, obs: np.ndarray) -> tuple:
        """Run one inference step.

        Parameters
        ----------
        obs : Normalized observation vector of shape (obs_dim,).

        Returns
        -------
        raw_action : tuple of (raw_accl, raw_steer), each in [-1, 1].
        """
        obs = obs[np.newaxis, :].astype(np.float32)
        raw_action = self.session.run(["action"], {"obs": obs})[0]
        raw_action = np.clip(raw_action[0], -1.0, 1.0)
        return raw_action[0], raw_action[1]

    def denorm_steering(self, raw_steer: float) -> float:
        """Convert normalized steering [-1, 1] to radians."""
        return raw_steer * self.s_max
