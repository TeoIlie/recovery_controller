# Plan: Add Stanley Steering Controller to Recovery Controller

## Context
The recovery controller currently only supports a learned (ONNX) RL policy for autonomous recovery. We need to add a Stanley path-tracking controller as an alternative, selectable via config. This enables a classical control baseline that maintains the car's initial speed while correcting heading and cross-track errors.

## Files to Modify
1. **`recovery_controller/config/recovery.yaml`** — Add `controller` param and Stanley gains
2. **`recovery_controller/recovery_controller/stanley_controller.py`** — New file: Stanley controller class
3. **`recovery_controller/recovery_controller/recovery_node.py`** — Add controller selection logic
4. **`recovery_controller/test/test_stanley_controller.py`** — New file: unit tests

## Design Decision: Raw Values vs Normalized Obs
The `obs` array from `ObservationBuilder.build()` is normalized to [-1, 1]. The Stanley controller needs raw physical values. Since `vx`, `frenet_u`, and `frenet_n` are already computed as raw values in `_tick()` before the obs is built, the Stanley controller will receive these raw values directly rather than extracting and denormalizing from the obs vector. This is simpler and avoids coupling to normalization bounds.

## Implementation Steps

### Step 1: Add config parameters to `recovery.yaml`
Add to `recovery_controller/config/recovery.yaml`:
```yaml
# Controller type: "learned" (ONNX RL policy) or "stanley" (classical)
controller: "learned"

# Stanley controller gains (only used when controller: "stanley")
stanley_k: 0.1           # Cross-track gain
stanley_k_soft: 0.1      # Velocity softening constant [m/s]
stanley_k_heading: 1.0   # Heading error gain
```

### Step 2: Create `stanley_controller.py`
New file at `recovery_controller/recovery_controller/stanley_controller.py`:

```python
import numpy as np

class StanleyController:
    def __init__(self, k, k_soft, k_heading, target_speed):
        # Store gains and target speed

    def compute_steering(self, vx, heading_error, cross_track_error) -> float:
        # heading_term = k_heading * heading_error
        # cross_track_term = arctan(k * cross_track_error / (|vx| + k_soft))
        # return -heading_term - cross_track_term

    def get_action(self, vx, frenet_u, frenet_n) -> tuple[float, float]:
        # Returns (speed, steering_angle) as raw physical values
        # speed = self.target_speed
        # steering_angle = self.compute_steering(vx, frenet_u, frenet_n)
```

### Step 3: Modify `recovery_node.py`
- Declare new parameters: `controller`, `stanley_k`, `stanley_k_soft`, `stanley_k_heading`
- Validate `controller` is one of `["learned", "stanley"]` in `__init__`, raise error if not
- For `"learned"`: initialize `PolicyRunner` and `ObservationBuilder` as currently done
- For `"stanley"`: initialize `StanleyController` (target_speed set on zone entry via reset)
- In `_tick()` drive zone block:
  - If `"learned"`: existing logic (build obs → predict → step → publish)
  - If `"stanley"`: call `get_action(vx, frenet_u, frenet_n)` → publish speed + steering directly
- The `ObservationBuilder` and debug obs publishing should still run for both controllers (useful for debugging)
- Stanley controller's `target_speed` is set on activation (same as how `_obs_builder.reset(vx)` captures initial speed)

### Step 4: Create `test_stanley_controller.py`
Test cases:
- **Zero errors** → steering angle is 0.0
- **Positive heading error** → negative steering correction
- **Positive cross-track error** → negative steering correction
- **Speed returned matches target_speed**
- **Cross-track term saturates** via arctan (large error, small speed)
- **k_soft prevents division by zero** when vx=0

## Verification
1. **Unit tests**: `cd /home/jetson/f1tenth_ws/src/F1TENTH_System/recovery_controller && python -m pytest test/test_stanley_controller.py -v`
2. **Config validation**: Run node with invalid `controller` value, confirm it errors
3. **Integration**: Build with `colcon build --packages-select recovery_controller`, launch with `controller: "stanley"` in config, verify steering commands on `/drive` topic
