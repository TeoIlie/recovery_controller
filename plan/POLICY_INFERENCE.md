# Phase 3: Policy Inference Wiring

## Context
The recovery node already has zone monitoring, state estimation, observation building, and debug publishing. The `_in_drive_zone` branch in `_tick()` currently publishes a dummy `speed=0.5, steer=0.0`. This plan replaces that with real policy inference.

## Files to Create
- `recovery_controller/recovery_controller/policy_runner.py` — pure Python, wraps SB3 `PPO.load()` + `predict()`

## Files to Modify
- `recovery_controller/recovery_controller/recovery_node.py` — wire policy into `_tick()`
- `recovery_controller/config/recovery.yaml` — add `model_path` parameter

## Step-by-Step

### Step 1: Create `policy_runner.py`
Pure Python class (no ROS deps):
- `__init__(model_path: str)` — calls `PPO.load(model_path)`, stores model
- `predict(obs: np.ndarray) -> np.ndarray` — calls `self.model.predict(obs, deterministic=True)`, returns raw action `[accl_norm, steer_norm]` both in `[-1, 1]`

### Step 2: Add activation tracking to `recovery_node.py`
- Add `self._active = False` flag in `__init__`
- When entering drive zone and `not self._active`:
  - Set `self._active = True`
  - Capture initial speed: `ERPM / speed_to_erpm_gain` (from `self._latest_erpm`)
  - Call `self._obs_builder.reset(initial_speed)` to initialize `curr_vel_cmd` and zero out action history
- When leaving drive zone (ebrake zone or teleop zone): set `self._active = False`

### Step 3: Wire inference into the drive zone branch
Replace the placeholder in `elif self._in_drive_zone(x, y):` with:
```python
# On first tick in zone, activate
if not self._active:
    initial_speed = self._latest_erpm / speed_to_erpm_gain if self._latest_erpm else 0.0
    self._obs_builder.reset(initial_speed)
    self._active = True

# Inference
raw_action = self._policy.predict(obs)   # [accl_norm, steer_norm]

# Update obs builder internal state (prev_steering_cmd, prev_accl_cmd, curr_vel_cmd)
self._obs_builder.step(raw_action)

# Denormalize for AckermannDriveStamped
steering_angle = raw_action[1] * s_max    # s_max = 0.5 rad
speed = self._obs_builder.curr_vel_cmd    # integrated velocity

# Publish
msg = AckermannDriveStamped()
msg.header.stamp = ...
msg.drive.speed = speed
msg.drive.steering_angle = steering_angle
self._drive_pub.publish(msg)
```

### Step 4: Deactivate when leaving drive zone
In both the ebrake branch and the implicit teleop branch (the `else`), set `self._active = False` so the next entry re-triggers activation.

### Step 5: Add `model_path` parameter
- Declare `model_path` (string) in the node
- Read from `recovery.yaml` — points to the `.zip` file in `models/`
- Construct `PolicyRunner` in `__init__` using this path

### Step 6: Store `speed_to_erpm_gain` and `s_max` as instance attributes
These are needed at inference time for initial speed capture and steering denormalization. `s_max = 0.5` (from plan spec) and `speed_to_erpm_gain` is already a constructor parameter but not stored.

## Constants (from SIM_TO_REAL_IMPL.md)
- `s_max = 0.5` rad (max steering angle)
- `a_max = 5.0` m/s² (already in obs builder)
- `action[0]` = accl (normalized), `action[1]` = steering (normalized)
- `curr_vel_cmd += action[0] * a_max * dt` (handled by `obs_builder.step()`)

## Verification
1. Unit test `policy_runner.py`: mock SB3 model, verify `predict()` passes through correctly
2. On-car: set `debug: false`, launch with a trained model — verify `/drive` publishes non-zero speed/steering when car is in zone
3. Verify `curr_vel_cmd` integration: check `/recovery/debug_obs` shows evolving velocity
