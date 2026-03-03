# Sim-to-Real Transfer: Recovery Policy Deployment Plan

## Context

The simulator (`github.com/TeoIlie/F1TENTH_Gym`) trains PPO policies for two tasks: (1) racing with
drifting, and (2) recovering from out-of-control states. This plan covers deploying the **recovery
policy** onto a real 1/10-scale F1TENTH car to test sim-to-real transfer.

In simulation, a vehicle is initialized on a straight segment of the IMS track with an out-of-control
state (combinations of velocity, heading error, sideslip β, and yaw rate r from curriculum ranges:
`v: 5–9 m/s`, `beta: ±0.1–0.35 rad`, `r: ±0.2–0.79 rad/s`). The policy uses steering and acceleration
to return the car to controlled travel. Success is defined by `f110_env.py:_check_recovery_success()`.

In real life, the out-of-control state is created by **manually driving at high speed from outside the
Vicon space and performing a sudden cornering manoeuvre on a slippery strip of plastic** as the car
enters the capture volume. This generates the target heading error, β, and r values. As soon as the car
crosses the entry threshold into the Vicon space, the autonomous recovery controller takes over.

**Simulator key config** (`train/config/gym_config.yaml`):

- `action_input: [accl, steering_angle]`, `normalize_act: true`
- `normalize_obs: true`, `lookahead_n_points: 5`, `sparse_width_obs: true`, `timestep: 0.01`

**Deployment contract**: `deployment_config.yaml` (generated in Phase 0, committed to
`recovery_controller/models/`) is the single source of truth for all parameters. No simulator code runs
on the car.

---

## Architecture

```
Vicon ──►                         /ebrake ──► ackermann_mux (priority 200)
VESC  ──► [recovery_node] ──► /drive ──► ackermann_mux (priority 10)
IMU   ──►        │
                 │          Joystick ──► /teleop ──► ackermann_mux (priority 100)
            arm/disarm                                      │
            (services)                                      ▼
                                                      /ackermann_cmd ──► vesc
```

**Single node** (`recovery_node`) handles everything: zone monitoring, state estimation, observation
building, inference, and e-brake. The existing `ackermann_mux` is the safety arbitrator — no custom
safety gate node needed.

**Mux priority scheme** (add `ebrake` topic to existing `mux.yaml`):

```yaml
ackermann_mux:
  topics:
    navigation:
      topic:    drive
      timeout:  0.2
      priority: 10      # recovery publishes here (same as any autonomous controller)
    joystick:
      topic:    teleop
      timeout:  0.2
      priority: 100     # operator overrides with LB deadman
    ebrake:
      topic:    ebrake
      timeout:  0.5
      priority: 200     # overrides everything; recovery_node publishes here when unsafe
```

If the `recovery_node` crashes entirely: it stops publishing, mux times out `/drive` within 200 ms, car
is under joystick control — operator must hold LB deadman.

---

## Node Specification: `recovery_node` (100 Hz)

**Subscribes**: `/vrpn_mocap/<name>/pose` (PoseStamped), `/sensors/core` (VescStateStamped),
`/sensors/imu/raw` (Imu)
**Publishes**: `/drive` (AckermannDriveStamped), `/ebrake` (AckermannDriveStamped)
**Services**: `/recovery/arm` (Trigger), `/recovery/disarm` (Trigger)

**State machine**:

```
IDLE ─[arm]─► ARMED ─[crosses entry line]─► ACTIVE
                                               │
              E_BRAKE ◄─[exits bounds OR Vicon dropout > 50ms]
                 │
              [disarm] ──► IDLE
```

**Per-tick pipeline (100 Hz)**:

1. Zone check: read latest Vicon pose, update state machine
2. If `E_BRAKE`: publish `speed=0, steer=0` on `/ebrake` (priority 200), return
3. If not `ACTIVE`: return (mux masks `/drive` via 200 ms timeout)
4. State estimation: differentiate Vicon position → body-frame vx/vy, Butterworth LPF
5. Observation: build 18-element normalized vector (spec below)
6. Inference: `model.predict(obs, deterministic=True)`
7. Update internal state: `prev_steering_cmd`, `prev_accl_cmd`, `curr_vel_cmd`
8. Publish `AckermannDriveStamped` on `/drive`

**On activation**: initialize `curr_vel_cmd` to current VESC speed (`ERPM / speed_to_erpm_gain`).

**Timing**: The node runs at 100 Hz (dt = 0.01 s, matching the sim timestep). VESC data arrives at
~50 Hz, so every other tick will use stale VESC data for `wheel_omega` and `delta`. This is acceptable
since those signals change slowly relative to 100 Hz; Vicon position (the primary state input) updates
every tick at 100–200 Hz.

---

## Recovery Success (real-car equivalent)

In simulation, `_check_recovery_success()` checks that the car has stabilized (low β, low r, heading
aligned with track, sufficient forward speed) and reached a downstream threshold point.

On the real car, success is defined operationally: the car exits the recovery zone through the **exit
line** (rather than the lateral bounds) while still under control. The mux then automatically stops
forwarding `/drive` commands when the node stops publishing (200 ms timeout), and the car coasts/brakes.
Log β and r traces from the bag for each trial and compare to simulation trajectories.

---

## Observation Vector (18 elements)

Matches `observation_factory("drift")` → `VectorObservation` in `observation.py`.
Normalization formula (from `utils.py:normalize_feature`):
`norm = clip(2*(val − min) / (max − min) − 1, −1, 1)`

| Idx | Feature | Real-car source | Norm bounds |
|-----|---------|-----------------|-------------|
| 0 | `linear_vel_x` | Vicon Δpos/Δt → body frame vx | `[−5.0, 20.0]` |
| 1 | `linear_vel_y` | Vicon Δpos/Δt → body frame vy | `[−10.0, 10.0]` |
| 2 | `frenet_u` | `wrap(yaw − zone_heading, −π, π)` | `[−π, π]` |
| 3 | `frenet_n` | signed perp. distance to centerline | `[−1.1, 1.1]` |
| 4 | `ang_vel_z` | `Imu.angular_velocity.z` (rad/s, direct) | `[−5.0, 5.0]` |
| 5 | `delta` | VESC: `(servo_pos − 0.512) / (−0.673)` | `[−0.5, 0.5]` |
| 6 | `beta` | `atan2(vy, vx)`; use 0 if `vx < 0.5 m/s` | `[−π/3, π/3]` |
| 7 | `prev_steering_cmd` | last raw `action[1]` from network (`[−1, 1]`) | `[−1.0, 1.0]` |
| 8 | `prev_accl_cmd` | last `action[0] × a_max` (m/s², denorm.) | `[−5.0, 5.0]` |
| 9 | `prev_avg_wheel_omega` | `ERPM / (4600 × 0.049)` (rad/s) | `[0.0, 2612.24]` |
| 10 | `curr_vel_cmd` | integrated (see below) | `[−5.0, 20.0]` |
| 11–15 | `lookahead_curvatures` ×5 | `[0, 0, 0, 0, 0]` (straight line) | `[−1.95, 1.95]` |
| 16–17 | `lookahead_widths` ×2 (sparse) | `[zone_full_width, zone_full_width]` | `[1.2, 2.2]` |

### Per-feature derivation details

**Body-frame velocity from Vicon** — differentiate world-frame position, then rotate:

```python
world_vx = (x[t] - x[t-1]) / dt   # low-pass filter before rotation
world_vy = (y[t] - y[t-1]) / dt
vx =  world_vx * cos(yaw) + world_vy * sin(yaw)
vy = -world_vx * sin(yaw) + world_vy * cos(yaw)
```

Apply 2nd-order Butterworth low-pass (~20 Hz cutoff, tunable) on `world_vx/vy` before rotation.
Validate `vx` against VESC `/odom` during straight-line manual driving.

**Frenet coordinates** — straight-line zone, pure vector math (no spline needed):

```python
zone_heading = atan2(end[1]-start[1], end[0]-start[0])
frenet_u = wrap_angle(car_yaw - zone_heading)   # heading error

L = norm(end - start)
ux, uy = (end - start) / L          # unit vector along zone
nx, ny = -uy, ux                    # unit normal (90° CCW = left)
frenet_n = (car_x - start[0]) * nx + (car_y - start[1]) * ny
```

**Steering angle from VESC** — invert the servo mapping in `vesc.yaml`:

```
steering_angle = (servo_position - 0.512) / (-0.673)
```

`VescState.servo_position` is in [0, 1]. Note the gain is negative (left steer = higher servo value).
Alternative: track last commanded `action[1] × s_max` to avoid servo feedback lag.

**Wheel angular velocity from VESC ERPM**:

```
omega_wheel = ERPM / (speed_to_erpm_gain × R_w) = ERPM / (4600 × 0.049) = ERPM / 225.4
```

`VescState.speed` is signed ERPM — take `abs()` since norm bounds are `[0, 2612.24]`.
This avoids needing the gear ratio explicitly; `speed_to_erpm_gain` already encodes it.

**Action ordering** (`action = model.predict(obs)` output, from `action.py` + `gym_config.yaml`):

- `action[0]` = **longitudinal (accl)**, normalized `[−1, 1]` → `action[0] × a_max` m/s²
- `action[1]` = **steering angle**, normalized `[−1, 1]` → `action[1] × s_max` rad

`CarAction` maps `control_mode[0]="accl"` → `action[0]` and `control_mode[1]="steering_angle"` →
`action[1]`. The `AckermannDriveStamped` fields are:
- `drive.steering_angle = action[1] × s_max`  (s_max = 0.5 rad)
- `drive.speed = curr_vel_cmd`  (integrated, not direct acceleration)

**`curr_vel_cmd` integration** (mirrors `base_classes.py:update_pose`):

```python
# At activation: initialize from measured speed (mirrors base_classes.py reset: curr_vel_cmd = state[3])
curr_vel_cmd = ERPM / speed_to_erpm_gain

# Each tick (dt = 0.01 s):
curr_vel_cmd += action[0] * a_max * dt
curr_vel_cmd = clip(curr_vel_cmd, v_min, v_max)   # v_min=-5.0, v_max=20.0
```

**`prev_steering_cmd`**: store raw `action[1]` (not denormalized). Initialized to 0.0 at activation.
Mirrors `self.curr_steering_cmd = raw_steer` in `base_classes.py`.

**`prev_accl_cmd`**: store `action[0] × a_max` (denormalized m/s²). Initialized to 0.0 at activation.
Mirrors `self.curr_accl_cmd = accl` where `accl = AcclAction.act(action[0]) = action[0] * a_max`.

---

## Package Structure

```
recovery_controller/
  package.xml
  setup.py / setup.cfg
  resource/recovery_controller
  recovery_controller/
    __init__.py
    recovery_node.py          # ROS2 node (thin wrapper, no logic)
    state_estimator.py        # Pure Python class — no ROS deps, unit-testable
    observation_builder.py    # Pure Python class — no ROS deps, unit-testable
    policy_runner.py          # Pure Python class wrapping SB3
  config/
    recovery.yaml             # Zone geometry, model path
  models/
    deployment_config.yaml    # Generated by Phase 0 export script
    ppo_recover_<run_id>.zip  # Trained model weights
  test/
    test_state_estimator.py
    test_observation_builder.py
```

Core logic in pure Python classes (no ROS2 deps) so unit tests run without a ROS2 environment.

New files in `f1tenth_stack`:
- `launch/recovery_bringup_launch.py` — includes base bringup + recovery node + updated mux config
- `config/recovery_mux.yaml` — existing mux config extended with `ebrake` topic at priority 200

---

## Phase 0: Export Deployment Config (in F1TENTH_Gym repo)

Add `train/export_deployment_config.py` — run once after training, commits alongside the model zip.

```yaml
# deployment_config.yaml (schema)
obs:
  features: [linear_vel_x, linear_vel_y, frenet_u, frenet_n, ang_vel_z, delta, beta,
             prev_steering_cmd, prev_accl_cmd, prev_avg_wheel_omega, curr_vel_cmd,
             lookahead_curvatures, lookahead_widths]
  lookahead_n_points: 5
  lookahead_ds: 0.5
  sparse_width_obs: true
  norm_bounds: { ... }        # exact values from calculate_norm_bounds()
action:
  input: [accl, steering_angle]   # action[0]=accl, action[1]=steering_angle
  normalize: true
vehicle:
  a_max: 5.0
  s_max: 0.5
  v_min: -5.0
  v_max: 20.0
  R_w: 0.049
  timestep: 0.01
```

---

## Implementation Phases

### Phase 1: Package scaffold + zone monitoring ✓

No state machine, no arm/disarm services. The node is purely reactive using the `/ebrake` topic
(priority 200): when the car is **outside** the recovery zone, the node publishes `speed=0,
steering_angle=0` on `/ebrake` to override all other commands. When **inside** the zone, it
publishes nothing — the mux times out `/ebrake` within 500 ms and `/drive` + `/teleop` regain
control.

1. ✓ Created `recovery_controller` package structure
2. ✓ Defined zone geometry in `config/recovery.yaml` as a half-bounded rectangle
   (`zone_x_max`, `zone_y_min`, `zone_y_max` in Vicon world frame — no `x_min`, unbounded on
   entry side) plus the Vicon rigid-body name and timer rate
3. ✓ Implemented `recovery_node.py`:
   - Subscribes to `/vrpn_mocap/<name>/pose` (PoseStamped, BEST_EFFORT QoS)
   - At configured rate: checks if car (x, y) is inside the zone; if **outside**, publishes
     `AckermannDriveStamped(speed=0, steering_angle=0)` on `/ebrake` with throttled warning log;
     if inside, publishes nothing
4. ✓ Wired into `recovery_bringup_launch.py` (launch file in `f1tenth_stack`)
5. **Test**: Carry car through zone manually → verify `/ebrake` publishes only while outside;
   verify `/drive` + `/teleop` regain control within 500 ms of entering the zone

### Phase 2: State estimator + observation builder (pure Python)

1. Implement `state_estimator.py` and `observation_builder.py` as pure Python classes
2. **Unit test**: Known inputs → verify normalization matches `utils.py:normalize_feature` exactly
3. **Unit test**: Frenet math with car at known positions relative to zone
4. **Manual drive test**: Record bag, replay through estimator, compare `vx` to `/odom`, `yaw_rate` to
   IMU gyro; expect close agreement. Tune Butterworth cutoff.

### Phase 3: Policy inference + full pipeline

1. Implement `policy_runner.py`, wire into `recovery_node.py`
   - **Zone-conditional publishing**: `recovery_node` must publish on `/drive` **only** while in
     `ACTIVE` state (autonomous zone, after crossing the entry line). In `ARMED`/`IDLE` (entry zone),
     it must not publish on `/drive` so that the 200 ms mux timeout fires and `/teleop` (priority 100)
     wins. This is what makes `drive` priority 150 > `teleop` priority 100 safe: the higher priority
     is only exercised when the recovery policy is actually running.
2. **Test**: Known obs vector → verify action matches `model.predict()` in Python directly
3. **Test**: Full pipeline on stand (wheels off ground) with mock Vicon data; verify motor/servo response
4. **Test**: Mux arbitration — recovery active → joystick override → e-brake override (all priority
   levels work as expected)

### Phase 4: Real-world testing

1. Calibrate zone in Vicon frame with a ruler; commit final `recovery.yaml`
2. Low-speed tests first (push car slowly into zone by hand to verify controller engages)
3. Gradually increase entry speed and out-of-control severity; use curriculum ranges from training as
   a guide: `v: 5–9 m/s`, `beta: ±0.1 rad`, `r: ±0.2 rad/s` initially
4. Record every trial as a ROS2 bag

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Vicon dropout during recovery | 50 ms freshness check → publish on `/ebrake` |
| `vx/vy` from differentiation are noisy | Butterworth LPF; validate vs VESC odom before any policy run |
| `delta` mismatch: servo feedback lags policy output | Consider tracking last commanded `action[1] × s_max` instead of servo readback |
| `curr_vel_cmd` initialization causes jerk | Init from VESC ERPM at activation (matches sim reset `curr_vel_cmd = state[3]`) |
| `beta` undefined at near-zero speed | Guard: `beta = atan2(vy, vx) if vx > 0.5 else 0` |
| `wheel_omega` is signed from VESC | Take `abs(ERPM)` before conversion — bounds are `[0, 2612]` |
| Action ordering confusion | Lock in via `deployment_config.yaml`; assert at node startup |

---

## Sim-to-Real Gap

Even with a correct observation pipeline the policy may fail to transfer. Test without modifications
first, then apply these in order if performance is unsatisfactory:

1. **Domain randomization on friction** — re-train with randomized `mu`, `C_Sf`, `C_Sr` (tire/road
   friction params in `f1tenth_std_vehicle_params()`). Broad friction ranges make the policy more robust
   to the real surface.

2. **System identification** — measure actual tire parameters for the real car's surface (floor material,
   tire compound). Replace the default params in the sim and re-train.

3. **Observation noise injection** — add Gaussian noise to Vicon position and IMU signals during
   training to match real sensor noise characteristics.

---

## Verification Strategy

1. **Unit tests**: Pure Python classes tested with pytest (no ROS2 required)
2. **Cross-validation**: Python script reads a ROS2 bag, replays state through the sim's
   `VectorObservation.observe()` and the ROS2 observation builder, diffs outputs — must match to
   float32 precision
3. **Mux arbitration test**: Verify all three priority levels behave correctly
4. **Stand test**: Full pipeline wheels off ground + mock Vicon bag
5. **Every real trial**: ROS2 bag recording + post-analysis (β vs r phase plane, comparison to sim
   training distribution)
