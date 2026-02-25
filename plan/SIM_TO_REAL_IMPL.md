# Recovery Policy Deployment: Revised Implementation Plan

## Context

Deploy a trained PPO recovery policy from simulation onto a real F1TENTH car. The existing `SIM_TO_REAL_IMPL.md` is strong on observation pipeline details (normalization bounds, feature ordering, state estimation) but over-engineers the node architecture and misuses `ackermann_mux`. This revised plan simplifies to a single node, properly integrates with the existing mux, and resolves frequency/timing issues.

---

## Key Changes from Original Plan

### 1. Use `ackermann_mux` instead of a custom safety gate (from 4 nodes to 1)

The original plan has `safety_monitor_node` as the "sole publisher to `/drive`", which **bypasses the existing ackermann_mux**. The mux already provides priority-based arbitration with timeout/deadman behavior.

**Revised approach**: Publish recovery commands to the existing `/drive` topic (already in the mux at priority 10). Add only one new mux topic for e-brake:

```yaml
ackermann_mux:
  ros__parameters:
    topics:
      navigation:
        topic   : drive
        timeout : 0.2
        priority: 10        # recovery publishes here (existing topic)
      joystick:
        topic   : teleop
        timeout : 0.2
        priority: 100
      ebrake:
        topic   : ebrake    # NEW - only addition
        timeout : 0.5
        priority: 200        # overrides everything
```

**Priority logic**:
- Recovery publishes to `/drive` (priority 10) — same as any autonomous controller
- Joystick (100) > drive (10): operator can override by holding LB deadman
- E-brake (200) > all: recovery_node publishes speed=0 here when out of bounds or Vicon dropout
- Timeouts: if recovery_node stops publishing, mux masks `/drive` within 200ms automatically

### 2. Single node design

The 4-node pipeline adds unnecessary message hops and complexity. Everything runs in **1 node** (`recovery_node`):
- State estimation + observation building + policy inference + zone monitoring + e-brake
- If the node crashes, it stops publishing, the mux times out `/drive` within 200ms, and the car coasts to a stop
- The mux's built-in timeout is the safety net — no active e-brake if the node dies, but also no stale commands

### 3. Run control at 100 Hz (match simulation dt=0.01s)

The simulation requires dt=0.01s — the policy was trained at this rate and behaves erratically at other timesteps. VESC data arrives at 50 Hz, so half the cycles will use stale VESC data. Vicon at 100-200 Hz provides fresh position data every tick. This is acceptable since:
- Position/velocity (from Vicon) update every tick — these are the primary state inputs
- Wheel omega / motor RPM (from VESC) change slowly relative to 100 Hz — stale-by-one-tick is fine
- The `curr_vel_cmd` integration uses the correct dt=0.01s matching simulation

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

---

## Node Specification: `recovery_node` (100 Hz)

Single node handling everything: zone monitoring, state estimation, observation building, inference, and e-brake.

**Subscribes**: `/vrpn_mocap/<name>/pose` (PoseStamped), `/sensors/core` (VescStateStamped), `/sensors/imu/raw` (Imu)
**Publishes**: `/drive` (AckermannDriveStamped, existing mux topic), `/ebrake` (AckermannDriveStamped, new mux topic)
**Services**: `/recovery/arm` (Trigger), `/recovery/disarm` (Trigger)

**State machine**:
```
IDLE ─[arm service]─► ARMED ─[crosses entry line]─► ACTIVE
                                                       │
                      E_BRAKE ◄─[exits bounds OR Vicon dropout > 50ms]
                         │
                      [disarm service] ──► IDLE
```

**Pipeline per 100 Hz tick**:
1. **Zone check**: Read latest Vicon pose, check state machine transitions
2. If E_BRAKE: publish speed=0 on `/ebrake` (priority 200), return
3. If not ACTIVE: skip (mux auto-masks `/drive` via timeout)
4. **State estimation**: Differentiate Vicon position, rotate to body frame, Butterworth LPF
5. **Observation**: Build 13-feature normalized vector (same spec as original plan)
6. **Inference**: `model.predict(obs, deterministic=True)`
7. **Publish**: AckermannDriveStamped on `/drive`

**On activation**: Initialize `curr_vel_cmd` to measured vx from Vicon.

**Safety via mux**: If node crashes, it stops publishing on both topics. The mux times out `/drive` (0.2s) and `/ebrake` (0.5s). After timeout the car is under joystick control — operator must be ready with deadman.

**Message type**: `ackermann_msgs/msg/AckermannDriveStamped` with `drive.speed` (m/s) and `drive.steering_angle` (rad) — matches what `ackermann_to_vesc_node` expects.

---

## Package Structure

Create `recovery_controller` as a **new git repo, added as a submodule** (follows existing pattern: vesc, ackermann_mux, teleop_tools are all submodules).

```
recovery_controller/
  package.xml
  setup.py / setup.cfg
  resource/recovery_controller
  recovery_controller/
    __init__.py
    recovery_node.py           # ROS2 node (thin wrapper around pure classes)
    state_estimator.py         # Pure Python class (no ROS deps, testable)
    observation_builder.py     # Pure Python class (no ROS deps, testable)
    policy_runner.py           # Pure Python class wrapping SB3
  config/
    recovery.yaml              # Zone geometry, norm bounds, model path
  test/
    test_state_estimator.py
    test_observation_builder.py
```

Core logic in pure Python classes (no ROS2 deps) = unit-testable with pytest.

New launch file: `f1tenth_stack/launch/recovery_bringup_launch.py` — includes base bringup via `IncludeLaunchDescription` + adds recovery node + loads updated mux config.

New config: `f1tenth_stack/config/recovery_mux.yaml` (existing mux config + ebrake topic).

---

## Implementation Phases

### Phase 0: Export deployment config (simulation repo)
Export `deployment_config.yaml` from training code with normalization bounds, feature ordering, action space config, vehicle params. Single source of truth.

### Phase 1: Package scaffold + zone monitoring + e-brake
1. Create repo, package structure, add as submodule to F1TENTH_System
2. Implement `recovery_node.py` with zone geometry, state machine, arm/disarm services, e-brake publishing
3. Create `recovery_mux.yaml` (adds ebrake topic to existing mux config)
4. Create `recovery_bringup_launch.py` (includes base bringup + recovery node)
5. **Test**: Push car through zone manually, verify entry/exit/bounds/dropout detection, verify e-brake on `/ebrake`

### Phase 2: State estimator + observation builder (pure Python classes)
1. Implement `state_estimator.py` and `observation_builder.py` as pure Python classes
2. **Test**: Unit tests with known inputs matching simulation's `VectorObservation.observe()`
3. **Test**: Drive manually, record bag, replay through estimator, compare vx to `/odom`

### Phase 3: Policy inference + full pipeline
1. Implement `policy_runner.py`, wire into `recovery_node.py`
2. **Test**: Known sim observation → verify action matches sim output
3. **Test**: Full pipeline on stand (wheels off ground), verify motor/servo response
4. **Test**: Mux priority arbitration — recovery active, joystick override, e-brake override

### Phase 4: Real-world testing
1. Calibrate zone in Vicon frame
2. Low-speed tests, gradually increase
3. Record every trial as ROS2 bag

---

## Verification

1. **Unit tests**: Pure Python classes tested with pytest (no ROS2 needed)
2. **Cross-validation**: Same state inputs through sim obs pipeline and real obs builder — outputs must match
3. **Mux arbitration**: Verify all priority scenarios (recovery, joystick override, e-brake override)
4. **Stand test**: Full pipeline, wheels off ground, mock Vicon
5. **Every trial**: ROS2 bag recording + post-analysis

---

## Key Files to Modify

| File | Change |
|------|--------|
| `.gitmodules` | Add recovery_controller submodule |
| `f1tenth_stack/config/recovery_mux.yaml` | New file: existing mux config + ebrake topic at priority 200 |
| `f1tenth_stack/launch/recovery_bringup_launch.py` | New launch file including base bringup + recovery nodes |
