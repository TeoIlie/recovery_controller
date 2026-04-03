# Plan: Trajectory Error Measurement (TRAJ_ERROR.md)

## Context
Quantify the sim2real gap by recording control commands on the real F1TENTH car, replaying them open-loop in the Gym-Khana STD simulator, and comparing the resulting trajectories against Vicon ground truth. The deliverable is `recovery_controller/plan/TRAJ_ERROR.md` — a step-by-step plan with script design for each stage.

## Document structure

### Step 1: Record a ROS2 bag on the real car
- Topics: `/drive`, `/vrpn_mocap/f110/pose`, `/vrpn_mocap/f110/twist`, `/sensors/core`, `/odom`
- One-liner command, no script needed

### Step 2: Parse bag → numpy (`parse_bag.py`)
- Standalone Python script using `rosbag2_py` + `rclpy.serialization`
- Reads the bag, deserializes each message, extracts:
  - From `/drive` (AckermannDriveStamped): timestamp, speed, steering_angle
  - From `/vrpn_mocap/f110/pose` (PoseStamped): timestamp, x, y, quaternion → yaw
  - From `/vrpn_mocap/f110/twist` (TwistStamped): timestamp, vx_world, vy_world, yaw_rate
- Saves to a single `.npz` file with arrays: `cmd_t, cmd_speed, cmd_steer, vicon_t, vicon_x, vicon_y, vicon_yaw, vicon_vx, vicon_vy, vicon_r`
- Quaternion → yaw via `transforms3d.euler.quat2euler`
- Body-frame velocity: rotate Vicon world-frame twist using yaw

### Step 3: Resample & replay in sim (`replay_sim.py`)
- Loads the `.npz` from Step 2
- Resamples commands to 100 Hz via zero-order hold (for each sim tick, find last command before that time)
- Computes initial state from Vicon: `[x0, y0, delta=0, v0, yaw0, r0, beta0]`
  - v0 from body-frame vx/vy magnitude, beta0 from atan2(vy, vx), r0 from yaw_rate
- Creates Gym-Khana env: `model='std'`, `control_input=['speed', 'steering_angle']`, `timestep=0.01`, `params=f1tenth_std_vehicle_params()` with overrides
- `env.reset(options={'states': initial_state})`
- Steps through resampled commands, recording sim state (x, y, yaw, vx, vy) per step
- Also resamples Vicon trajectory to same 100 Hz grid for aligned comparison
- Saves results to `.npz`: `sim_t, sim_x, sim_y, sim_yaw, sim_vx, sim_vy, real_t, real_x, real_y, real_yaw, real_vx, real_vy`

### Step 4: Plot comparison (`plot_comparison.py`)
- Loads replay `.npz` from Step 3
- Produces 4 plots:
  1. XY trajectory overlay (sim vs real)
  2. Heading (yaw) vs time
  3. Velocity vs time
  4. Position error (Euclidean distance between sim and real) vs time
- Saves figures as PNGs

### Step 5: Compute metrics (`compute_metrics.py`)
- Loads replay `.npz`
- Computes and prints:
  - Position RMSE (m)
  - Heading RMSE (rad)
  - Velocity RMSE (m/s)
  - Max position error and time it occurs
  - Final position error
- Optionally outputs a summary JSON

### Key considerations section
- Speed control mismatch (sim P-controller vs VESC speed controller)
- Option to replay actual VESC telemetry to isolate dynamics gap
- Coordinate frame alignment
- Vehicle param tuning (steering limits, mass, friction)
- Drive slowly first to validate pipeline before high-speed tests

## Files to create
- **Create**: `recovery_controller/plan/TRAJ_ERROR.md`

## Verification
- Document is self-contained with enough detail to implement each script
- Consistent with existing SIM_TO_REAL.md conventions
