#!/usr/bin/env python3
"""Validate Phase 2 observations by comparing recovery node debug output
against independent recomputation from raw sensor topics in the same bag.

Usage:
    python3 scripts/validate_obs.py <bag_dir>

Example:
    python3 scripts/validate_obs.py ~/f1tenth_ws/bags/test_drive_phase_2
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rclpy.serialization import deserialize_message

from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64, String
from vesc_msgs.msg import VescStateStamped

# ---------------------------------------------------------------------------
# Bag reading via sqlite3 (no rosbags dependency needed)
# ---------------------------------------------------------------------------
import sqlite3

# Map ROS2 type strings to message classes
TYPE_MAP = {
    "geometry_msgs/msg/PoseStamped": PoseStamped,
    "geometry_msgs/msg/TwistStamped": TwistStamped,
    "nav_msgs/msg/Odometry": Odometry,
    "sensor_msgs/msg/Imu": Imu,
    "std_msgs/msg/Float64": Float64,
    "std_msgs/msg/String": String,
    "vesc_msgs/msg/VescStateStamped": VescStateStamped,
}

TOPICS_OF_INTEREST = {
    "/recovery/debug_obs",
    "/vrpn_mocap/f110/pose",
    "/vrpn_mocap/f110/twist",
    "/odom",
    "/sensors/imu/raw",
    "/sensors/core",
    "/sensors/servo_position_command",
}


def read_bag(bag_dir: str) -> dict[str, list[tuple[float, object]]]:
    """Read a ROS2 bag (sqlite3) and return {topic: [(t_sec, msg), ...]}."""
    bag_path = Path(bag_dir)
    db_files = sorted(bag_path.glob("*.db3"))
    if not db_files:
        sys.exit(f"No .db3 files found in {bag_path}")

    results: dict[str, list] = {t: [] for t in TOPICS_OF_INTEREST}

    for db_file in db_files:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()

        # Build topic_id → (name, type_class) map
        cursor.execute("SELECT id, name, type FROM topics")
        topic_map = {}
        for tid, name, typename in cursor.fetchall():
            if name in TOPICS_OF_INTEREST and typename in TYPE_MAP:
                topic_map[tid] = (name, TYPE_MAP[typename])

        cursor.execute(
            "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp"
        )
        for tid, timestamp, data in cursor.fetchall():
            if tid not in topic_map:
                continue
            name, msg_class = topic_map[tid]
            msg = deserialize_message(data, msg_class)
            t_sec = timestamp * 1e-9
            results[name].append((t_sec, msg))

        conn.close()

    return results


# ---------------------------------------------------------------------------
# State estimation (mirrors recovery_controller/state_estimator.py)
# ---------------------------------------------------------------------------
# VESC config values (from vesc.yaml)
SERVO_OFFSET = 0.512
SERVO_GAIN = -0.673
SPEED_TO_ERPM_GAIN = 4000.0
WHEEL_RADIUS = 0.049

# Zone config (from recovery.yaml)
ZONE_Y_MIN = -1.1
ZONE_Y_MAX = 1.1
CENTER_Y = (ZONE_Y_MIN + ZONE_Y_MAX) / 2.0


def yaw_from_quat(q) -> float:
    from transforms3d.euler import quat2euler

    _, _, yaw = quat2euler([q.w, q.x, q.y, q.z])
    return yaw


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def body_vel(world_vx, world_vy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return world_vx * c + world_vy * s, -world_vx * s + world_vy * c


def steering_angle(servo_pos):
    return (servo_pos - SERVO_OFFSET) / SERVO_GAIN


def wheel_omega(erpm):
    if erpm <= 0:
        return 0.0
    return (erpm / SPEED_TO_ERPM_GAIN) / WHEEL_RADIUS


def sideslip(vx, vy):
    if vx < 0.5:
        return 0.0
    return math.atan2(vy, vx)


# ---------------------------------------------------------------------------
# Parse debug_obs String messages
# ---------------------------------------------------------------------------
def parse_debug_obs(msg_str: str) -> dict[str, float]:
    d = {}
    for line in msg_str.strip().split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = float(v.strip())
    return d


# ---------------------------------------------------------------------------
# Nearest-neighbour time matching
# ---------------------------------------------------------------------------
def match_nearest(ts_target, ts_source, values_source, max_dt=0.05):
    """For each target timestamp, find the nearest source value within max_dt."""
    src_t = np.array(ts_source)
    matched = []
    for t in ts_target:
        idx = np.argmin(np.abs(src_t - t))
        if abs(src_t[idx] - t) <= max_dt:
            matched.append(values_source[idx])
        else:
            matched.append(np.nan)
    return np.array(matched)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Validate Phase 2 observations")
    parser.add_argument("bag_dir", help="Path to the ROS2 bag directory")
    args = parser.parse_args()

    print(f"Reading bag from {args.bag_dir} ...")
    data = read_bag(args.bag_dir)

    for topic, msgs in data.items():
        print(f"  {topic}: {len(msgs)} messages")

    if not data["/recovery/debug_obs"]:
        sys.exit("No /recovery/debug_obs messages found — was debug mode enabled?")

    # -----------------------------------------------------------------------
    # Extract debug_obs timeseries
    # -----------------------------------------------------------------------
    debug_t = []
    debug = {
        k: []
        for k in [
            "yaw",
            "vx",
            "vy",
            "frenet_u",
            "frenet_n",
            "beta",
            "ang_vel_z",
            "delta",
            "wheel_omega",
        ]
    }
    for t, msg in data["/recovery/debug_obs"]:
        parsed = parse_debug_obs(msg.data)
        debug_t.append(t)
        for k in debug:
            debug[k].append(parsed.get(k, np.nan))
    debug_t = np.array(debug_t)
    for k in debug:
        debug[k] = np.array(debug[k])

    # Normalize time to start at 0
    t0 = debug_t[0]
    debug_t_rel = debug_t - t0

    # -----------------------------------------------------------------------
    # Recompute from raw topics and compare
    # -----------------------------------------------------------------------

    # --- 1. vx: debug vs /odom twist.linear.x ---
    odom_t = [t for t, _ in data["/odom"]]
    odom_vx = [m.twist.twist.linear.x for _, m in data["/odom"]]
    odom_vx_matched = match_nearest(debug_t, odom_t, odom_vx)

    # --- 2. vx recomputed: from Vicon pose + twist ---
    vicon_pose_t = [t for t, _ in data["/vrpn_mocap/f110/pose"]]
    vicon_yaws = [
        yaw_from_quat(m.pose.orientation) for _, m in data["/vrpn_mocap/f110/pose"]
    ]
    vicon_twist_t = [t for t, _ in data["/vrpn_mocap/f110/twist"]]
    vicon_world_vx = [m.twist.linear.x for _, m in data["/vrpn_mocap/f110/twist"]]
    vicon_world_vy = [m.twist.linear.y for _, m in data["/vrpn_mocap/f110/twist"]]

    recomp_vx, recomp_vy, recomp_beta = [], [], []
    for i, t in enumerate(debug_t):
        # Get nearest yaw
        yi = np.argmin(np.abs(np.array(vicon_pose_t) - t))
        ti = np.argmin(np.abs(np.array(vicon_twist_t) - t))
        if abs(vicon_pose_t[yi] - t) > 0.05 or abs(vicon_twist_t[ti] - t) > 0.05:
            recomp_vx.append(np.nan)
            recomp_vy.append(np.nan)
            recomp_beta.append(np.nan)
            continue
        bvx, bvy = body_vel(vicon_world_vx[ti], vicon_world_vy[ti], vicon_yaws[yi])
        recomp_vx.append(bvx)
        recomp_vy.append(bvy)
        recomp_beta.append(sideslip(bvx, bvy))
    recomp_vx = np.array(recomp_vx)
    recomp_vy = np.array(recomp_vy)
    recomp_beta = np.array(recomp_beta)

    # --- 3. ang_vel_z: debug vs IMU and Vicon ---
    imu_t = [t for t, _ in data["/sensors/imu/raw"]]
    # IMU angular_velocity.z — the node converts deg/s → rad/s when source=imu
    imu_gz_raw = [m.angular_velocity.z for _, m in data["/sensors/imu/raw"]]
    imu_gz_rads = [np.deg2rad(g) for g in imu_gz_raw]
    imu_gz_matched = match_nearest(debug_t, imu_t, imu_gz_rads)

    vicon_angz = [m.twist.angular.z for _, m in data["/vrpn_mocap/f110/twist"]]
    vicon_angz_matched = match_nearest(debug_t, vicon_twist_t, vicon_angz)

    # --- 4. delta: debug vs recomputed from servo ---
    servo_t = [t for t, _ in data["/sensors/servo_position_command"]]
    servo_delta = [
        steering_angle(m.data) for _, m in data["/sensors/servo_position_command"]
    ]
    servo_delta_matched = match_nearest(debug_t, servo_t, servo_delta)

    # --- 5. wheel_omega: debug vs recomputed from ERPM ---
    core_t = [t for t, _ in data["/sensors/core"]]
    core_omega = [wheel_omega(m.state.speed) for _, m in data["/sensors/core"]]
    core_omega_matched = match_nearest(debug_t, core_t, core_omega)

    # --- 6. frenet_n: debug vs recomputed from Vicon y ---
    vicon_y = [m.pose.position.y for _, m in data["/vrpn_mocap/f110/pose"]]
    recomp_frenet_n = match_nearest(
        debug_t, vicon_pose_t, [y - CENTER_Y for y in vicon_y]
    )

    # -----------------------------------------------------------------------
    # Print summary statistics
    # -----------------------------------------------------------------------
    def rmse(a, b):
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() == 0:
            return np.nan
        return np.sqrt(np.mean((a[mask] - b[mask]) ** 2))

    print("\n=== Validation Summary ===")
    print(
        f"  vx (debug vs odom):           RMSE = {rmse(debug['vx'], odom_vx_matched):.4f} m/s"
    )
    print(
        f"  vx (debug vs recomputed):      RMSE = {rmse(debug['vx'], recomp_vx):.6f} m/s"
    )
    print(
        f"  vy (debug vs recomputed):      RMSE = {rmse(debug['vy'], recomp_vy):.6f} m/s"
    )
    print(
        f"  beta (debug vs recomputed):    RMSE = {rmse(debug['beta'], recomp_beta):.6f} rad"
    )
    print(
        f"  ang_vel_z (debug vs IMU):      RMSE = {rmse(debug['ang_vel_z'], imu_gz_matched):.4f} rad/s"
    )
    print(
        f"  ang_vel_z (debug vs Vicon):    RMSE = {rmse(debug['ang_vel_z'], vicon_angz_matched):.4f} rad/s"
    )
    print(
        f"  delta (debug vs recomputed):   RMSE = {rmse(debug['delta'], servo_delta_matched):.6f} rad"
    )
    print(
        f"  wheel_omega (debug vs recomp): RMSE = {rmse(debug['wheel_omega'], core_omega_matched):.6f} rad/s"
    )
    print(
        f"  frenet_n (debug vs recomp):    RMSE = {rmse(debug['frenet_n'], recomp_frenet_n):.6f} m"
    )

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    fig.suptitle("Phase 2 Observation Validation", fontsize=14)

    # 1. vx comparison
    ax = axes[0, 0]
    ax.plot(debug_t_rel, debug["vx"], label="debug vx", linewidth=1)
    ax.plot(debug_t_rel, odom_vx_matched, label="odom vx", linewidth=1, alpha=0.7)
    ax.plot(
        debug_t_rel,
        recomp_vx,
        label="recomputed vx",
        linewidth=1,
        linestyle="--",
        alpha=0.7,
    )
    ax.set_ylabel("m/s")
    ax.set_title("Longitudinal velocity (vx)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. vy comparison
    ax = axes[0, 1]
    ax.plot(debug_t_rel, debug["vy"], label="debug vy", linewidth=1)
    ax.plot(
        debug_t_rel,
        recomp_vy,
        label="recomputed vy",
        linewidth=1,
        linestyle="--",
        alpha=0.7,
    )
    ax.set_ylabel("m/s")
    ax.set_title("Lateral velocity (vy)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. ang_vel_z comparison
    ax = axes[1, 0]
    ax.plot(debug_t_rel, debug["ang_vel_z"], label="debug ang_vel_z", linewidth=1)
    ax.plot(debug_t_rel, imu_gz_matched, label="IMU (deg->rad)", linewidth=1, alpha=0.7)
    ax.plot(
        debug_t_rel,
        vicon_angz_matched,
        label="Vicon twist.angular.z",
        linewidth=1,
        alpha=0.7,
    )
    ax.set_ylabel("rad/s")
    ax.set_title("Yaw rate (ang_vel_z)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. delta comparison
    ax = axes[1, 1]
    ax.plot(debug_t_rel, debug["delta"], label="debug delta", linewidth=1)
    ax.plot(
        debug_t_rel,
        servo_delta_matched,
        label="recomputed delta",
        linewidth=1,
        linestyle="--",
        alpha=0.7,
    )
    ax.set_ylabel("rad")
    ax.set_title("Steering angle (delta)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. beta comparison
    ax = axes[2, 0]
    ax.plot(debug_t_rel, debug["beta"], label="debug beta", linewidth=1)
    ax.plot(
        debug_t_rel,
        recomp_beta,
        label="recomputed beta",
        linewidth=1,
        linestyle="--",
        alpha=0.7,
    )
    ax.set_ylabel("rad")
    ax.set_title("Sideslip (beta)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. wheel_omega comparison
    ax = axes[2, 1]
    ax.plot(debug_t_rel, debug["wheel_omega"], label="debug wheel_omega", linewidth=1)
    ax.plot(
        debug_t_rel,
        core_omega_matched,
        label="recomputed wheel_omega",
        linewidth=1,
        linestyle="--",
        alpha=0.7,
    )
    ax.set_ylabel("rad/s")
    ax.set_title("Wheel angular velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 7. frenet_n comparison
    ax = axes[3, 0]
    ax.plot(debug_t_rel, debug["frenet_n"], label="debug frenet_n", linewidth=1)
    ax.plot(
        debug_t_rel,
        recomp_frenet_n,
        label="recomputed frenet_n",
        linewidth=1,
        linestyle="--",
        alpha=0.7,
    )
    ax.set_ylabel("m")
    ax.set_title("Lateral offset (frenet_n)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (s)")

    # 8. frenet_u / yaw
    ax = axes[3, 1]
    ax.plot(debug_t_rel, debug["frenet_u"], label="debug frenet_u", linewidth=1)
    ax.plot(debug_t_rel, debug["yaw"], label="debug yaw", linewidth=1, alpha=0.7)
    ax.set_ylabel("rad")
    ax.set_title("Heading error (frenet_u) and yaw")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (s)")

    plt.tight_layout()

    out_path = Path(args.bag_dir) / "validation_plots.png"
    fig.savefig(str(out_path), dpi=150)
    print(f"\nPlots saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
