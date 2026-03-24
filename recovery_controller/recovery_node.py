import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64, String
from ackermann_msgs.msg import AckermannDriveStamped
from vesc_msgs.msg import VescStateStamped
from recovery_controller.state_estimator import StateEstimator
from recovery_controller.observation_builder import (
    ObservationBuilder,
    parse_norm_bounds,
)
from recovery_controller.policy_runner import PolicyRunner
from recovery_controller.stanley_controller import StanleyController


class RecoveryNode(Node):
    """Phase 1: When the car is outside the recovery zone, publish speed=0,
    steer=0 on /ebrake (priority 200) to override all other commands.
    When inside the zone, publish nothing — the mux times out within
    500 ms and /drive + /teleop regain control.
    """

    def __init__(self):
        super().__init__("recovery_node")

        # Declare parameters (no defaults — recovery.yaml is the single source of truth)
        self.declare_parameter("vicon_body_name", rclpy.Parameter.Type.STRING)
        self.declare_parameter("zone_x_min", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("zone_x_max", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("zone_y_min", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("zone_y_max", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("rate", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("debug", rclpy.Parameter.Type.BOOL)
        self.declare_parameter("yaw_rate_source", rclpy.Parameter.Type.STRING)
        self.declare_parameter(
            "steering_angle_to_servo_gain", rclpy.Parameter.Type.DOUBLE
        )
        self.declare_parameter(
            "steering_angle_to_servo_offset", rclpy.Parameter.Type.DOUBLE
        )
        self.declare_parameter("speed_to_erpm_gain", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("wheel_radius", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("model_path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("norm_bounds_path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("controller", rclpy.Parameter.Type.STRING)
        self.declare_parameter("stanley_k", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("stanley_k_soft", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("stanley_k_heading", rclpy.Parameter.Type.DOUBLE)

        # Read parameters
        body_name = self.get_parameter("vicon_body_name").value
        self.zone_x_min = self.get_parameter("zone_x_min").value
        self.zone_x_max = self.get_parameter("zone_x_max").value
        self.zone_y_min = self.get_parameter("zone_y_min").value
        self.zone_y_max = self.get_parameter("zone_y_max").value
        rate = self.get_parameter("rate").value
        self.debug = self.get_parameter("debug").value
        self.yaw_rate_source = self.get_parameter("yaw_rate_source").value
        servo_gain = self.get_parameter("steering_angle_to_servo_gain").value
        servo_offset = self.get_parameter("steering_angle_to_servo_offset").value
        self._speed_to_erpm_gain = self.get_parameter("speed_to_erpm_gain").value
        wheel_radius = self.get_parameter("wheel_radius").value
        model_path = self.get_parameter("model_path").value
        norm_bounds_path = self.get_parameter("norm_bounds_path").value
        self._controller_type = self.get_parameter("controller").value

        # Validate controller type
        valid_controllers = ["learned", "stanley"]
        if self._controller_type not in valid_controllers:
            raise ValueError(
                f"Invalid controller type '{self._controller_type}'. "
                f"Must be one of {valid_controllers}"
            )

        # Load norm bounds from YAML file (output by simulator training script)
        with open(norm_bounds_path, "r") as f:
            norm_bounds = parse_norm_bounds(yaml.safe_load(f))
        self.get_logger().info(f"Loaded norm bounds from {norm_bounds_path}")

        if self.debug:
            self.get_logger().info(f"Normalized bounds from yaml:{norm_bounds}")

        # Derive s_max from norm_bounds (single source of truth)
        s_max = norm_bounds["delta"][1]

        # Activation flag — True while policy is running inside the drive zone
        self._autonomy_active = False

        # Latest sensor data (None until first message arrives)
        self._latest_pose = None
        self._latest_twist = None
        self._latest_imu = None
        self._latest_servo_pos = None
        self._latest_erpm = None

        # validate yaw_rate_source set correctly
        if self.yaw_rate_source not in ["imu", "vicon"]:
            self.get_logger().warn(
                f"yaw_rate_source {self.yaw_rate_source} not set correctly, defaulting to 'vicon' source"
            )

        # Subscribe to Vicon pose and twist (VRPN client publishes BEST_EFFORT)
        vrpn_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        pose_topic = f"/vrpn_mocap/{body_name}/pose"
        twist_topic = f"/vrpn_mocap/{body_name}/twist"
        self.create_subscription(PoseStamped, pose_topic, self._pose_cb, vrpn_qos)
        self.create_subscription(TwistStamped, twist_topic, self._twist_cb, vrpn_qos)

        # Subscribe to VESC IMU
        self.create_subscription(Imu, "/sensors/imu/raw", self._imu_cb, 10)

        # Subscribe to VESC state (ERPM, servo position, etc.)
        self.create_subscription(
            VescStateStamped, "/sensors/core", self._vesc_state_cb, 10
        )

        # Subscribe to VESC servo position (event-driven, published on each servo command)
        self.create_subscription(
            Float64, "/sensors/servo_position_command", self._servo_cb, 10
        )

        # Publisher on /ebrake for safety stop
        self._ebrake_pub = self.create_publisher(AckermannDriveStamped, "/ebrake", 10)

        # Publisher on /drive for autonomous recovery control
        self._drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        # State estimator
        self._estimator = StateEstimator(
            zone_y_min=self.zone_y_min,
            zone_y_max=self.zone_y_max,
            servo_offset=servo_offset,
            servo_gain=servo_gain,
            speed_to_erpm_gain=self._speed_to_erpm_gain,
            wheel_radius=wheel_radius,
        )

        # Observation builder
        zone_width = self.zone_y_max - self.zone_y_min
        self._obs_builder = ObservationBuilder(
            norm_bounds=norm_bounds,
            zone_width=zone_width,
            dt=1.0 / rate,
        )

        # Controller setup
        self._stanley = None
        self._policy = None
        if self._controller_type == "learned":
            self._policy = PolicyRunner(model_path, s_max=s_max)
            self.get_logger().info(f"Loaded ONNX policy model from {model_path}")
        else:
            stanley_k = self.get_parameter("stanley_k").value
            stanley_k_soft = self.get_parameter("stanley_k_soft").value
            stanley_k_heading = self.get_parameter("stanley_k_heading").value
            self._stanley = StanleyController(
                k=stanley_k,
                k_soft=stanley_k_soft,
                k_heading=stanley_k_heading,
                target_speed=0.0,
            )
            self.get_logger().info(
                f"Stanley controller initialized (k={stanley_k}, "
                f"k_soft={stanley_k_soft}, k_heading={stanley_k_heading})"
            )

        # Debug publisher — raw (pre-normalization) observation values as JSON
        self._debug_pub = self.create_publisher(String, "/recovery/debug_obs", 10)

        # Timer at configured rate
        dt = 1.0 / rate
        self.create_timer(dt, self._tick)

        self.get_logger().info(
            f"Recovery node started — autonomous zone [{self.zone_x_min}, {self.zone_x_max}] x "
            f"[{self.zone_y_min}, {self.zone_y_max}], "
            f"listening on {pose_topic}"
        )

    def _pose_cb(self, msg: PoseStamped):
        self._latest_pose = msg

    def _twist_cb(self, msg: TwistStamped):
        self._latest_twist = msg

    def _imu_cb(self, msg: Imu):
        self._latest_imu = msg

    def _vesc_state_cb(self, msg: VescStateStamped):
        self._latest_erpm = msg.state.speed

    def _servo_cb(self, msg: Float64):
        self._latest_servo_pos = msg.data

    def _compute_control(self, obs, vx, frenet_u, frenet_n):
        """Return (speed, steering_angle) from the active controller."""
        if self._controller_type == "learned":
            raw_accl, raw_steer = self._policy.predict(obs)
            self._obs_builder.step(raw_accl, raw_steer)
            steering_angle = self._policy.denorm_steering(raw_steer)
            speed = self._obs_builder.curr_vel_cmd
        else:
            speed, steering_angle = self._stanley.get_action(vx, frenet_u, frenet_n)
        return speed, steering_angle

    def _in_ebrake_zone(self, x: float, y: float) -> bool:
        """Vehicle is in safety ebrake zone"""
        return (
            x >= self.zone_x_max or y <= self.zone_y_min or y >= self.zone_y_max
        ) and x >= self.zone_x_min

    def _in_drive_zone(self, x: float, y: float) -> bool:
        """Vehicle is in autonomous drive zone"""
        return (
            self.zone_x_min <= x <= self.zone_x_max
            and self.zone_y_min <= y <= self.zone_y_max
        )

    def _tick(self):
        if self._latest_pose is None:
            return

        # --- Compute observation features ---
        q = self._latest_pose.pose.orientation
        x = self._latest_pose.pose.position.x
        y = self._latest_pose.pose.position.y
        yaw = self._estimator.yaw_from_quaternion(q.x, q.y, q.z, q.w)

        # Body-frame velocity from Vicon twist
        vx, vy = 0.0, 0.0
        if self._latest_twist is not None:
            vx, vy = self._estimator.body_frame_velocity(
                self._latest_twist.twist.linear.x,
                self._latest_twist.twist.linear.y,
                yaw,
            )

        # Frenet coordinates (heading error + lateral offset)
        frenet_u, frenet_n = self._estimator.frenet_coords(y, yaw)

        # Sideslip
        beta = self._estimator.sideslip(vx, vy)

        # get yaw rate from Vicon or IMU depending on config
        ang_vel_z = 0.0
        if self.yaw_rate_source == "imu":
            if self._latest_imu is not None:
                ang_vel_z = self._estimator.yaw_rate(
                    self._latest_imu.angular_velocity.z
                )
        else:
            if self._latest_twist is not None:
                ang_vel_z = self._latest_twist.twist.angular.z

        delta = 0.0
        if self._latest_servo_pos is not None:
            delta = self._estimator.steering_angle(self._latest_servo_pos)

        wheel_omega = 0.0
        if self._latest_erpm is not None:
            wheel_omega = self._estimator.wheel_omega(self._latest_erpm)

        obs = self._obs_builder.build(
            vx=vx,
            vy=vy,
            frenet_u=frenet_u,
            frenet_n=frenet_n,
            ang_vel_z=ang_vel_z,
            delta=delta,
            beta=beta,
            wheel_omega=wheel_omega,
        )

        # --- Debug: publish raw (pre-normalization) values ---
        debug_msg = String()
        debug_msg.data = (
            f"yaw={yaw:.4f}\n"
            f"vx={vx:.4f}\n"
            f"vy={vy:.4f}\n"
            f"frenet_u={frenet_u:.4f}\n"
            f"frenet_n={frenet_n:.4f}\n"
            f"beta={beta:.4f}\n"
            f"ang_vel_z={ang_vel_z:.4f}\n"
            f"delta={delta:.4f}\n"
            f"wheel_omega={wheel_omega:.4f}"
        )
        self._debug_pub.publish(debug_msg)

        if not self.debug:
            if self._in_ebrake_zone(x, y):
                self._autonomy_active = False
                self.get_logger().warn(
                    f"Outside safety zone at ({x:.2f}, {y:.2f}) — e-braking",
                    throttle_duration_sec=0.5,
                )
                msg = AckermannDriveStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.drive.speed = 0.0
                msg.drive.steering_angle = 0.0
                self._ebrake_pub.publish(msg)

            elif self._in_drive_zone(x, y):
                # On first tick in zone: capture initial speed and reset obs builder, stanley target speed
                if not self._autonomy_active:
                    self._obs_builder.reset(vx)
                    if self._stanley is not None:
                        self._stanley.set_target_speed(vx)
                    self._autonomy_active = True
                    self.get_logger().info(
                        f"Recovery activated — initial speed {vx:.2f} m/s"
                    )

                speed, steering_angle = self._compute_control(
                    obs, vx, frenet_u, frenet_n
                )

                msg = AckermannDriveStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.drive.speed = float(speed)
                msg.drive.steering_angle = float(steering_angle)

                if self._controller_type == "learned":
                    self.get_logger().info(f"Observation: {obs}")
                else:
                    self.get_logger().info(
                        f"vx: {vx}, frenet_u: {frenet_u}, frenet_n: {frenet_n}"
                    )
                self.get_logger().info(
                    f"Control command: Speed: {speed}, Steer angle: {steering_angle}"
                )

                self._drive_pub.publish(msg)

            else:
                # Outside both zones (teleop region) — deactivate
                self._autonomy_active = False


def main(args=None):
    rclpy.init(args=args)
    node = RecoveryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
