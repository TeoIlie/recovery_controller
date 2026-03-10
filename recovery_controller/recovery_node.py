import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64, String
from ackermann_msgs.msg import AckermannDriveStamped
from vesc_msgs.msg import VescStateStamped
from recovery_controller.state_estimator import StateEstimator
from recovery_controller.observation_builder import ObservationBuilder


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
        self.declare_parameter(
            "steering_angle_to_servo_gain", rclpy.Parameter.Type.DOUBLE
        )
        self.declare_parameter(
            "steering_angle_to_servo_offset", rclpy.Parameter.Type.DOUBLE
        )
        self.declare_parameter("speed_to_erpm_gain", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("wheel_radius", rclpy.Parameter.Type.DOUBLE)

        # Read parameters
        body_name = self.get_parameter("vicon_body_name").value
        self.zone_x_min = self.get_parameter("zone_x_min").value
        self.zone_x_max = self.get_parameter("zone_x_max").value
        self.zone_y_min = self.get_parameter("zone_y_min").value
        self.zone_y_max = self.get_parameter("zone_y_max").value
        rate = self.get_parameter("rate").value
        self.debug = self.get_parameter("debug").value
        servo_gain = self.get_parameter("steering_angle_to_servo_gain").value
        servo_offset = self.get_parameter("steering_angle_to_servo_offset").value
        speed_to_erpm_gain = self.get_parameter("speed_to_erpm_gain").value
        wheel_radius = self.get_parameter("wheel_radius").value

        # Latest sensor data (None until first message arrives)
        self._latest_pose = None
        self._latest_twist = None
        self._latest_imu = None
        self._latest_servo_pos = None
        self._latest_erpm = None

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
            zone_start=(self.zone_x_min, self.zone_y_min),
            zone_end=(self.zone_x_max, self.zone_y_max),
            servo_offset=servo_offset,
            servo_gain=servo_gain,
            speed_to_erpm_gain=speed_to_erpm_gain,
            wheel_radius=wheel_radius,
        )

        # Observation builder
        zone_width = self.zone_y_max - self.zone_y_min
        self._obs_builder = ObservationBuilder(
            zone_width=zone_width,
            a_max=5.0,
            v_min=-5.0,
            v_max=20.0,
            dt=1.0 / rate,
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

    def _in_ebrake_zone(self, x: float, y: float) -> bool:
        """Vehicle is in safety ebrake zone"""
        return x >= self.zone_x_max or y <= self.zone_y_min or y >= self.zone_y_max

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
        
        # TODO remove logging after testing
        # self.get_logger().info(f"yaw={yaw:.4f}", throttle_duration_sec=0.5)

        # Body-frame velocity from Vicon twist
        vx, vy = 0.0, 0.0
        if self._latest_twist is not None:
            vx, vy = self._estimator.body_frame_velocity(
                self._latest_twist.twist.linear.x,
                self._latest_twist.twist.linear.y,
                yaw,
            )

        # Frenet coordinates (heading error + lateral offset)
        frenet_u, frenet_n = self._estimator.frenet_coords(x, y, yaw)

        # Sideslip
        beta = self._estimator.sideslip(vx, vy)

        ang_vel_z = 0.0
        if self._latest_imu is not None:
            ang_vel_z = self._estimator.yaw_rate(self._latest_imu.angular_velocity.z)

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
                # if in ebrake zone, publish 0 speed command on /ebrake topic
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
                # if in autonomous drive zone, publish autonomous drive command on /drive topic
                # TODO replace with actual autonomous drive command
                self.get_logger().info(
                    f"Autonomous recovery control active",
                    throttle_duration_sec=0.5,
                )
                # TODO replace placeholder code with autonomous driving command
                msg = AckermannDriveStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.drive.speed = 0.5
                msg.drive.steering_angle = 0.0
                self._drive_pub.publish(msg)

            # else, teleop takes precedence


def main(args=None):
    rclpy.init(args=args)
    node = RecoveryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
