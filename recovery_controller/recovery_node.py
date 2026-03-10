import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped
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

        # Read parameters
        body_name = self.get_parameter("vicon_body_name").value
        self.zone_x_min = self.get_parameter("zone_x_min").value
        self.zone_x_max = self.get_parameter("zone_x_max").value
        self.zone_y_min = self.get_parameter("zone_y_min").value
        self.zone_y_max = self.get_parameter("zone_y_max").value
        rate = self.get_parameter("rate").value
        self.debug = self.get_parameter("debug").value

        # Latest sensor data (None until first message arrives)
        self._latest_pose = None
        self._latest_imu = None

        # Subscribe to Vicon pose (VRPN client publishes BEST_EFFORT)
        pose_topic = f"/vrpn_mocap/{body_name}/pose"
        vrpn_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PoseStamped, pose_topic, self._pose_cb, vrpn_qos)

        # Subscribe to VESC IMU
        self.create_subscription(Imu, "/sensors/imu/raw", self._imu_cb, 10)

        # Publisher on /ebrake for safety stop
        self._ebrake_pub = self.create_publisher(AckermannDriveStamped, "/ebrake", 10)

        # Publisher on /drive for autonomous recovery control
        self._drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        # State estimator (zone geometry placeholder — will be configured later)
        self._estimator = StateEstimator(
            zone_start=(self.zone_x_min, self.zone_y_min),
            zone_end=(self.zone_x_max, self.zone_y_max),
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

    def _imu_cb(self, msg: Imu):
        self._latest_imu = msg

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
        ang_vel_z = 0.0
        if self._latest_imu is not None:
            ang_vel_z = self._estimator.yaw_rate(self._latest_imu.angular_velocity.z)

        obs = self._obs_builder.build(
            vx=0.0,
            vy=0.0,
            frenet_u=0.0,
            frenet_n=0.0,
            ang_vel_z=ang_vel_z,
            delta=0.0,
            beta=0.0,
            wheel_omega=0.0,
        )

        # --- Debug: publish raw (pre-normalization) values ---
        debug_msg = String()
        debug_msg.data = f'{{"ang_vel_z": {ang_vel_z:.4f}, "obs[4]": {obs[4]:.4f}}}'
        self._debug_pub.publish(debug_msg)

        # Update pose from Vicon
        x = self._latest_pose.pose.position.x
        y = self._latest_pose.pose.position.y

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
