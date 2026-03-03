import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped


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

        # Read parameters
        body_name = self.get_parameter("vicon_body_name").value
        self.zone_x_min = self.get_parameter("zone_x_min").value
        self.zone_x_max = self.get_parameter("zone_x_max").value
        self.zone_y_min = self.get_parameter("zone_y_min").value
        self.zone_y_max = self.get_parameter("zone_y_max").value
        rate = self.get_parameter("rate").value

        # Latest pose from Vicon (None until first message arrives)
        self._latest_pose = None

        # Subscribe to Vicon pose (VRPN client publishes BEST_EFFORT)
        pose_topic = f"/vrpn_mocap/{body_name}/pose"
        vrpn_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PoseStamped, pose_topic, self._pose_cb, vrpn_qos)

        # Publisher on /ebrake (priority 200 in mux — overrides /drive and /teleop)
        self._ebrake_pub = self.create_publisher(AckermannDriveStamped, "/ebrake", 10)
        self._drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

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

        x = self._latest_pose.pose.position.x
        y = self._latest_pose.pose.position.y

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
