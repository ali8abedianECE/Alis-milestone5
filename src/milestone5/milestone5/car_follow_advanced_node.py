#used for math based functions
import math
import os

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


class CarFollowAdvancedNode(Node):
    """
    AutoPass Follow Mode with a raceline and pure-pursuit.
    Steers along the pre-built raceline (not directly at the car),
    and uses ygap (path distance to opponent) to control speed.
    """

    def __init__(self):
        """Loads raceline, sets up subscriptions, publisher, and parameters.

        ARGS: None
        RETURNS: None
        """
        super().__init__('car_follow_advanced_node')

        self.declare_parameter('raceline_path', os.path.expanduser('~/raceline.csv'))
        self.declare_parameter('lookahead', 0.8)    # pure-pursuit lookahead distance (m)
        self.declare_parameter('wheelbase', 0.32)   # F1Tenth wheelbase (m)
        self.declare_parameter('max_speed', 1.5)
        self.declare_parameter('min_speed', 0.7)
        self.declare_parameter('d_close', 0.4)      # DF_min: ygap below this → slow down
        self.declare_parameter('d_far', 1.5)        # DF_max: ygap above this → speed up
        self.declare_parameter('d_stop', 0.1)
        self.declare_parameter('d_clear', 0.3)
        self.declare_parameter('tau_down', 1.5)
        self.declare_parameter('tau_up', 0.4)

        self._waypoints = self._load_raceline(
            self.get_parameter('raceline_path').value
        )

        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(Bool, '/car_detector/car_detected', self._detected_cb, 10)
        self.create_subscription(Float32, '/car_detector/theta', self._theta_cb, 10)
        self.create_subscription(Float32, '/car_detector/distance', self._distance_cb, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/state/car_follow_advanced', 10)

        self._car_visible = False
        self._theta = 0.0
        self._d_car = None
        self._ego_x = 0.0
        self._ego_y = 0.0
        self._ego_yaw = 0.0
        self._last_stamp = self.get_clock().now()
        self._v_cmd = 0.0

        self.get_logger().info(f'[CarFollowAdvancedNode] Loaded {len(self._waypoints)} waypoints')

    # ---------- raceline ----------

    def _load_raceline(self, path: str):
        """Loads x,y waypoints from a CSV file (one waypoint per row: x,y).

        ARGS: path (str) — absolute path to the raceline CSV
        RETURNS: numpy.ndarray — shape (N, 2) array of [x, y] waypoints
        """
        if not os.path.exists(path):
            self.get_logger().warn(f'Raceline not found at {path}, using empty raceline')
            return np.zeros((1, 2))
        waypoints = np.loadtxt(path, delimiter=',', usecols=(0, 1))
        return waypoints if waypoints.ndim == 2 else waypoints[np.newaxis, :]

    def _closest_idx(self, x: float, y: float) -> int:
        """Returns the index of the waypoint closest to (x, y).

        ARGS:
            x (float) — query x position in map frame
            y (float) — query y position in map frame
        RETURNS: int — index into self._waypoints
        """
        dists = np.linalg.norm(self._waypoints - np.array([x, y]), axis=1)
        return int(np.argmin(dists))

    def _ygap(self, ego_idx: int, opp_idx: int) -> float:
        """Computes signed path distance from ego to opponent along the raceline.

        Positive means opponent is ahead of ego.

        ARGS:
            ego_idx (int) — raceline index of the ego car
            opp_idx (int) — raceline index of the opponent car
        RETURNS: float — path distance in metres (approx, assumes uniform waypoint spacing)
        """
        n = len(self._waypoints)
        diff = (opp_idx - ego_idx) % n
        # spacing between consecutive waypoints
        spacing = float(np.mean(np.linalg.norm(np.diff(self._waypoints, axis=0), axis=1)))
        return diff * spacing

    def _lookahead_point(self, ego_idx: int, lookahead: float):
        """Finds the waypoint at approximately lookahead distance ahead of ego_idx.

        ARGS:
            ego_idx (int) — current ego waypoint index
            lookahead (float) — desired lookahead distance in metres
        RETURNS: numpy.ndarray — [x, y] of the lookahead waypoint
        """
        n = len(self._waypoints)
        dist = 0.0
        idx = ego_idx
        while dist < lookahead:
            next_idx = (idx + 1) % n
            dist += float(np.linalg.norm(self._waypoints[next_idx] - self._waypoints[idx]))
            idx = next_idx
        return self._waypoints[idx]

    # ---------- subscribers ----------

    def _odom_cb(self, msg: Odometry):
        """Caches ego position and yaw from odometry and triggers control output.

        ARGS: msg (Odometry) — ego odometry from /odom
        RETURNS: None
        """
        self._ego_x = msg.pose.pose.position.x
        self._ego_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # yaw from quaternion
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._ego_yaw = math.atan2(siny, cosy)
        self._publish_drive()

    def _detected_cb(self, msg: Bool):
        """Updates car visibility flag.

        ARGS: msg (Bool) — True if car_detector sees a car
        RETURNS: None
        """
        self._car_visible = msg.data

    def _theta_cb(self, msg: Float32):
        """Caches latest bearing to the opponent.

        ARGS: msg (Float32) — bearing in radians, LiDAR convention (left=+, right=-)
        RETURNS: None
        """
        self._theta = msg.data

    def _distance_cb(self, msg: Float32):
        """Caches latest straight-line distance to the opponent.

        ARGS: msg (Float32) — distance in metres from /car_detector/distance
        RETURNS: None
        """
        self._d_car = msg.data if msg.data > 0.0 else None

    # ---------- control ----------

    def _publish_drive(self):
        """Computes pure-pursuit steering and speed then publishes AckermannDriveStamped.

        Uses ygap (path distance along raceline) for speed control instead of raw d_car,
        matching AutoPass Follow Mode. Steering follows the raceline via pure-pursuit.

        PUBLISHES:
            /state/car_follow_advanced (AckermannDriveStamped) — speed and steering
        RETURNS: None
        """
        if len(self._waypoints) < 2:
            return

        now = self.get_clock().now()
        dt = max(1e-4, min((now - self._last_stamp).nanoseconds * 1e-9, 0.5))
        self._last_stamp = now

        ego_idx = self._closest_idx(self._ego_x, self._ego_y)

        # project opponent position from bearing + distance in map frame
        # theta is LiDAR convention so negate to get camera-right bearing
        bearing_map = self._ego_yaw - self._theta
        opp_x = self._ego_x + self._d_car * math.cos(bearing_map) if self._d_car else None
        opp_y = self._ego_y + self._d_car * math.sin(bearing_map) if self._d_car else None

        # ygap: path distance from ego to opponent along the raceline
        if self._car_visible and opp_x is not None:
            opp_idx = self._closest_idx(opp_x, opp_y)
            gap = self._ygap(ego_idx, opp_idx)
        else:
            gap = None

        steering = self._pure_pursuit(ego_idx)
        speed = self._speed_cal(dt, gap)

        msg = AckermannDriveStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = speed
        msg.drive.steering_angle = steering
        self.drive_pub.publish(msg)

    def _pure_pursuit(self, ego_idx: int) -> float:
        """Computes Ackermann steering angle toward the lookahead point on the raceline.

        steering = atan2(2 * L * sin(alpha), lookahead)
        where alpha is the angle from car heading to the lookahead point.

        ARGS: ego_idx (int) — current ego waypoint index
        RETURNS: float — steering angle in radians
        """
        L = self.get_parameter('wheelbase').value
        ld = self.get_parameter('lookahead').value

        target = self._lookahead_point(ego_idx, ld)
        dx = target[0] - self._ego_x
        dy = target[1] - self._ego_y

        # angle from car heading to lookahead point
        alpha = math.atan2(dy, dx) - self._ego_yaw
        # wrap to [-pi, pi]
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        return math.atan2(2.0 * L * math.sin(alpha), ld)

    def _speed_cal(self, dt: float, gap) -> float:
        """Speed control based on ygap (path distance) matching AutoPass Follow Mode.

        Uses ygap instead of raw d_car so distance is measured along the track, not straight-line.
        Falls back to min_speed if car is not visible.

        ARGS:
            dt  (float)      — seconds since last call
            gap (float|None) — ygap in metres, or None if car not visible
        RETURNS: float — smoothed speed in m/s
        """
        V_MAX = self.get_parameter('max_speed').value
        V_MIN = self.get_parameter('min_speed').value
        D_CLOSE = self.get_parameter('d_close').value
        D_FAR = self.get_parameter('d_far').value
        TAU_DOWN = self.get_parameter('tau_down').value
        TAU_UP = self.get_parameter('tau_up').value

        if gap is not None:
            f_vis = max(0.0, min(1.0, (gap - D_CLOSE) / (D_FAR - D_CLOSE)))
        else:
            f_vis = 0.0  # car not visible, coast at min speed

        v_target = V_MIN + (V_MAX - V_MIN) * f_vis
        tau = TAU_DOWN if v_target < self._v_cmd else TAU_UP
        alpha = 1.0 - math.exp(-dt / tau)
        self._v_cmd = max(V_MIN, min(V_MAX, self._v_cmd + alpha * (v_target - self._v_cmd)))
        return self._v_cmd


def main(args=None):
    """Entry point.

    ARGS: args (list) — optional CLI args passed to rclpy
    RETURNS: None
    """
    rclpy.init(args=args)
    node = CarFollowAdvancedNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
