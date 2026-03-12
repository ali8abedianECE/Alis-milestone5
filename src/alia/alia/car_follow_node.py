#used for math based functions
import math

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32


class CarFollowNode(Node):
    """Follow Mode from AutoPass: maintains a safe following distance [DF_min, DF_max] behind the opponent."""

    def __init__(self):
        """Sets up subscriptions, publisher, and follow/speed parameters.

        ARGS: None
        RETURNS: None
        """
        super().__init__('car_follow_node')

        # from car_detector_node — no need to recompute
        self.create_subscription(Bool, '/car_detector/car_detected', self._detected_cb, 10)
        self.create_subscription(Float32, '/car_detector/theta', self._theta_cb, 10)
        self.create_subscription(Float32, '/car_detector/distance', self._distance_cb, 10)
        # lidar still needed for d_obs (nearest obstacle, not the car)
        self.create_subscription(LaserScan, '/scan', self._lidar_cb, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/state/car_follow', 10)

        self._car_visible = False
        self._theta = 0.0   # bearing in LiDAR convention (left=+, right=-)
        self._d_car = None  # metres, from car_detector
        self._d_obs = 1e6
        self._last_stamp = self.get_clock().now()
        self._v_cmd = 0.0
        self._t_unseen = 0.0

        # speed parameters — d_close = DF_min, d_far = DF_max from AutoPass paper
        self.declare_parameter('max_speed', 1.5)  # max speed we go
        self.declare_parameter('min_speed', 0.7)  # min speed we can go
        self.declare_parameter('d_close', 0.2)    # DF_min: too close, slow down
        self.declare_parameter('d_far', 1.0)      # DF_max: too far, speed up
        self.declare_parameter('tau_rise', 1.0)   # how fast we speed up if car is unseen
        self.declare_parameter('d_stop', 0.1)     # obstacles this close -> go min speed
        self.declare_parameter('d_clear', 0.3)    # no obstacles in this range = clear
        self.declare_parameter('tau_down', 1.5)   # how fast we slow down
        self.declare_parameter('tau_up', 0.4)     # how fast we speed up
        self.declare_parameter('k_steer', 1.0)    # proportional steering gain, flip sign if wrong

        self.get_logger().info('[CarFollowNode] Ready')

    # ---------- subscribers ----------

    def _detected_cb(self, msg: Bool):
        """Updates car visibility flag.

        ARGS: msg (Bool) — True if car_detector sees a car
        RETURNS: None
        """
        self._car_visible = msg.data

    def _theta_cb(self, msg: Float32):
        """Caches latest bearing to the car.

        ARGS: msg (Float32) — bearing in radians, LiDAR convention (left=+, right=-)
        RETURNS: None
        """
        self._theta = msg.data

    def _distance_cb(self, msg: Float32):
        """Caches latest distance to the car from car_detector_node.

        ARGS: msg (Float32) — distance in metres from /car_detector/distance
        RETURNS: None
        """
        self._d_car = msg.data if msg.data > 0.0 else None

    def _lidar_cb(self, scan: LaserScan):
        """Computes d_obs then publishes AckermannDriveStamped if car is visible.

        ARGS: scan (LaserScan) — raw scan used only for nearest-obstacle distance
        PUBLISHES:
            /state/car_follow (AckermannDriveStamped) — speed and steering to follow the car
        RETURNS: None
        """
        now = self.get_clock().now()
        dt = max(1e-4, min((now - self._last_stamp).nanoseconds * 1e-9, 0.5))
        self._last_stamp = now

        ranges = np.array(scan.ranges)
        finite = ranges[np.isfinite(ranges)]
        self._d_obs = float(np.min(finite)) if len(finite) > 0 else 1e6

        if not self._car_visible:
            return

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = self._speed_cal(dt)
        msg.drive.steering_angle = self._steer_cal()
        self.drive_pub.publish(msg)

    # ---------- control ----------

    def _speed_cal(self, dt: float) -> float:
        """Maintains following distance between DF_min (d_close) and DF_max (d_far) per AutoPass Follow Mode.

        Speed equation:
            v_target = v_min + (v_max - v_min) * f_vis * f_obs
            f_vis = clip((d_car - d_close) / (d_far - d_close), 0, 1)  if car visible
                  = 1 - exp(-t_unseen / tau_rise)                       otherwise
            f_obs = clip((d_obs - d_stop) / (d_clear - d_stop), 0, 1)
            v_cmd += (1 - exp(-dt/tau)) * (v_target - v_cmd)

        ARGS: dt (float) — seconds since last call
        RETURNS: float — smoothed speed in m/s
        """
        V_MAX = self.get_parameter('max_speed').value
        V_MIN = self.get_parameter('min_speed').value
        D_CLOSE = self.get_parameter('d_close').value
        D_FAR = self.get_parameter('d_far').value
        TAU_RISE = self.get_parameter('tau_rise').value
        D_STOP = self.get_parameter('d_stop').value
        D_CLEAR = self.get_parameter('d_clear').value
        TAU_DOWN = self.get_parameter('tau_down').value
        TAU_UP = self.get_parameter('tau_up').value

        self._t_unseen = 0.0 if self._car_visible else self._t_unseen + dt

        if self._car_visible and self._d_car is not None:
            f_vis = max(0.0, min(1.0, (self._d_car - D_CLOSE) / (D_FAR - D_CLOSE)))
        else:
            f_vis = 1.0 - math.exp(-self._t_unseen / TAU_RISE)

        f_obs = max(0.0, min(1.0, (self._d_obs - D_STOP) / (D_CLEAR - D_STOP)))
        v_target = V_MIN + (V_MAX - V_MIN) * f_vis * f_obs
        tau = TAU_DOWN if v_target < self._v_cmd else TAU_UP
        alpha = 1.0 - math.exp(-dt / tau)
        self._v_cmd = max(V_MIN, min(V_MAX, self._v_cmd + alpha * (v_target - self._v_cmd)))
        return self._v_cmd

    def _steer_cal(self) -> float:
        """Proportional steering toward the car bearing.

        theta is LiDAR convention (left=+, right=-), so k_steer * theta steers toward the car.
        NOTE: flip sign of k_steer if the car steers the wrong way.

        ARGS: None
        RETURNS: float — steering angle in radians
        """
        return float(self.get_parameter('k_steer').value * self._theta)


def main(args=None):
    """Entry point.

    ARGS: args (list) — optional CLI args passed to rclpy
    RETURNS: None
    """
    rclpy.init(args=args)
    node = CarFollowNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
