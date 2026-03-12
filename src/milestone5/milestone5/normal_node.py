import math

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class NormalNode(Node):
    """Follow-the-Gap with dynamic exponential speed control based on obstacle clearance."""

    def __init__(self):
        """Sets up subscriber, publisher, gap follow, and dynamic speed parameters.

        ARGS: None
        RETURNS: None
        """
        super().__init__('normal_node')

        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/state/normal_node', 10)

        # gap follow parameters
        self.declare_parameter('max_range', 3.0)
        self.declare_parameter('cone_fov', np.deg2rad(180))
        self.declare_parameter('window_size', 3)
        self.declare_parameter('radius_of_car', 0.10)
        self.declare_parameter('best_point_window', 5)
        self.declare_parameter('p_factor', 1.2)
        self.declare_parameter('steering_gain', 1.0)
        self.declare_parameter('steering_max', np.radians(25))

        # dynamic speed parameters — no car distance, just obstacle clearance
        self.declare_parameter('max_speed', 1.5)
        self.declare_parameter('min_speed', 0.7)
        self.declare_parameter('d_stop', 0.1)   # obstacle this close -> min speed
        self.declare_parameter('d_clear', 0.3)  # obstacle this far -> max speed
        self.declare_parameter('tau_down', 1.5)
        self.declare_parameter('tau_up', 0.4)

        self._v_cmd = 0.0
        self._last_stamp = self.get_clock().now()

        self.get_logger().info('[NormalNode] Ready')

    # ---------- gap follow pipeline ----------

    def preprocess_scan(self, scan: LaserScan):
        """Cleans, clips, and crops the scan to the forward FOV, then smooths with a moving average.

        ARGS: scan (LaserScan) — raw scan from /scan
        RETURNS: (ranges, angles) — matched numpy arrays cropped to CONE_FOV
        """
        ranges = np.array(scan.ranges)
        ranges = np.nan_to_num(ranges, nan=scan.range_max, posinf=scan.range_max, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.get_parameter('max_range').value)

        cone_fov = self.get_parameter('cone_fov').value
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        mask = (angles >= -cone_fov / 2.0) & (angles <= cone_fov / 2.0)
        ranges = ranges[mask]
        angles = angles[mask]

        window_size = self.get_parameter('window_size').value
        if window_size > 1:
            half_w = window_size // 2
            n = len(ranges)
            smoothed = np.empty(n)
            for i in range(n):
                smoothed[i] = np.mean(ranges[max(0, i - half_w):min(n - 1, i + half_w) + 1])
            ranges = smoothed

        return ranges, angles

    def find_closest_point(self, ranges: np.ndarray):
        """Finds the index and distance of the nearest non-zero beam.

        ARGS: ranges (np.ndarray) — preprocessed range array
        RETURNS: (closest_idx, closest_dist) — int and float
        """
        closest_idx = -1
        closest_dist = float('inf')
        for i, r in enumerate(ranges):
            if 0.0 < r < closest_dist:
                closest_dist = r
                closest_idx = i
        return closest_idx, closest_dist

    def apply_safety_bubble(self, ranges: np.ndarray, closest_idx: int, closest_dist: float, angle_increment: float):
        """Zeros out beams within the safety bubble around the closest obstacle.

        ARGS:
            ranges (np.ndarray) — preprocessed ranges
            closest_idx (int) — index of closest point
            closest_dist (float) — distance at closest_idx
            angle_increment (float) — scan.angle_increment in radians
        RETURNS: np.ndarray — ranges with bubble zeroed out
        """
        bubbled = np.copy(ranges)
        if closest_idx == -1 or closest_dist <= 0.0:
            return bubbled
        half_angle = np.arctan(self.get_parameter('radius_of_car').value / closest_dist)
        n_zero = int(np.ceil(half_angle / angle_increment))
        start = max(0, closest_idx - n_zero)
        end = min(len(bubbled) - 1, closest_idx + n_zero)
        bubbled[start:end + 1] = 0.0
        return bubbled

    def find_max_gap(self, bubbled_ranges: np.ndarray):
        """Finds the start and end indices of the longest free (non-zero) gap.

        ARGS: bubbled_ranges (np.ndarray) — ranges after bubble applied
        RETURNS: (gap_start, gap_end) — int indices
        """
        gap_start = gap_end = 0
        best = cur = 0
        cur_start = 0
        for i, r in enumerate(bubbled_ranges):
            if r > 0.0:
                if cur == 0:
                    cur_start = i
                cur += 1
            else:
                if cur > best:
                    best = cur
                    gap_start = cur_start
                    gap_end = i - 1
                cur = 0
        if cur > best:
            gap_start = len(bubbled_ranges) - cur
            gap_end = len(bubbled_ranges) - 1
        return gap_start, gap_end

    def find_best_point(self, bubbled_ranges: np.ndarray, gap_start: int, gap_end: int):
        """Smooths within the gap and picks the weighted centre-of-mass target index.

        ARGS:
            bubbled_ranges (np.ndarray) — ranges after bubble
            gap_start (int) — start index of max gap
            gap_end (int) — end index of max gap
        RETURNS: int — best beam index to aim for
        """
        if gap_end < gap_start:
            return len(bubbled_ranges) // 2

        copy_r = np.copy(bubbled_ranges)
        orig = np.copy(bubbled_ranges)
        half_w = self.get_parameter('best_point_window').value // 2

        for i in range(gap_start, gap_end + 1):
            vals = [orig[j] for j in range(i - half_w, i + half_w + 1) if gap_start <= j <= gap_end]
            copy_r[i] = np.mean(vals) if vals else 0.0

        gap_len = gap_end - gap_start + 1
        weights = [max(0.0, float(copy_r[gap_start + j])) ** self.get_parameter('p_factor').value for j in range(gap_len)]
        sum_w = sum(weights)

        if sum_w <= 1e-9:
            chosen = gap_len // 2
        else:
            chosen = int(round(sum(j * w for j, w in enumerate(weights)) / sum_w))
            chosen = max(0, min(chosen, gap_len - 1))

        return gap_start + chosen

    # ---------- speed control ----------

    def _speed_cal(self, dt: float, d_obs: float) -> float:
        """Dynamic speed based on obstacle clearance with exponential smoothing.

        Replaces the stepped MAX/MID/MIN speed logic — speed rises smoothly as
        obstacles clear, and drops quickly when something gets close.

        Speed equation:
            f_obs    = clip((d_obs - d_stop) / (d_clear - d_stop), 0, 1)
            v_target = v_min + (v_max - v_min) * f_obs
            v_cmd   += (1 - exp(-dt/tau)) * (v_target - v_cmd)

        ARGS:
            dt    (float) — seconds since last call
            d_obs (float) — nearest obstacle distance in metres
        RETURNS: float — smoothed speed in m/s
        """
        V_MAX = self.get_parameter('max_speed').value
        V_MIN = self.get_parameter('min_speed').value
        D_STOP = self.get_parameter('d_stop').value
        D_CLEAR = self.get_parameter('d_clear').value
        TAU_DOWN = self.get_parameter('tau_down').value
        TAU_UP = self.get_parameter('tau_up').value

        f_obs = max(0.0, min(1.0, (d_obs - D_STOP) / (D_CLEAR - D_STOP)))
        v_target = V_MIN + (V_MAX - V_MIN) * f_obs
        tau = TAU_DOWN if v_target < self._v_cmd else TAU_UP
        alpha = 1.0 - math.exp(-dt / tau)
        self._v_cmd = max(V_MIN, min(V_MAX, self._v_cmd + alpha * (v_target - self._v_cmd)))
        return self._v_cmd

    # ---------- drive ----------

    def publish_drive(self, best_idx: int, angles: np.ndarray, d_obs: float, dt: float):
        """Publishes AckermannDriveStamped with dynamic speed and gap-follow steering.

        ARGS:
            best_idx (int) — beam index to aim for
            angles (np.ndarray) — beam angles aligned with processed ranges
            d_obs (float) — nearest obstacle distance for speed control
            dt (float) — seconds since last scan
        PUBLISHES:
            /state/gap_follow (AckermannDriveStamped) — steering and speed
        RETURNS: None
        """
        steering = float(np.clip(
            self.get_parameter('steering_gain').value * angles[best_idx],
            -self.get_parameter('steering_max').value,
            self.get_parameter('steering_max').value
        ))
        speed = self._speed_cal(dt, d_obs)

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = steering
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

    def lidar_callback(self, scan: LaserScan):
        """Main control loop triggered on every LiDAR scan.

        ARGS: scan (LaserScan) — incoming scan from /scan
        PUBLISHES: /state/gap_follow via publish_drive
        RETURNS: None
        """
        now = self.get_clock().now()
        dt = max(1e-4, min((now - self._last_stamp).nanoseconds * 1e-9, 0.5))
        self._last_stamp = now

        ranges, angles = self.preprocess_scan(scan)
        if len(ranges) == 0:
            self.get_logger().warn('[NormalNode] No LIDAR data received!')
            return

        closest_idx, closest_dist = self.find_closest_point(ranges)
        bubbled = self.apply_safety_bubble(ranges, closest_idx, closest_dist, scan.angle_increment)
        gap_start, gap_end = self.find_max_gap(bubbled)
        best_idx = self.find_best_point(bubbled, gap_start, gap_end)
        best_idx = max(0, min(best_idx, len(angles) - 1))

        # nearest obstacle for speed control
        finite = ranges[np.isfinite(ranges) & (ranges > 0.0)]
        d_obs = float(np.min(finite)) if len(finite) > 0 else self.get_parameter('max_range').value

        self.publish_drive(best_idx, angles, d_obs, dt)


def main(args=None):
    """Entry point.

    ARGS: args (list) — optional CLI args passed to rclpy
    RETURNS: None
    """
    rclpy.init(args=args)
    node = NormalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
