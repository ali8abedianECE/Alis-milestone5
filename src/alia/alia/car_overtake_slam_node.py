import math
import os
from enum import Enum

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


class State(Enum):
    FOLLOW = 'follow'
    OVERTAKE = 'overtake'
    MERGE_FRONT = 'merge_front'
    FALLBACK = 'fallback'


class CarOvertakeSlamNode(Node):
    """
    AutoPass overtake using raceline CSV + occupancy grid map.
    Generates left/right offset trajectories (multiples of DsepL),
    checks feasibility against the map, and pure-pursuits the best one.
    """

    def __init__(self):
        """Loads raceline, sets up subscriptions, publisher, and AutoPass parameters.

        ARGS: None
        RETURNS: None
        """
        super().__init__('car_overtake_slam_node')

        self.declare_parameter('raceline_path', os.path.expanduser('~/raceline.csv'))
        self.declare_parameter('lookahead', 0.8)      # pure-pursuit lookahead (m)
        self.declare_parameter('wheelbase', 0.32)     # F1Tenth wheelbase (m)

        # AutoPass distance thresholds (paper Table 2)
        self.declare_parameter('df_min', 0.4)         # DF_min: min follow distance
        self.declare_parameter('df_max', 1.5)         # DF_max: max follow distance
        self.declare_parameter('dsep_l', 0.35)        # DsepL: lateral separation between cars
        self.declare_parameter('dsep_f', 0.3)         # DsepF: min frontal sep to merge front
        self.declare_parameter('dsep_b', 0.2)         # DsepB: min rear sep before fallback

        # speed
        self.declare_parameter('follow_speed', 1.2)
        self.declare_parameter('overtake_speed', 2.0)  # boost
        self.declare_parameter('merge_speed', 1.4)
        self.declare_parameter('fallback_speed', 0.5)

        self._raceline = self._load_raceline(self.get_parameter('raceline_path').value)
        self._map: OccupancyGrid = None

        self.create_subscription(OccupancyGrid, '/map', self._map_cb, 1)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(Bool, '/car_detector/car_detected', self._detected_cb, 10)
        self.create_subscription(Float32, '/car_detector/theta', self._theta_cb, 10)
        self.create_subscription(Float32, '/car_detector/distance', self._distance_cb, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/state/car_overtake', 10)
        self.done_pub = self.create_publisher(Bool, '/car_overtake/done', 10)

        self._car_visible = False
        self._theta = 0.0
        self._d_car = None
        self._ego_x = 0.0
        self._ego_y = 0.0
        self._ego_yaw = 0.0
        self._v_cmd = 0.0
        self._last_stamp = self.get_clock().now()

        self._state = State.FOLLOW
        self._active_traj = None   # np array of (N,2) waypoints currently being tracked
        self._overtake_side = None

        self.get_logger().info(f'[CarOvertakeSlamNode] Loaded {len(self._raceline)} waypoints')

    # ---------- raceline ----------

    def _load_raceline(self, path: str) -> np.ndarray:
        """Loads x,y waypoints from a CSV file.

        ARGS: path (str) — path to raceline CSV
        RETURNS: np.ndarray — shape (N,2)
        """
        if not os.path.exists(path):
            self.get_logger().warn(f'Raceline not found at {path}')
            return np.zeros((2, 2))
        wps = np.loadtxt(path, delimiter=',', usecols=(0, 1))
        return wps if wps.ndim == 2 else wps[np.newaxis, :]

    def _waypoint_headings(self, waypoints: np.ndarray) -> np.ndarray:
        """Computes forward heading at each waypoint using finite differences.

        ARGS: waypoints (np.ndarray) — shape (N,2)
        RETURNS: np.ndarray — shape (N,) headings in radians
        """
        n = len(waypoints)
        headings = np.zeros(n)
        for i in range(n):
            nxt = waypoints[(i + 1) % n]
            cur = waypoints[i]
            headings[i] = math.atan2(nxt[1] - cur[1], nxt[0] - cur[0])
        return headings

    def _offset_trajectory(self, waypoints: np.ndarray, offset: float) -> np.ndarray:
        """Shifts each waypoint perpendicular to the path by offset metres.

        Positive offset = left, negative = right (standard ROS convention).

        ARGS:
            waypoints (np.ndarray) — shape (N,2) raceline
            offset (float) — lateral offset in metres
        RETURNS: np.ndarray — shape (N,2) shifted trajectory
        """
        headings = self._waypoint_headings(waypoints)
        # perpendicular direction is heading + 90°
        shifted = np.copy(waypoints)
        shifted[:, 0] += offset * (-np.sin(headings))
        shifted[:, 1] += offset * np.cos(headings)
        return shifted

    def _trajectory_feasible(self, trajectory: np.ndarray) -> bool:
        """Checks all waypoints against the occupancy grid map.

        ARGS: trajectory (np.ndarray) — shape (N,2) waypoints in map frame
        RETURNS: bool — True if no waypoint lands in an occupied cell
        """
        if self._map is None:
            return False

        info = self._map.info
        data = self._map.data

        for x, y in trajectory:
            # convert world (x,y) to grid cell
            col = int((x - info.origin.position.x) / info.resolution)
            row = int((y - info.origin.position.y) / info.resolution)

            if col < 0 or col >= info.width or row < 0 or row >= info.height:
                return False  # outside map = infeasible

            cell = data[row * info.width + col]
            if cell > 50:  # occupied (0=free, 100=occupied, -1=unknown)
                return False

        return True

    def _best_overtake_trajectory(self):
        """Generates left and right offset trajectories and returns the feasible one.

        Tries left first (inner line is typically shorter), then right.
        Returns None if neither is feasible.

        ARGS: None
        RETURNS: (np.ndarray, str) | (None, None) — (trajectory, side) or (None, None)
        """
        dsep_l = self.get_parameter('dsep_l').value

        left_traj = self._offset_trajectory(self._raceline, dsep_l)
        right_traj = self._offset_trajectory(self._raceline, -dsep_l)

        if self._trajectory_feasible(left_traj):
            return left_traj, 'left'
        if self._trajectory_feasible(right_traj):
            return right_traj, 'right'
        return None, None

    # ---------- geometry ----------

    def _closest_idx(self, waypoints: np.ndarray, x: float, y: float) -> int:
        """Returns index of the waypoint closest to (x, y).

        ARGS:
            waypoints (np.ndarray) — shape (N,2)
            x (float) — query x
            y (float) — query y
        RETURNS: int
        """
        dists = np.linalg.norm(waypoints - np.array([x, y]), axis=1)
        return int(np.argmin(dists))

    def _ygap(self, ego_idx: int, opp_idx: int) -> float:
        """Path distance from ego to opponent along the raceline (positive = opponent ahead).

        ARGS:
            ego_idx (int) — ego waypoint index
            opp_idx (int) — opponent waypoint index
        RETURNS: float — metres
        """
        n = len(self._raceline)
        diff = (opp_idx - ego_idx) % n
        spacing = float(np.mean(np.linalg.norm(np.diff(self._raceline, axis=0), axis=1)))
        return diff * spacing

    def _lookahead_point(self, waypoints: np.ndarray, ego_idx: int) -> np.ndarray:
        """Finds the waypoint at lookahead distance ahead of ego_idx.

        ARGS:
            waypoints (np.ndarray) — trajectory to follow
            ego_idx (int) — current ego index on that trajectory
        RETURNS: np.ndarray — [x, y] lookahead point
        """
        ld = self.get_parameter('lookahead').value
        n = len(waypoints)
        dist = 0.0
        idx = ego_idx
        while dist < ld:
            nxt = (idx + 1) % n
            dist += float(np.linalg.norm(waypoints[nxt] - waypoints[idx]))
            idx = nxt
        return waypoints[idx]

    def _pure_pursuit(self, waypoints: np.ndarray, ego_idx: int) -> float:
        """Computes Ackermann steering angle toward the lookahead point.

        ARGS:
            waypoints (np.ndarray) — trajectory to follow
            ego_idx (int) — current ego index on that trajectory
        RETURNS: float — steering angle in radians
        """
        L = self.get_parameter('wheelbase').value
        ld = self.get_parameter('lookahead').value
        target = self._lookahead_point(waypoints, ego_idx)
        dx = target[0] - self._ego_x
        dy = target[1] - self._ego_y
        alpha = math.atan2(dy, dx) - self._ego_yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        return math.atan2(2.0 * L * math.sin(alpha), ld)

    def _opp_position(self):
        """Projects opponent position into map frame from ego pose + theta + d_car.

        ARGS: None
        RETURNS: (float, float) | (None, None) — (x, y) or None if no detection
        """
        if self._d_car is None or not self._car_visible:
            return None, None
        bearing = self._ego_yaw - self._theta
        opp_x = self._ego_x + self._d_car * math.cos(bearing)
        opp_y = self._ego_y + self._d_car * math.sin(bearing)
        return opp_x, opp_y

    # ---------- subscribers ----------

    def _map_cb(self, msg: OccupancyGrid):
        """Caches the occupancy grid map.

        ARGS: msg (OccupancyGrid) — map from /map
        RETURNS: None
        """
        self._map = msg
        self.get_logger().info('[CarOvertakeSlamNode] Map received')

    def _odom_cb(self, msg: Odometry):
        """Caches ego pose and triggers the state machine.

        ARGS: msg (Odometry) — ego odometry from /odom
        RETURNS: None
        """
        self._ego_x = msg.pose.pose.position.x
        self._ego_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._ego_yaw = math.atan2(siny, cosy)

        now = self.get_clock().now()
        dt = max(1e-4, min((now - self._last_stamp).nanoseconds * 1e-9, 0.5))
        self._last_stamp = now

        self._step(dt)

    def _detected_cb(self, msg: Bool):
        """Caches car visibility.

        ARGS: msg (Bool) — True if car detected
        RETURNS: None
        """
        self._car_visible = msg.data

    def _theta_cb(self, msg: Float32):
        """Caches bearing to opponent.

        ARGS: msg (Float32) — bearing in radians, LiDAR convention
        RETURNS: None
        """
        self._theta = msg.data

    def _distance_cb(self, msg: Float32):
        """Caches straight-line distance to opponent.

        ARGS: msg (Float32) — distance in metres
        RETURNS: None
        """
        self._d_car = msg.data if msg.data > 0.0 else None

    # ---------- state machine ----------

    def _step(self, dt: float):
        """Runs one tick of the AutoPass state machine and publishes a drive command.

        ARGS: dt (float) — seconds since last call
        PUBLISHES:
            /state/car_overtake_slam (AckermannDriveStamped)
        RETURNS: None
        """
        DF_MIN = self.get_parameter('df_min').value
        DF_MAX = self.get_parameter('df_max').value
        DSEP_F = self.get_parameter('dsep_f').value
        DSEP_B = self.get_parameter('dsep_b').value

        ego_idx = self._closest_idx(self._raceline, self._ego_x, self._ego_y)

        opp_x, opp_y = self._opp_position()
        if opp_x is not None:
            opp_idx = self._closest_idx(self._raceline, opp_x, opp_y)
            ygap = self._ygap(ego_idx, opp_idx)
        else:
            ygap = DF_MAX  # assume far if not visible

        # OTF: overtake is feasible if a trajectory exists and ygap is in follow range
        otf_traj, otf_side = self._best_overtake_trajectory()
        OTF = otf_traj is not None and DF_MIN <= ygap <= DF_MAX

        # PMF: position match, we've pulled alongside
        PMF = ygap <= DSEP_F

        # SFB: safe to fall back, enough separation behind
        SFB = ygap > DSEP_B

        if self._state == State.FOLLOW:
            if OTF:
                self._active_traj = otf_traj
                self._overtake_side = otf_side
                self._state = State.OVERTAKE
                self.get_logger().info(f'[CarOvertakeSlamNode] Overtaking {otf_side}')
            else:
                steering = self._pure_pursuit(self._raceline, ego_idx)
                speed = self._follow_speed(ygap, DF_MIN, DF_MAX, dt)
                self._publish(speed, steering)
                return

        if self._state == State.OVERTAKE:
            if not OTF:
                self._state = State.FALLBACK
                self.get_logger().info('[CarOvertakeSlamNode] Overtake infeasible, falling back')
            elif PMF:
                self._state = State.MERGE_FRONT
                self.get_logger().info('[CarOvertakeSlamNode] Position matched, merging front')
            else:
                traj_idx = self._closest_idx(self._active_traj, self._ego_x, self._ego_y)
                steering = self._pure_pursuit(self._active_traj, traj_idx)
                self._publish(self.get_parameter('overtake_speed').value, steering)
                return

        if self._state == State.MERGE_FRONT:
            if not OTF or not PMF:
                self._state = State.FALLBACK
            elif ygap > DSEP_F:
                self._state = State.FOLLOW
                self.get_logger().info('[CarOvertakeSlamNode] Overtake complete')
                done_msg = Bool()
                done_msg.data = True
                self.done_pub.publish(done_msg)
            else:
                steering = self._pure_pursuit(self._raceline, ego_idx)
                self._publish(self.get_parameter('merge_speed').value, steering)
                return

        if self._state == State.FALLBACK:
            if SFB:
                self._state = State.FOLLOW
                self.get_logger().info('[CarOvertakeSlamNode] Safe, returning to follow')
            else:
                steering = self._pure_pursuit(self._raceline, ego_idx)
                self._publish(self.get_parameter('fallback_speed').value, steering)
                return

        # default: follow raceline
        steering = self._pure_pursuit(self._raceline, ego_idx)
        speed = self._follow_speed(ygap, DF_MIN, DF_MAX, dt)
        self._publish(speed, steering)

    def _follow_speed(self, ygap: float, df_min: float, df_max: float, dt: float) -> float:
        """Smooth speed based on ygap: faster when far, slower when close.

        ARGS:
            ygap (float) — path distance to opponent in metres
            df_min (float) — DF_min threshold
            df_max (float) — DF_max threshold
            dt (float) — seconds since last call
        RETURNS: float — smoothed speed in m/s
        """
        V_MAX = self.get_parameter('follow_speed').value
        V_MIN = V_MAX * 0.5
        f = max(0.0, min(1.0, (ygap - df_min) / (df_max - df_min)))
        v_target = V_MIN + (V_MAX - V_MIN) * f
        alpha = 1.0 - math.exp(-dt / 0.4)
        self._v_cmd = self._v_cmd + alpha * (v_target - self._v_cmd)
        return self._v_cmd

    def _publish(self, speed: float, steering: float):
        """Publishes an AckermannDriveStamped command.

        ARGS:
            speed (float) — target speed in m/s
            steering (float) — steering angle in radians
        PUBLISHES:
            /state/car_overtake_slam (AckermannDriveStamped)
        RETURNS: None
        """
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = speed
        msg.drive.steering_angle = steering
        self.drive_pub.publish(msg)


def main(args=None):
    """Entry point.

    ARGS: args (list) — optional CLI args passed to rclpy
    RETURNS: None
    """
    rclpy.init(args=args)
    node = CarOvertakeSlamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
