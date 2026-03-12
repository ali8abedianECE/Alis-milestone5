from enum import Enum

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32


class State(Enum):
    FOLLOW = 'follow' # behind car, waiting for a clear side
    OVERTAKE = 'overtake' # committed to a side, boosting past
    MERGE = 'merge' # we passed, steering back to centre


class CarOvertakeNode(Node):
    """Reactive overtake — no map needed. Uses LiDAR sectors to check left/right clearance."""

    def __init__(self):
        """Sets up subscriptions, publisher, and overtake parameters.

        ARGS: None
        RETURNS: None
        """
        super().__init__('car_overtake_node')

        self.create_subscription(Bool, '/car_detector/car_detected', self._detected_cb, 10)
        self.create_subscription(Float32, '/car_detector/theta', self._theta_cb, 10)
        self.create_subscription(Float32, '/car_detector/distance', self._distance_cb, 10)
        self.create_subscription(LaserScan, '/scan', self._lidar_cb, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/state/car_overtake', 10)
        self.done_pub = self.create_publisher(Bool, '/car_overtake/done', 10)

        # follow parameters
        self.declare_parameter('follow_speed', 1.2)  # speed while following
        self.declare_parameter('k_steer', 1.0)  # proportional steering gain toward car
        self.declare_parameter('overtake_trigger_dist', 0.8)  # start overtake when car is this close

        # overtake parameters
        self.declare_parameter('overtake_speed', 2.0)  # boost speed during overtake
        self.declare_parameter('overtake_steer', 0.3)  # fixed steering angle to the side (rad)
        self.declare_parameter('overtake_timeout', 3.0)  # max seconds to spend in overtake
        self.declare_parameter('clearance_min', 1.2)  # min LiDAR range needed to attempt a side
        self.declare_parameter('clearance_angle', 1.0)  # angular width (rad) of each side check

        # merge parameters
        self.declare_parameter('merge_speed', 1.4)  # speed while merging back
        self.declare_parameter('merge_timeout', 2.0)  # seconds to spend merging back to centre

        self._car_visible = False
        self._theta = 0.0
        self._d_car = None
        self._scan: LaserScan = None

        self._state = State.FOLLOW
        self._overtake_side = None # 'left' or 'right'
        self._state_timer = 0.0 # how long we have been in current state
        self._last_stamp = self.get_clock().now()

        self.get_logger().info('[CarOvertakeNode] Ready')

    # ---------- subscribers ----------

    def _detected_cb(self, msg: Bool):
        """Caches car visibility.

        ARGS: msg (Bool) — True if car detected
        RETURNS: None
        """
        self._car_visible = msg.data

    def _theta_cb(self, msg: Float32):
        """Caches bearing to car.

        ARGS: msg (Float32) — bearing in radians, LiDAR convention (left=+, right=-)
        RETURNS: None
        """
        self._theta = msg.data

    def _distance_cb(self, msg: Float32):
        """Caches distance to car.

        ARGS: msg (Float32) — distance in metres from /car_detector/distance
        RETURNS: None
        """
        self._d_car = msg.data if msg.data > 0.0 else None

    def _lidar_cb(self, scan: LaserScan):
        """Caches scan and triggers the state machine.

        ARGS: scan (LaserScan) — raw scan from /scan
        PUBLISHES:
            /state/car_overtake (AckermannDriveStamped) — speed and steering
        RETURNS: None
        """
        self._scan = scan

        now = self.get_clock().now()
        dt = max(1e-4, min((now - self._last_stamp).nanoseconds * 1e-9, 0.5))
        self._last_stamp = now
        self._state_timer += dt

        self._step(dt)

    # ---------- clearance check ----------

    def _side_clear(self, side: str) -> bool:
        """Checks if a LiDAR sector on the given side is free of obstacles.

        Looks at beams from the car bearing outward by clearance_angle in the
        given direction and checks the minimum range against clearance_min.

        ARGS: side (str) — 'left' or 'right'
        RETURNS: bool — True if the sector is clear
        """
        if self._scan is None:
            return False

        clearance_min = self.get_parameter('clearance_min').value
        clearance_angle = self.get_parameter('clearance_angle').value

        scan = self._scan
        n = len(scan.ranges)
        ranges = np.array(scan.ranges)

        # sector starts from the car bearing and sweeps outward to the side
        if side == 'left':
            a_start = self._theta
            a_end = self._theta + clearance_angle
        else:
            a_start = self._theta - clearance_angle
            a_end = self._theta

        idx_start = int((a_start - scan.angle_min) / scan.angle_increment)
        idx_end = int((a_end - scan.angle_min) / scan.angle_increment)
        idx_start = max(0, min(idx_start, n - 1))
        idx_end = max(0, min(idx_end, n - 1))

        if idx_start > idx_end:
            idx_start, idx_end = idx_end, idx_start

        window = ranges[idx_start:idx_end + 1]
        valid = window[np.isfinite(window) & (window > 0.0)]

        if len(valid) == 0:
            return False

        return float(np.min(valid)) > clearance_min

    # ---------- state machine ----------

    def _step(self, _dt: float):
        """Runs one tick of the overtake state machine and publishes a drive command.

        States:
            FOLLOW — steer toward car at follow speed, transition when close + side clear
            OVERTAKE — boost speed, steer to chosen side, transition when past or timed out
            MERGE — steer back to centre at merge speed, transition when timer expires

        ARGS: dt (float) — seconds since last call
        PUBLISHES:
            /state/car_overtake (AckermannDriveStamped)
        RETURNS: None
        """
        if self._state == State.FOLLOW:
            steering = self.get_parameter('k_steer').value * self._theta
            speed = self.get_parameter('follow_speed').value

            # attempt overtake if car is close and a side is clear
            trigger = self.get_parameter('overtake_trigger_dist').value
            if self._car_visible and self._d_car is not None and self._d_car < trigger:
                left_clear = self._side_clear('left')
                right_clear = self._side_clear('right')

                if left_clear or right_clear:
                    # prefer the clearer side; if both clear prefer left
                    self._overtake_side = 'left' if left_clear else 'right'
                    self._transition(State.OVERTAKE)
                    self.get_logger().info(f'[CarOvertakeNode] Overtaking {self._overtake_side}')

        elif self._state == State.OVERTAKE:
            overtake_steer = self.get_parameter('overtake_steer').value
            steering = overtake_steer if self._overtake_side == 'left' else -overtake_steer
            speed = self.get_parameter('overtake_speed').value

            # we've passed when the car is no longer visible (out of camera FOV = behind us)
            # or the overtake has timed out
            timeout = self.get_parameter('overtake_timeout').value
            if not self._car_visible or self._state_timer > timeout:
                self._transition(State.MERGE)
                self.get_logger().info('[CarOvertakeNode] Overtake done, merging')

        elif self._state == State.MERGE:
            # steer back toward centre
            merge_steer = self.get_parameter('overtake_steer').value
            # if we went left, steer back right, and vice versa
            steering = -merge_steer if self._overtake_side == 'left' else merge_steer
            speed = self.get_parameter('merge_speed').value

            merge_timeout = self.get_parameter('merge_timeout').value
            if self._state_timer > merge_timeout:
                self._transition(State.FOLLOW)
                self.get_logger().info('[CarOvertakeNode] Merged, back to follow')
                done_msg = Bool()
                done_msg.data = True
                self.done_pub.publish(done_msg)

        self._publish(speed, steering)

    def _transition(self, new_state: State):
        """Transitions to a new state and resets the state timer.

        ARGS: new_state (State) — state to transition to
        RETURNS: None
        """
        self._state = new_state
        self._state_timer = 0.0

    def _publish(self, speed: float, steering: float):
        """Publishes an AckermannDriveStamped command.

        ARGS:
            speed (float) — target speed in m/s
            steering (float) — steering angle in radians
        PUBLISHES:
            /state/car_overtake (AckermannDriveStamped)
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
    node = CarOvertakeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
