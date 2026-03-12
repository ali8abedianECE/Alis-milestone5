from enum import Enum

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


class State(Enum):
    NORMAL = 'normal'      # no car visible, use gap follow
    FOLLOW = 'follow'      # car visible, maintain gap behind it
    OVERTAKE = 'overtake'  # car close, attempt to pass


class ArbiterNode(Node):
    """Top-level arbiter that selects among normal, follow, and overtake drive commands."""

    def __init__(self):
        """Sets up subscriptions, publisher, and state machine parameters.

        ARGS: None
        RETURNS: None
        """
        super().__init__('arbiter_node')

        self.declare_parameter('overtake_trigger_dist', 0.8)  # switch to OVERTAKE below this (m)
        self.declare_parameter('lost_timeout', 1.0)           # seconds before dropping to NORMAL
        self.declare_parameter('overtake_max_duration', 8.0)  # hard timeout for OVERTAKE (s)
        self.declare_parameter('timer_hz', 20.0)

        self.create_subscription(AckermannDriveStamped, '/state/normal_node', self._normal_cb, 10)
        self.create_subscription(AckermannDriveStamped, '/state/car_follow', self._follow_cb, 10)
        self.create_subscription(AckermannDriveStamped, '/state/car_overtake', self._overtake_cb, 10)
        self.create_subscription(Bool, '/car_detector/car_detected', self._detected_cb, 10)
        self.create_subscription(Float32, '/car_detector/distance', self._distance_cb, 10)
        self.create_subscription(Bool, '/car_overtake/done', self._overtake_done_cb, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/state_machine_decision', 10)

        self._normal_cmd: AckermannDriveStamped = None
        self._follow_cmd: AckermannDriveStamped = None
        self._overtake_cmd: AckermannDriveStamped = None

        self._car_visible = False
        self._d_car = None
        self._last_seen = None       # rclpy.Time when car was last detected
        self._overtake_start = None  # rclpy.Time when OVERTAKE state was entered
        self._overtake_done = False  # set True when overtake node signals completion

        self._state = State.NORMAL

        hz = self.get_parameter('timer_hz').value
        self.create_timer(1.0 / hz, self._timer_cb)

        self.get_logger().info('[ArbiterNode] Ready')

    # ---------- subscribers ----------

    def _normal_cb(self, msg: AckermannDriveStamped):
        """Caches latest normal node drive command.

        ARGS: msg (AckermannDriveStamped) — from /state/normal_node
        RETURNS: None
        """
        self._normal_cmd = msg

    def _follow_cb(self, msg: AckermannDriveStamped):
        """Caches latest car follow drive command.

        ARGS: msg (AckermannDriveStamped) — from /state/car_follow
        RETURNS: None
        """
        self._follow_cmd = msg

    def _overtake_cb(self, msg: AckermannDriveStamped):
        """Caches latest car overtake drive command.

        ARGS: msg (AckermannDriveStamped) — from /state/car_overtake
        RETURNS: None
        """
        self._overtake_cmd = msg

    def _detected_cb(self, msg: Bool):
        """Caches car visibility and timestamps last detection.

        ARGS: msg (Bool) — True if car detected
        RETURNS: None
        """
        self._car_visible = msg.data
        if msg.data:
            self._last_seen = self.get_clock().now()

    def _distance_cb(self, msg: Float32):
        """Caches distance to opponent.

        ARGS: msg (Float32) — distance in metres
        RETURNS: None
        """
        self._d_car = msg.data if msg.data > 0.0 else None

    def _overtake_done_cb(self, msg: Bool):
        """Receives completion signal from the overtake node.

        Published by car_overtake_node or car_overtake_slam_node when the
        merge phase completes and they return to their internal FOLLOW state.

        ARGS: msg (Bool) — True when overtake+merge is done
        RETURNS: None
        """
        if msg.data:
            self._overtake_done = True

    # ---------- state machine ----------

    def _car_lost(self) -> bool:
        """Returns True if car has not been detected recently.

        ARGS: None
        RETURNS: bool
        """
        if self._last_seen is None:
            return True
        elapsed = (self.get_clock().now() - self._last_seen).nanoseconds * 1e-9
        return elapsed > self.get_parameter('lost_timeout').value

    def _step(self):
        """Runs one tick of the arbiter state machine.

        Transitions:
            NORMAL   -> FOLLOW:   car detected
            FOLLOW   -> NORMAL:   car lost for lost_timeout seconds
            FOLLOW   -> OVERTAKE: car visible and d_car < overtake_trigger_dist
            OVERTAKE -> NORMAL:   overtake node signals done via /car_overtake/done
            OVERTAKE -> NORMAL:   overtake_max_duration exceeded (hard timeout)
            NOTE: car disappearing during OVERTAKE is expected (car is behind us
            during merge), so it is intentionally ignored.

        ARGS: None
        RETURNS: None
        """
        trigger = self.get_parameter('overtake_trigger_dist').value
        max_dur = self.get_parameter('overtake_max_duration').value

        if self._state == State.NORMAL:
            if self._car_visible:
                self._state = State.FOLLOW
                self.get_logger().info('[ArbiterNode] Car detected -> FOLLOW')

        elif self._state == State.FOLLOW:
            if self._car_lost():
                self._state = State.NORMAL
                self.get_logger().info('[ArbiterNode] Car lost -> NORMAL')
            elif self._car_visible and self._d_car is not None and self._d_car < trigger:
                self._state = State.OVERTAKE
                self._overtake_start = self.get_clock().now()
                self._overtake_done = False
                self.get_logger().info('[ArbiterNode] Car close -> OVERTAKE')

        elif self._state == State.OVERTAKE:
            elapsed = 0.0
            if self._overtake_start is not None:
                elapsed = (self.get_clock().now() - self._overtake_start).nanoseconds * 1e-9

            timed_out = elapsed > max_dur

            if self._overtake_done:
                self._state = State.NORMAL
                self._overtake_done = False
                self.get_logger().info('[ArbiterNode] Overtake done -> NORMAL')
            elif timed_out:
                self._state = State.NORMAL
                self.get_logger().info('[ArbiterNode] Overtake timeout -> NORMAL')

    def _select_cmd(self) -> AckermannDriveStamped:
        """Returns the cached command for the current state.

        ARGS: None
        RETURNS: AckermannDriveStamped | None
        """
        if self._state == State.NORMAL:
            return self._normal_cmd
        if self._state == State.FOLLOW:
            return self._follow_cmd
        if self._state == State.OVERTAKE:
            return self._overtake_cmd
        return None

    def _timer_cb(self):
        """Runs state machine and forwards selected command to /final_decision.

        ARGS: None
        PUBLISHES:
            /final_decision (AckermannDriveStamped) — selected drive command
        RETURNS: None
        """
        self._step()
        cmd = self._select_cmd()
        if cmd is None:
            return

        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.drive.speed = cmd.drive.speed
        out.drive.steering_angle = cmd.drive.steering_angle
        self.drive_pub.publish(out)


def main(args=None):
    """Entry point.

    ARGS: args (list) — optional CLI args passed to rclpy
    RETURNS: None
    """
    rclpy.init(args=args)
    node = ArbiterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
