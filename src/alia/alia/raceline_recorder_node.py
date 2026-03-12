import math
import os

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class RacelineRecorderNode(Node):
    """Records ego odometry to a CSV while driving manually. Run once per track."""

    def __init__(self):
        """Opens the output CSV and subscribes to odometry.

        ARGS: None
        RETURNS: None
        """
        super().__init__('raceline_recorder_node')

        self.declare_parameter('output_path', os.path.expanduser('~/raceline.csv'))
        self.declare_parameter('min_spacing', 0.05)  # metres between saved waypoints

        path = self.get_parameter('output_path').value
        self._min_spacing = self.get_parameter('min_spacing').value
        self._last = None
        self._count = 0

        self._file = open(path, 'w')
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        self.get_logger().info(f'[RacelineRecorderNode] Recording to {path} — drive one lap then Ctrl+C')

    def _odom_cb(self, msg: Odometry):
        """Saves x,y to CSV if the car has moved more than min_spacing since the last saved point.

        ARGS: msg (Odometry) — ego odometry from /odom
        RETURNS: None
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self._last is None or math.dist([x, y], self._last) > self._min_spacing:
            self._file.write(f'{x},{y}\n')
            self._file.flush()
            self._last = [x, y]
            self._count += 1
            self.get_logger().info(f'Saved waypoint {self._count}: ({x:.3f}, {y:.3f})', throttle_duration_sec=1.0)

    def destroy_node(self):
        """Closes the CSV file on shutdown.

        ARGS: None
        RETURNS: None
        """
        self._file.close()
        self.get_logger().info(f'[RacelineRecorderNode] Saved {self._count} waypoints')
        super().destroy_node()


def main(args=None):
    """Entry point.

    ARGS: args (list) — optional CLI args passed to rclpy
    RETURNS: None
    """
    rclpy.init(args=args)
    node = RacelineRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
