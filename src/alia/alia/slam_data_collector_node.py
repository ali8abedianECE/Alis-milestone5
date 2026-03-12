import os
from datetime import datetime

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import serialize_message
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

import rosbag2_py


class SlamDataCollectorNode(Node):
    """Records /scan, /odom, /tf, and /tf_static to a ROS2 bag for SLAM map building."""

    def __init__(self):
        """Opens the bag writer and subscribes to all SLAM-required topics.

        ARGS: None
        RETURNS: None
        """
        super().__init__('slam_data_collector_node')

        self.declare_parameter('output_dir', os.path.expanduser('~/slam_bags'))

        output_dir = self.get_parameter('output_dir').value
        os.makedirs(output_dir, exist_ok=True)

        # bag name includes timestamp so multiple runs don't overwrite each other
        bag_name = datetime.now().strftime('slam_%Y%m%d_%H%M%S')
        self._bag_path = os.path.join(output_dir, bag_name)

        self._writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri=self._bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        self._writer.open(storage_options, converter_options)

        # register topics in the bag
        self._register_topic('/scan', 'sensor_msgs/msg/LaserScan')
        self._register_topic('/odom', 'nav_msgs/msg/Odometry')
        self._register_topic('/tf', 'tf2_msgs/msg/TFMessage')
        self._register_topic('/tf_static', 'tf2_msgs/msg/TFMessage')

        # /tf_static is latched — needs TransientLocal durability or we miss
        # transforms that were published before this node started
        static_qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # subscribers
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(TFMessage, '/tf', self._tf_cb, 100)
        self.create_subscription(TFMessage, '/tf_static', self._tf_static_cb, static_qos)

        self._scan_count = 0

        self.get_logger().info(f'[SlamDataCollectorNode] Recording to {self._bag_path} — drive one lap then Ctrl+C')

    def _register_topic(self, topic: str, msg_type: str):
        """Registers a topic with the bag writer.

        ARGS:
            topic (str) — ROS2 topic name
            msg_type (str) — fully qualified message type string
        RETURNS: None
        """
        # use attribute assignment — keyword constructor not stable across ROS2 versions
        meta = rosbag2_py.TopicMetadata()
        meta.name = topic
        meta.type = msg_type
        meta.serialization_format = 'cdr'
        self._writer.create_topic(meta)

    def _write(self, topic: str, msg):
        """Serializes and writes a message to the bag.

        ARGS:
            topic (str) — topic the message was received on
            msg — ROS2 message to serialize and write
        RETURNS: None
        """
        stamp = self.get_clock().now().nanoseconds
        self._writer.write(topic, serialize_message(msg), stamp)

    # ---------- subscribers ----------

    def _scan_cb(self, msg: LaserScan):
        """Writes incoming LiDAR scan to the bag.

        ARGS: msg (LaserScan) — scan from /scan
        RETURNS: None
        """
        self._write('/scan', msg)
        self._scan_count += 1
        if self._scan_count % 40 == 0:  # log roughly every second at 40Hz
            self.get_logger().info(f'Recorded {self._scan_count} scans', throttle_duration_sec=1.0)

    def _odom_cb(self, msg: Odometry):
        """Writes incoming odometry to the bag.

        ARGS: msg (Odometry) — odometry from /odom
        RETURNS: None
        """
        self._write('/odom', msg)

    def _tf_cb(self, msg: TFMessage):
        """Writes incoming TF transforms to the bag.

        ARGS: msg (TFMessage) — transforms from /tf
        RETURNS: None
        """
        self._write('/tf', msg)

    def _tf_static_cb(self, msg: TFMessage):
        """Writes incoming static TF transforms to the bag.

        ARGS: msg (TFMessage) — static transforms from /tf_static
        RETURNS: None
        """
        self._write('/tf_static', msg)

    def destroy_node(self):
        """Closes the bag on shutdown.

        ARGS: None
        RETURNS: None
        """
        del self._writer
        self.get_logger().info(f'[SlamDataCollectorNode] Bag saved to {self._bag_path}')
        self.get_logger().info(f'[SlamDataCollectorNode] Total scans recorded: {self._scan_count}')
        self.get_logger().info('To build the map run:')
        self.get_logger().info(f'  ros2 bag play {self._bag_path}  (in one terminal)')
        self.get_logger().info('  ros2 launch slam_toolbox online_async_launch.py  (in another)')
        super().destroy_node()


def main(args=None):
    """Entry point.

    ARGS: args (list) — optional CLI args passed to rclpy
    RETURNS: None
    """
    rclpy.init(args=args)
    node = SlamDataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
