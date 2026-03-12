import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, LaserScan

class DataFuserNode(Node):
    """
    Fuses camera and LiDAR data and re-publishes at 15 Hz.
    """
    def __init__(self):
        """Sets up subscribers, publishers, and 15 Hz timer.

        ARGS: None
        RETURNS: None
        """
        super().__init__('data_fuser_node')

        # Latest received messages
        self._image: Image = None
        self._camera_info: CameraInfo = None
        self._scan: LaserScan = None

        # Subscribers
        self.create_subscription(Image, '/camera/color/image_raw',
                                 self._image_cb, 10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info',
                                 self._camera_info_cb, 10)
        self.create_subscription(LaserScan, '/scan',
                                 self._scan_cb, 10)

        # Publishers (fused / re-stamped topics)
        self._image_pub = self.create_publisher(Image, '/fused/image', 10)
        self._info_pub = self.create_publisher(CameraInfo, '/fused/camera_info', 10)
        self._scan_pub = self.create_publisher(LaserScan, '/fused/scan', 10)

        # 15 Hz timer
        self.create_timer(1.0 / 15.0, self._timer_cb)

        self.get_logger().info('DataFuserNode started at 15 Hz')

    # ---------- callbacks ----------

    def _image_cb(self, msg: Image):
        """Stores latest color image.

        ARGS: msg (Image) — raw color image from /camera/color/image_raw
        RETURNS: None
        """
        self._image = msg

    def _camera_info_cb(self, msg: CameraInfo):
        """Stores latest camera info.

        ARGS: msg (CameraInfo) — intrinsics from /camera/color/camera_info
        RETURNS: None
        """
        self._camera_info = msg

    def _scan_cb(self, msg: LaserScan):
        """Stores latest LiDAR scan.

        ARGS: msg (LaserScan) — scan from /scan
        RETURNS: None
        """
        self._scan = msg

    # ---------- timer ----------

    def _timer_cb(self):
        """Publishes all three sensors at 15 Hz with a shared timestamp.

        ARGS: None
        PUBLISHES:
            /fused/image (Image) — re-stamped color image
            /fused/camera_info (CameraInfo) — re-stamped camera intrinsics
            /fused/scan (LaserScan) — re-stamped LiDAR scan
        RETURNS: None
        """
        if self._image is None or self._camera_info is None or self._scan is None:
            self.get_logger().warn('Waiting for all sensors...', throttle_duration_sec=2.0)
            return

        now = self.get_clock().now().to_msg()

        # Stamp everything with the same time so consumers see them as synced
        self._image.header.stamp = now
        self._camera_info.header.stamp = now
        self._scan.header.stamp = now

        self._image_pub.publish(self._image)
        self._info_pub.publish(self._camera_info)
        self._scan_pub.publish(self._scan)


def main(args=None):
    """Entry point.

    ARGS: args (list) — optional CLI args passed to rclpy
    RETURNS: None
    """
    rclpy.init(args=args)
    node = DataFuserNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
