import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

import cv2


# Speed multipliers for each ROI zone (index 0 = top/far, index 3 = bottom/closest)
_ZONE_MULTIPLIERS = [1.0, 0.75, 0.5, 0.0]


class CameraAebNode(Node):
    """
    Staged camera AEB. Watches 4 stacked rectangular ROIs centered on the
    image. If an obstacle (dark pixels) is detected in a zone the outgoing
    speed is scaled down. The closest zone triggers a full stop. Steering is
    never modified.

    Zones (top to bottom, front of car is at the bottom):
        Zone 0 (top, far)   -- monitor only, no speed change
        Zone 1              -- speed x 0.75
        Zone 2              -- speed x 0.50
        Zone 3 (bottom, near) -- full stop (speed = 0)

    The node subscribes to /state_machine_decision (arbiter output), applies
    the safety multiplier, and republishes on /drive.
    """

    def __init__(self):
        """Sets up subscriptions, publisher, and AEB parameters.

        ARGS: None
        RETURNS: None
        """
        super().__init__('camera_aeb_node')

        self.declare_parameter('brightness_threshold', 80)   # same as vision.py
        self.declare_parameter('min_white_ratio', 0.50)      # below this = obstacle in zone
        self.declare_parameter('roi_width_fraction', 0.30)   # fraction of image width for ROI
        self.declare_parameter('roi_height_fraction', 0.50)  # fraction of image height for total ROI block
        self.declare_parameter('debug_enable', False)

        self.create_subscription(Image, '/camera/color/image_raw', self._image_cb, 10)
        self.create_subscription(
            AckermannDriveStamped, '/state_machine_decision', self._drive_cb, 10
        )

        self.drive_pub = self.create_publisher(AckermannDriveStamped, 'safety', 10)

        self._bridge = CvBridge()
        self._pending_cmd: AckermannDriveStamped = None  # latest command from arbiter
        self._multiplier = 1.0  # most recent safety multiplier
        self._active_zone = -1  # highest triggered zone index (-1 = none)

        self.get_logger().info('[CameraAebNode] Ready')

    # ---------- subscribers ----------

    def _drive_cb(self, msg: AckermannDriveStamped):
        """Caches the latest drive command from the arbiter.

        ARGS: msg (AckermannDriveStamped) — from /state_machine_decision
        RETURNS: None
        """
        self._pending_cmd = msg

    def _image_cb(self, msg: Image):
        """Processes camera frame, computes safety multiplier, and publishes safe command.

        ARGS: msg (Image) — from /camera/color/image_raw
        PUBLISHES:
            /drive (AckermannDriveStamped) — speed-clamped command with original steering
        RETURNS: None
        """
        if self._pending_cmd is None:
            return

        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self._multiplier, self._active_zone = self._check_zones(bgr)

        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.drive.steering_angle = self._pending_cmd.drive.steering_angle
        out.drive.speed = self._pending_cmd.drive.speed * self._multiplier
        self.drive_pub.publish(out)

        if self.get_parameter('debug_enable').value and self._active_zone >= 0:
            self.get_logger().info(
                f'[CameraAebNode] Zone {self._active_zone} triggered, '
                f'multiplier={self._multiplier:.2f}'
            )

    # ---------- zone check ----------

    def _check_zones(self, bgr: np.ndarray):
        """Computes which ROI zone is the deepest triggered and its speed multiplier.

        The ROI block is centered horizontally. It is divided into 4 equal
        horizontal bands from top (far) to bottom (near car). The lowest
        band that falls below min_white_ratio determines the multiplier.

        ARGS: bgr (np.ndarray) — full camera frame in BGR
        RETURNS: (float, int) — (speed_multiplier, active_zone_index) where
                  active_zone_index is -1 if no zone triggered
        """
        h, w = bgr.shape[:2]

        roi_w = int(w * self.get_parameter('roi_width_fraction').value)
        roi_h = int(h * self.get_parameter('roi_height_fraction').value)
        thresh = self.get_parameter('brightness_threshold').value
        min_ratio = self.get_parameter('min_white_ratio').value

        # ROI block: centered horizontally, anchored to the bottom of the image
        x0 = (w - roi_w) // 2
        x1 = x0 + roi_w
        y_block_top = h - roi_h
        band_h = roi_h // 4

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

        # zones: 0 = top/far, 3 = bottom/near
        active_zone = -1
        multiplier = 1.0

        for zone in range(4):
            y0 = y_block_top + zone * band_h
            y1 = y0 + band_h
            band = binary[y0:y1, x0:x1]
            if band.size == 0:
                continue
            white_ratio = float(np.count_nonzero(band)) / band.size
            if white_ratio < min_ratio:
                active_zone = zone
                multiplier = _ZONE_MULTIPLIERS[zone]
                # keep going to find the nearest (highest index) triggered zone

        return multiplier, active_zone


def main(args=None):
    """Entry point.

    ARGS: args (list) — optional CLI args passed to rclpy
    RETURNS: None
    """
    rclpy.init(args=args)
    node = CameraAebNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
