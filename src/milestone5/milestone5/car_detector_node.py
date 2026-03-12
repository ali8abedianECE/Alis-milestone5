import math
import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Bool, Float32
from ultralytics import YOLO

MODEL_PATH = '/Users/mohammadaliabedian/Downloads/Alis-milestone5/best.pt'
CAR_CLASS_ID = 0  # class 0 in custom best.pt
CONF = 0.5
IMG_SIZE = 320

class CarDetectorNode(Node):
    """Detects cars via YOLO and estimates bearing and range using camera intrinsics and LiDAR."""

    def __init__(self):
        """Loads YOLO model, sets up subscribers and publishers.

        ARGS: None
        RETURNS: None
        """
        super().__init__('car_detector_node')

        self._bridge = CvBridge()
        self._model = YOLO(MODEL_PATH)

        self._camera_info: CameraInfo = None
        self._scan: LaserScan = None

        self.create_subscription(Image, '/fused/image', self._image_cb, 10)
        self.create_subscription(CameraInfo, '/fused/camera_info', self._camera_info_cb, 10)
        self.create_subscription(LaserScan, '/fused/scan', self._scan_cb, 10)

        self._detected_pub = self.create_publisher(Bool, '/car_detector/car_detected', 10)
        self._theta_pub = self.create_publisher(Float32, '/car_detector/theta', 10)
        self._distance_pub = self.create_publisher(Float32, '/car_detector/distance', 10)

        self._debug_dir = os.path.expanduser('~/car_detector_debug')
        os.makedirs(self._debug_dir, exist_ok=True)
        self._debug_frame = 0

        self.get_logger().info('CarDetectorNode ready')

    def _image_cb(self, msg: Image):
        """Triggers detection on each incoming fused image.

        ARGS: msg (Image) — fused color image from /fused/image
        RETURNS: None
        """
        self._run_detection(msg)

    def _camera_info_cb(self, msg: CameraInfo):
        """Stores latest camera intrinsics.

        ARGS: msg (CameraInfo) — intrinsics from /fused/camera_info
        RETURNS: None
        """
        self._camera_info = msg

    def _scan_cb(self, msg: LaserScan):
        """Stores latest LiDAR scan.

        ARGS: msg (LaserScan) — scan from /fused/scan
        RETURNS: None
        """
        self._scan = msg

    def _run_detection(self, img_msg: Image):
        """Runs YOLO and publishes theta, distance, and detection flag for the best detection.

        ARGS: img_msg (Image) — color image to run inference on
        PUBLISHES:
            /car_detector/car_detected (Bool) — True if at least one car found
            /car_detector/theta (Float32) — horizontal bearing to best detection (radians)
            /car_detector/distance (Float32) — LiDAR range to best detection (metres)
        RETURNS: None
        """
        if self._camera_info is None or self._scan is None:
            self.get_logger().warn('Waiting for camera info and scan...', throttle_duration_sec=2.0)
            return

        cv_img = self._bridge.imgmsg_to_cv2(img_msg, 'bgr8')

        # K is row-major: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        fx = self._camera_info.k[0]
        cx = self._camera_info.k[2]

        results = self._model.predict(source=cv_img, conf=CONF, imgsz=IMG_SIZE, verbose=False)

        car_detected = False
        best_conf = 0.0
        best_theta = 0.0
        best_dist = 0.0

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                x1, _, x2, _ = box.xyxy[0].tolist()

                theta = self._pixel_to_angle((x1 + x2) / 2.0, cx, fx)
                distance = self._lidar_distance_for_box(x1, x2, cx, fx)

                if distance is None:
                    continue

                car_detected = True
                if conf > best_conf:
                    best_conf = conf
                    best_theta = theta
                    best_dist = distance

        detected_msg = Bool()
        detected_msg.data = car_detected
        self._detected_pub.publish(detected_msg)

        theta_msg = Float32()
        theta_msg.data = float(best_theta)
        self._theta_pub.publish(theta_msg)

        dist_msg = Float32()
        dist_msg.data = float(best_dist)
        self._distance_pub.publish(dist_msg)

        self._save_debug(cv_img, results, best_theta, best_dist, cx)

    def _save_debug(self, img, results, theta: float, dist: float, cx: float):
        """Draws detections and bearing arrow on the image and saves it to ~/car_detector_debug.
        ARGS:
            img (ndarray) — BGR image
            results — YOLO result list
            theta (float) — bearing to best detection in radians
            dist (float) — LiDAR distance to best detection in metres
            cx (float) — principal point x for drawing the centre line
        RETURNS: None
        """
        dbg = img.copy()
        h, w = dbg.shape[:2]

        # draw all bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(dbg, f'{conf:.2f}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # draw bearing arrow from bottom-centre toward the detected car
        cx_img = int(cx)
        base = (cx_img, h - 10)
        tip_x = int(cx_img - math.tan(theta) * 80)  # negate back: theta is LiDAR convention
        tip = (tip_x, h - 90)
        cv2.arrowedLine(dbg, base, tip, (0, 0, 255), 2, tipLength=0.3)
        cv2.putText(dbg, f'th={math.degrees(theta):.1f}d  d={dist:.2f}m',
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        path = os.path.join(self._debug_dir, f'frame_{self._debug_frame:05d}.jpg')
        cv2.imwrite(path, dbg)
        self._debug_frame += 1

    def _pixel_to_angle(self, cx_pixel: float, cx_intr: float, fx: float) -> float:
        """Converts a pixel x coordinate to a horizontal bearing using intrinsics.

        ARGS:
            cx_pixel (float) — pixel x of the detection centre
            cx_intr (float) — principal point x from CameraInfo.K
            fx (float) — focal length x from CameraInfo.K
        RETURNS: 
            float — bearing in radians, positive = right of centre
        """
        return math.atan2(cx_intr - cx_pixel, fx)

    def _lidar_distance_for_box(self, x1: float, x2: float, cx_intr: float, fx: float):
        """
        Returns IQR-filtered median LiDAR range over the angular span of the bounding box.

        ARGS:
            x1 (float) — left pixel x of bounding box
            x2 (float) — right pixel x of bounding box
            cx_intr (float) — principal point x from CameraInfo.K
            fx (float) — focal length x from CameraInfo.K
        RETURNS: 
            float | None — median range in metres, or None if no valid beams
        """
        scan = self._scan

        # negate: camera right (x > cx) = positive pixel offset, but LiDAR CCW right = negative angle
        theta_left = math.atan2(cx_intr - x1, fx)
        theta_right = math.atan2(cx_intr - x2, fx)

        # assumes camera and LiDAR share the same yaw axis
        n = len(scan.ranges)
        idx_left = int((theta_left - scan.angle_min) / scan.angle_increment)
        idx_right = int((theta_right - scan.angle_min) / scan.angle_increment)

        idx_left = max(0, min(idx_left, n - 1))
        idx_right = max(0, min(idx_right, n - 1))

        if idx_left > idx_right:
            idx_left, idx_right = idx_right, idx_left

        window = np.array(scan.ranges[idx_left:idx_right + 1], dtype=np.float32)

        valid = window[np.isfinite(window) & (window >= scan.range_min) & (window <= scan.range_max)]

        if len(valid) == 0:
            return None

        q1, q3 = np.percentile(valid, [25, 75])
        iqr = q3 - q1
        filtered = valid[(valid >= q1 - 1.5 * iqr) & (valid <= q3 + 1.5 * iqr)]

        if len(filtered) == 0:
            return None

        return float(np.median(filtered))


def main(args=None):
    """Entry point.

    ARGS: args (list) — optional CLI args passed to rclpy
    RETURNS: None
    """
    rclpy.init(args=args)
    node = CarDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
