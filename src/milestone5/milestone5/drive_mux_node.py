import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped


class DriveMuxNode(Node):
    """
    Drive Multiplexer Node

    This node manages all the nodes that want to publish to /drive.
    It assigns each node a priority and forwards the node with the highest priority
    that has a recent message to the /drive topic.
    Since safety is the highest priority, it will always override other nodes.
    """

    UPDATE_PERIOD = 0.02  # 20ms

    def __init__(self):
        """
        Creates a DriveMuxNode instance.

        Subscribes to each of the nodes in the self.nodes config and sets up a publisher
        for the /drive topic. Also creates a timer to periodically check and publish
        the highest priority message.
        """
        super().__init__("drive_mux_node")

        # Define controlled nodes with their topics and priorities
        self.nodes = {
            "safety": {
                "subscription": None,
                "msg": None,  # Latest message from this node
                "time": 0.0,  # Timestamp of the latest message
                "topic": "safety",
                "priority": 100,
            },
            
            "state_machine_decision" : {
                "subscription": None,
                "msg": None,
                "time": 0.0,
                "topic": "/state_machine_decision",
                "priority": 99,
            },
            
        }

        # Subscribe to each of the controlled nodes
        for name, node in self.nodes.items():
            self.nodes[name]["subscription"] = self.create_subscription(
                AckermannDriveStamped,
                node["topic"],
                lambda msg, n=name: self.topic_callback(msg, n),
                10,
            )

        # Publisher for drive commands
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        # Create a timer to control publishing rate
        self.timer = self.create_timer(
            self.UPDATE_PERIOD, self.publish_highest_priority
        )

        # Timeout in seconds before source is considered stale and ignored
        self.timeout = 0.35

        self.stop_msg = AckermannDriveStamped()
        self.stop_msg.drive.speed = 0.0
        self.stop_msg.drive.steering_angle = 0.0
        
    # Now in seconds
    def now(self):
        """
        Returns the robots current time in seconds.
        """
        return self.get_clock().now().nanoseconds / 1e9

    # Update messages when received
    def topic_callback(self, msg, source_name):
        """
        Each topic is subscribed to this common callback to update the latest message
        and timestamp for that source.

        :param msg: The message received from the topic
        :param source_name: The name of the source node
        """
        self.nodes[source_name]["msg"] = msg
        self.nodes[source_name]["time"] = self.now()

    def publish_highest_priority(self):
        """
        Determins the highest priority valid message and publishes it to /drive.
        """
        current_time = self.now()

        active_node = None

        # Sort nodes by priority (highest first)
        sorted_nodes = sorted(
            self.nodes.items(), key=lambda x: x[1]["priority"], reverse=True
        )

        for name, info in sorted_nodes:
            if info["msg"] is not None:
                if (current_time - info["time"]) <= self.timeout:
                    active_node = info["msg"]
                    break  # Found the highest priority active message

        if active_node:
            self.drive_pub.publish(active_node)
        else:
            self.drive_pub.publish(self.stop_msg)
            self.get_logger().error(
                "[ERROR] No valid drive command; publishing STOP.",
                throttle_duration_sec=1.0,
            )


def main(args=None):
    rclpy.init(args=args)
    safety = DriveMuxNode()
    rclpy.spin(safety)
    safety.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
