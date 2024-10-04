#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class ConstantControl(Node):
    def __init__(self) -> None:
        super().__init__("constant_control")
        self.get_logger().info("Publisher has been created!")
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.subscriber = self.create_subscription(Bool, '/kill', self.kill_callback, 10)
        self.timer = self.create_timer(0.2, self.control_callback)

    def control_callback(self):
        msg = Twist()
        msg.linear.x = float(10)
        msg.angular.z = float(5)
        self.publisher.publish(msg)

    def kill_callback(self, msg):
        if msg.data:
            self.timer.cancel()
            stop = Twist()
            stop.linear.x = 0.0
            stop.angular.z = 0.0
            self.publisher.publish(stop)

def main(args=None):
    rclpy.init(args=args)
    publisher = ConstantControl()
    rclpy.spin(publisher)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
    