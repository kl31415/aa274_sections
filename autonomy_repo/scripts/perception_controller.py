#!/usr/bin/env python3

import rclpy
from asl_tb3_lib.control import BaseController
from asl_tb3_msgs.msg import TurtleBotControl
from std_msgs.msg import Bool

class PerceptionController(BaseController):
    def __init__(self):
        super().__init__("perception_controller")
        self.declare_parameter("active", True)
        self.start_time = None
        self.detector_start_time = None
        self.stopsign = self.create_subscription(Bool, "/detector_bool", self.stopsign_callback, 10)

    @property
    def active(self) -> bool:
        return self.get_parameter("active").value
    
    def stopsign_callback(self, msg: Bool) -> None:
        if msg.data and self.active:
            if self.detector_start_time is None:
                self.detector_start_time = self.get_clock().now().nanoseconds / 1e9
                self.set_parameters([rclpy.Parameter("active", value=False)])
            else:
                if (self.get_clock().now().nanoseconds / 1e9 - self.detector_start_time >= 8):
                    # self.set_parameters([rclpy.Parameter("active", value=True)])
                    self.detector_start_time = None
        elif not msg.data:
            self.detector_start_time = None

    
    def compute_control(self) -> TurtleBotControl:
        control_msg = TurtleBotControl()
        if self.active:
            control_msg.omega = 0.5
        else:
            control_msg.omega = 0.0
            if (self.start_time == None):
                self.start_time = self.get_clock().now().nanoseconds / 1e9
            else:
                if (self.get_clock().now().nanoseconds / 1e9 - self.start_time >= 5):
                    self.set_parameters([rclpy.Parameter("active", value=True)])
                    self.start_time = None
        return control_msg
    

    
def main(args=None):
    rclpy.init(args=args)
    perception_controller = PerceptionController()
    rclpy.spin(perception_controller)
    rclpy.shutdown()

if __name__ == "__main__":
    main()