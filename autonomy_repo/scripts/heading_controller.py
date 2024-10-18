import rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):
    def __init__(self):
        super().__init__()
        self.kp = 2.0

    def compute_control_with_goal(self, current_state: TurtleBotState, desired_state: TurtleBotState) -> TurtleBotControl:
        heading_error = wrap_angle(desired_state.theta - current_state.theta)

        omega = self.kp * heading_error

        control_msg = TurtleBotControl()
        control_msg.omega = omega

        return control_msg

def main(args=None):
    rclpy.init(args=args)
    heading_controller = HeadingController()
    rclpy.spin(heading_controller)
    rclpy.shutdown()

if __name__ == "__main__":
        main()