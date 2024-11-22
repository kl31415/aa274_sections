#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
import numpy as np
from scipy.signal import convolve2d

from asl_tb3_msgs.msg import TurtleBotState
from asl_tb3_lib.grids import StochOccupancyGrid2D

from asl_tb3_msgs.msg import TurtleBotControl
from std_msgs.msg import Bool

class FrontierExplorationController(Node):
    def __init__(self):
        super().__init__('frontier_exploration_node')
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.nav_success_sub = self.create_subscription(Bool, '/nav_success', self.nav_success_callback, 10)
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
        
        self.declare_parameter("active", True)
        self.start_time = None
        self.detector_start_time = None
        self.stopsign = self.create_subscription(Bool, "/detector_bool", self.stopsign_callback, 10)

        self.nav_success = True
        self.occupancy = None
        self.state = None
        self.prev_frontier_states = None
        
    @property
    def active(self):
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

    def map_callback(self, msg):
        self.occupancy = StochOccupancyGrid2D(
            resolution = msg.info.resolution,
            size_xy = np.array([msg.info.width, msg.info.height]),
            origin_xy = np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size = 13,
            probs = msg.data,
        )

        if(self.nav_success and (self.occupancy != None) and (self.state != None)):
            self.explore(self.occupancy)
            self.nav_success = False

    def nav_success_callback(self, msg):
        self.explore(self.occupancy)

    def state_callback(self, msg):
        if(self.nav_success and (self.occupancy != None) and (self.state != None)):
            self.explore(self.occupancy)
            self.nav_success = False
        self.state = msg

    def explore(self, occupancy):
        if not self.active:
            self.get_logger().info("stop sign detected")
            cur_state = TurtleBotState()
            self.cmd_nav_pub.publish(TurtleBotState(x = cur_state.x, y = cur_state.y))
            if (self.start_time == None):
                self.start_time = self.get_clock().now().nanoseconds / 1e9
            else:
                if (self.get_clock().now().nanoseconds / 1e9 - self.start_time >= 5):
                    self.set_parameters([rclpy.Parameter("active", value=True)])
                    self.start_time = None
            return self.prev_frontier_states
            
        else:
            window_size = 13

            occ_mask = (occupancy.probs >= 0.5).astype(float)
            unknown_mask = (occupancy.probs == -1).astype(float)
            unocc_mask = ((occupancy.probs < 0.5) & (occupancy.probs != -1)).astype(float)

            k = np.ones((window_size, window_size))
            unknown_c = convolve2d(unknown_mask, k, mode = 'same', boundary = 'fill', fillvalue = 0)
            unocc_c = convolve2d(unocc_mask, k, mode = 'same', boundary = 'fill', fillvalue = 0)
            occ_c = convolve2d(occ_mask, k, mode = 'same', boundary = 'fill', fillvalue = 0)

            fr_mask = ((occ_c == 0) & (unocc_c >= (window_size ** 2) * 0.3) & (unknown_c >= (window_size ** 2) * 0.2))
            frontier_states = occupancy.grid2state(np.column_stack((np.where(fr_mask)[1], np.where(fr_mask)[0])))
            self.prev_frontier_states = frontier_states

            if len(frontier_states) == 0:
                self.get_logger().info("Finished exploring")
                return []

            dists = np.linalg.norm(frontier_states - np.array([self.state.x, self.state.y]), axis = 1)
            nearest = frontier_states[np.argmin(dists)]
            
            self.cmd_nav_pub.publish(TurtleBotState(x = nearest[0], y = nearest[1]))
        
        return frontier_states

def main(args = None):
    rclpy.init(args = args)
    node = FrontierExplorationController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
