#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
import numpy as np
from scipy.signal import convolve2d

from asl_tb3_msgs.msg import TurtleBotState
from asl_tb3_lib.grids import StochOccupancyGrid2D

class FrontierExplorationController(Node):
    def __init__(self):
        super().__init__('frontier_exploration_node')
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.nav_success_sub = self.create_subscription(Bool, '/nav_success', self.nav_success_callback, 10)
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
        
        self.active = True
        self.stopsign = self.create_subscription(Bool, "/detector_bool", self.stopsign_callback, 10)

        self.nav_success = True
        self.occupancy = None
        self.state = None
        self.detector_start_time = None
    
    def stopsign_callback(self, msg: Bool) -> None:
        cur_time = self.get_clock().now().nanoseconds / 1e9
        if msg.data:
            if self.active:
                self.get_logger().info("Stop sign detected. Stopping.")
                self.active = False
                self.detector_start_time = cur_time
                if self.state:
                    stop_cmd = TurtleBotState()
                    stop_cmd.x = self.state.x
                    stop_cmd.y = self.state.y
                    self.cmd_nav_pub.publish(stop_cmd)
        else:
            if self.detector_start_time is not None:
                if (cur_time - self.detector_start_time) >= 5.0:
                    self.get_logger().info("Stop sign duration elapsed. Continuing.")
                    self.active = True
                    self.detector_start_time = None
                    if self.occupancy and self.state:
                        self.explore(self.occupancy)

    def map_callback(self, msg):
        self.occupancy = StochOccupancyGrid2D(
            resolution = msg.info.resolution,
            size_xy = np.array([msg.info.width, msg.info.height]),
            origin_xy = np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size = 13,
            probs = msg.data,
        )

        if(self.nav_success and self.occupancy and self.state):
            self.explore(self.occupancy)
            self.nav_success = False

    def nav_success_callback(self, msg):
        if msg.data and self.occupancy:
            self.explore(self.occupancy)

    def state_callback(self, msg):
        self.state = msg
        if(self.nav_success and self.occupancy and self.state):
            self.explore(self.occupancy)
            self.nav_success = False

    def explore(self, occupancy):
        if not self.active:
            self.get_logger().info("Paused (stop sign)")
            return
            
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

        if len(frontier_states) == 0:
            self.get_logger().info("Finished exploring!")
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