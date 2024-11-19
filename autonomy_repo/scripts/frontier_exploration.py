#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import numpy as np
from scipy.signal import convolve2d

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from scipy.signal import convolve2d
from std_msgs.msg import Bool

from asl_tb3_msgs.msg import TurtleBotState
from asl_tb3_lib.grids import StochOccupancyGrid2D

class FrontierExplorationController(Node):
    def __init__(self):
        super().__init__('frontier_exploration_node')
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.nav_success_sub = self.create_subscription(Bool, '/nav_success', self.nav_success_callback, 10)
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)

        self.nav_success = True
        self.occupancy = None
        self.state = None

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
