#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from asl_tb3_lib.navigation import BaseNavigator
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_lib.navigation import TrajectoryPlan
from scipy.interpolate import splrep, splev
import numpy as np

class AStar:
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution = 1):
        self.statespace_lo = statespace_lo
        self.statespace_hi = statespace_hi
        self.occupancy = occupancy
        self.resolution = resolution
        self.x_offset = x_init
        self.x_init = self.snap_to_grid(x_init)
        self.x_goal = self.snap_to_grid(x_goal)

        self.closed_set = set()
        self.open_set = set()

        self.est_cost_through = {}
        self.cost_to_arrive = {}
        self.came_from = {}

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init, self.x_goal)

        self.path = None

    def is_free(self, x):
        return self.occupancy.is_free(x)

    def distance(self, x1, x2):
        return np.linalg.norm(np.array(x2) - np.array(x1))

    def snap_to_grid(self, x):
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        neighbors = []
        res_list = [
            (-self.resolution, self.resolution), (0, self.resolution), (self.resolution, self.resolution), 
            (-self.resolution, 0), (self.resolution, 0), 
            (-self.resolution, -self.resolution), (0, -self.resolution), (self.resolution, -self.resolution)
        ]

        for res in res_list:
            n = (x[0] + res[0], x[1] + res[1])
            n = self.snap_to_grid(n)
            if self.is_free(n):
                neighbors.append(n)
        return neighbors

    def find_best_est_cost_through(self):
        return min(self.open_set, key = lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def solve(self):
        while self.open_set:
            cur = self.find_best_est_cost_through()
            if cur == self.x_goal:
                self.path = self.reconstruct_path()
                return True

            self.open_set.remove(cur)
            self.closed_set.add(cur)

            for n in self.get_neighbors(cur):
                if n in self.closed_set:
                    continue

                tcta = self.cost_to_arrive[cur] + self.distance(cur, n)
                if n not in self.open_set:
                    self.open_set.add(n)
                elif tcta >= self.cost_to_arrive[n]:
                    continue

                self.came_from[n] = cur
                self.cost_to_arrive[n] = tcta
                self.est_cost_through[n] = tcta + self.distance(n, self.x_goal)
        
        return False

class Navigator(BaseNavigator):
    def __init__(self, kp: float, kpx: float, kdx: float, kpy: float, kdy: float, V_max: float, om_max: float):
        super().__init__("navigator")
        self.kp = kp

        self.kpx = kpx
        self.kdx = kdx

        self.kpy = kpy
        self.kdy = kdy

        self.V_max = V_max
        self.om_max = om_max

        self.V_prev = 0.0
        self.t_prev = 0.0
        self.V_PREV_THRES = 0.1

        self.x_init = None
        self.x_goal = None
        self.occupancy = None
        self.astar = None

    def compute_trajectory_plan(self, x_init, x_goal, occupancy):
        self.x_init = x_init
        self.x_goal = x_goal
        self.occupancy = occupancy
        
        self.astar = AStar(
            statespace_lo = (0, 0), 
            statespace_hi = (occupancy.width, occupancy.height), 
            x_init = self.x_init, 
            x_goal = self.x_goal, 
            occupancy = self.occupancy, 
            resolution = 0.1
        )

        if not self.astar.solve():
            print("No valid path")
            return None
        else:
            print("Valid A* path")
            astar_path = np.asarray(self.astar.path)
            return self.compute_smooth_plan(astar_path)

    def compute_smooth_plan(self, path, v_desired = 0.15, spline_alpha = 0.05):
        path = np.asarray(path)

        dist = np.sqrt(np.diff(path[:, 0]) ** 2 + np.diff(path[:, 1]) ** 2)
        ts = np.zeros(len(path))
        ts[1:] = np.cumsum(dist / v_desired)

        path_x_spline = splrep(ts, path[:, 0], s = spline_alpha)
        path_y_spline = splrep(ts, path[:, 1], s = spline_alpha)

        return TrajectoryPlan(
            path = path,
            path_x_spline = path_x_spline,
            path_y_spline = path_y_spline,
            duration = ts[-1],
        )

    def compute_heading_control(self, current_state: TurtleBotState, desired_state: TurtleBotState) -> TurtleBotControl:
        current_yaw = quaternion_to_yaw(current_state.orientation)
        desired_yaw = quaternion_to_yaw(desired_state.orientation)

        heading_error = wrap_angle(desired_yaw - current_yaw)
        omega = self.kp * heading_error

        control_msg = TurtleBotControl()
        control_msg.omega = omega

        return control_msg
    
    def get_desired_state(self, t: float, plan):
        x_d = splev(t, plan.spline_x)
        xd_d = splev(t, plan.spline_x, der = 1)
        xdd_d = splev(t, plan.spline_x, der = 2)

        y_d = splev(t, plan.spline_y)
        yd_d = splev(t, plan.spline_y, der = 1)
        ydd_d = splev(t, plan.spline_y, der = 2)

        return x_d, xd_d, xdd_d, y_d, yd_d, ydd_d

    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan, t: float) -> TurtleBotControl:
        x, y, th = state.x, state.y, state.theta

        dt = t - self.t_prev

        x_d, xd_d, xdd_d, y_d, yd_d, ydd_d = self.get_desired_state(t, plan)

        x_dot = self.V_prev * np.cos(th)
        y_dot = self.V_prev * np.sin(th)
        
        u1 = xdd_d + self.kpx * (x_d - x) + self.kdx * (xd_d - x_dot)
        u2 = ydd_d + self.kpy * (y_d - y) + self.kdy * (yd_d - y_dot)

        v_dot = u1 * np.cos(th) + u2 * np.sin(th)
        V = max(self.V_prev, self.V_PREV_THRES) + v_dot * dt
        V = max(V, self.V_PREV_THRES)

        om = (u2 * np.cos(th) - u1 * np.sin(th)) / V

        self.t_prev = t
        self.V_prev = V

        control_msg = TurtleBotControl()
        control_msg.V = V
        control_msg.omega = om

        return control_msg

def main(args = None):
    rclpy.init(args = args)
    node = Navigator(kp = 1.0, kpx = 1.0, kdx = 0.1, kpy = 1.0, kdy = 0.1, V_max = 0.5, om_max = 1.0)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

