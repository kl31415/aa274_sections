#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from asl_tb3_lib.navigation import BaseNavigator
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.navigation import TrajectoryPlan
from scipy.interpolate import splrep, splev
import numpy as np

class Navigator(BaseNavigator):
    def __init__(self, kp: float, kpx: float, kdx: float, kpy: float, kdy: float):
        super().__init__("navigator")
        self.kp = kp
        self.kpx = kpx
        self.kdx = kdx
        self.kpy = kpy
        self.kdy = kdy

        self.V_prev = 0.0
        self.t_prev = 0.0
        self.om_prev = 0.0
        self.V_PREV_THRES = 0.0001

        self.v_desired = 0.15
        self.spline_alpha = 0.05

        self.resolution = 1
        self.horizon = 5
        self.astar = None

    class AStar(object):
        def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
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
            self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

            self.path = None      

        def is_free(self, x):
            if x==self.x_init or x==self.x_goal:
                return True
            for dim in range(len(x)):
                if x[dim] < self.statespace_lo[dim]:
                    return False
                if x[dim] > self.statespace_hi[dim]:
                    return False
            if not self.occupancy.is_free(np.asarray(x)):
                return False
            return True

        def distance(self, x1, x2):
            return np.linalg.norm(np.array(x1)-np.array(x2))

        def snap_to_grid(self, x):
            return (
                self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
                self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
            )

        def get_neighbors(self, x):
            neighbors = []
            for dx1 in [-self.resolution, 0, self.resolution]:
                for dx2 in [-self.resolution, 0, self.resolution]:
                    if dx1==0 and dx2==0:
                        continue
                    new_x = (x[0]+dx1,x[1]+dx2)
                    if self.is_free(new_x):
                        neighbors.append(self.snap_to_grid(new_x))
            return neighbors
        
        def find_best_est_cost_through(self):
            return min(self.open_set, key=lambda x: self.est_cost_through[x])

        def reconstruct_path(self):
            path = [self.x_goal]
            current = path[-1]
            while current != self.x_init:
                path.append(self.came_from[current])
                current = path[-1]
            return list(reversed(path))

        def solve(self):
            while len(self.open_set)>0:
                current = self.find_best_est_cost_through()
                if current == self.x_goal:
                    self.path = self.reconstruct_path()
                    return True
                self.open_set.remove(current)
                self.closed_set.add(current)
                for n in self.get_neighbors(current):
                    if n in self.closed_set:
                        continue
                    tentative_cost_to_arrive = self.cost_to_arrive[current] + self.distance(current,n)
                    if n not in self.open_set:
                        self.open_set.add(n)
                    elif tentative_cost_to_arrive >= self.cost_to_arrive[n]:
                        continue
                    self.came_from[n] = current
                    self.cost_to_arrive[n] = tentative_cost_to_arrive
                    self.est_cost_through[n] = self.cost_to_arrive[n] + self.distance(n,self.x_goal)

            return False

    def compute_trajectory_plan(self, state, goal, occupancy, resolution, horizon) -> TrajectoryPlan:
        lo = (state.x - horizon, state.y - horizon)
        hi = (state.x + horizon, state.y + horizon)

        astar_state = (state.x, state.y)
        astar_goal = (goal.x, goal.y)

        astar = self.AStar(lo, hi, astar_state, astar_goal, occupancy, resolution)
        if not astar.solve() or len(astar.path) < 4:
            return None
        
        path = np.asarray(astar.path)

        ts_n = np.shape(path)[0]
        ts = np.zeros(ts_n)

        for i in range(ts_n-1):
            ts[i+1] = astar.distance(path[i+1], path[i]) / self.v_desired
            ts[i+1] += ts[i]

        path_x_spline = splrep(ts, path[: ,0], k=3, s=self.spline_alpha)
        path_y_spline = splrep(ts, path[: ,1], k=3, s=self.spline_alpha)

        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )

    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        control = TurtleBotControl()

        err = wrap_angle(goal.theta - state.theta)
        control.omega = self.kp * err

        return control

    def compute_trajectory_tracking_control(self, state, plan, t: float) -> TurtleBotControl:
        x, y, th = state.x, state.y, state.theta

        dt = t - self.t_prev

        x_d = splev(t, plan.spline_x)
        xd_d = splev(t, plan.spline_x, der=1)
        xdd_d = splev(t, plan.spline_x, der=2)

        y_d = splev(t, plan.spline_y)
        yd_d = splev(t, plan.spline_y, der=1)
        ydd_d = splev(t, plan.spline_y, der=2)

        x_dot = self.V_prev * np.cos(th)
        y_dot = self.V_prev * np.sin(th)
        
        u1 = xdd_d + self.kpx * (x_d - x) + self.kdx * (xd_d - x_dot)
        u2 = ydd_d + self.kpy * (y_d - y) + self.kdy * (yd_d - y_dot)

        v_dot = u1 * np.cos(th) + u2 * np.sin(th)
        V = max(self.V_prev, self.V_PREV_THRES) 
        V = self.V_prev + v_dot * dt
        V = max(V, self.V_PREV_THRES)

        om = (u2 * np.cos(th) - u1 * np.sin(th)) / V

        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        control = TurtleBotControl()
        control.V = V
        control.omega = om

        return control

def main(args=None):
    rclpy.init(args=args)
    node = Navigator(kp=2.0, kpx=2.0, kdx=2.0, kpy=2.0, kdy=2.0)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
