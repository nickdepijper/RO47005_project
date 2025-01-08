

import cvxpy as cp
import numpy as np
import pybullet as p
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class SimpleMPCPlanner:
    def __init__(self, horizon=500, timestep=1/10, m=0.027, g=9.81):
        self.horizon = horizon
        self.timestep = timestep
        self.m = m
        self.g = g

        """
        Continuous state space without considering yawing(assuming yaw angle is 0 for all the time)
        state = [x y z dx dy dz]
        dot_state = [dx dy dz ddx ddy ddz]
        u = [ddx, ddy, ddz]
        """

        self.A_c = np.zeros((6,6))
        self.A_c[:3, 3:] = np.eye(3)
        self.B_c = np.vstack((np.zeros((3, 3)), np.eye(3)))
        self.C_c = np.eye(6)
        self.D_c = np.zeros((1,6))

        # Discretization state space
        self.A = np.eye(6) + self.A_c * self.timestep
        self.B = self.B_c * self.timestep
        self.C = self.C_c
        self.D = self.D_c

    def compute_control(self, current_state_planner, target_state):
        # Weight on the input
        current_state_planner = current_state_planner[:6]
        cost = 0.
        constraints = []

        # Create the optimization variables
        x = cp.Variable((6, self.horizon + 1)) # cp.Variable((dim_1, dim_2))
        u = cp.Variable((3, self.horizon))

        # Initial state
        constraints += [x[:, 0] == current_state_planner.flatten()]

        # For each stage in the MPC horizon
        Q = np.diag([1, 1, 1, 1, 1, 10])  # High weight on position/orientation
        R = np.eye(3)  # Lower weight on control effort

        for n in range(self.horizon):
            cost += (cp.quad_form((x[:,n+1]-target_state),Q)  + cp.quad_form(u[:,n], R))
            constraints += [x[:,n+1] == self.A @ x[:,n] + self.B @ u[:,n]]

            # State and input constraints
            constraints += [x[3, n + 1] <= 5]
            constraints += [x[4, n + 1] <= 5]
            constraints += [x[5, n + 1] <= 5]
            constraints += [x[3, n + 1] >= -5]
            constraints += [x[4, n + 1] >= -5]
            constraints += [x[5, n + 1] >= -5]

            constraints += [u[:, n] >= -10.1]
            constraints += [u[:, n] <= 10.1]

            #Obstacle avoidance
            #constraints += [A_obs @ x[:2,n] <= b_obs.flatten()]

        # Solve the problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, verbose=False) # solver=cp.OSQP

        # We return the MPC input
        return x.value
