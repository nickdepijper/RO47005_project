

import cvxpy as cp
import numpy as np
import pybullet as p
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class SimpleMPC:
    def __init__(self, horizon=500, timestep=1/60, m=0.027, g=9.8, Ixx=1.4e-5, Iyy=1.4e-5, Izz=2.17e-5):
        self.horizon = horizon
        self.timestep = timestep
        self.m = m
        self.g = g
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz  = Izz
        CD = 7.9379e-12
        CT = 3.1582e-10
        L = 39.73e-3
        gamma_torque = CD/(CT*Izz)

        """
        Define model parameters and state space for linear quadrotor dynamics
        """
        self.arm_length      = L  # meters
        g = 9.81 # m/s^2

        """
        Continuous state space without considering yawing(assuming yaw angle is 0 for all the time)
        state = [x y z dx dy dz phi theta phi_dot theta_dot]
        dot_state = [dx dy dz ddx ddy ddz phi_dot theta_dot phi_ddot theta_ddot]
        u = [F1 F2 F3 F4]
        """
        #                     x  y  z  dx dy dz phi theta   xi   phi_dot theta_dot  xi_dot
        self.A_c = np.array([[0, 0, 0, 1, 0, 0, 0,  0,      0,    0,      0,        0 ],    #dx
                             [0, 0, 0, 0, 1, 0, 0,  0,      0,    0,      0,        0 ],    #dy
                             [0, 0, 0, 0, 0, 1, 0,  0,      0,    0,      0,        0 ],    #dz
                             [0, 0, 0, 0, 0, 0, 0,  g,      0,    0,      0,        0 ],    #ddx
                             [0, 0, 0, 0, 0, 0, -g, 0,      0,    0,      0,        0 ],    #ddy
                             [0, 0, 0, 0, 0, 0, 0,  0,      0,    0,      0,        0 ],    #ddz
                             [0, 0, 0, 0, 0, 0, 0,  0,      0,    1,      0,        0 ],    #phi_dot
                             [0, 0, 0, 0, 0, 0, 0,  0,      0,    0,      1,        0 ],    #theta_dot
                             [0, 0, 0, 0, 0, 0, 0,  0,      0,    0,      0,        1 ],    #xi_dot
                             [0, 0, 0, 0, 0, 0, 0,  0,      0,    0,      0,        0 ],    #phi_ddot
                             [0, 0, 0, 0, 0, 0, 0,  0,      0,    0,      0,        0 ],    #theta_ddot
                             [0, 0, 0, 0, 0, 0, 0,  0,      0,    0,      0,        0 ]])   #xi_ddot

        self.B_c = np.array([[0, 0, 0, 0], #dx
                             [0, 0, 0, 0], #dy
                             [0, 0, 0, 0], #dz
                             [0, 0, 0, 0], #ddx
                             [0, 0, 0, 0], #ddy
                             [1/self.m, 1/self.m, 1/self.m, 1/self.m], #ddz
                             [0, 0, 0, 0],           #phi_dot
                             [0, 0, 0, 0],           #theta_dot
                             [0, 0, 0, 0],  # xi_dot
                             [0, L/self.Ixx, 0, -L/self.Ixx],  #phi_ddot
                             [-L/self.Iyy, 0, L/self.Iyy, 0],  #theta_ddot
                             [-gamma_torque, gamma_torque, -gamma_torque, gamma_torque]  # xi_ddot
                             ])
        self.C_c = np.identity(12)
        self.D_c = np.zeros((1,4))

        # Discretization state space
        self.A = np.eye(12) + self.A_c * self.timestep
        self.B = self.B_c * self.timestep
        self.C = self.C_c
        self.D = self.D_c


    def compute_control(self, current_state, target_state):
           # Weight on the input
        cost = 0.
        constraints = []

        # Create the optimization variables
        x = cp.Variable((12, self.horizon + 1)) # cp.Variable((dim_1, dim_2))
        u = cp.Variable((4, self.horizon))

        # Initial state
        constraints += [x[:, 0] == current_state.flatten()]

        # For each stage in the MPC horizon
        u_target=np.array([self.m*9.81/4,self.m*9.81/4,self.m*9.81/4,self.m*9.81/4])
        Q = np.diag([150, 150, 100, 1, 1, 10, 1, 1, 10, 1, 1, 1])  # High weight on position/orientation
        R = 0.01 * np.eye(4)  # Lower weight on control effort
        for n in range(self.horizon):
            cost += (cp.quad_form((x[:,n+1]-target_state),Q)  + cp.quad_form(u[:,n]-u_target, R))
            constraints += [x[:,n+1] == self.A @ x[:,n] + self.B @ u[:,n]]
            # State and input constraints
            #constraints += [x[3, n + 1] <= 1]
            #constraints += [x[4, n + 1] <= 1]
            #constraints += [x[5, n + 1] <= 1]
            #constraints += [x[6, n + 1] <= 0.6]
            #constraints += [x[7, n + 1] <= 0.6]
            constraints += [x[8, n + 1] <= 0.3]
            #constraints += [x[3, n + 1] >= -1]
            #constraints += [x[4, n + 1] >= -1]
            #constraints += [x[4, n + 1] >= -1]
            #constraints += [x[6, n + 1] >= -0.6]
            #constraints += [x[7, n + 1] >= -0.6]
            constraints += [x[8, n + 1] >= -0.3]

            #constraints += [u[:, n] >= -0.07 * 30]
            #constraints += [u[:, n] <= 0.07 * 30]

            #Obstacle avoidance
            #constraints += [A_obs @ x[:2,n] <= b_obs.flatten()]




        # Solves the problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, verbose=False) # solver=cp.OSQP
        # We return the MPC input
        return u[:, 0].value, x[:3, :].value


class DSLMPCControl(BaseControl):
    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 10000
        self.MAX_PWM = 65535
        self.KF = 3.16e-10  # Motor thrust coefficient from specs
        self.KM = 7.94e-12  # Motor torque coefficient from specs
        self.ARM_LENGTH = 0.0397  # Arm length from specs

        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])

        # Initialize MPC
        self.mpc = SimpleMPC(horizon=40, timestep=1/60, m=0.027, g=g, Ixx=1.4e-5, Iyy=1.4e-5, Izz=2.17e-5)

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        current_state = np.hstack((cur_pos, cur_vel, p.getEulerFromQuaternion(cur_quat), cur_ang_vel))
        target_state = np.hstack((target_pos, target_vel, target_rpy, target_rpy_rates))
        #target_state = np.hstack(([0,0,5], target_rpy, target_vel, target_rpy_rates))

        # Compute MPC control inputs
        thrusts, path = self.mpc.compute_control(current_state, target_state)
        #thrusts = np.clip(thrusts, 0, self.MAX_PWM / self.PWM2RPM_SCALE)
        rpms = []
        
        for thrust in thrusts:
            #print("thrust isssssssssssss",thrust)
            if thrust < 0:
                rpm = -np.sqrt(-thrust / self.KF)  # radians per second
            else: rpm = np.sqrt(thrust / self.KF)  # radians per second
            rpm = rpm * (60 / (2 * np.pi))  # convert to RPM
            rpms.append(rpm)
        pos_error = target_pos - cur_pos
        yaw_error = target_rpy[2] - p.getEulerFromQuaternion(cur_quat)[2]

        return rpms, pos_error, yaw_error, path


