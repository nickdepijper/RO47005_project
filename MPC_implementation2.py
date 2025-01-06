

import cvxpy as cp
import numpy as np
import pybullet as p
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class SimpleMPC:
    def __init__(self, horizon=40, timestep=1/240, m=0.027, g=9.8, Ixx=1.4e-5, Iyy=1.4e-5, Izz=2.17e-5):
        self.horizon = horizon
        self.timestep = timestep
        self.m = m
        self.g = g
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz  = Izz
        CD = 7.9379e-12
        CT = 3.1582e-10
        d= 39.73e-3

        """
        Define model parameters and state space for linear quadrotor dynamics
        """
        self.arm_length      = d  # meters
        g = 9.81 # m/s^2
        L = d
        self.to_TM = np.array([[1,  1,  1,  1],
                               [ 0,  L,  0, -L],
                               [-L,  0,  L,  0]])
        self.t_step = 0.1

        """
        Continuous state space without considering yawing(assuming yaw angle is 0 for all the time)
        state = [x y z dx dy dz phi theta phi_dot theta_dot]
        dot_state = [dx dy dz ddx ddy ddz phi_dot theta_dot phi_ddot theta_ddot]
        u = [F1 F2 F3 F4]
        """
        #                     x  y  z  dx dy dz phi theta phi_dot theta_dot
        self.A_c = np.array([[0, 0, 0, 1, 0, 0, 0,  0,    0,      0        ],    #dx
                             [0, 0, 0, 0, 1, 0, 0,  0,    0,      0        ],    #dy
                             [0, 0, 0, 0, 0, 1, 0,  0,    0,      0        ],    #dz
                             [0, 0, 0, 0, 0, 0, 0,  g,    0,      0        ],    #ddx
                             [0, 0, 0, 0, 0, 0, -g, 0,    0,      0        ],    #ddy
                             [0, 0, 0, 0, 0, 0, 0,  0,    0,      0        ],    #ddz
                             [0, 0, 0, 0, 0, 0, 0,  0,    1,      0        ],    #phi_dot
                             [0, 0, 0, 0, 0, 0, 0,  0,    0,      1        ],    #theta_dot
                             [0, 0, 0, 0, 0, 0, 0,  0,    0,      0        ],    #phi_ddot
                             [0, 0, 0, 0, 0, 0, 0,  0,    0,      0        ]])   #theta_ddot
        self.B_c = np.array([[0, 0, 0], #dx
                             [0, 0, 0], #dy
                             [0, 0, 0], #dz
                             [0, 0, 0], #ddx
                             [0, 0, 0], #ddy
                             [1/self.m, 0, 0], #ddz
                             [0, 0, 0],           #phi_dot
                             [0, 0, 0],           #theta_dot
                             [0, 1/self.Ixx, 0],  #phi_ddot
                             [0, 0, 1/self.Iyy]  #theta_ddot
                             ]) @ self.to_TM
        self.C_c = np.identity(10)
        self.D_c = np.zeros((1,4))

        # Discretization state space
        self.A = np.eye(10) + self.A_c * self.timestep
        self.B = self.B_c * self.timestep
        self.C = self.C_c
        self.D = self.D_c


    def compute_control(self, current_state, target_state):
        import cvxpy as cp

           # Weight on the input
        cost = 0.
        constraints = []
        
        # Create the optimization variables
        x = cp.Variable((10, self.horizon + 1)) # cp.Variable((dim_1, dim_2))
        u = cp.Variable((4, self.horizon))

        # Get constraints from obstacle list
        

        # For each stage in the MPC horizon
        u_target=np.array([14479,14479,14479,14479])
        Q = np.diag([2,2,1,1,1,1,1,1,1,1])
        R = 0.02*np.eye(4) 
        for n in range(self.horizon):
            cost += (cp.quad_form((x[:,n+1]-target_state),Q)  + cp.quad_form(u[:,n]-u_target, R))
            constraints += [x[:,n+1] == self.A @ x[:,n] + self.B @ u[:,n]]
            # State and input constraints
            #constraints += [x[6,n+1] <= 0.5]
            #constraints += [x[7,n+1] <= 0.5]
            #constraints += [x[6,n+1] >= -0.5]
            #constraints += [x[7,n+1] >= -0.5]
            #constraints += [u[:,n] >= -0.07*30]
            #constraints += [u[:,n] <= 0.07*30]
            #Obstacle avoidance
            #constraints += [A_obs @ x[:2,n] <= b_obs.flatten()]
            
        #Initial state
        constraints += [x[:,0] == current_state.flatten()]
        
        
        # Solves the problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP)
        # We return the MPC input
        return u[:, 0].value, x[:3, :].value


class DSLMPCControl(BaseControl):
    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.KF = 3.16e-10  # Motor thrust coefficient from specs
        self.KM = 7.94e-12  # Motor torque coefficient from specs
        self.ARM_LENGTH = 0.0397  # Arm length from specs

        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
        #if self.DRONE_MODEL == DroneModel.CF2X:
        #    self.MIXER_MATRIX = np.array([ 
        #                            [-.5, -.5, -1],
        #                            [-.5,  .5,  1],
        #                            [.5, .5, -1],
        #                            [.5, -.5,  1]
        #                            ])

        # Initialize MPC
        self.mpc = SimpleMPC(horizon=50, timestep=1/5, m=0.027, g=g, Ixx=1.4e-5, Iyy=1.4e-5, Izz=2.17e-5)

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
        current_state = np.hstack((cur_pos, cur_vel, p.getEulerFromQuaternion(cur_quat)[:2], cur_ang_vel[:2]))
        target_state = np.hstack(([0,0.3,0.3], target_vel, target_rpy[:2], target_rpy_rates[:2]))
        #target_state = np.hstack(([0,0,5], target_rpy, target_vel, target_rpy_rates))

        
        # Compute MPC control inputs
        rpm, path = self.mpc.compute_control(current_state, target_state)
       
        pos_error = target_pos - cur_pos
        yaw_error = target_rpy[2] - p.getEulerFromQuaternion(cur_quat)[2]

        return path #rpm, pos_error, yaw_error, path


