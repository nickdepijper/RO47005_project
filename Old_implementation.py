

import cvxpy as cp
import numpy as np
import pybullet as p
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class SimpleMPC:
    def __init__(self, horizon=50, timestep=1/48, m=0.027, g=9.8, Ixx=1.4e-5, Iyy=1.4e-5, Izz=2.17e-5):
        self.horizon = horizon
        self.timestep = timestep
        self.m = m
        self.g = g
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz  = Izz
        CD = 7.9379*10**-12
        CT = 3.1582*10**-10
        d= 39.73 * 10**-3

        # Linearized dynamics matrices
        self.A = np.zeros((12, 12))
        self.B = np.zeros((12, 4))

        # Fill A matrix
      

        #                     x  y  z  roll pitch yaw dx dy dz roll_dot pitch_dot yaw_dot
        self.A_c = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #x
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #y
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #z
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], #roll
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], #pitch
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], #yaw
            [0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0, 0], #dx
            [0, 0, 0, 0, 0, -g, 0, 0, 0, 0, 0, 0], #dy
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #dz
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #roll_dot
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #pitch_dot
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #yaw_dot
            ])
        self.B_c = np.sqrt(m*g/(4*CT))*np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2 * CT / m, 2 * CT / m, 2 * CT / m, 2 * CT / m],
            [-np.sqrt(2) * d * CT / Ixx, -np.sqrt(2) * d * CT / Ixx, np.sqrt(2) * d * CT / Ixx, np.sqrt(2) * d * CT / Ixx],
            [-2 * CD / Izz, 2 * CD / Izz, -2 * CD / Izz, 2 * CD / Izz],
            [-np.sqrt(2) * d * CT / Iyy, np.sqrt(2) * d * CT / Iyy, np.sqrt(2) * d * CT / Iyy, -np.sqrt(2) * d * CT / Iyy]
            ])
        self.C_c = np.identity(12)
        self.D_c = np.zeros((1,4))

        # Discretization state space
        self.A = np.eye(12) + self.A_c * self.timestep
        self.B = self.B_c * self.timestep 
        self.C = self.C_c
        self.D = self.D_c

    def compute_control(self, current_state, target_state):
        import cvxpy as cp

        # Optimization variables
        
        #x = cp.Variable((self.horizon + 1, 12))  # State trajectory
        #u = cp.Variable((self.horizon, 4))       # Control inputs (thrusts and torques)
        x = cp.Variable((12, self.horizon  + 1)) # cp.Variable((dim_1, dim_2))
        u = cp.Variable((4, self.horizon ))

        # Constraints and cost
        constraints = []
        cost = 0

        u_ref=np.array([16073,16073,16073,16073])*0.95
        Q = np.diag([10, 10, 10, 1, 1, 0.1, 1, 1, 1, 0.5, 0.5, 0.5])
        R = np.eye(4) * 0.00001  # Reduce penalty on control effort
        constraints += [x[:,0] == current_state.flatten()]
        for t in range(self.horizon):
            # Dynamics constraint: x[t+1] = A * x[t] + B * u[t]
             # Create the optimization variables

            constraints += [x[:,t+1] == self.A @ x[:,t] + self.B @ u[:,t]]
            constraints += [x[3,t+1] <= 0.3]
            constraints += [x[4,t+1] <= 0.3]
            constraints += [x[3,t+1] >= -0.3]
            constraints += [x[4,t+1] >= -0.3]
            #constraints += [u[:,t] >= 16073-3000]
            #constraints += [u[t,:] <= 0.07*30]

            # Physical constraints on thrust inputs
            #constraints.append(u[t] >= 0)  # Non-negativity for all control inputs
            #constraints.append(u[t][0] <= 2 * self.m * self.g)  # Thrust limit
            #constraints.append(cp.abs(u[t][1:]) <= 10)          # Torque limits

            # Cost: position error + control effort
            cost += (cp.quad_form((x[:,t+1]-target_state),Q)  + cp.quad_form(u[:,t]-u_ref, R))
            #position_error = cp.norm(x[t, :3] - target_state[:3], 2)  # Positional deviation
            #control_effort = cp.norm(u[t], 2)                        # Energy minimization
            #cost += position_error + 0.01 * control_effort

        # Terminal cost to drive final state close to the target state
        #cost += cp.norm(x[:3, self.horizon] - target_state[:3], 2)
        #print('DEZZZZZZZZEEEEEEEEISHETVERSCHILLLLLLLLLLLLLLLL',(np.array(current_state[:])-np.array(target_state[:])))
        # Solve the optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        # Check for solver issues
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Solver Status: {problem.status}")
            print("Thrusts may be invalid due to infeasibility.")
            return np.zeros(4)  # Default safe output in case of solver failure

        # Return the first control input
        return u[:, 0].value#, x[:3, :].value


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
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])

        # Initialize MPC
        self.mpc = SimpleMPC(horizon=50, timestep=1/48, m=0.027, g=g, Ixx=1.4e-5, Iyy=1.4e-5, Izz=2.17e-5)

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy,
                       target_vel,
                       target_rpy_rates
                       ):
        current_state = np.hstack((cur_pos, p.getEulerFromQuaternion(cur_quat), cur_vel, cur_ang_vel))
        target_state = np.hstack((target_pos, target_rpy, target_vel, target_rpy_rates))

        #print("SSSSSSSSSSTAAAAAAAAAAAAAAAAAAATEEEEEEEEEEESSSSSSSSSss",p.getEulerFromQuaternion(cur_quat))
        
        # Compute MPC control inputs
        rpm = self.mpc.compute_control(current_state, target_state)
       
        #print("TAAAAAAAAARRRRRRRRGETTTTTTTTTTPOSSSS",target_pos)
        pos_error = target_pos - cur_pos
        yaw_error = target_rpy[2] - p.getEulerFromQuaternion(cur_quat)[2]

        return rpm, pos_error, yaw_error#, path


