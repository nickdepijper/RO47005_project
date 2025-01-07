import cvxpy as cp
import numpy as np
import pybullet as p
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class SimpleMPC:
    def __init__(self, horizon=10, timestep=1/240, m=0.027, g=9.8, Ixx=1.4e-5, Iyy=1.4e-5, Izz=2.17e-5):
        self.horizon = horizon
        self.timestep = timestep
        self.m = m
        self.g = g
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        CD = 7.9379e-12
        CT = 3.1582e-10
        d = 39.73e-3

        self.arm_length = d  # meters
        L = d
        self.to_TM = np.array([[1,  1,  1,  1],
                               [0,  L,  0, -L],
                               [-L,  0,  L,  0],
                               [CT, -CT, CT, -CT]])

        # Continuous state-space matrices including yaw dynamics
        self.A_c = np.array([
            [0, 0, 0, 1, 0, 0, 0,  0,    0,      0,      0,      0],
            [0, 0, 0, 0, 1, 0, 0,  0,    0,      0,      0,      0],
            [0, 0, 0, 0, 0, 1, 0,  0,    0,      0,      0,      0],
            [0, 0, 0, 0, 0, 0, 0,  self.g, 0,      0,      0,      0],
            [0, 0, 0, 0, 0, 0, -self.g, 0,    0,      0,      0,      0],
            [0, 0, 0, 0, 0, 0, 0,  0,    0,      0,      0,      0],
            [0, 0, 0, 0, 0, 0, 0,  0,    1,      0,      0,      0],
            [0, 0, 0, 0, 0, 0, 0,  0,    0,      1,      0,      0],
            [0, 0, 0, 0, 0, 0, 0,  0,    0,      0,      1,      0],
            [0, 0, 0, 0, 0, 0, 0,  0,    0,      0,      0,      1],
            [0, 0, 0, 0, 0, 0, 0,  0,    0,      0,      0,      0],
            [0, 0, 0, 0, 0, 0, 0,  0,    0,      0,      0,      0],
        ])

        self.B_c = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1/self.m, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1/self.Ixx, 0, 0],
            [0, 0, 1/self.Iyy, 0],
            [0, 0, 0, 1/self.Izz],
            [0, 0, 0, 0],
        ]) @ self.to_TM

        self.C_c = np.identity(12)
        self.D_c = np.zeros((12, 4))

        # Discretize state-space matrices
        self.A = np.eye(12) + self.A_c * self.timestep
        self.B = self.B_c * self.timestep
        self.C = self.C_c
        self.D = self.D_c

    def compute_control(self, current_state, target_state):
        cost = 0.0
        constraints = []

        # Create optimization variables
        x = cp.Variable((12, self.horizon + 1))
        u = cp.Variable((4, self.horizon))

        # Define cost matrices
        Q = np.diag([1, 1, 0.5, 1, 1, 1, 800, 800, 5, 25, 25, 14000])  
        R = 0.02 * np.eye(4)

        u_target = np.array([self.m * self.g / 4] * 4)

        # Add constraints and cost
        for n in range(self.horizon):
            cost += cp.quad_form(x[:, n + 1] - target_state, Q) + cp.quad_form(u[:, n] - u_target, R)
            constraints += [x[:, n + 1] == self.A @ x[:, n] + self.B @ u[:, n]]
            constraints += [cp.abs(x[6, n + 1]) <= 28.5]  # Limit pitch (phi)
            constraints += [cp.abs(x[7, n + 1]) <= 28.5]  # Limit roll (theta)
            constraints += [cp.abs(x[10, n + 1]) <= 28.5]  # Limit yaw (psi)
            constraints += [u[:, n] >= -0.07 * 30]
            constraints += [u[:, n] <= 0.07 * 30]

        # Initial state constraint
        constraints += [x[:, 0] == current_state.flatten()]

        # Solve the optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP)

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
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or CF2P")
            exit()

        self.mpc = SimpleMPC(horizon=90, timestep=1/45, m=0.027, g=g, Ixx=1.4e-5, Iyy=1.4e-5, Izz=2.17e-5)

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos, target_rpy=np.zeros(3), target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        current_state = np.hstack((cur_pos, cur_vel, p.getEulerFromQuaternion(cur_quat), cur_ang_vel))
        target_state = np.hstack((target_pos, target_vel, target_rpy, target_rpy_rates))

        thrusts, path = self.mpc.compute_control(current_state, target_state)
        rpms = []
        for thrust in thrusts:
            if thrust < 0:
                rpm = -np.sqrt(-thrust / self.KF)
            else:
                rpm = np.sqrt(thrust / self.KF)
            rpm = rpm * (60 / (2 * np.pi))
            rpms.append(rpm)

        pos_error = target_pos - cur_pos
        yaw_error = target_rpy[2] - p.getEulerFromQuaternion(cur_quat)[2]

        return rpms, pos_error, yaw_error, path
