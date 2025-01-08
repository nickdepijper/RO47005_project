import cvxpy as cp
import numpy as np
import pybullet as p
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class SimpleMPC:
    def __init__(self, horizon=10, timestep=1/240, m=0.027, g=9.8, Ixx=1.4e-5, Iyy=1.4e-5):
        self.horizon = horizon
        self.timestep = timestep
        self.m = m
        self.g = g
        self.Ixx = Ixx
        self.Iyy = Iyy
        CD = 7.9379e-12
        CT = 3.1582e-10
        d = 39.73e-3

        self.arm_length = d  # meters
        L = d
        self.to_TM = np.array([[1,  1,  1,  1],
                               [0,  L,  0, -L],
                               [-L,  0,  L,  0]])

        # Continuous state-space matrices excluding yaw dynamics
        self.A_c = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        self.B_c = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1/self.m, 0, 0, 0],
            [0, 1/self.Ixx, 0, 0],
            [0, 0, 1/self.Iyy, 0]
        ]) @ self.to_TM

        self.C_c = np.identity(6)
        self.D_c = np.zeros((6, 4))

        # Discretize state-space matrices
        self.A = np.eye(6) + self.A_c * self.timestep
        self.B = self.B_c * self.timestep
        self.C = self.C_c
        self.D = self.D_c

    def compute_control(self, current_state, target_state):
        cost = 0.0
        constraints = []

        # Updated cost matrices
        Q = np.diag([10, 10, 100, 1, 1, 1])  # Penalize position and angles heavily
        R = 0.002 * np.eye(4)  # Penalize control inputs lightly

        u_target = np.array([self.m * self.g / 4] * 4)

        # Create optimization variables
        x = cp.Variable((6, self.horizon + 1))
        u = cp.Variable((4, self.horizon))

        for n in range(self.horizon):
            cost += cp.quad_form(x[:, n + 1] - target_state[:6], Q) + cp.quad_form(u[:, n] - u_target, R)

            # Dynamics constraints
            constraints += [x[:, n + 1] == self.A @ x[:, n] + self.B @ u[:, n]]

            # Input bounds
            constraints += [u[:, n] >= -0.02 * 30]
            constraints += [u[:, n] <= 0.02 * 30]

        # Initial state constraint
        constraints += [x[:, 0] == current_state[:6].flatten()]

        # Solve the optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP)

        if problem.status != cp.OPTIMAL:
            print(f"Optimization failed with status: {problem.status}")
            return np.zeros(4), None

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

        self.mpc = SimpleMPC(horizon=30, timestep=1/45, m=0.027, g=g, Ixx=1.4e-5, Iyy=1.4e-5)

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos, target_rpy=np.zeros(3), target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        # Fix yaw to zero
        current_state = np.hstack((cur_pos, cur_vel, p.getEulerFromQuaternion(cur_quat)[:2], cur_ang_vel[:2]))
        target_state = np.hstack((target_pos, target_vel, target_rpy[:2], target_rpy_rates[:2]))

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

        return rpms, pos_error, path
