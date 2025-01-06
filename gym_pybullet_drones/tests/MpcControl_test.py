import numpy as np
from gym_pybullet_drones.control.MpcControl import MPCController, DroneDynamics

print("Started")
drone = DroneDynamics()
start_pos = np.array([[0],
                    [0],
                    [0]])

target_pos = np.array([[10],
                    [10],
                    [10]])

obstacles = np.array([
    [[1, 1, 1, 0.5], [1, 2, 1, 0.5]],
    [[3, 1, 4, 2], [3, 1, 3, 2]],
    [[2, 3, 1, 1], [2, 1, 1, 1]]
])

MPC = MPCController(drone, start_pos, target_pos, obstacles)

constraints =  MPC._get_obstacle_constraints(0)

print(constraints)