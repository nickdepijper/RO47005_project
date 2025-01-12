"""Script demonstrating the joint use of simulation and control.

The simulation is run by a customized 'CtrlAviary` environment.
The control is given by the MPC Controller and Planner Implementation in MPC_controller & Planner.

The file is based on examples/pid.py, from the gym-pybullet-drones repository
https://github.com/utiasDSL/gym-pybullet-drones

Example
-------
In a terminal, run as:

    $ python3 simulation_manager.py

Notes
-----
The drone navigates an environment with random obstacles from a start to an end position

"""

import time
import argparse
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from mpc import DSLMPCControl #, MPCPlanner

DEFAULT_DRONES = DroneModel("cf2p")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 120
DEFAULT_CONTROL_FREQ_HZ = 60# 48
DEFAULT_DURATION_SEC = 45
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

# Environment Setup
WORLD_SIZE=np.array([3,3,1]),
N_OBSTACLES_STATIC=25
N_OBSTACLES_DYNAMIC=0
N_OBSTACLES_FALLING=0
N_OBSTACLES_PILLAR=0
N_OBSTACLES_CUBOID_FLOOR=0
N_OBSTACLES_CUBOID_CEILING=0
SPHERE_SIZE_ARRAY=np.array([0.05, 0.1, 0.15])
CUBOID_SIZE_ARRAY=np.array([0.05, 0.075, 0.1])
PILLAR_SIZE_ARRAY=np.array([0.05])

# Debug functionality
DEFAULT_GUI = True
DEFAULT_USER_DEBUG_GUI = False
MPC_TRAJECTORY = False

# Automated Test Setups
N_OBSTACLE_PROGRESSION = [25, 50, 75, 100]
N_RUNS = 50

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
        world_size=WORLD_SIZE,
        n_obstacles_static=N_OBSTACLES_STATIC,
        n_obstacles_dynamic=N_OBSTACLES_DYNAMIC,
        n_obstacles_falling=N_OBSTACLES_FALLING,
        n_obstacles_pillar=N_OBSTACLES_PILLAR,
        n_obstacles_cuboid_floor=N_OBSTACLES_CUBOID_FLOOR,
        n_obstacles_cuboid_ceiling=N_OBSTACLES_CUBOID_CEILING,
        sphere_size_array=SPHERE_SIZE_ARRAY,
        cuboid_size_array=CUBOID_SIZE_ARRAY,
        pillar_size_array=PILLAR_SIZE_ARRAY,
        obstacle_avoidance_mode=True
        ):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3

    # Generate a start and goal position at opposite ends of the working volume
    INIT_XYZS= (np.random.rand(3) * WORLD_SIZE - np.array([0.5, 0.5, 0]) * WORLD_SIZE) * 0.5
    INIT_XYZS[0][0] = -0.4 * WORLD_SIZE[0][0]

    TARGET_POS = ((np.random.rand(3) * WORLD_SIZE- np.array([0.5, 0.5, 0]) * WORLD_SIZE) * 0.5)[0]
    TARGET_POS[0] = 0.4 * WORLD_SIZE[0][0]
    TARGET_STATE = np.concatenate((TARGET_POS.copy(), np.array([0,0,0])))

    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])
    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = control_freq_hz*PERIOD
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        target_pos=TARGET_POS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui,
                        world_size=WORLD_SIZE,
                        n_obstacles_static=n_obstacles_static,
                        n_obstacles_dynamic=n_obstacles_dynamic,
                        n_obstacles_falling=n_obstacles_falling,
                        n_obstacles_pillar=n_obstacles_pillar,
                        n_obstacles_cuboid_floor=n_obstacles_cuboid_floor,
                        n_obstacles_cuboid_ceiling=n_obstacles_cuboid_ceiling,
                        sphere_size_array=sphere_size_array,
                        cuboid_size_array=cuboid_size_array,
                        pillar_size_array=pillar_size_array
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    ctrl_MPC = DSLMPCControl(drone_model=drone, horizon=40, timestep=1/60, obstacles=env.environment_description.obstacles)

    previous_debug_lines = []

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    start_time = time.time()
    
    average_computation_time = 0
    collision = None
    goal_reached = None
    
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        #### Remove previous debug lines ##########################
        for debug_item in previous_debug_lines:
            p.removeUserDebugItem(debug_item)
        previous_debug_lines = []

        #### Collision & Success Checks ###############

        # Check for collisions
        for obstacle_id in env.obstacle_ids:
            drone_id = 1
            contact_points = p.getContactPoints(bodyA=drone_id, bodyB=obstacle_id)
            if len(contact_points) > 0:
                collision = True
                goal_reached = False
                print("Detected object collision")
                elapsed_time = time.time() - start_time
                print(f"Drone crashed after {elapsed_time:.2f} seconds.")
                env.close()
                return elapsed_time, average_computation_time, collision, goal_reached

        # Check if goal has been reached 
        for j in range(num_drones):
            distance_to_goal = np.linalg.norm(obs[j][:3] - TARGET_POS)
            if distance_to_goal < 0.1:  # Drone reached the goal
                elapsed_time = time.time() - start_time
                collision = False
                goal_reached = True
                print(f"Drone reached the goal in {elapsed_time:.2f} seconds.")
                env.close()
                return elapsed_time, average_computation_time, collision, goal_reached
            
            calc_start = time.time() 
            
            # Compute Control
            action[j, :], pos_error, _, path = ctrl_MPC.computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    target_pos=TARGET_STATE[0:3],
                                                                    target_rpy=INIT_RPYS[j, :],
                                                                    target_vel=TARGET_STATE[3:],
                                                                    obstacle_avoidance=obstacle_avoidance_mode
                                                                    )
            calc_end = time.time()
            average_computation_time += (calc_end - calc_start) / int(duration_sec*env.CTRL_FREQ)

            
            
            env._showDroneLocalAxes(j)

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    collision = False
    goal_reached = False
    elapsed_time = time.time() - start_time
    print(f"Drone has failed to reach the goal in {elapsed_time:.2f} seconds.")
    env.close()
    return elapsed_time, average_computation_time, collision, goal_reached


if __name__ == "__main__":
    for n_obstacles_static in N_OBSTACLE_PROGRESSION:
        print(f"Running test series with {n_obstacles_static} obstacles")
        for obstacle_avoidance_mode in [True,False]:
            with open("results.txt", "a") as file:
                file.write("\n")
                file.write(f"New run with {n_obstacles_static} obstacles and {obstacle_avoidance_mode} Obstacle Avoidance")
                file.write("\n")
                header = ["Elapsed_time", "Average_computation_time", "Collision", "Goal_reached", "Obstacle Avoidance"]
                file.write(",".join(map(str, header)) + "\n")
            for run_index in range(N_RUNS):
                print(f"Starting run {run_index + 1} / {N_RUNS}")
                try:
                    elapsed_time, average_computation_time, collision, goal_reached = run(n_obstacles_static=n_obstacles_static, obstacle_avoidance_mode=obstacle_avoidance_mode)  # Pass filtered arguments
                except Exception as e:
                    print(f"Run {run_index + 1} failed with error: {e}")
                    collision = True
                
                with open("results.txt", "a") as file:
                    data = [round(elapsed_time,3), round(average_computation_time,5), collision, goal_reached, obstacle_avoidance_mode]
                    file.write(",".join(map(str, data)) + "\n")
                print("Data of run saved to results.txt")
        

    print("All runs completed.")


