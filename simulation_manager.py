"""Script demonstrating the joint use of simulation and control.

The simulation is run by a customized 'CtrlAviary` environment.
The control is given by the MPC Controller and Planner Implementation in MPC_controller & Planner

Example
-------
In a terminal, run as:

    $ python3 simulation_manager.py

Notes
-----
The drone navigates an environment with random obstacles from a start to an end position

"""

import os
import time
import argparse
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from mpc import DSLMPCControl, MPCPlanner

DEFAULT_DRONES = DroneModel("cf2p")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
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
N_OBSTACLES_STATIC=5,
N_OBSTACLES_DYNAMIC=0,
N_OBSTACLES_FALLING=0,
N_OBSTACLES_PILLAR=3,
N_OBSTACLES_CUBOID_FLOOR=10,
N_OBSTACLES_CUBOID_CEILING=5,
SPHERE_SIZE_ARRAY=np.array([0.05, 0.1, 0.15]),
CUBOID_SIZE_ARRAY=np.array([0.05, 0.075, 0.1]),
PILLAR_SIZE_ARRAY=np.array([0.05])

# Debug functionality
DEFAULT_USER_DEBUG_GUI = False
MPC_TRAJECTORY = False

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
        pillar_size_array=PILLAR_SIZE_ARRAY
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
                        world_size=np.array([3,3,1]),
                        n_obstacles_static=5,
                        n_obstacles_dynamic=0,
                        n_obstacles_falling=0,
                        n_obstacles_pillar=3,
                        n_obstacles_cuboid_floor=10,
                        n_obstacles_cuboid_ceiling=5,
                        sphere_size_array=np.array([0.05, 0.1, 0.15]),
                        cuboid_size_array=np.array([0.05, 0.075, 0.1]),
                        pillar_size_array=np.array([0.05])
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
    ctrl_MPC = DSLMPCControl(drone_model=drone)
    planner_MPC = MPCPlanner(horizon=20, timestep=1, m=0.027, g=9.81)

    previous_debug_lines = []

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    used = False
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        #### Remove previous debug lines ##########################
        for debug_item in previous_debug_lines:
            p.removeUserDebugItem(debug_item)
        previous_debug_lines = []

        # Visualization code for planner MPC
        if not used:
            used = True
            dots = planner_MPC.compute_control(current_state_planner=obs[0], target_state=TARGET_STATE)
            dots = dots.T
            for dot_coords in dots:
                id = p.createVisualShape(p.GEOM_SPHERE,
                                                radius=0.02,
                                                visualFramePosition=[0, 0, 0],
                                                rgbaColor=[0, 1, 0, 0.5],
                                                )
                p.createMultiBody(
                    baseMass=0,  # Setting mass to 0 disables physics (but not collisions)
                    baseVisualShapeIndex=id,
                    basePosition=dot_coords[:3],
                    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                    physicsClientId=env.CLIENT
                        )

        #### Compute control for the current way point ############# 
        
        for j in range(num_drones):
            particle_state_traj = planner_MPC.compute_control(current_state_planner=obs[j],
                                               target_state=TARGET_STATE)
            target_from_planner_pos = particle_state_traj.T[2, :3]
            target_from_planner_vel = particle_state_traj.T[2, 3:]
            action[j, :], pos_error, _, path = ctrl_MPC.computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    target_pos=target_from_planner_pos,
                                                                    target_rpy=INIT_RPYS[j, :],
                                                                    target_vel=target_from_planner_vel
                                                                    )

            env._showDroneLocalAxes(j)
        # Extract target position for visualization
        # target_pos = target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :]   
        
        if MPC_TRAJECTORY:
        # Draw MPC path as a line 
            for idx in range(len(path.T) - 1):
                line_id = p.addUserDebugLine(
                    lineFromXYZ=path.T[idx],
                    lineToXYZ=path.T[idx + 1],
                    lineColorRGB=[1, 0, 0],  # Red line
                    lineWidth=1.5
                )
                previous_debug_lines.append(line_id)
            
            # Draw line to target position
            current_pos = obs[j][:3]  # Drone's current position (x, y, z)
            target_line_id = p.addUserDebugLine(
                lineFromXYZ=current_pos,
                lineToXYZ=TARGET_POS,
                lineColorRGB=[0, 1, 0],  # Green line
                lineWidth=3
            )
            previous_debug_lines.append(target_line_id)

        #### Go to the next way point and loop #####################
        # for j in range(num_drones):
        #     wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        # for j in range(num_drones):
        #     logger.log(drone=j,
        #                timestamp=i/env.CTRL_FREQ,
        #                state=obs[j],
        #                control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
        #                #control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
        #                )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and MPC Control & Planner')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
