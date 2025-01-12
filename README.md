# RO47005_project
## Quadcopter project
Authors: Max Dam, Nick de Pijper, Freek Müller, Tobias Knell

## Introduction
This repository introduces a convex optimization-based approach for quadcopter control using MPC for use in static environments. 
The proposed motion planner calculates an optimal control input sequence for drone rotor RPM using a hierachical planning and control architecture. A demonstration can be found here: https://www.youtube.com/watch?v=9irkxh63B6c.

This README file focuses on guiding the user through the installation and running of the simulation, demonstrating the results, that are shown in the project report. For detailed information on the implementation, please consult the report.

## Installation
The simulation environment is based on the gym-pybullet-drones environment https://github.com/utiasDSL/gym-pybullet-drones

This installation guide assumes, that the user is on Linux, has the RO47005_project repository available and has navigated into the base of it with the terminal. 

The first part of the installation process is analogous to that of the gym-pybullet-drones repository, and is thus copied here:
```bash
conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . 
```


After having completed these steps, some additional packages are required, to run the simulation:
```bash
conda install -c conda-forge libgcc=5.2.0
conda install -c anaconda libstdcxx-ng
conda install -c conda-forge gcc=12.1.0
conda install -c conda-forge cvxpy
```
## Starting the simulation
The simulation is started by running the simulation_manager.py file. Executing it will start a simulation test with 400 iterations, showing the drone navigate various simulated environments with an increasing number of obstacles.

```bash
python3 simulation_manager.py
```

## File Guide
Since this  project heavily relies on gym-pybullet-drones for simulation but also has a lot of own contribution, it was necessary to adjust some of the pybullet-drones files. To make it easier to identify the authors' contribution, below is a list of the files they have added or changed, together with a short explanation:

- **simulation_manager.py**: This file is based on gym_pybullet_drones/examples/pid.py, but was heavily modified. It organizes and runs the simulation. Here all changes to the simulation, like changing the number of obstacles, or activating/deactivating obstacle avoidance, can be made.
- **mpc.py**: This file is fully created by the authors and includes all the code necessary for the MPC controller
- **Environment/environment_classes.py**: This file is fully created by the authors. It includes all code necessary for creating the randomly generated environment, used for the simulation
- **gym_pybullet_drones/env/BaseAviary.py**: This file was modified by the authors to call the code created in **Environment/environment_classes.py**, to generate the needed environment
- **gym_pybullet_drones/env/CtrlAviary.py**: This file was slightly modified to be able to pass it another argument, but was otherwise unchanged.
- **gym_pybullet_drones/control/BaseControl.py**: This file was slightly modified to be able to pass it another argument, but otherwise was unchanged.
