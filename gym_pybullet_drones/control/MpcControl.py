import cvxpy as cp
import numpy as np


class MPCController:
    """MPC controller for controlling a drone in a dynamic environment.
    
    Parameters
    ----------
    drone: DroneDynamics Object
        Drone Dynamics object containing the state space dynamics model of the drone
    start_pos: float array(1,3)
        X,Y,Z coordinates of start position of the drone
    target_pos: float array(1,3)
        X,Y,Z coordinates of the end position of the drone
    obstacles: float array(N,M,4)
        X,Y,Z coordinates and radius of N obstacles over M time steps
    #TODO: If we also have stationary column obstacles, we  probably need to treat them separately somehow
    """
    def __init__(self,drone, start_pos, target_pos, obstacles):
        """Setup of class variables needed for the controller"""
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.obstacles = obstacles
        self.current_pos = start_pos
        
        self.drone_radius = 2
    
    def _get_obstacle_constraints(self, time_id):
        """Computes the obstacle constraints for the current time id (column in obstacles array)
        
        Parameter
        ----------
        time_id integer
            Id for which time step (row in the obstacle array) to compute the obstacle constraints for

        Returns
        ----------
        float list(N)
            List with all obstacle constraints
        """
        # Get current position of drone in simulation
        drone_x, drone_y, drone_z = self.current_pos
        # Get radius of drone
        r_drone = self.drone_radius
        # Get list of obstacles at requested time ID
        obstacles = self.obstacles[:,time_id]
        # Set Safety margin
        epsilon = 0.1

        constraints = []

        for obstacle in obstacles:
            obs_x, obs_y, obs_z, r_obs = obstacle
            constraints.append(
                cp.maximum(drone_x - (obs_x + r_obs + r_drone + epsilon),  # Right side
                        (obs_x - r_obs - r_drone - epsilon) - drone_x,  # Left side
                        drone_y - (obs_y + r_obs + r_drone + epsilon),  # Back side
                        (obs_y - r_obs - r_drone - epsilon) - drone_y,  # Front side
                        drone_z - (obs_z + r_obs + r_drone + epsilon),  # Top side
                        (obs_z - r_obs - r_drone - epsilon) - drone_z) >= 0
            )
        
        return constraints


    def _set_constraints(self):
        """Sets the constraints of the MPC controller"""
    
    def _set_cost_function(self):
        """"Sets the cost function of the MPC controller"""
    
    def get_control_signal(self):
        """Computes next control signal
        Returns
        ----------
        float array(4)
            Thrust, yaw, pitch, roll control signal
        """
    
    def _update_pos(self, pos):
        """"Placeholder function to ensure we remember to update the position"""
        raise NotImplementedError()
        


class DroneDynamics():
    """Defines the dynamics of a drone in state space representation
    
    Parameters
    ----------
    A: numpy array
        State space matrix A. Do not discretize beforehand!
    B: numpy array
        State space matrix B. Do not discretize beforehand!
    C: numpy array
        State space matrix C. Do not discretize beforehand!
    D: numpy array
        State space matrix D. Do not discretize beforehand!
    """
    def __init__(self, A=None, B=None, C=None, D=None):
        """Set state space matrices of the drone."""
        # If matrices are passed during init, they are used instead of the default ones
        if not all(matrix is None for matrix in [A,B,C,D]):
            print("Using custom matrices")
            self.A_c = A 
            self.B_c = B
            self.C_c = C
            self.D_c = D 
        else:
            m=0.027
            g=9.8
            CD = 7.9379*10**-12
            CT = 3.1582*10**-10
            d= 39.73 * 10**-3
            Ixx=1.4e-5
            Iyy=1.4e-5
            Izz=2.17e-5
            horizon=50
            timestep=1/48
            self.timestep = timestep
            # Set default matrices
            print("Using default matrices")
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

        # Discretizing state space model
        self.A = np.eye(12) + self.A_c * self.timestep
        self.B = self.B_c * self.timestep 
        self.C = self.C_c
        self.D = self.D_c
