import pybullet as p
import pybullet_data
import time

class Drone:
    def __init__(self, drone_id):
        self.drone_id = drone_id

    def get_state(self):
        pos, ori = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang_vel = p.getBaseVelocity(self.drone_id)
        return {
            "position": pos,
            "orientation": ori,
            "velocity": vel,
            "angular_velocity": ang_vel,
        }


class Obstacle:
    def __init__(self, obstacle_id):
        self.obstacle_id = obstacle_id

    def get_state(self):
        pos, ori = p.getBasePositionAndOrientation(self.obstacle_id)
        return {
            "position": pos,
            "orientation": ori,
        }


class SimulationUpdater:
    def __init__(self, drones, obstacles):
        self.drones = drones  # List of Drone objects
        self.obstacles = obstacles  # List of Obstacle objects
        self.parameters = []  # List to hold updated parameters

    def update_parameters(self):
        self.parameters.clear()  # Clear the list to avoid duplication
        
        # Update drone parameters
        for drone in self.drones:
            state = drone.get_state()
            self.parameters.append({"type": "drone", **state})
        
        # Update obstacle parameters
        for obstacle in self.obstacles:
            state = obstacle.get_state()
            self.parameters.append({"type": "obstacle", **state})

    def get_parameters(self):
        return self.parameters


# Example usage
if __name__ == "__main__":
    # Initialize PyBullet simulation
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Load drones and obstacles
    plane_id = p.loadURDF("plane.urdf")
    drone_id = p.loadURDF("quadrotor.urdf", [0, 0, 1])
    obstacle_id = p.loadURDF("cube.urdf", [1, 1, 0.5])

    # Create Drone and Obstacle objects
    drones = [Drone(drone_id)]
    obstacles = [Obstacle(obstacle_id)]

    # Initialize the updater
    updater = SimulationUpdater(drones, obstacles)

    # Simulation loop
    for _ in range(100):
        p.stepSimulation()
        updater.update_parameters()
        print(updater.get_parameters())  # Print the updated state
        time.sleep(0.1)

    # Disconnect from simulation
    p.disconnect()
