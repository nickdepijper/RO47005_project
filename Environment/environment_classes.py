import numpy as np

class WorldDescription:
    "A description of the world in terms of its size, the obstacles in it, and the start and goal positions"
    def __init__(self, world_size, n_obstacles_static, n_obstacles_dynamic, obstacle_size_array):
        self.world_size = world_size
        self.n_obstacles_static = n_obstacles_static
        self.n_obstacles_dynamic = n_obstacles_dynamic
        self.n_obstacles = self.n_obstacles_static + self.n_obstacles_dynamic
        self.obstacles = []
        self.obstacle_size_array = obstacle_size_array
        self.startpos = None
        self.goalpos = None
        self.generate_start_and_goal_pos()

    def generate_world_description(self):
        if len(self.obstacles) == 0:
            self.generate_static_obstacles()
            self.generate_dynamic_obstacles()

    def generate_static_obstacles(self):
        for i in range(self.n_obstacles_static):
            staticpath = self.generate_valid_position()
            obstacle = ObstacleDescription(shape="sphere",
                                           size_array=self.obstacle_size_array,
                                           path_description=PathDescription(start_pos=staticpath,
                                                                            end_pos=staticpath,
                                                                            n_timesteps=1))
            self.obstacles.append(obstacle)

    def generate_dynamic_obstacles(self):
        for i in range(self.n_obstacles_dynamic):
            path_start = self.generate_valid_position()
            path_end = self.generate_valid_position()
            n_steps = int(np.sqrt(np.sum((path_start - path_end)**2))*300)
            obstacle = ObstacleDescription(shape="sphere",
                                           size_array=self.obstacle_size_array,
                                           path_description=PathDescription(start_pos=path_start,
                                                                            end_pos=path_end,
                                                                            n_timesteps=n_steps))
            self.obstacles.append(obstacle)

    def generate_start_and_goal_pos(self):
        self.startpos = (np.random.rand(3) * self.world_size - [0.5, 0.5, 0] * self.world_size) * 0.5
        self.startpos[0] = -0.5 * self.world_size[0]

        self.goalpos = (np.random.rand(3) * self.world_size - [0.5, 0.5, 0] * self.world_size) * 0.5
        self.goalpos[0] = 0.5 * self.world_size[0]

    def generate_valid_position(self):
        """Generates a position within the world boundaries"""
        return np.random.rand(3)*self.world_size - [0.5, 0.5, 0] * self.world_size

    def update_positions(self):
        """Updates all object positions for the next simulation timestep"""
        for obstacle in self.obstacles:
            obstacle.update_position()


class ObstacleDescription:
    """A description used to build an obstacle in pybullet"""
    def __init__(self, shape, path_description, size_array):
        self.shape = shape
        self.geometric_description = None
        self.generate_geometric_description(size_array)
        self.path_description = path_description
        self.path = path_description.generate_3d_path()
        self.current_position_index = 0
        self.max_position_index = self.path.shape[0] - 1
        self.current_position = self.path[0, :]
        self.move_forward = True

    def generate_geometric_description(self, sphere_size_array):
        """Generates a description of the geometric object in terms of its shape and size"""
        if self.shape == "sphere":
            r = np.random.choice(sphere_size_array)
            self.geometric_description = {"shape_type" : self.shape, "radius" : r}

    def update_position(self):
        if self.move_forward:
            if self.current_position_index < self.max_position_index:
                self.current_position_index += 1
                self.current_position = self.path[self.current_position_index]
            else:
                self.move_forward = False
                self.current_position_index -= 1
                self.current_position = self.path[self.current_position_index]
        else:  # Moving backward
            if self.current_position_index > 0:
                self.current_position_index -= 1
                self.current_position = self.path[self.current_position_index]
            else:
                self.move_forward = True
                self.current_position_index += 1
                self.current_position = self.path[self.current_position_index]



class PathDescription:
    """A description of a straight line path for obstacles"""
    def __init__(self, start_pos, end_pos, n_timesteps):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.n_timesteps = n_timesteps


    def generate_3d_path(self):
        """Takes a start and end position and
        generates a numpy array of shape n_timestepsx3 of points sampled along the line.
        """
        # Turn into numpy arrays for broadcasting
        start_point = np.array(self.start_pos)
        end_point = np.array(self.end_pos)

        # Generate interpolation factors (t values) between 0 and 1
        t = np.linspace(0, 1, self.n_timesteps).reshape(-1, 1)

        # Interpolate linearly between the two points
        points = start_point + t * (end_point - start_point)
        return points

