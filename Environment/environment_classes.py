import numpy as np

class WorldDescription:
    "A description of the world in terms of its size, the obstacles in it, and the start and goal positions"
    def __init__(self, world_size,
                 n_obstacles_static, n_obstacles_dynamic, n_obstacles_falling,
                 n_obstacles_pillar, n_obstacles_cuboid_floor, n_obstacles_cuboid_ceiling,
                 sphere_size_array, cuboid_size_array, pillar_size_array):

        self.world_size = world_size
        self.n_obstacles_static = n_obstacles_static
        self.n_obstacles_dynamic = n_obstacles_dynamic
        self.n_obstacles_falling = n_obstacles_falling
        self.n_obstacles_pillar = n_obstacles_pillar
        self.n_obstacles_cuboid_floor = n_obstacles_cuboid_floor
        self.n_obstacles_cuboid_ceiling = n_obstacles_cuboid_ceiling
        self.n_obstacles = self.n_obstacles_static + self.n_obstacles_dynamic + self.n_obstacles_falling \
                           + self.n_obstacles_pillar + self.n_obstacles_cuboid_floor + self.n_obstacles_cuboid_ceiling
        self.obstacles = []
        self.sphere_size_array = sphere_size_array
        self.cuboid_size_array = cuboid_size_array
        self.pillar_size_array = pillar_size_array
        self.startpos = None
        self.goalpos = None
        self.generate_start_and_goal_pos()

    def generate_world_description(self):
        if len(self.obstacles) == 0:
            self.generate_static_obstacles()
            self.generate_dynamic_obstacles()
            self.generate_falling_obstacles()
            self.generate_obstacles_pillar()
            self.generate_obstacles_cuboid_floor()
            self.generate_obstacles_cuboid_ceiling()

            self.randomize_object_start_positions()

    def generate_static_obstacles(self):
        for i in range(self.n_obstacles_static):
            staticpath = self.generate_valid_position()
            obstacle = ObstacleDescription(shape="sphere",
                                           sphere_size_array=self.sphere_size_array,
                                           cuboid_size_array=self.cuboid_size_array,
                                           path_description=PathDescription(start_pos=staticpath,
                                                                            end_pos=staticpath,
                                                                            n_timesteps=1),
                                           is_falling=False)
            self.obstacles.append(obstacle)

    def generate_dynamic_obstacles(self):
        for i in range(self.n_obstacles_dynamic):
            path_start = self.generate_valid_position()
            path_end = self.generate_valid_position()
            n_steps = int(np.sqrt(np.sum((path_start - path_end)**2))*300)
            obstacle = ObstacleDescription(shape="sphere",
                                           sphere_size_array=self.sphere_size_array,
                                           cuboid_size_array=self.cuboid_size_array,
                                           path_description=PathDescription(start_pos=path_start,
                                                                            end_pos=path_end,
                                                                            n_timesteps=n_steps),
                                           is_falling=False)
            self.obstacles.append(obstacle)

    def generate_falling_obstacles(self):
        for i in range(self.n_obstacles_falling):
            startpos = self.generate_valid_position()
            startpos[2] = self.world_size[2]
            obstacle = ObstacleDescription(shape="cuboid",
                                           sphere_size_array=self.sphere_size_array,
                                           cuboid_size_array=self.cuboid_size_array,
                                           path_description=PathDescription(start_pos=startpos,
                                                                            end_pos=startpos,
                                                                            n_timesteps=1),
                                           is_falling=True)
            self.obstacles.append(obstacle)

    def generate_obstacles_pillar(self):
        for i in range(self.n_obstacles_pillar):
            startpos = self.generate_valid_position()
            startpos[2] = self.world_size[2] / 2
            obstacle = ObstacleDescription(shape="cuboid",
                                           sphere_size_array=self.sphere_size_array,
                                           cuboid_size_array=self.pillar_size_array,
                                           path_description=PathDescription(start_pos=startpos,
                                                                            end_pos=startpos,
                                                                            n_timesteps=1),
                                           is_falling=False)
            obstacle.geometric_description["xyz_dims"][2] = self.world_size[2] / 2
            self.obstacles.append(obstacle)

    def generate_obstacles_cuboid_floor(self):
        for i in range(self.n_obstacles_cuboid_floor):
            staticpath = self.generate_valid_position()
            obstacle = ObstacleDescription(shape="cuboid",
                                           sphere_size_array=self.sphere_size_array,
                                           cuboid_size_array=self.cuboid_size_array,
                                           path_description=PathDescription(start_pos=staticpath,
                                                                            end_pos=staticpath,
                                                                            n_timesteps=1),
                                           is_falling=False)
            obstacle.path[:, 2] = obstacle.geometric_description["xyz_dims"][2]
            self.obstacles.append(obstacle)

    def generate_obstacles_cuboid_ceiling(self):
        for i in range(self.n_obstacles_cuboid_ceiling):
            staticpath = self.generate_valid_position()
            obstacle = ObstacleDescription(shape="cuboid",
                                           sphere_size_array=self.sphere_size_array,
                                           cuboid_size_array=self.cuboid_size_array,
                                           path_description=PathDescription(start_pos=staticpath,
                                                                            end_pos=staticpath,
                                                                            n_timesteps=1),
                                           is_falling=False)
            obstacle.path[:, 2] = self.world_size[2] - obstacle.geometric_description["xyz_dims"][2]
            self.obstacles.append(obstacle)

    def randomize_object_start_positions(self):
        for obstacle in self.obstacles:
            if obstacle.max_position_index > 0:
                obstacle.current_position_index = np.random.randint(0, obstacle.max_position_index)
                obstacle.current_position = obstacle.path[obstacle.current_position_index, :]

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
    def __init__(self, shape, path_description, sphere_size_array, cuboid_size_array, is_falling):
        self.shape = shape
        self.geometric_description = None
        self.generate_geometric_description(sphere_size_array, cuboid_size_array)
        self.path_description = path_description
        if is_falling:
            self.path = path_description.generate_3d_path_gravity()
        else:
            self.path = path_description.generate_3d_path()
        self.current_position_index = 0
        self.max_position_index = self.path.shape[0] - 1
        self.current_position = self.path[0, :]
        self.move_forward = True
        self.is_falling = is_falling

    def generate_geometric_description(self, sphere_size_array, cuboid_size_array):
        """Generates a description of the geometric object in terms of its shape and size"""
        if self.shape == "sphere":
            r = np.random.choice(sphere_size_array)
            self.geometric_description = {"shape_type" : self.shape, "radius" : r}
        elif self.shape == "cuboid":
            xyz_dims = np.random.choice(a=cuboid_size_array, size = 3)
            self.geometric_description = {"shape_type" : self.shape, "xyz_dims" : xyz_dims}


    def update_position(self):
        if not self.is_falling:
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
        else:
            self.current_position_index += 1
            self.current_position_index = self.current_position_index % len(self.path)
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

    def generate_3d_path_gravity(self):
        g = 9.81
        start_point = np.array(self.start_pos)
        end_time = np.sqrt(2*self.start_pos[2] / g)

        t = np.linspace(0, end_time, int(end_time*240)).reshape(-1, 1)

        points = start_point - 0.5 * np.array([0., 0., 1.]) * g * t**2
        return points