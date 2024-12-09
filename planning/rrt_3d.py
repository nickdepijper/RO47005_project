import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline

def plot_sphere(center_x, center_y, center_z, r):
    # Create the sphere coordinates
    phi = np.linspace(0, np.pi, 8)  # Polar angle
    theta = np.linspace(0, 2 * np.pi, 8)  # Azimuthal angle
    phi, theta = np.meshgrid(phi, theta)

    x = r * np.sin(phi) * np.cos(theta) + center_x
    y = r * np.sin(phi) * np.sin(theta) + center_y
    z = r * np.cos(phi) + center_z

    return x, y, z


class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None


def distance(node1, node2):
    return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2 + (node1.z - node2.z) ** 2)


def sample_point(bounds, goal, goal_bias=0.1):
    if np.random.rand() < goal_bias:
        return goal.x, goal.y, goal.z
    return np.random.uniform(bounds[0], bounds[1]), \
           np.random.uniform(bounds[2], bounds[3]), \
           np.random.uniform(bounds[4], bounds[5])


def steer(from_node, to_point, step_size):
    direction = np.array([to_point[0] - from_node.x, to_point[1] - from_node.y, to_point[2] - from_node.z])
    length = np.linalg.norm(direction)
    direction = direction / length  # Normalize
    new_x = from_node.x + step_size * direction[0]
    new_y = from_node.y + step_size * direction[1]
    new_z = from_node.z + step_size * direction[2]
    new_node = Node(new_x, new_y, new_z)
    new_node.parent = from_node
    return new_node


def is_in_collision(node, obstacles):
    for obs in obstacles:
        if np.sqrt((node.x - obs[0])**2 + (node.y - obs[1])**2 + (node.z - obs[2])**2) < obs[3]:
            return True
    return False


def rrt_3d(start, goal, bounds, obstacles, max_iters, step_size):
    tree = [start]
    for _ in range(max_iters):
        rnd_point = sample_point(bounds, goal)
        nearest_node = min(tree, key=lambda n: distance(n, Node(*rnd_point)))
        new_node = steer(nearest_node, rnd_point, step_size)

        if not is_in_collision(new_node, obstacles):
            tree.append(new_node)

            if distance(new_node, goal) < step_size:
                goal.parent = new_node
                tree.append(goal)
                return tree
    return None


def extract_path(goal):
    path = []
    node = goal
    while node is not None:
        path.append([node.x, node.y, node.z])
        node = node.parent
    return path[::-1]


start = Node(0, 0, 0)
goal = Node(9, 9, 4)
bounds = [0, 10, 0, 10, 0, 5]  # x_min, x_max, y_min, y_max, z_min, z_max
obstacles = [[5, 5, 5, 1]]  # List of spheres [x, y, z, r]

for i in range(30):
    obstacles.append(list(np.random.rand(4) * [bounds[1], bounds[3], bounds[5], 0]))
    obstacles[i][3] = 0.2



def get_rrt_path_visual(start, goal, bounds, obstacles, max_iters, step_size):
    tree = rrt_3d(start, goal, bounds, obstacles, max_iters, step_size)
    if tree:
        path = extract_path(goal)
        print("Path found!")
        # Visualization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_zlim(bounds[4], bounds[5])

        for obs in obstacles:
            x, y, z = plot_sphere(obs[0], obs[1], obs[2], obs[3])
            ax.plot_surface(x, y, z, color='red', alpha=0.5, edgecolor='k')
            plot_sphere(obs[0], obs[1], obs[2], obs[3])

        for node in tree:
            if node.parent:
                ax.plot([node.x, node.parent.x], [node.y, node.parent.y], [node.z, node.parent.z], 'b')

        path = np.array(path)

        spline_x, spline_y, spline_z = interpolate_spline(path)

        # Sample the spline
        t_new = np.linspace(0, 1, 100)
        x_new = spline_x(t_new)
        y_new = spline_y(t_new)
        z_new = spline_z(t_new)

        ax.scatter(path[:, 0], path[:, 1], path[:, 2], color='red', label='Original Points')

        # Plot the interpolated path as a line
        ax.plot(x_new, y_new, z_new, color='g', label='Interpolated Path')

        plt.show()
    else:
        print("Path not found.")

def get_rrt_path(start, goal, bounds, obstacles, max_iters, step_size):
    tree = rrt_3d(start, goal, bounds, obstacles, max_iters, step_size)
    if tree:
        path = extract_path(goal)
        path = np.array(path)
        print("Path found!")
        print(path)
        return path
    else:
        print("Path not found.")


def interpolate_spline(path):
    """
    Interpolates a cubic spline through the given 3D path.

    Parameters:
        path (numpy.ndarray): An N x 3 array where each row is a point [x, y, z].

    Returns:
        tuple: Three CubicSpline objects for x(t), y(t), and z(t).
    """
    if path.shape[1] != 3:
        raise ValueError("Input path must be an N x 3 array.")

    # Compute parameter t based on cumulative distances along the path
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    t = np.concatenate(([0], np.cumsum(distances)))

    # Normalize t to a range from 0 to 1 for numerical stability
    t /= t[-1]

    # Extract x, y, z coordinates
    x, y, z = path[:, 0], path[:, 1], path[:, 2]

    # Create cubic splines for x(t), y(t), and z(t)
    spline_x = CubicSpline(t, x)
    spline_y = CubicSpline(t, y)
    spline_z = CubicSpline(t, z)

    return spline_x, spline_y, spline_z




max_iters = 3000
step_size = 0.5

get_rrt_path_visual(start, goal, bounds, obstacles, max_iters, step_size)
