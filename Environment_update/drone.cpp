import pybullet as p

# Drone Class
class Drone:
    def __init__(self, object_id):
        self._object_id = object_id
        self._position = None
        self._orientation = None
        self._speed = 0
        self._acceleration = 0

    def update(self):
        self._position, self._orientation = p.getBasePositionAndOrientation(self._object_id)

    # Getters
    def get_position(self):
        return self._position

    def get_orientation(self):
        return self._orientation

    def get_speed(self):
        return self._speed

    def get_acceleration(self):
        return self._acceleration

    # Setters
    def set_position(self, position):
        self._position = position

    def set_orientation(self, orientation):
        self._orientation = orientation

    def set_speed(self, speed):
        self._speed = speed

    def set_acceleration(self, acceleration):
        self._acceleration = acceleration
