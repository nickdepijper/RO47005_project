# Obstacle Class
import pybullet as p

class Obstacle:
    def __init__(self, object_id, radius, is_dynamic=False, movement_formula=None):
        self._object_id = object_id
        self._radius = radius
        self._is_dynamic = is_dynamic
        self._movement_formula = movement_formula
        self._position = None
        self._orientation = None

    def update(self):
        self._position, self._orientation = p.getBasePositionAndOrientation(self._object_id)
        if self._is_dynamic and self._movement_formula:
            self._position = self._movement_formula(self._position)

    # Getters
    def get_position(self):
        return self._position

    def get_orientation(self):
        return self._orientation

    def get_radius(self):
        return self._radius

    def is_dynamic(self):
        return self._is_dynamic

    def get_movement_formula(self):
        return self._movement_formula

    # Setters
    def set_position(self, position):
        self._position = position

    def set_orientation(self, orientation):
        self._orientation = orientation

    def set_radius(self, radius):
        self._radius = radius

    def set_dynamic(self, is_dynamic):
        self._is_dynamic = is_dynamic

    def set_movement_formula(self, movement_formula):
        self._movement_formula = movement_formula
