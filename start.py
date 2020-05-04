import numpy as np

class Planet:
    def set_mass(self, mass):
        self.mass = mass
    def set_position(self, x, y, z):
        self.position = np.array([x, y, z])
    def set_velocity(self, vx, vy, vz):
        self.velocity = np.array([vx, vy, vz])
    def __init__(self, mass, x, y, z, vx, vy, vz):
        self.set_mass(mass)
        self.set_position(x, y, z)
        self.set_velocity(vx, vy, vz)

if __name__ == "__main__":
    planet1 = Planet(1, 1, 0, 0, 1, 0)
    # planet1.set_position(2, 3, 0)
