import numpy as np

class Planet:
    # def next_timestep(self, delta_t, planets):
    #     self.new_position = self.position + self.velocity * delta_t

    #     a = np.array([0., 0., 0.])
    #     for planet in planets:
    #         if planet.name == self.name:
    #             continue
    #         rel_position = planet.position - self.position
    #         denominator = np.linalg.norm(rel_position, ord = 2)
    #         a += self.G * planet.mass * rel_position / denominator
        
    #     self.new_velocity = self.velocity + a * delta_t

    #     # print("----------------")
    #     # print(self.name)
    #     # print("v_0: {}, v_1: {}".format(self.velocity, self.new_velocity))
    #     # print("x_0: {}, x_1: {}".format(self.position, self.new_position))

    # def update_position_and_velocity(self):
    #     self.new_position, self.position = self.position, self.new_position
    #     self.new_velocity, self.velocity = self.velocity, self.new_velocity

    def __init__(self, mass, position, velocity, name):
        self.G = 6.67430 * 1E-11 # m^3 / kg * s^2
        self.mass = mass
        self.set_position(position)
        self.set_velocity(velocity)
        self.name = name

    @property
    def mass(self):
        return self.__mass
    
    @mass.setter
    def mass(self, value):
        if value < 0:
            self.__mass = 0
        else:
            self.__mass = value

    def set_position(self, vector):
        self.position = vector
    
    def set_velocity(self, vector):
        self.velocity = vector

    def __str__(self):
        return "name: " + str(self.name) + "\nmass: " + str(self.mass) + "\nposition: " + str(self.position) + "\nvelocity: " + str(self.velocity)