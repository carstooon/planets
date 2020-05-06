import numpy as np

class Planet:
    def set_mass(self, mass):
        self.mass = mass
    
    def next_timestep(self, delta_t, planets):
        self.new_position = self.position + self.velocity * delta_t

        a = np.array([0., 0., 0.])
        for planet in planets:
            if planet.name == self.name:
                continue
            rel_position = planet.position - self.position
            denominator = np.linalg.norm(rel_position, ord = 3)
            a += self.G * planet.mass * rel_position / denominator
        
        self.new_velocity = self.velocity + a * delta_t

        print("----------------")
        print(self.name)
        print("v_0: {}, v_1: {}".format(self.velocity, self.new_velocity))
        print("x_0: {}, x_1: {}".format(self.position, self.new_position))

    def update_position_and_velocity(self):
        self.new_position, self.position = self.position, self.new_position
        self.new_velocity, self.velocity = self.velocity, self.new_velocity

    def set_position(self, vector):
        self.position = vector
    
    def set_velocity(self, vector):
        self.velocity = vector

    def __init__(self, mass, position, velocity, name):
        self.G = 6.67430 * 1E-11 # m^3 / kg * s^2
        self.set_mass(mass)
        self.set_position(position)
        self.set_velocity(velocity)
        self.name = name

    def __str__(self):
        return "name: " + str(self.name) + "\nmass: " + str(self.mass) + "\nposition: " + str(self.position) + "\nvelocity: " + str(self.velocity)

if __name__ == "__main__":

    position1 = np.array([0., 0., 0.])
    velocity1 = np.array([0., 0., 0.])
    planet1   = Planet(1E10, position1, velocity1, "Finera")
    
    position2 = np.array([1., 0., 0.])
    velocity2 = np.array([0., 0., 0.])
    planet2   = Planet(1E10, position2, velocity2, "Genobi")

    # position3 = np.array([0., 2., 0.])
    # velocity3 = np.array([0., 0., 0.])
    # planet3   = Planet(20.0, position3, velocity3, "Altazar")

    planets = [planet1, planet2]

    number_timesteps = 100
    delta_t = 0.01

    for t_i in range(number_timesteps):
        for planet in planets:
            planet.next_timestep(delta_t, planets)
        for planet in planets:
            planet.update_position_and_velocity()
