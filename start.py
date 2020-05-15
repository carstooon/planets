import numpy as np
import planet
import simulation

if __name__ == "__main__":
    position1 = np.array([0., 0., 0.])
    velocity1 = np.array([0., 0., 0.])
    planet1   = planet.Planet(1E10, position1, velocity1, "Finera")
    
    position2 = np.array([1., 0., 0.])
    velocity2 = np.array([0., 0., 0.])
    planet2   = planet.Planet(1E10, position2, velocity2, "Genobi")
    planet2.mass = 10
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
