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
    
    position3 = np.array([0., 2., 0.])
    velocity3 = np.array([0., 0., 0.])
    planet3   = planet.Planet(20.0, position3, velocity3, "Altazar")

    planets2 = [planet1, planet2]
    planets3 = [planet1, planet2, planet3]

    number_timesteps = 100
    delta_t = 0.01

    sim = simulation.Simulation(planets2, number_timesteps, delta_t)
    sim.run_simulation()
