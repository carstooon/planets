import numpy as np
import planet
import simulation

if __name__ == "__main__":
    # position1 = np.array([0., 0., 0.])
    # velocity1 = np.array([0, 0., 0.])
    # planet1   = planet.Planet(1.989E30, position1, velocity1, "Finera")
    
    # position2 = np.array([1.5e8, 0., 0.])
    # velocity2 = np.array([0., 29.78, 0.])
    # planet2   = planet.Planet(5.972E24, position2, velocity2, "Genobi")

    position1 = np.array([ .491403347836458E+05, -3.632715592552171E+05, -1.049148558556447E+04])
    velocity1 = np.array([6.242516505718979E-3, 1.163260794114081E-2, -2.475374674040771E-4])
    planet1   = planet.Planet(1.988544E+30, position1, velocity1, "Sun")
    
    position2 = np.array([-1.418287679667581E+08, 4.126884716814923E+07, -1.125285256157373E+04])
    velocity2 = np.array([-8.830883522656004E+0, -2.868395352996171E+01, -1.085239827735510E-05])
    planet2   = planet.Planet(5.97219E+24, position2, velocity2, "Genobi")

    position3 = np.array([0., 2., 0.])
    velocity3 = np.array([0., 0., 0.])
    planet3   = planet.Planet(20.0, position3, velocity3, "Altazar")

    planets1 = [planet1]
    planets2 = [planet1, planet2]
    planets3 = [planet1, planet2, planet3]

    number_timesteps = 365
    delta_t = 1*24*60*60.

    sim = simulation.Simulation(planets2, number_timesteps, delta_t)
    sim.run_simulation()
    sim.save_dataframes()
    sim.print_plots()
