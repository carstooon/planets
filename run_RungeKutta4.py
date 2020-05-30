import numpy as np
import planet
import simulation

if __name__ == "__main__":

    # Epi_posi  = np.array([151422000.0, 0.0, 0.0]) # position in m relative to Saturn
    # Epi_velo  = np.array([0.0, 15829.70992, 0.0]) # in m/s
    # Epimetheus= planet.Planet(5.304E17, Epi_posi, Epi_velo, "Epimetheus") # mass in kg
    
    # Jan_posi = np.array([-151472000.0, 0.0, 0.0])
    # Jan_velo = np.array([0.0, -15827.09706, 0.0])
    # Janus    = planet.Planet(1.98E18, Jan_posi, Jan_velo, "Janus")

    # Sat_posi = np.array([0., 0., 0.])
    # Sat_velo = np.array([0., 0., 0.])
    # Saturn   = planet.Planet(5.683E26, Sat_posi, Sat_velo, "Saturn")

    Epi_posi  = np.array([151422.0, 0.0, 0.0]) # position in km relative to Saturn
    Epi_velo  = np.array([0.0, 1.*15.82970992, 0.0]) # in km/s
    Epimetheus= planet.Planet(5.304E17, Epi_posi, Epi_velo, "Epimetheus") # mass in tons
    
    Jan_posi = np.array([-151472.0, 0.0, 0.0])
    Jan_velo = np.array([0.0, -15.82709706, 0.0])
    Janus    = planet.Planet(1.98E18, Jan_posi, Jan_velo, "Janus")

    Sat_posi = np.array([0., 0., 0.])
    Sat_velo = np.array([0., 0., 0.])
    Saturn   = planet.Planet(5.683E26, Sat_posi, Sat_velo, "Saturn")



    Planets = [Epimetheus, Janus, Saturn]
    #Planets = [Epimetheus, Saturn]

    position1 = np.array([ .491403347836458E+05, -3.632715592552171E+05, -1.049148558556447E+04])
    velocity1 = np.array([6.242516505718979E-3, 1.163260794114081E-2, -2.475374674040771E-4])
    planet1   = planet.Planet(1.988544E+30, position1, velocity1, "Sun")
    
    position2 = np.array([-1.418287679667581E+08, 4.126884716814923E+07, -1.125285256157373E+04])
    velocity2 = np.array([-8.830883522656004E+0, -2.868395352996171E+01, -1.085239827735510E-05])
    planet2   = planet.Planet(5.97219E+24, position2, velocity2, "Earth")

    position3 = np.array([0., 2., 0.])
    velocity3 = np.array([0., 0., 0.])
    planet3   = planet.Planet(20.0, position3, velocity3, "Altazar")

    planets1 = [planet1]
    planets2 = [planet2, planet1]
    planets3 = [planet1, planet2, planet3]





    time_simulation = 3*365*24*60*60.0
    delta_t = 10
    precision = 1.E-6

    sim = simulation.RungeKutta4(Planets, delta_t, precision)
    energy_before = sim.Energy()
    radius_before = np.sqrt(sim.planets[0].position.dot(sim.planets[0].position))
    posiX_before = sim.planets[0].position[0]
    posiY_before = sim.planets[0].position[1]
    sim.RunSimulation(time_simulation)

    print(sim.Energy()/energy_before)
    print(np.sqrt(sim.planets[0].position.dot(sim.planets[0].position))/radius_before)
    print(sim.planets[0].position[0]/posiX_before)
    #print(sim.planets[0].position[1]/posiY_before)