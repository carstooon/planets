import numpy as np
import planet
import math
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#carsten implementation
#eventuell eine virtuelle class implementieren von der verschiedene sim implementations erben koennen
class Simulation:
    
    def __init__(self, planets, timesteps = 10, delta_t = 0.01):        
        self.timesteps = timesteps
        self.delta_t = delta_t
        self.planets = planets
        self.Grav_const = 6.67430 * 1E-11 # m^3 / kg * s^2
        
    @property
    def timesteps(self):
        return self.__timesteps
    
    @timesteps.setter
    def timesteps(self, value):
        if value < 0:
            self.__timesteps = 0
        else:
            self.__timesteps = value


    @property
    def delta_t(self):
        return self.__delta_t
    
    @delta_t.setter
    def delta_t(self, value):
        if value < 0:
            self.__delta_t = 0
        else:
            self.__delta_t = value

    
    @property
    def Grav_const(self):
        return self.__Grav_const
    
    @Grav_const.setter
    def Grav_const(self, value):
        if value < 0:
            self.__Grav_const = 6.67430 * 1E-11 # m^3 / kg * s^2
        else:
            self.__Grav_const = value


    def propagate_planets(self):
        for planet in self.planets:
            planet.position = planet.position + planet.velocity * self.delta_t
        

    def calculate_acceleration(self):
        # calculates the acceleration by newtons gravity law as "a = F/m"
        # F = G * m1 * m2 * (r2-r1) / |r2 - r1|^3
        # keep in mind that r2 and r1 are vectors
        for i in range(len(self.planets)):
            a_i = np.array([0., 0., 0.])

            for j in range(0, len(self.planets)):
                # do not allow self-interaction
                if i == j:
                    continue;
                # print("Acceleration for planet{} and planet{}".format(i, j))
                rel_pos = self.planets[j].position - self.planets[i].position
                denominator = np.sqrt(rel_pos.dot(rel_pos))**3
                a_i += self.Grav_const * self.planets[j].mass * rel_pos / denominator
    
            # v_new = v_old + a * delta_t
            self.planets[i].velocity = self.planets[i].velocity + a_i * self.delta_t

    def calculate_energy(self):
        # E = E_kin + E_pot
        # E_kin = sum_i( 0.5 * m_i * v_i * v_i)
        E_kin = 0
        for planet in self.planets:
            # abs_v2 = planet.velocity.dot(planet.velocity)
            E_kin += 0.5 * planet.mass * planet.velocity.dot(planet.velocity)
        
        E_pot = 0
        for i in range(len(self.planets)):
            for j in range(i, len(self.planets)):
                if i == j:
                    continue
                rel_pos = self.planets[j].position - self.planets[i].position
                denominator = np.sqrt(rel_pos.dot(rel_pos))
                E_pot += -1. * self.Grav_const * self.planets[i].mass * self.planets[j].mass / denominator if denominator > 1e-30 else 0
        if self.timestep == 0:
            self.gauge_potential_energy = E_pot

        E_pot = E_pot - self.gauge_potential_energy
        return (E_kin + E_pot, E_kin, E_pot)

    def run_simulation(self):
        """
        Idea:
        1. Propagate planets according to their velocity
        2. Calculate the 3D-acceleration via newtons gravity laws per planet
        3. Apply the acceleration to the velocity vectors of the planets
        """
        print("Start simulation")

        list_energy = []
        list_energy_kinetic = []
        list_energy_potential = []
        list_timestep = []
        list_x_planet1 = []
        list_y_planet1 = []
        list_x_planet2 = []
        list_y_planet2 = []

        # for planet in self.planets:
        #     print(planet)

        for self.timestep in range(self.timesteps):
            if self.timestep % 1000 == 0:
                print("Time step {}".format(self.timestep))
            E, E_kin, E_pot = self.calculate_energy()
            
            list_timestep.append(self.timestep)
            list_energy.append(E)
            list_energy_kinetic.append(E_kin)
            list_energy_potential.append(E_pot)
            list_x_planet1.append(self.planets[0].position[0])
            list_y_planet1.append(self.planets[0].position[1])
            list_x_planet2.append(self.planets[1].position[0])
            list_y_planet2.append(self.planets[1].position[1])
            # print("E = {}, E_kin = {}, E_pot = {}".format(E, E_kin, E_pot))
            self.propagate_planets()
            self.calculate_acceleration()

        # for planet in self.planets:
        #     print(planet)

        df_energy = pd.DataFrame(data = {'timestep': list_timestep, 
                                         'energy': list_energy, 
                                         'kinetic_energy': list_energy_kinetic, 
                                         'potential_energy': list_energy_potential})


        sns.set_style("darkgrid")
        f, (ax1, ax2, ax3) = plt.subplots(3)
        f.set_size_inches(8, 11)

        sns.lineplot(x="timestep", y="energy", data=df_energy, ax=ax1)
        sns.lineplot(x="timestep", y="kinetic_energy", data=df_energy, ax=ax2)
        sns.lineplot(x="timestep", y="potential_energy", data=df_energy, ax=ax3)
        f.savefig("001_energy.png")

        df_position = pd.DataFrame(data= {'timestep': list_timestep,
                                            '1_position_x': list_x_planet1,
                                            '1_position_y': list_y_planet1,
                                            '2_position_x': list_x_planet2,
                                            '2_position_y': list_y_planet2})
        fig, ax1 = plt.subplots()
        sns.lineplot(x="1_position_x", y="1_position_y", data=df_position, ax=ax1, palette=['green'])
        sns.lineplot(x="2_position_x", y="2_position_y", data=df_position, ax=ax1, palette=['blue'])
        fig.savefig("002_position.png")

        df_position.to_csv("position.csv")
        