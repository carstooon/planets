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
        self.Grav_const = 6.67430 * 1E-20 # m^3 / kg * s^2
        
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
            self.__Grav_const = 6.67430 * 1E-20 # km^3 / kg * s^2
        else:
            self.__Grav_const = value


    def propagate_planets(self, factor = 1):
        for planet in self.planets:
            planet.position = planet.position + planet.velocity * factor * self.delta_t
        

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
                abs_r = np.sqrt(rel_pos.dot(rel_pos))
                a_i += self.Grav_const * self.planets[j].mass * rel_pos / abs_r**3
    
            # v_new = v_old + a * delta_t
            self.planets[i].velocity = self.planets[i].velocity + a_i * self.delta_t

    def calculate_energy(self):
        # E = E_kin + E_pot
        # E_kin = sum_i( 0.5 * m_i * v_i * v_i)
        # E_pot = G * m1 * m2 / r
        E_kin = 0
        for planet in self.planets:
            E_kin += 0.5 * planet.mass * planet.velocity.dot(planet.velocity)
        
        E_pot = 0
        for i in range(len(self.planets)):
            for j in range(i, len(self.planets)):
                if i == j:
                    continue
                rel_pos = self.planets[j].position - self.planets[i].position
                r = np.sqrt(rel_pos.dot(rel_pos))
                E_pot += -1. * self.Grav_const * self.planets[i].mass * self.planets[j].mass / r if r > 1e-30 else 0

        return (E_kin + E_pot, E_kin, E_pot)

    def run_simulation(self):
        """
        Idea:
        1. Propagate planets according to their velocity
        2. Calculate the 3D-acceleration via newtons gravity laws per planet
        3. Apply the acceleration to the velocity vectors of the planets
        """
        print("Start simulation")

        self.list_energy = []
        self.list_energy_kinetic = []
        self.list_energy_potential = []
        self.list_timestep = []
        self.list_x_planet1 = []
        self.list_y_planet1 = []
        self.list_x_planet2 = []
        self.list_y_planet2 = []


        for self.timestep in range(self.timesteps):
            if self.timestep % 1000 == 0:
                print("Time step {}".format(self.timestep))

            self.list_x_planet1.append(self.planets[0].position[0])
            self.list_y_planet1.append(self.planets[0].position[1])
            self.list_x_planet2.append(self.planets[1].position[0])
            self.list_y_planet2.append(self.planets[1].position[1])            

            ##### LEAPFROG INTEGRATION
            self.propagate_planets(0.5)
            self.calculate_acceleration()
            self.propagate_planets(0.5)

            ##### ENERGY CALCULATION
            E, E_kin, E_pot = self.calculate_energy()
            
            self.list_timestep.append(self.timestep)
            self.list_energy.append(E)
            self.list_energy_kinetic.append(E_kin)
            self.list_energy_potential.append(E_pot)


    def save_dataframes(self):
        print("Save dataframes")
        self.df_energy = pd.DataFrame(data = {'timestep': self.list_timestep, 
                                         'energy': self.list_energy, 
                                         'kinetic_energy': self.list_energy_kinetic, 
                                         'potential_energy': self.list_energy_potential})

        self.df_position = pd.DataFrame(data= {'timestep':  self.list_timestep,
                                            '1_position_x': self.list_x_planet1,
                                            '1_position_y': self.list_y_planet1,
                                            '2_position_x': self.list_x_planet2,
                                            '2_position_y': self.list_y_planet2})

    def print_energy(self, filename="001_energy.png"):
        print("Print Plots")
        sns.set()
        sns.set_style("white")
        f, (ax1, ax2, ax3) = plt.subplots(3)
        f.set_size_inches(8, 11)

        sns.lineplot(x="timestep", y="energy", data=self.df_energy, ax=ax1)
        sns.lineplot(x="timestep", y="kinetic_energy", data=self.df_energy, ax=ax2)
        sns.lineplot(x="timestep", y="potential_energy", data=self.df_energy, ax=ax3)
        ax1.set_ylabel('energy')
        ax2.set_ylabel('kinetic energy')
        ax3.set_ylabel('potential energy')
        ax1.set_xlabel('')
        ax2.set_xlabel('')
        ax3.set_xlabel('t [d]')
        
        f.savefig(filename)

    def print_position(self, filename="002_position.png"):
        sns.set()
        sns.set_style("white")    
        fig, ax1 = plt.subplots()
        fig.set_size_inches(8,8)
        sns.scatterplot(x="1_position_x", y="1_position_y", data=self.df_position, ax=ax1, palette='green')
        sns.scatterplot(x="2_position_x", y="2_position_y", hue="timestep", palette="Blues", data=self.df_position, ax=ax1)
        plt.xlabel('x [km]')
        plt.ylabel('y [km]')

        fig.savefig(filename)        