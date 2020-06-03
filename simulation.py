import numpy as np
import planet
import math
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import copy

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

        self.list_planets_positions = []

        for self.timestep in range(self.timesteps):
            if self.timestep % 1000 == 0:
                print("Time step {}".format(self.timestep))

            list_position = []
            for planet in self.planets:
                list_position.append(planet.position)
            self.list_planets_positions.append(list_position)
            
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

        self.save_dataframes()


    def save_dataframes(self):
        print("Save dataframes")
        self.df_energy = pd.DataFrame(data = {'timestep': self.list_timestep, 
                                         'energy': self.list_energy, 
                                         'kinetic_energy': self.list_energy_kinetic, 
                                         'potential_energy': self.list_energy_potential})

        ### SAVE POSITION
        data = []
        i = 0
        for timestep in self.list_planets_positions:
            data_timestep = [self.list_timestep[i]]
            for planet in timestep:
                data_timestep.append(planet[0])
                data_timestep.append(planet[1])
                data_timestep.append(planet[2])
            data.append(data_timestep)
            i += 1

        columns = ["timestep"]
        for i in range(len(self.planets)):
            columns.append("planet{}_x".format(i))
            columns.append("planet{}_y".format(i))
            columns.append("planet{}_z".format(i))
        self.df_position = pd.DataFrame(data=data, columns=columns)
        # print(self.df_position)



class RungeKutta4:

    def __init__(self, planets, delta_t = 0.01, precision = 0.01):        
        self.delta_t = delta_t
        self.planets = planets
        self.precision = precision
        self.Grav_const = 6.67430 * 1E-20 # m^3 / kg * s^2
        self.simulated_time = 0.


    @property
    def time(self):
        return self.__simulated_time

    @property
    def delta_t(self):
        return self.__delta_t
    
    @delta_t.setter
    def delta_t(self, value):
        if value < 0:
            self.__delta_t = 0.01
        else:
            self.__delta_t = value


    @property
    def precision(self):
        return self.__precision
    
    @precision.setter
    def precision(self, value):
        if value < 0:
            self.__precision = 0.01
        else:
            self.__precision = value

    
    @property
    def Grav_const(self):
        return self.__Grav_const
    
    @Grav_const.setter
    def Grav_const(self, value):
        if value < 0:
            self.__Grav_const = 6.67430 * 1E-20 # km^3 / kg * s^2
        else:
            self.__Grav_const = value

    def __str__(self):
        return "simulated time: " + str(self.simulated_time) + "\ndelta t: " + str(self.delta_t)


    #returns the d-th component of the force on the i-th body caused by the j-th body at the time t
    def fInternal(self, i, j, dim, planets):

        #in case that the force of a body on itself would be calculated
        if planets[i] is planets[j]: 
            return 0

        #evaluates the distance between the interacting bodies
        rel_position = planets[j].position - planets[i].position
        r2 = np.sum(rel_position**2)
        #r2 = np.linalg.norm(rel_position, ord = 2)

        #evaluates gravitational force in d-th dimension
        grav_force_ij = self.Grav_const * (planets[i].mass * planets[j].mass) * (planets[j].position[dim]-planets[i].position[dim]) / math.fabs(math.pow(math.sqrt(r2),3))
        return grav_force_ij


    #returns the d-th component of the total force on the i-th body caused by exteral sources at the time t
    def fExternal(self, i, dim, planet):
        return 0


    # returns the total energy of the system at the time t
    def Energy(self):

        E_tot = 0
        for planet_i in self.planets:

            #Evaluating the potential Energy    
            E_pot = 0
            for planet_j in self.planets:
            
                #preventing that E_pot of a body is evaluated in its own gravitational field
                if planet_i is planet_j: 
	                continue

                #evaluates the distance between the interacting bodies
                rel_position = planet_i.position - planet_j.position
                r2 = np.sum(rel_position**2)
                #r2 = rel_position.dot(rel_position)
                #r2 = np.linalg.norm(rel_position, ord = 2)

                #summing pot energies of two body interaction
                E_pot += -1. * self.Grav_const * planet_i.mass * planet_j.mass / math.sqrt(r2)
            

            #Evaluating the kinetic energy    
            v2 = np.sum(planet_i.velocity**2)

            #Evaluating the total Energy  
            E_tot += 0.5 * planet_i.mass *(v2) + E_pot

        return E_tot


    # returns the d-th component of the acceleration of the i-th body at the time t
    def CalcAcceleration(self, i, dim, planets):
        fInternal_tot = 0
        for j in range(len(planets)):
            fInternal_tot += self.fInternal(i, j, dim, planets)
       # print("body: " + str(i) + "  dim: " + str(dim) + "  a: " + str(fInternal_tot/planets[i].mass))
        return fInternal_tot/planets[i].mass
    

    

    # This function will update everything by integrating the differential equations
    def NextStep(self):  

        N = len(self.planets)
        dim = len(self.planets[0].position)

        k1 = np.zeros((N,dim))
        k2 = np.zeros((N,dim))
        k3 = np.zeros((N,dim))
        k4 = np.zeros((N,dim))

        w1 = np.zeros((N,dim))
        w2 = np.zeros((N,dim))
        w3 = np.zeros((N,dim))
        w4 = np.zeros((N,dim))

        planets_tmp = copy.deepcopy(self.planets) 
        
        for (body, dim), x in np.ndenumerate(w1):
            k1[body][dim] = self.delta_t * self.planets[body].velocity[dim]
            w1[body][dim] = self.delta_t * self.CalcAcceleration(body, dim, self.planets)

     #   for (body, dim), x in np.ndenumerate(w1):
            planets_tmp[body].position[dim] = self.planets[body].position[dim] + k1[body][dim]/2
            planets_tmp[body].velocity[dim] = self.planets[body].velocity[dim] + w1[body][dim]/2

    #    for (body, dim), x in np.ndenumerate(w2):
            k2[body][dim] = self.delta_t * planets_tmp[body].velocity[dim]
            w2[body][dim] = self.delta_t * self.CalcAcceleration(body, dim, planets_tmp)

    #    for (body, dim), x in np.ndenumerate(w2):
            planets_tmp[body].position[dim] = self.planets[body].position[dim] + k2[body][dim]/2
            planets_tmp[body].velocity[dim] = self.planets[body].velocity[dim] + w2[body][dim]/2

    #    for (body, dim), x in np.ndenumerate(w3):
            k3[body][dim] = self.delta_t * planets_tmp[body].velocity[dim]
            w3[body][dim] = self.delta_t * self.CalcAcceleration(body, dim, planets_tmp)

    #    for (body, dim), x in np.ndenumerate(w3):
            planets_tmp[body].position[dim] = self.planets[body].position[dim] + k3[body][dim]
            planets_tmp[body].velocity[dim] = self.planets[body].velocity[dim] + w3[body][dim]

    #    for (body, dim), x in np.ndenumerate(w4):
            k4[body][dim] = self.delta_t * planets_tmp[body].velocity[dim]
            w4[body][dim] = self.delta_t * self.CalcAcceleration(body, dim, planets_tmp)

     #   for (body, dim), x in np.ndenumerate(w3):
            self.planets[body].position[dim] += k1[body][dim]/6 + k2[body][dim]/3 + k3[body][dim]/3 + k4[body][dim]/6
            self.planets[body].velocity[dim] += w1[body][dim]/6 + w2[body][dim]/3 + w3[body][dim]/3 + w4[body][dim]/6

        self.simulated_time += self.delta_t






    #Methode1: 1vs2 steps
    def NextStepError1(self):

        planets_orig = copy.deepcopy(self.planets)
        planets_tmp1 = copy.deepcopy(self.planets)
        planets_tmp2 = copy.deepcopy(self.planets)

        simulated_time_orig = copy.copy(self.simulated_time)
        simulated_time_tmp1 = copy.copy(self.simulated_time)
        simulated_time_tmp2 = copy.copy(self.simulated_time)

        delta_t_orig = copy.copy(self.delta_t)
        delta_t_tmp1 = copy.copy(self.delta_t)
        delta_t_tmp2 = copy.copy(self.delta_t)/2.

        
        self.planets = planets_tmp1
        self.simulated_time = simulated_time_tmp1
        self.delta_t = delta_t_tmp1  
        self.NextStep()

        self.planets = planets_tmp2
        self.simulated_time = simulated_time_tmp2
        self.delta_t = delta_t_tmp2
        self.NextStep()
        self.NextStep()

        #distances between the bodies
        rel_distances = []
        for i in range(len(self.planets)):
                
            rel_position = planets_tmp1[i].position - planets_tmp2[i].position
            r2 = np.sum(rel_position**2)
            #r2 = np.linalg.norm(rel_position, ord = 2)
            rel_distances.append(math.sqrt(r2))

        #print(rel_distances)
        #If the distance between each body to itself is bigger than the precision the "NextError" function is evaluated again until the desired precision is reached.            
        passTest = True
        for dist in rel_distances:
            if(dist >  self.precision):
                passTest = False


        self.planets = planets_orig
        self.simulated_time = simulated_time_orig
        self.delta_t = delta_t_orig

        #evalution of the new coordinates and velocities. These are consistent with x_(n+1) and v_(n+1) from the instruction.    
        if(passTest):
            self.NextStep()
            self.delta_t *= 2 #Doubling the initial delta_t to test wether a bigger iteration step gives enough precise 

        else:
            self.delta_t /= 2 #delta_t is halved to improve precision
            self.NextStepError1()

        

        #End of 1 vs. 2 step method. The Programm returns to the while(1) loop in EJS.c



    #Starting point of the energy conservation criterion.
    def NextStepError2(self):

        planets_orig = copy.deepcopy(self.planets)
        planets_tmp1 = copy.deepcopy(self.planets)

        simulated_time_orig = copy.copy(self.simulated_time)
        simulated_time_tmp1 = copy.copy(self.simulated_time)

        delta_t_orig = copy.copy(self.delta_t)
        delta_t_tmp1 = copy.copy(self.delta_t)
        
        #Total energy of the system before our iteration step    
        energy_before = self.Energy()

        #updating t, x and v and storing them in ttmp1, xtmp1, vtmp1
        self.planets = planets_tmp1
        self.simulated_time = simulated_time_tmp1
        self.delta_t = delta_t_tmp1  
        self.NextStep()

        #Difference between the initial energy and energy after one step of iteration    
        delta_Energy = math.fabs(self.Energy() - energy_before)
        #print(energy_before)
        #print(delta_Energy)

        #Comparing the relative energy error with the precision
        if(math.fabs(delta_Energy / energy_before) <= self.precision):    
            self.delta_t *= 2 #Doubling the initial delta_t to test wether a bigger iteration step gives enough precise

        #in case of no energy conservation
        else:
            self.planets = planets_orig
            self.simulated_time = simulated_time_orig
            self.delta_t = delta_t_orig
            self.delta_t /= 2 #delta_t is halved to improve precision
            self.NextStepError2()


    def CreateDataFrame(self, time, posX_0, posY_0, posX_1, posY_1):
        print("Save dataframes")
        
        dataframe = pd.DataFrame(data= {'time':  time,
                                        '1_position_x': posX_0,
                                        '1_position_y': posY_0,
                                        '2_position_x': posX_1,
                                        '2_position_y': posY_1})
        return dataframe

    def PrintDataFrame(self, dataframe, filename):
        sns.set()
        sns.set_style("white")    
        fig, ax1 = plt.subplots()
        fig.set_size_inches(8,8)
        sns.scatterplot(x="1_position_x", y="1_position_y", data=dataframe, ax=ax1, palette='Reds')
        sns.scatterplot(x="2_position_x", y="2_position_y", hue="time", palette="Blues", data=dataframe, ax=ax1)
        #sns.scatterplot(x="2_position_x", y="2_position_y", hue="time", palette="Red", data=dataframe, ax=ax1)
        plt.xlabel('x [km]')
        plt.ylabel('y [km]')

        fig.savefig(filename)        



    def RunSimulation(self, run_time):

        list_sim_time     = []
        list_planet0_posX = []
        list_planet0_posY = []
        list_planet1_posX = []
        list_planet1_posY = []
        list_planet2_posX = []
        list_planet2_posY = []



        while(1):
            self.NextStepError2()

            #print(self.planets[0])
            #print(self.planets[2]) 
            #print(self)

            list_sim_time.append(self.simulated_time)
            list_planet0_posX.append(self.planets[0].position[0])
            list_planet0_posY.append(self.planets[0].position[1])
            list_planet1_posX.append(self.planets[1].position[0])
            list_planet1_posY.append(self.planets[1].position[1])
            #list_planet2_posX.append(self.planets[2].position[0])
            #list_planet2_posY.append(self.planets[2].position[1])



            #print(self.simulated_time/run_time)

            if(self.simulated_time>=run_time):
                break

        df_position = self.CreateDataFrame(list_sim_time,list_planet0_posX,list_planet0_posY,list_planet1_posX,list_planet1_posY)
        self.PrintDataFrame(df_position, "002_position.png")

