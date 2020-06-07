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
    
    def __init__(self, bodies, delta_t = 0.01):        
        self.delta_t = delta_t
        self.bodies = bodies
        self.Grav_const = 6.67430 * 1E-20 # m^3 / kg * s^2
        self.simulated_time = 0.

        self.list_energy = []
        self.list_energy_kinetic = []
        self.list_energy_potential = []
        self.list_time_simulated = []

        self.list_bodies_positions = []
        self.list_bodies_velocity  = []

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
    def bodies(self):
        return self.__bodies
    
    @bodies.setter
    def bodies(self, value):
        self.__bodies = value


    @property
    def Grav_const(self):
        return self.__Grav_const
    
    @Grav_const.setter
    def Grav_const(self, value):
        if value < 0:
            self.__Grav_const = 6.6738480 * 1E-20 # km^3 / kg * s^2
        else:
            self.__Grav_const = value

    @property
    def simulated_time(self):
        return self.__simulated_time

    @simulated_time.setter
    def simulated_time(self, value):
        self.__simulated_time = value



    def __str__(self):
        return "simulated time: " + str(self.simulated_time) + "\ndelta t: " + str(self.delta_t)


    # returns the total energy of the system at the time t
    def SystemEnergy(self):

        E_tot = 0
        for body_i in self.bodies:

            #Evaluating the potential Energy    
            E_pot = 0
            for body_j in self.bodies:
            
                #preventing that E_pot of a body is evaluated in its own gravitational field
                if body_i is body_j: 
	                continue

                #evaluates the distance between the interacting bodies
                rel_position = body_i.position - body_j.position
                r2 = np.sum(rel_position**2)
                #r2 = rel_position.dot(rel_position)
                #r2 = np.linalg.norm(rel_position, ord = 2)

                #summing pot energies of two body interaction
                E_pot += -1. * self.Grav_const * body_i.mass * body_j.mass / math.sqrt(r2)
            

            #Evaluating the kinetic energy    
            v2 = np.sum(body_i.velocity**2)

            #Evaluating the total Energy  
            E_tot += 0.5 * body_i.mass *(v2) + E_pot

        return E_tot


    # returns the kinetic energy of body i 
    def KineticEnergy(self, body_i):

        #Evaluating the kinetic energy    
        v2 = np.sum(body_i.velocity**2)  
        E_kin = 0.5 * body_i.mass *(v2)

        return E_kin


    # returns the potential energy of body i
    def PotentialEnergy(self, body_i):

        #Evaluating the potential Energy    
        E_pot = 0
        for body_j in self.bodies:
            
            #preventing that E_pot of a body is evaluated in its own gravitational field
            if body_i is body_j: 
	            continue

            #evaluates the distance between the interacting bodies
            rel_position = body_i.position - body_j.position
            r2 = np.sum(rel_position**2)
                
            #summing pot energies of two body interaction
            E_pot += -1. * self.Grav_const * body_i.mass * body_j.mass / math.sqrt(r2)

        return E_pot

    
    def save_dataframes(self):
        print("Save dataframes")
        self.df_energy = pd.DataFrame(data = {'time': self.list_time_simulated, 
                                              'energy': self.list_energy})

        ### SAVE POSITION
        data = []
        for timestep_i in range(len(self.list_bodies_positions)):
            data_timestep = [self.list_time_simulated[timestep_i]]
            data_timestep.append(self.list_energy[timestep_i])
            #data_timestep.append(self.list_energy_kinetic[timestep_i])
            #data_timestep.append(self.list_energy_potential[timestep_i])
            for planet_i in range(len(self.list_bodies_positions[timestep_i])):
                data_timestep.append(self.bodies[planet_i].mass)
                data_timestep.append(self.list_bodies_positions[timestep_i][planet_i][0])
                data_timestep.append(self.list_bodies_positions[timestep_i][planet_i][1])
                data_timestep.append(self.list_bodies_positions[timestep_i][planet_i][2])
                data_timestep.append(self.list_bodies_velocity[timestep_i][planet_i][2])
                data_timestep.append(self.list_bodies_velocity[timestep_i][planet_i][2])
                data_timestep.append(self.list_bodies_velocity[timestep_i][planet_i][2])
            data.append(data_timestep)


        #columns = ["timestep", "energy", "kinetic energy", "potential energy"]
        columns = ["timestep", "energy"]
        for i in range(len(self.bodies)):
            columns.append("planet{}_m".format(i))
            columns.append("planet{}_x".format(i))
            columns.append("planet{}_y".format(i))
            columns.append("planet{}_z".format(i))
            columns.append("planet{}_vx".format(i))
            columns.append("planet{}_vy".format(i))
            columns.append("planet{}_vz".format(i))
        self.df = pd.DataFrame(data=data, columns=columns)

        self.df.to_csv("output.csv")
        self.df.to_pickle("output.pkl")
        




class EulerLeapfrog(Simulation):

    def __init__(self, bodies, timesteps = 10, delta_t = 0.01):        

        super().__init__(bodies, delta_t)

        self.timesteps = timesteps
         
    @property
    def timesteps(self):
        return self.__timesteps
    
    @timesteps.setter
    def timesteps(self, value):
        if value < 0:
            self.__timesteps = 0
        else:
            self.__timesteps = value


    def propagate_bodies(self, factor = 1):
        for planet in self.bodies:
            planet.position = planet.position + planet.velocity * factor * self.delta_t
        

    def calculate_acceleration(self):
        # calculates the acceleration by newtons gravity law as "a = F/m"
        # F = G * m1 * m2 * (r2-r1) / |r2 - r1|^3
        # keep in mind that r2 and r1 are vectors
        for i in range(len(self.bodies)):
            a_i = np.array([0., 0., 0.])

            for j in range(0, len(self.bodies)):
                # do not allow self-interaction
                if i == j:
                    continue;
                # print("Acceleration for planet{} and planet{}".format(i, j))
                rel_pos = self.bodies[j].position - self.bodies[i].position
                abs_r = np.sqrt(rel_pos.dot(rel_pos))
                a_i += self.Grav_const * self.bodies[j].mass * rel_pos / abs_r**3
    
            # v_new = v_old + a * delta_t
            self.bodies[i].velocity = self.bodies[i].velocity + a_i * self.delta_t

    def calculate_energy(self):
        # E = E_kin + E_pot
        # E_kin = sum_i( 0.5 * m_i * v_i * v_i)
        # E_pot = G * m1 * m2 / r
        E_kin = 0
        for planet in self.bodies:
            E_kin += 0.5 * planet.mass * planet.velocity.dot(planet.velocity)
        
        E_pot = 0
        for i in range(len(self.bodies)):
            for j in range(i, len(self.bodies)):
                if i == j:
                    continue
                rel_pos = self.bodies[j].position - self.bodies[i].position
                r = np.sqrt(rel_pos.dot(rel_pos))
                E_pot += -1. * self.Grav_const * self.bodies[i].mass * self.bodies[j].mass / r if r > 1e-30 else 0

        return (E_kin + E_pot, E_kin, E_pot)

    def run_simulation(self):
        """
        Idea:
        1. Propagate bodies according to their velocity
        2. Calculate the 3D-acceleration via newtons gravity laws per planet
        3. Apply the acceleration to the velocity vectors of the bodies
        """
        print("Start simulation")

        for timestep in range(self.timesteps):
            if timestep % 1000 == 0:
                print("Time step {}".format(timestep))


            ##### ENERGY CALCULATION
            E, E_kin, E_pot = self.calculate_energy()

            list_position = []
            for planet in self.bodies:
                list_position.append(planet.position)
            self.list_bodies_positions.append(list_position)
            list_velocity = []
            for planet in self.bodies:
                list_velocity.append(planet.velocity)
            self.list_bodies_velocity.append(list_velocity)
            
            self.list_time_simulated.append(self.simulated_time)
            self.list_energy.append(E)
            self.list_energy_kinetic.append(E_kin)
            self.list_energy_potential.append(E_pot)

            ##### LEAPFROG INTEGRATION
            self.propagate_bodies(0.5)
            self.calculate_acceleration()
            self.propagate_bodies(0.5)
            self.simulated_time = timestep*self.delta_t

        self.save_dataframes()



class RungeKutta4(Simulation):

    def __init__(self, bodies, delta_t = 0.01, precision = 0.01):   

        super().__init__(bodies, delta_t)

        self.precision = precision
        

    @property
    def precision(self):
        return self.__precision
    
    @precision.setter
    def precision(self, value):
        if value < 0:
            self.__precision = 0.01
        else:
            self.__precision = value



    #returns the d-th component of the force on the i-th body caused by the j-th body at the time t
    def fInternal(self, i, j, dim, bodies):

        #in case that the force of a body on itself would be calculated
        if bodies[i] is bodies[j]: 
            return 0

        #evaluates the distance between the interacting bodies
        rel_position = bodies[j].position - bodies[i].position
        r2 = np.sum(rel_position**2)

        #evaluates gravitational force in d-th dimension
        grav_force_ij = self.Grav_const * (bodies[i].mass * bodies[j].mass) * (bodies[j].position[dim]-bodies[i].position[dim]) / math.fabs(math.pow(math.sqrt(r2),3))

        return grav_force_ij


    #returns the d-th component of the force on the i-th body caused by the j-th body at the time t
    def fInternal3D(self, i, j, bodies):

        #in case that the force of a body on itself would be calculated
        if bodies[i] is bodies[j]: 
            return 0

        #evaluates the distance between the interacting bodies
        rel_position = bodies[j].position - bodies[i].position
        r2 = np.sum(rel_position**2)
        #r2 = np.linalg.norm(rel_position, ord = 2)

        grav_force_ij = np.empty(len(self.bodies[0].position))
        #evaluates gravitational force in d-th dimension
        for dim in range(len(self.bodies[0].position)): 
            grav_force_ij[dim] = (self.Grav_const * (bodies[i].mass * bodies[j].mass) * (bodies[j].position[dim]-bodies[i].position[dim]) / math.fabs(math.pow(math.sqrt(r2),3)))

        return grav_force_ij


    #returns the d-th component of the total force on the i-th body caused by exteral sources at the time t
    def fExternal(self, i, dim, bodies):
        return 0

    #returns the d-th component of the total force on the i-th body caused by exteral sources at the time t
    def fExternal3D(self, i, dim, bodies):
        f_external = np.zeros(len(self.bodies[0].position))
        
        return f_external

    # returns the d-th component of the acceleration of the i-th body at the time t
    def CalcAcceleration(self, i, dim, bodies):
        fInternal_tot = 0
        for j in range(len(bodies)):
            fInternal_tot += self.fInternal(i, j, dim, bodies)
       # print("body: " + str(i) + "  dim: " + str(dim) + "  a: " + str(fInternal_tot/bodies[i].mass))
        return fInternal_tot/bodies[i].mass
    

    # returns the acceleration of the i-th body at the time t
    def CalcAcceleration3D(self, i, bodies):
        fInternal_tot = np.zeros(len(self.bodies[0].position))

        for j in range(len(bodies)):
            fInternal_tot += self.fInternal3D(i, j, bodies)
            
        return fInternal_tot / bodies[i].mass
    
    

    # This function will update everything by integrating the differential equations
    def NextStep(self):  

        N = len(self.bodies)
        dim = len(self.bodies[0].position)

        k1 = np.zeros((N,dim))
        k2 = np.zeros((N,dim))
        k3 = np.zeros((N,dim))
        k4 = np.zeros((N,dim))

        w1 = np.zeros((N,dim))
        w2 = np.zeros((N,dim))
        w3 = np.zeros((N,dim))
        w4 = np.zeros((N,dim))

        bodies_tmp = copy.deepcopy(self.bodies) 
        
        for (body, dim), x in np.ndenumerate(w1):
            k1[body][dim] = self.delta_t * self.bodies[body].velocity[dim]
            w1[body][dim] = self.delta_t * self.CalcAcceleration(body, dim, self.bodies)

    #   for (body, dim), x in np.ndenumerate(w1):
            bodies_tmp[body].position[dim] = self.bodies[body].position[dim] + k1[body][dim]/2
            bodies_tmp[body].velocity[dim] = self.bodies[body].velocity[dim] + w1[body][dim]/2

    #    for (body, dim), x in np.ndenumerate(w2):
            k2[body][dim] = self.delta_t * bodies_tmp[body].velocity[dim]
            w2[body][dim] = self.delta_t * self.CalcAcceleration(body, dim, bodies_tmp)

    #    for (body, dim), x in np.ndenumerate(w2):
            bodies_tmp[body].position[dim] = self.bodies[body].position[dim] + k2[body][dim]/2
            bodies_tmp[body].velocity[dim] = self.bodies[body].velocity[dim] + w2[body][dim]/2

    #    for (body, dim), x in np.ndenumerate(w3):
            k3[body][dim] = self.delta_t * bodies_tmp[body].velocity[dim]
            w3[body][dim] = self.delta_t * self.CalcAcceleration(body, dim, bodies_tmp)

    #    for (body, dim), x in np.ndenumerate(w3):
            bodies_tmp[body].position[dim] = self.bodies[body].position[dim] + k3[body][dim]
            bodies_tmp[body].velocity[dim] = self.bodies[body].velocity[dim] + w3[body][dim]

    #    for (body, dim), x in np.ndenumerate(w4):
            k4[body][dim] = self.delta_t * bodies_tmp[body].velocity[dim]
            w4[body][dim] = self.delta_t * self.CalcAcceleration(body, dim, bodies_tmp)

     #   for (body, dim), x in np.ndenumerate(w3):
            self.bodies[body].position[dim] += k1[body][dim]/6 + k2[body][dim]/3 + k3[body][dim]/3 + k4[body][dim]/6
            self.bodies[body].velocity[dim] += w1[body][dim]/6 + w2[body][dim]/3 + w3[body][dim]/3 + w4[body][dim]/6

        self.simulated_time += self.delta_t


    # This function will update everything by integrating the differential equations
    def NextStep3D(self):  

        N = len(self.bodies)
        dim = len(self.bodies[0].position)

        k1 = np.zeros((N,dim))
        k2 = np.zeros((N,dim))
        k3 = np.zeros((N,dim))
        k4 = np.zeros((N,dim))

        w1 = np.zeros((N,dim))
        w2 = np.zeros((N,dim))
        w3 = np.zeros((N,dim))
        w4 = np.zeros((N,dim))

        bodies_tmp = copy.deepcopy(self.bodies) 
        
        for body in range(len(self.bodies)):
            k1[body] = self.delta_t * self.bodies[body].velocity
            w1[body] = self.delta_t * self.CalcAcceleration3D(body, self.bodies)

            bodies_tmp[body].position = self.bodies[body].position + k1[body]/2
            bodies_tmp[body].velocity = self.bodies[body].velocity + w1[body]/2

            k2[body] = self.delta_t * bodies_tmp[body].velocity
            w2[body] = self.delta_t * self.CalcAcceleration3D(body, bodies_tmp)

            bodies_tmp[body].position = self.bodies[body].position + k2[body]/2
            bodies_tmp[body].velocity = self.bodies[body].velocity + w2[body]/2

            k3[body] = self.delta_t * bodies_tmp[body].velocity
            w3[body] = self.delta_t * self.CalcAcceleration3D(body, bodies_tmp)

            bodies_tmp[body].position = self.bodies[body].position + k3[body]
            bodies_tmp[body].velocity = self.bodies[body].velocity + w3[body]

            k4[body] = self.delta_t * bodies_tmp[body].velocity
            w4[body] = self.delta_t * self.CalcAcceleration3D(body, bodies_tmp)

            self.bodies[body].position += k1[body]/6 + k2[body]/3 + k3[body]/3 + k4[body]/6
            self.bodies[body].velocity += w1[body]/6 + w2[body]/3 + w3[body]/3 + w4[body]/6

        self.simulated_time += self.delta_t




    #Methode1: 1vs2 steps
    def NextStepError1(self):

        bodies_orig = copy.deepcopy(self.bodies)
        bodies_tmp  = copy.deepcopy(self.bodies)
        bodies_tmp2 = copy.deepcopy(self.bodies)

        simulated_time_orig = copy.deepcopy(self.simulated_time)
        simulated_time_tmp  = copy.deepcopy(self.simulated_time)

        delta_t_orig = copy.deepcopy(self.delta_t)
        delta_t_tmp  = copy.deepcopy(self.delta_t)


        self.bodies = bodies_tmp2
        self.delta_t /= 2.  
        self.NextStep()
        self.NextStep()

        self.bodies = bodies_tmp
        self.simulated_time = simulated_time_tmp
        self.delta_t = delta_t_tmp
        self.NextStep()

        #distances between the bodies
        passTest = True
        for i in range(len(self.bodies)):
            radius = math.sqrt(np.sum(bodies_orig[i].position**2)) 

            rel_position = bodies_tmp2[i].position - bodies_tmp[i].position
            r2 = np.sum(rel_position**2)
            #r2 = np.linalg.norm(rel_position, ord = 2)
            rel_distance = math.sqrt(r2)

            #If the distance between each body to itself is bigger than the precision the "NextError" function is evaluated again until the desired precision is reached.            
            if(rel_distance > self.precision):
                passTest = False
                

        #evalution of the new coordinates and velocities. These are consistent with x_(n+1) and v_(n+1) from the instruction.    
        if(passTest):
            self.delta_t = delta_t_orig * 2 #Doubling the initial delta_t to test wether a bigger iteration step gives enough precise 

        else:
            self.bodies = bodies_orig
            self.simulated_time = simulated_time_orig
            self.delta_t = delta_t_orig/2. #delta_t is halved to improve precision
            self.NextStepError1()

        

        #End of 1 vs. 2 step method. The Programm returns to the while(1) loop in EJS.c



    #Starting point of the energy conservation criterion.
    def NextStepError2(self):

        bodies_orig = copy.deepcopy(self.bodies)

        simulated_time_orig = copy.deepcopy(self.simulated_time)
        delta_t_orig = copy.deepcopy(self.delta_t)
        
        #Total energy of the system before our iteration step    
        energy_before = self.SystemEnergy()

        #updating t, x and v and storing them in ttmp1, xtmp1, vtmp1
        self.NextStep3D()

        #Difference between the initial energy and energy after one step of iteration    
        delta_Energy = math.fabs(self.SystemEnergy() - energy_before)

        rel_energy_change = math.fabs(delta_Energy / energy_before) 
        #Comparing the relative energy error with the precision
        if(rel_energy_change <= self.precision):    
            self.delta_t *= 2 #Doubling the initial delta_t to test wether a bigger iteration step gives enough precise

        #elif(rel_energy_change <= self.precision):    
        #    pass 

        #in case of no energy conservation
        else:
            self.bodies = bodies_orig
            self.simulated_time = simulated_time_orig
            self.delta_t = delta_t_orig/2. #delta_t is halved to improve precision
            self.NextStepError2()


    def RunSimulation(self, run_time):

        passed = False
        while(1):
            self.NextStepError2()

            #print(self.bodies[0])
            #print(self.bodies[2]) 
            #print(self)

            #print(self.simulated_time/run_time)

            #rel_position = self.bodies[0].position - self.bodies[1].position
            #r2 = np.sum(rel_position**2)     
            #rel_distance = math.sqrt(r2)


            #if(rel_distance < 30000):
            #    print(rel_distance)
            #    print(self.simulated_time)

            self.list_time_simulated.append(self.simulated_time)
            self.list_energy.append(self.SystemEnergy)

            list_position = []
            for planet in self.bodies:
                list_position.append(planet.position)
            self.list_bodies_positions.append(list_position)
            list_velocity = []
            for planet in self.bodies:
                list_velocity.append(planet.velocity)
            self.list_bodies_velocity.append(list_velocity)
            
            self.save_dataframes()


            
            if(self.bodies[0].GetDistanceToOrigin() > self.bodies[1].GetDistanceToOrigin() and passed == False):
                print("Epimetheus passed")
                print(self.simulated_time/60./60./24./365.)
                passed = True

            if(self.bodies[0].GetDistanceToOrigin() < self.bodies[1].GetDistanceToOrigin() and passed == True):
                print("Janus passed")
                print(self.simulated_time/60./60./24./365.)
                passed = False

            if(self.simulated_time>=run_time):
                break

