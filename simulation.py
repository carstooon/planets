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
            for j in range(0, len(self.planets)):
                if i == j:
                    continue
                # rel_pos = self.planets[j].position - self.planets[i].position
                rel_pos = self.planets[j].position - self.planets[i].position
                denominator = np.sqrt(rel_pos.dot(rel_pos))
                E_pot += self.Grav_const * self.planets[i].mass * self.planets[j].mass / denominator if denominator > 1e-30 else 0
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
        list_x_planet2 = []

        for planet in self.planets:
            print(planet)

        for i in range(self.timesteps):
            E, E_kin, E_pot = self.calculate_energy()
            
            list_timestep.append(i)
            list_energy.append(E)
            list_energy_kinetic.append(E_kin)
            list_energy_potential.append(E_pot)
            list_x_planet1.append(self.planets[0].position[0])
            list_x_planet2.append(self.planets[1].position[0])
            print("E = {}, E_kin = {}, E_pot = {}".format(E, E_kin, E_pot))
            self.propagate_planets()
            self.calculate_acceleration()

        for planet in self.planets:
            print(planet)

        df_energy = pd.DataFrame(data = {'timestep': list_timestep, 
                                         'energy': list_energy, 
                                         'kinetic_energy': list_energy_kinetic, 
                                         'potential_energy': list_energy_potential})

        f, (ax1, ax2, ax3) = plt.subplots(3)
        sns.lineplot(x="timestep", y="energy", data=df_energy, ax=ax1)
        sns.lineplot(x="timestep", y="kinetic_energy", data=df_energy, ax=ax2)
        sns.lineplot(x="timestep", y="potential_energy", data=df_energy, ax=ax3)
        f.savefig("energy.png")

        df_position_x = pd.DataFrame(data= {'timestep': list_timestep,
                                            '1_position_x': list_x_planet1,
                                            '2_position_x': list_x_planet2})
        f, (ax1, ax2) = plt.subplots(2)
        sns.lineplot(x="timestep", y="1_position_x", data=df_position_x, ax=ax1)
        sns.lineplot(x="timestep", y="2_position_x", data=df_position_x, ax=ax2)
        f.savefig("position_x.png")

#erbt dann von der virtuellen Klasse. planets grav_const etc muss dann entsprechend angepasst werden 
#kann definitiv noch besser gemacht werden. Habe erstmal den code nur uebersetzt 
# class runge_kutta_4:

    #returns the d-th component of the force on the i-th body caused by the j-th body at the time t
    # def fInternal(self, planet_i, planet_j, d, t):
  
    #     #in case that the force of a body on itself would be calculated
    #     if planet_i is planet_j: 
    #         return 0
     
    #     #evaluates the distance between the interacting bodies
    #     rel_position = planet_i.position - planet_j.position
    #     r2 += np.linalg.norm(rel_position, ord = 2)
             
    #     #evaluates gravitational force in d-th dimension
    #     grav_force_ij= self.Grav_const * planet_i.mass*planet_j.mass * (planet_j.position[d]-planet_i.position[d]) / math.fabs(math.pow(math.sqrt(r2),3))
    #     return grav_force_ij
    
    #returns the d-th component of the total force on the i-th body caused by exteral sources at the time t
    # def fExternal(self, i, d, t, planets):
    #     return 0.0;

    # returns the total energy of the system at the time t
    # def Energy(self, t, planets):
  
    #     E_tot = 0
    #     for planet_i in planets:

    #         #Evaluating the potential Energy    
    #         E_pot = 0
    #         for planet_j in planets:
    #             if planet_i == planet_j: #preventing that E_pot of a body is evaluated in its own gravitational field.
	#                 continue

    #             #evaluates the distance between the interacting bodies
    #             rel_position = planets[i].position - planets[j].position
    #             r2 += np.linalg.norm(rel_position, ord = 2)
                
    #             #summing pot energies of two body interaction
    #             E_pot += Grav_const * planet_i.mass * planet_j.mass / math.sqrt(r2);
    #         }
    
    #         #Evaluating the kinetic energy    
    #         v2 = np.sum(planet_i**2)

    #         #Evaluating the total Energy  
    #         E_tot += 0.5 * planet_i.mass *(v2) + E_pot
        
    #     return E_tot
    

    # returns the d-th component of the acceleration of the i-th body at the time t
    # def acceleration(self, i, d, t, planets):
    #     fInternal_tot = 0
    #     for planet_j in planets:
    #         fInternal_tot += fInternal(planets[i], planet_j, d, t)
    #     return fInternal_tot/planets[i].mass;
    # }



    # This function will update everything by integrating the differential equations
    # def NextStep(self, t, h, planets):  

    #     N = planets.len()
    #     dim = planets[0].position.len()

    #     k1 = np.zeros((N,dim))
    #     k2 = np.zeros((N,dim))
    #     k3 = np.zeros((N,dim))
    #     k4 = np.zeros((N,dim))

    #     w1 = np.zeros((N,dim))
    #     w2 = np.zeros((N,dim))
    #     w3 = np.zeros((N,dim))
    #     w4 = np.zeros((N,dim))

    #     posi_tmp = np.zeros((N,dim))
    #     velo_tmp = np.zeros((N,dim))

    #     velo_list = []
    #     for planet in planets:
    #         velo_list.append(planet.velocity)
 
    #     k1 = np.vstack(velo_list)
    #     k1 *= h

        

#        for(int n=0; n<N; n++){
#            for(int d=0; d<D; d++){
#                k1[n][d]=h*v[n][d];
#                w1[n][d]=h*acceleration( n, d, *t, x, v);
#            }
#        }
#      
#        for(int n=0; n<N; n++){
#            for(int d=0; d<D; d++){   
#                xtmp[n][d]=x[n][d]+k1[n][d]/2;
#                vtmp[n][d]=v[n][d]+w1[n][d]/2;
#            }
#        }
#  
#        for(int n=0; n<N; n++){
#            for(int d=0; d<D; d++){
#                k2[n][d]=h*vtmp[n][d];  
#                w2[n][d]=h*acceleration(n, d, *t+h/2, xtmp, vtmp);      
#            }
#        }
#  
#        for(int n=0; n<N; n++){
#            for(int d=0; d<D; d++){
#                xtmp[n][d]=x[n][d]+k2[n][d]/2;
#                vtmp[n][d]=v[n][d]+w2[n][d]/2;
#            }
#        }
#
#  
#        for(int n=0; n<N; n++){
#            for(int d=0; d<D; d++){
#                k3[n][d]=h*vtmp[n][d];
#                w3[n][d]=h*acceleration(n, d, *t+h/2, xtmp, vtmp);
#            }
#        }      
#      
#        for(int n=0; n<N; n++){
#            for(int d=0; d<D; d++){      
#                xtmp[n][d]=x[n][d]+k3[n][d];
#                vtmp[n][d]=v[n][d]+w3[n][d];
#            }
#        }
#
#        for(int n=0; n<N; n++){
#            for(int d=0; d<D; d++){
#                k4[n][d]=h*vtmp[n][d];
#                w4[n][d]=h*acceleration(n, d, *t+h, xtmp, vtmp);
#            }
#        }
#
#        # Final update of x_n and v_n to x_(n+1) and v_(n+1)
#        for(int n=0; n<N; n++){
#            for(int d=0; d<D; d++){
#                x[n][d]=x[n][d]+k1[n][d]/6+k2[n][d]/3+k3[n][d]/3+k4[n][d]/6;  
#                v[n][d]=v[n][d]+w1[n][d]/6+w2[n][d]/3+w3[n][d]/3+w4[n][d]/6;  
#            }
#        }    
#        # Updating t for the next step of our iteration
#         *t=*t+h;   
#
#    } 
