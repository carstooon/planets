import numpy as np
import planet

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

    def run_simulation(self):
        print("Start simulation")

    # def next_timestep(self, delta_t, planets):
    #     self.new_position = self.position + self.velocity * delta_t

    #     a = np.array([0., 0., 0.])
    #     for planet in planets:
    #         if planet.name == self.name:
    #             continue
    #         rel_position = planet.position - self.position
    #         denominator = np.linalg.norm(rel_position, ord = 2)
    #         a += self.G * planet.mass * rel_position / denominator
        
    #     self.new_velocity = self.velocity + a * delta_t

    #     # print("----------------")
    #     # print(self.name)
    #     # print("v_0: {}, v_1: {}".format(self.velocity, self.new_velocity))
    #     # print("x_0: {}, x_1: {}".format(self.position, self.new_position))

    # def update_position_and_velocity(self):
    #     self.new_position, self.position = self.position, self.new_position
    #     self.new_velocity, self.velocity = self.velocity, self.new_velocity



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
