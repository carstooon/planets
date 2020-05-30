import math
import numpy as np
class Planet:
    
    def __init__(self, mass, position, velocity, name):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.name = name
 #       self.radius = R

    @property
    def mass(self):
        return self.__mass
    
    @mass.setter
    def mass(self, value):
        if value < 0:
            self.__mass = 0
        else:
            self.__mass = value


    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, vector):
        self.__position = vector
    

    @property
    def velocity(self):
        return self.__velocity

    @velocity.setter
    def velocity(self, vector):
        self.__velocity = vector

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    def __str__(self):
        return "name: " + str(self.name) + "\nmass: " + str(self.mass) + "\nposition: " + str(self.position) + "\nvelocity: " + str(self.velocity) + "\nradius: " + str(np.sqrt(self.position.dot(self.position)))