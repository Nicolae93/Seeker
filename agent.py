import numpy as np
import math
from config import Config as cf
import functions as fn

# Input parameters
# n     -> Number of steps 
# m     -> Number of Dimensions 
# beta  -> Power law index  % Note: 1 < beta < 2

def levy_flight_steps(beta, n=100000, m=2):
    
    num = math.gamma(1+beta)*np.sin(np.pi*beta/2); # used for Numerator 
    
    den = math.gamma((1+beta)/2)*beta*2**((beta-1)/2); # used for Denominator

    sigma_u = (num/den)**(1/beta); # Standard deviation
    
    u = np.random.normal(0, sigma_u, n*m)

    v = np.random.normal(0, 1, n*m)
    
    z = u / np.power(np.fabs(v), 1 / beta)

    return z


class Agent:
    def __init__(self,function=fn.eggholder,bounds=[[-512, 512],[-512, 512]],my_id=1,dim=2):
        self.__id = my_id
        self.__initial_position = []
        for i in range(dim):
            self.__initial_position.append(np.random.uniform(low=bounds[i][0],high=bounds[i][1]))
        self.__initial_fitness = function(self.__initial_position)
        self.__bestPosition = self.__initial_position
        self.__bestFitness = self.__initial_fitness
        
    def get_id(self):
        return self.__id
    
    def get_best_position(self):
        return self.__bestPosition
    
    def set_best_position(self, best_position):
        self.__bestPosition = best_position
        
    def get_best_fitness(self):
        return self.__bestFitness
    
    def set_best_fitness(self, best_fitness):
        self.__bestFitness = best_fitness
        
    def get_initial_position(self):
        return self.__initial_position

    def set_initial_position(self, position):
        self.__initial_position = position

    def get_initial_fitness(self):
        return self.__initial_fitness

    def set_initial_fitness(self, fitness):
        self.__initial_fitness = fitness

if __name__ == '__main__':
    print(levy_flight_steps(cf.get_beta()))