import numpy as np
#from config import Config as cf
import functions as fn

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
