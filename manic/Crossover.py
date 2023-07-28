import random
import numpy as np

#TODO extend code to support extra number of parents.
class Crossover():
    def __init__(self, crossover_method, num_parents, population_size):
        self.crossover = self.set_crossover_method(crossover_method)
        self.num_parents = num_parents
        self.population_size = population_size
        self.parallel = False

        self.validate_self()

    def set_crossover_method(self, crossover_method):
        if(crossover_method == "single_point"):
            return self.single_point_crossover
        elif(crossover_method == "uniform"):
            return self.single_point_crossover
        elif(crossover_method == "two_point"):
             return self.two_point_crossover
        else:
            return self.uniform_crossover
        
    def single_point_crossover(self, parents):
        offspring = []
        
        for i in range(self.population_size):
            parent1 = parents[i % self.num_parents]
            parent2 = parents[(i + 1) % self.num_parents]
            cut_point = random.randint(1, len(parent1))
            child = np.concatenate((parent1[:cut_point], parent2[cut_point:]))
            offspring.append(child)
        return offspring
    
    def two_point_crossover(self, parents):
        offspring = []
        
        for i in range(self.population_size):
            parent1 = parents[i % self.num_parents]
            parent2 = parents[(i + 1) % self.num_parents]
            
            # Two random cut points
            cut_point1 = random.randint(0, len(parent1) - 1)
            cut_point2 = random.randint(cut_point1 + 1, len(parent1))
            
            # Create the first child using genetic material from both parents
            child1 = np.concatenate((parent1[:cut_point1], parent2[cut_point1:cut_point2], parent1[cut_point2:]))
            
            # Create the second child using genetic material from both parents
            child2 = np.concatenate((parent2[:cut_point1], parent1[cut_point1:cut_point2], parent2[cut_point2:]))
            
            offspring.append(child1)
            offspring.append(child2)
        
        return offspring
    
    def uniform_crossover(self, parents):
        offspring = []
        
        for i in range(self.population_size):
            parent1 = parents[i % self.num_parents]
            parent2 = parents[(i + 1) % self.num_parents]
            child = []
            
            for j in range(len(parent1)):
                # Randomly select genetic material from either parent1 or parent2
                if random.random() < 0.5:
                    child.append(parent1[j])
                else:
                    child.append(parent2[j])
            
            offspring.append(child)
        
        return offspring
    
    def validate_self(self):
        if(self.population_size < 2):
            raise ValueError("Population size must be at least 2.")
        
        if(self.num_parents < 2):
            raise ValueError("Number of parents must be at least 2.")