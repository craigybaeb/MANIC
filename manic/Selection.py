import numpy as np
import concurrent.futures

class Selection:
    def __init__(self, num_parents, target_class, population_size, predict_fn):
        self.num_parents = num_parents
        self.target_class = target_class
        self.population_size = population_size
        self.predict_fn = predict_fn
        self.parallel = False

        self.validate_self()
    
    def select_elites(self, population, fitness_scores):
      self.validate_population_and_fitness_scores(population, fitness_scores)
      
      elites = []
      num_elites = int(self.population_size / 10)  # Select top 10% as elites

      # Sort individuals based on fitness score
      sorted_indices = np.argsort(fitness_scores)
      elites_indices = sorted_indices[:num_elites]

      # Use parallel execution only if self.parallel is set to True
      if self.parallel:
          # Use ThreadPoolExecutor for parallel execution
          with concurrent.futures.ThreadPoolExecutor() as executor:
              elite_results = list(executor.map(self.is_elite, elites_indices, [population]*len(elites_indices)))

          # Add elite instances to the elites list
          for idx, is_elite_instance in zip(elites_indices, elite_results):
              if is_elite_instance:
                  elites.append(population[idx])
      else:
          # Run the selection in serial without parallelization
          for idx in elites_indices:
              elite_instance = population[idx]
              elite_class = self.predict_fn(elite_instance)
              if elite_class == self.target_class:
                  elites.append(elite_instance)

      return elites
    
    def is_elite(self, idx, population):
      elite_instance = population[idx]
      elite_class = self.predict_fn(elite_instance)
      return elite_class == self.target_class

    def select_parents(self, population, fitness_scores):
        self.validate_population_and_fitness_scores(population, fitness_scores)

        parents = []
        for _ in range(self.num_parents):
            idx = fitness_scores.index(min(fitness_scores))
            parents.append(population[idx])
            fitness_scores[idx] = float('inf')
        return parents
    
    def validate_population_and_fitness_scores(self, population, fitness_scores):
        if(len(population) == 0):
            raise ValueError(f"Population cannot be empty.")
        
        if(len(fitness_scores) == 0):
            raise ValueError(f"Fitness scores cannot be empty.")

        if(len(population) != len(fitness_scores)):
            raise ValueError(f"Size of population ({len(population)}) does not match size of fitness scores ({len(fitness_scores)}).")
        
        if(len(population) == 0):
            raise ValueError(f"Population cannot be empty.")
        
    def validate_self(self):
        if(self.num_parents < 2):
            raise ValueError("Minimum of 2 parents are required for selection.")
        
        if(self.population_size < 1):
            raise ValueError("Population size must be greater than 0.")
        
        if(self.predict_fn == None):
            raise ValueError("Predict function must be supplied.")
        
        if(self.parallel != True and self.parallel != False):
            raise ValueError("Parallel setting must be Boolean True or False.")
        
        if(not isinstance(self.target_class, int)):
            raise ValueError("Target class must be integer.")
        
 
        