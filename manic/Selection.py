import numpy as np

class Selection:
    def __init__(self, num_parents, target_class, population_size, predict_fn):
        self.num_parents = num_parents
        self.target_class = target_class
        self.population_size = population_size
        self.predict_fn = predict_fn
    
    def select_elites(self, population, fitness_scores):
      elites = []
      num_elites = int(self.population_size / 10)  # Select top 10% as elites

      # Sort individuals based on fitness score
      sorted_indices = np.argsort(fitness_scores)
      elites_indices = sorted_indices[:num_elites]

      for idx in elites_indices:
          elite_instance = population[idx]
          elite_class = self.predict_fn(elite_instance)
          if elite_class == self.target_class:
              elites.append(elite_instance)

      return elites
    
    def select_parents(self, population, fitness_scores):
        parents = []
        for _ in range(self.num_parents):
            idx = fitness_scores.index(min(fitness_scores))
            parents.append(population[idx])
            fitness_scores[idx] = float('inf')
        return parents