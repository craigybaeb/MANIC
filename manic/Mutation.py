
import random

class Mutation:
    def __init__(self, mutation_method, perturbation_fraction, feature_ranges):
        self.mutation_method = mutation_method
        self.mutate = self.set_mutation(mutation_method)
        self.perturbation_fraction = perturbation_fraction
        self.feature_ranges = feature_ranges
    
    def set_mutation(self, mutation_method):
            if(mutation_method == "random_resetting"):
                return self.random_resetting_mutation
            elif(mutation_method == "swap_mutation"):
                return self.swap_mutation
            else:
                return self.random_resetting_mutation
        
    def random_resetting_mutation(self, offspring):
        for i in range(len(offspring)):
            for j in range(len(offspring[i])):
                if random.random() < self.perturbation_fraction:
                    lower_bound, upper_bound = self.feature_ranges[j]
                    mutation_value = random.uniform(lower_bound, upper_bound)
                    offspring[i][j] = max(lower_bound, min(upper_bound, mutation_value))
                offspring[i][j] = offspring[i][j]
        return offspring
    
    #Needs constraining for valid ranges if using.
    def swap_mutation(self, offspring):
        for i in range(len(offspring)):
            # Randomly select two different feature indices
            feature_indices = random.sample(range(len(offspring[i])), 2)

            # Swap the values of the selected features
            offspring[i][feature_indices[0]], offspring[i][feature_indices[1]] = \
                offspring[i][feature_indices[1]], offspring[i][feature_indices[0]]

        return offspring