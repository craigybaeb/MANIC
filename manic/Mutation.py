
import random

class Mutation:
    def __init__(self, mutation_method, perturbation_fraction, feature_ranges):
        self.mutation_method = mutation_method
        self.mutate = self.set_mutation(mutation_method)
        self.perturbation_fraction = perturbation_fraction
        self.feature_ranges = feature_ranges

        self.validate_self()
    
    def set_mutation(self, mutation_method):
            if(mutation_method == "random_resetting"):
                return self.random_resetting_mutation
            elif(mutation_method == "swap_mutation"):
                return self.swap_mutation
            else:
                return self.random_resetting_mutation
        
    def random_resetting_mutation(self, offspring):
        new_offspring = []  # Store the mutated offspring
        for i in range(len(offspring)):
            mutated_instance = offspring[i].copy()  # Create a copy of the instance
            for j in range(len(offspring[i])):
                if random.random() < self.perturbation_fraction:
                    lower_bound, upper_bound = self.feature_ranges[j]
                    mutation_value = random.uniform(lower_bound, upper_bound)
                    mutated_instance[j] = max(lower_bound, min(upper_bound, mutation_value))
            new_offspring.append(mutated_instance)
        return new_offspring
    
    #Needs constraining for valid ranges if using.
    def swap_mutation(self, offspring):
        for i in range(len(offspring)):
            # Randomly select two different feature indices
            feature_indices = random.sample(range(len(offspring[i])), 2)

            # Swap the values of the selected features
            offspring[i][feature_indices[0]], offspring[i][feature_indices[1]] = \
                offspring[i][feature_indices[1]], offspring[i][feature_indices[0]]

        return offspring
    
    def validate_self(self):
        if(self.mutation_method not in ["random_resetting", "swap_mutation"]):
            raise Warning("Invalid mutation method given, it must be random_resetting or swap_mutation. Defaulting to random_resetting.")
        
        if(self.mutate not in [self.random_resetting_mutation, self.swap_mutation]):
            raise ValueError("Error initialising mutation method, no valid method was set.")
        
        if(self.perturbation_fraction == None):
            raise ValueError("Perturbation fraction cannot be None.")
        
        if(self.perturbation_fraction < 0 or self.perturbation_fraction > 1):
            raise ValueError("Perturbation fraction must be between 0 and 1.")
        
        if(self.feature_ranges == None or len(self.feature_ranges) == 0):
            raise ValueError("Feature ranges must be given.")
        