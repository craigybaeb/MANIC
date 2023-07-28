import time
import os

from Crossover import Crossover
from Initialise import Initialise
from Mutation import Mutation

class Manic:
    def __init__(self, data_instance, base_counterfactuals, categorical_features, immutable_features, feature_ranges, data, predict_fn, predict_proba_fn, class_labels, population_size=100, num_generations=50, alpha=0.5, beta=0.5, crossover_method="uniform", mutation_method="random_resetting", perturbation_fraction=0.1, num_parents=2, seed=42, verbose=1, early_stopping=None, max_time=None, disagreement_method="euclidean_distance", theta=0.3, labels=[]):
        self.initialise = Initialise(disagreement_method, data_instance, base_counterfactuals, predict_fn, predict_proba_fn, seed, population_size, categorical_features, feature_ranges, immutable_features, data, class_labels, theta, alpha, beta, num_parents, verbose, labels)
        self.immutable_features_set = self.initialise.immutable_features_set
        self.target_class = self.initialise.target_class
        self.instance_probability = self.initialise.instance_probability
        self.categories = self.initialise.categories
        self.feature_ranges = self.initialise.feature_ranges
        self.disagreement = self.initialise.disagreement
        self.evaluation = self.initialise.evaluation
        self.selection = self.initialise.selection
        self.utils = self.initialise.utils
        self.is_counterfactual_valid = self.utils.is_counterfactual_valid
        self.print_results = self.utils.print_results
        self.population = self.initialise.population
        self.baseline = self.initialise.baseline

        self.data_instance = data_instance
        self.base_counterfactuals = base_counterfactuals
        self.categorical_features = categorical_features
        self.immutable_features = immutable_features
        self.data = data
        self.predict_fn = predict_fn
        self.predict_proba_fn = predict_proba_fn
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover = Crossover(crossover_method, num_parents, population_size).crossover
        self.continuous_feature_ranges = feature_ranges
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.consecutive_generations_without_improvement = 0
        self.max_time = max_time
        self.mutate = Mutation(mutation_method, perturbation_fraction, self.feature_ranges).mutate
        self.best_counterfactual = None
        self.best_fitness = float('inf')
        self.generation_found = float('inf')
        self.time_found = float('inf')

    def __str__(self):
        attributes_str = [
            f"data_instance: {self.data_instance}",
            f"base_counterfactuals: {self.base_counterfactuals}",
            f"categorical_features: {self.categorical_features}",
            f"immutable_features: {self.immutable_features}",
            f"data: {self.data}",
            f"population_size: {self.population_size}",
            f"num_generations: {self.num_generations}",
            f"target_class: {self.target_class}",
            f"continuous_feature_ranges: {self.continuous_feature_ranges}",
            f"categories: {self.categories}",
            f"feature_ranges: {self.feature_ranges}",
            f"verbose: {self.verbose}"
        ]

        return "\n".join(attributes_str)

    def to_string(self):
        return str(self)

    def should_stop(self, generations, time_elapsed):
      if('found' in list(self.early_stopping.keys()) and self.early_stopping['found'] == True):
        if('patience_generations' in list(self.early_stopping.keys())):
          if(self.early_stopping['patience_generations'] <= generations):
            print(f"Early stopping at generation {generations}. No improvement for {self.early_stopping['patience_generations']} consecutive generations.")
            return True
        if('patience_time' in list(self.early_stopping.keys())):
          if(self.early_stopping['patience_time'] < time_elapsed / 60):
            print(f"Early stopping at time {(time_elapsed / 60):.2f} minutes. No improvement for {(self.early_stopping['patience_time']):.2f} minutes.")
            return True
        else:
          return False
      else:
        return False

    def get_cpu_time(self):
      return time.process_time()

    def get_cpu_cycles(self, cpu_time_seconds):
      cpu_clock_speed_hz = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
      cpu_cycles = cpu_time_seconds * cpu_clock_speed_hz
      
      return cpu_cycles

    def generate_counterfactuals(self):
        start_time = time.time()
        cpu_start_time = self.get_cpu_time()

        for generation in range(self.num_generations):
            if self.verbose == 1:
                print(f"Generation {generation + 1}")

            fitness_scores = self.evaluation.evaluate_population(self.population)

            #Maybe remove normalisation, not sure if ti works with our method
            unnormalised_fitness_scores = fitness_scores
            
            normalise=False
            if(normalise):
              min_score = min(fitness_scores)
              max_score = max(fitness_scores)

              if min_score == max_score:
                  # Handle the case when all scores are the same
                  fitness_scores = [1.0] * len(fitness_scores)
              else:
                  fitness_scores = [(score - min_score) / (max_score - min_score) for score in fitness_scores]
            
            parents = self.selection.select_parents(self.population, fitness_scores)
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring)
            
            # Combine elites and offspring
            elites = self.selection.select_elites(self.population, fitness_scores)
            self.population = elites + offspring

            best_idx = fitness_scores.index(min(fitness_scores))
            generation_best_counterfactual = self.population[best_idx]
            generation_best_fitness = fitness_scores[best_idx]

            if generation_best_fitness < self.best_fitness:
                # Check if the candidate counterfactual produces the target class
                formatted_counterfactual = self.utils.format_counterfactual(generation_best_counterfactual)
                prediction = self.predict_fn(formatted_counterfactual)
                if prediction == self.target_class:
                    self.best_fitness = generation_best_fitness
                    if(formatted_counterfactual != self.best_counterfactual):
                        self.best_counterfactual = formatted_counterfactual
                        self.generation_found = generation
                        self.time_found = time.time() - start_time

                    self.consecutive_generations_without_improvement = 0
                else:
                    self.consecutive_generations_without_improvement += 1
            else:
                self.consecutive_generations_without_improvement += 1

            if self.early_stopping is not None:
                time_elapsed = time.time() - start_time
                if(self.should_stop(generation, time_elapsed)):
                    if self.verbose > 0:
                      break

            if self.verbose == 2:
                print(f"Generation {generation+1}: Best Counterfactual = {self.best_counterfactual}, Fitness = {self.best_fitness}")

            if self.verbose == 3:
                print(f"Generation {generation+1}:")
                for idx, child in enumerate(offspring):
                    print(f"Child {idx+1}: {child}")

            # Check if the specified maximum time is exceeded
            if self.max_time is not None and (time.time() - start_time) > (self.max_time * 60):
                if self.verbose > 0:
                    print(f"Stopping search after {self.max_time} minutes.")
                break

        end_time = time.time()
        cpu_end_time = self.get_cpu_time()

                # Calculate elapsed CPU time in seconds
        elapsed_cpu_time_seconds = cpu_end_time - cpu_start_time

        cpu_cycles = self.get_cpu_cycles(elapsed_cpu_time_seconds)

        print(end_time - start_time)


        time_taken = end_time - start_time

        if self.verbose > 0:
            self.print_results(self.best_counterfactual, self.best_fitness, generation + 1, self.generation_found, time_taken, self.time_found, cpu_cycles)

        return self.best_counterfactual