import random
import time
from functools import lru_cache
import concurrent.futures

class Manic:
    def __init__(self, data_instance, base_counterfactuals, categorical_features, immutable_features, feature_ranges, data, predict_fn, population_size=100, num_generations=50, alpha=0.5, beta=0.5, perturbation_fraction=0.1, num_parents=2, seed=42, verbose=1, early_stopping=None, max_time=None):
        self.data_instance = data_instance
        self.base_counterfactuals = base_counterfactuals
        self.categorical_features = categorical_features
        self.immutable_features = immutable_features
        self.immutable_features_set = set(immutable_features)
        self.data = data
        self.predict_fn = predict_fn
        self.population_size = population_size
        self.num_generations = num_generations
        self.alpha = alpha
        self.beta = beta
        self.perturbation_fraction = perturbation_fraction
        self.num_parents = num_parents
        self.seed = seed
        self.target_class = 1 - predict_fn(data_instance)  # Assuming binary classification
        self.continuous_feature_ranges = feature_ranges
        self.categories = self.get_categories(categorical_features)
        self.feature_ranges = self.get_feature_ranges(feature_ranges)
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.consecutive_generations_without_improvement = 0
        self.max_time = max_time

    def __str__(self):
        attributes_str = [
            f"data_instance: {self.data_instance}",
            f"base_counterfactuals: {self.base_counterfactuals}",
            f"categorical_features: {self.categorical_features}",
            f"immutable_features: {self.immutable_features}",
            f"data: {self.data}",
            f"population_size: {self.population_size}",
            f"num_generations: {self.num_generations}",
            f"alpha: {self.alpha}",
            f"beta: {self.beta}",
            f"perturbation_fraction: {self.perturbation_fraction}",
            f"num_parents: {self.num_parents}",
            f"seed: {self.seed}",
            f"target_class: {self.target_class}",
            f"continuous_feature_ranges: {self.continuous_feature_ranges}",
            f"categories: {self.categories}",
            f"feature_ranges: {self.feature_ranges}",
            f"verbose: {self.verbose}"
        ]

        return "\n".join(attributes_str)

    def to_string(self):
        return str(self)

    def get_categories(self, categorical_features):
        categories = {}
        for feature in categorical_features:
            options = np.unique(self.data[:,feature])
            categories[feature] = options

        return categories

    def initialize_population(self):
        random.seed(self.seed)
        population = []
        feature_ranges = self.feature_ranges

        for _ in range(self.population_size):
            candidate_instance = self.generate_random_instance(feature_ranges)
            population.append(candidate_instance)

        return population

    def get_feature_ranges(self, continuous_feature_ranges):
        feature_ranges = []
        for i in range(len(self.data_instance)):
            if i in self.immutable_features_set:
                # For immutable features, the range is a single value (the current value)
                feature_ranges.append((self.data_instance[i], self.data_instance[i]))
            elif i in self.categorical_features:
                LOWER_BOUND = min(self.categories[i])
                UPPER_BOUND = max(self.categories[i])
                feature_ranges.append((LOWER_BOUND, UPPER_BOUND))
            elif i in list(continuous_feature_ranges.keys()):
                feature_ranges.append(continuous_feature_ranges[i])
            else:
                LOWER_BOUND = min(data[:, i]) - (min(data[:, i]) / 10)
                UPPER_BOUND = max(data[:, i]) + (max(data[:, i]) / 10)
                feature_ranges.append((LOWER_BOUND, UPPER_BOUND))

        return feature_ranges

    def generate_random_instance(self, feature_ranges):
        candidate_instance = []

        for i, (min_val, max_val) in enumerate(feature_ranges):
            if i in self.immutable_features_set:
                # For immutable features, use the original value
                candidate_instance.append(min_val)
            elif i in self.categorical_features:
                possible_values = sorted(set(int(data[i]) for data in self.data))
                candidate_instance.append(random.choice(possible_values))
            else:
                candidate_value = random.uniform(min_val, max_val)
                candidate_instance.append(max(min_val, min(max_val, candidate_value)))

        return candidate_instance

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

    def calculate_proximity(self, data_instance, counterfactual_instance):
      data_instance_tuple = tuple(data_instance)
      counterfactual_instance_tuple = tuple(counterfactual_instance)
      l1_norm_score = np.sum(np.abs(np.array(data_instance) - np.array(counterfactual_instance)))
      inverse_mad = 1 / self.median_absolute_deviation(data_instance_tuple)

      return l1_norm_score * inverse_mad

    @lru_cache(maxsize=None)
    def median_absolute_deviation(self, data):
        data_tuple = tuple(data)
        median = np.median(data)
        absolute_deviations = np.abs(np.array(data) - median)

        return np.median(absolute_deviations)

    def calculate_base_cf_scores(self, population, base_cf):
        base_cf_scores = []

        for candidate_instance in population:
            agreement_score = self.calculate_disagreement(candidate_instance, base_cf)
            base_cf_scores.append(agreement_score)

        return sum(base_cf_scores) / len(base_cf_scores)

    def calculate_disagreement(self, candidate_instance, base_cf):
        cf1_changed_features = set(i for i, cf1_val in enumerate(candidate_instance) if cf1_val != self.data_instance[i])
        cf2_changed_features = set(i for i, cf2_val in enumerate(base_cf) if cf2_val != self.data_instance[i])

        union_changed_features = cf1_changed_features.union(cf2_changed_features)
        total_features = len(self.data_instance)
        agreement_score = abs(len(union_changed_features)) / total_features

        return 1 - agreement_score

    def misclassification_penalty(self, counterfactual):
      prediction = self.predict_fn(counterfactual)

      if(prediction != self.target_class):
        return 1
      else:
        return 0

    def evaluate_population(self, population):
        combined_fitness_scores = []
        base_cf_scores = []

        for base_cf in self.base_counterfactuals:
            base_cf_scores.append(self.calculate_base_cf_scores(population, base_cf))

        for candidate_instance in population:
            avg_disagreement = sum(score for score in base_cf_scores) / len(base_cf_scores)
            proximity_score = self.calculate_proximity(self.data_instance, candidate_instance)
            prediction = self.predict_fn(candidate_instance)
            penalty = 0
            theta = 0.3

            if(prediction != self.target_class):
              penalty = 1

            combined_fitness = (self.alpha * avg_disagreement) + (self.beta * proximity_score) + (theta * penalty)
            combined_fitness_scores.append(combined_fitness)

        return combined_fitness_scores

    def select_parents(self, population, fitness_scores):
        parents = []
        for _ in range(self.num_parents):
            idx = fitness_scores.index(min(fitness_scores))
            parents.append(population[idx])
            fitness_scores[idx] = float('inf')
        return parents

    def crossover(self, parents):
        offspring = []
        for i in range(self.population_size):
            parent1 = parents[i % self.num_parents]
            parent2 = parents[(i + 1) % self.num_parents]
            cut_point = random.randint(0, len(parent1))
            child = parent1[:cut_point] + parent2[cut_point:]
            offspring.append(child)
        return offspring

    def mutate(self, offspring):
        for i in range(len(offspring)):
            for j in range(len(offspring[i])):
                if random.random() < self.perturbation_fraction:
                    lower_bound, upper_bound = self.feature_ranges[j]
                    mutation_value = random.uniform(lower_bound, upper_bound)
                    offspring[i][j] = max(lower_bound, min(upper_bound, mutation_value))
                offspring[i][j] = round(offspring[i][j])
        return offspring

    def calculate_sparsity(self, cf):
        num_changes = 0

        for i in range(len(cf)):
            if(cf[i] != self.data_instance[i]):
                num_changes += 1

        sparsity = 1 - (num_changes / len(cf))

        return sparsity, num_changes

    def print_results(self, best_counterfactual, best_fitness, num_generations, generation_found, time_taken, time_found):
        print("\n------ Counterfactual Generation Results ------")
        if best_counterfactual is not None:
            proximity_score = self.calculate_proximity(self.data_instance, best_counterfactual)
            sparsity_score, number_of_changes = self.calculate_sparsity(best_counterfactual)
            agreement_score = self.calculate_base_cf_scores([best_counterfactual], self.base_counterfactuals[0])
            print("Best Counterfactual:", best_counterfactual)
            print("Instance Explained:", self.data_instance)
            for i, counterfactual in enumerate(self.base_counterfactuals):
                print(f"Base Counterfactual ({i}): {counterfactual}")
            print("Proximity from Data Instance:", proximity_score)
            print("Sparsity:", sparsity_score)
            print("Number of changes made to produce the counterfactual:", number_of_changes)
            print("Agreement Score against Base Counterfactuals:", agreement_score)
            print("Number of Generations:", num_generations)
            print(f"Counterfactual found after {generation_found + 1} generations")
            print("Fitness Score:", best_fitness)
            print(f"Time taken to find counterfactual: {time_found:.4f} seconds")
            print(f"Total time searched: {time_taken:.4f} seconds")
        else:
            print("No valid counterfactual found within the specified number of generations.")
            print("Try increasing the number of generations or population size and/or altering alpha, beta and/or perturbation_fraction. As a last resort, you can also try changing the seed.")
        print("------ End of Results ------\n")

    def is_counterfactual_valid(self, counterfactual):
        # Check if the counterfactual is not None
        if counterfactual is None:
            if self.verbose > 0:
                print("Invalid Counterfactual: None value generated as counterfactual.")
            return False

        # Check if any immutable features are changed
        for i in self.immutable_features:
            if counterfactual[i] != self.data_instance[i]:
                if self.verbose > 0:
                    print(f"Invalid Counterfactual: Feature at index {i} is immutable and cannot be changed.")
                return False

        # Check if the class is equal to the target class
        prediction = self.predict_fn(counterfactual)
        if prediction != self.target_class:
            if self.verbose > 0:
                print(f"Invalid Counterfactual: Predicted class ({prediction}) is not the target class ({self.target_class}).")
            return False

        # All conditions are met, counterfactual is valid
        if self.verbose > 0:
            print("Valid Counterfactual: No immutable features were changed and the counterfactual causes the correct prediction change.")

        return True

    def generate_counterfactuals(self):
        population = self.initialize_population()
        best_counterfactual = None
        best_fitness = float('inf')
        generation_found = float('inf')
        time_found = float('inf')

        start_time = time.time()

        for generation in range(self.num_generations):
            if self.verbose == 1:
                print(f"Generation {generation+1}")

            fitness_scores = self.evaluate_population(population)
            parents = self.select_parents(population, fitness_scores)
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring)
            population = offspring

            best_idx = fitness_scores.index(min(fitness_scores))
            generation_best_counterfactual = population[best_idx]
            generation_best_fitness = fitness_scores[best_idx]

            if generation_best_fitness < best_fitness:
                # Check if the candidate counterfactual produces the target class
                prediction = self.predict_fn(generation_best_counterfactual)
                if prediction == self.target_class:
                    best_counterfactual = generation_best_counterfactual
                    best_fitness = generation_best_fitness
                    generation_found = generation
                    time_found = time.time() - start_time

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
                print(f"Generation {generation+1}: Best Counterfactual = {best_counterfactual}, Fitness = {best_fitness}")

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
        time_taken = end_time - start_time

        if self.verbose > 0:
            self.print_results(best_counterfactual, best_fitness, generation + 1, generation_found, time_taken, time_found)

        return best_counterfactual

    def generate_counterfactuals_parallel(self, data_instances, num_threads=2):
        counterfactuals = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_instance = {executor.submit(self.generate_counterfactuals_single, data_instance): data_instance for data_instance in data_instances}

            for future in concurrent.futures.as_completed(future_to_instance):
                data_instance = future_to_instance[future]
                counterfactual = future.result()
                counterfactuals[data_instance] = counterfactual

        return counterfactuals

    def generate_counterfactuals_single(self, data_instance):
        self.data_instance = data_instance
        return self.generate_counterfactuals()
