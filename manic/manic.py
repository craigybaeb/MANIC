import heapq
import random 
import numpy as np

from Baseline import Baseline
from Crossover import Crossover
from Disagreement import Disagreement
from Evaluation import Evaluation
from GeneticAlgorithm import GeneticAlgorithm
from Mutation import Mutation
from Replacement import Replacement
from Selection import Selection
from Utility import Utility

class Manic:
    def __init__(self, data_instance, base_counterfactuals, categorical_features, immutable_features, feature_ranges, data, predict_fn, predict_proba_fn, class_labels, population_size=100, num_generations=50, alpha=0.5, beta=0.5, crossover_method="uniform", mutation_method="random_resetting", perturbation_fraction=0.1, num_parents=2, seed=42, verbose=1, early_stopping=None, max_time=None, disagreement_method="euclidean_distance", theta=0.3, parallel=False, labels=[]):
        self.immutable_features_set = set(immutable_features)
        self.data = data
        self.data_instance = data_instance
        self.categorical_features = categorical_features
        self.base_counterfactuals = base_counterfactuals
        self.population_size = population_size
        self.labels = labels if labels else list(range(1, len(base_counterfactuals) + 1))
        self.class_labels = class_labels
        self.num_generations = num_generations
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.theta = theta
        self.target_class = 1 - predict_fn(data_instance) #TODO don't assume binary classification
        self.categories = self.get_categories(categorical_features)
        self.feature_ranges = self.get_feature_ranges(feature_ranges)
        self.disagreement = Disagreement(disagreement_method, data_instance, base_counterfactuals, categorical_features, feature_ranges, predict_fn, predict_proba_fn, self.target_class, self.feature_ranges)
        self.population = self.initialise_population()
        self.categorical_features = categorical_features
        self.immutable_features = immutable_features
        self.crossover = Crossover(crossover_method, num_parents, population_size, parallel).crossover
        self.mutate = Mutation(mutation_method, perturbation_fraction, self.feature_ranges).mutate
        self.selection = Selection(num_parents, self.target_class, population_size, predict_fn, parallel)
        self.instance_probability = predict_proba_fn(data_instance)
        self.evaluation = Evaluation(alpha, beta, predict_proba_fn, self.instance_probability, base_counterfactuals, self.disagreement, data_instance, theta, parallel)
        self.replacement = Replacement(self.crossover, self.mutate, self.evaluation, self.selection)
        self.baseline = Baseline(self.disagreement, base_counterfactuals, data_instance)
        self.utils = Utility(data_instance, self.categories, immutable_features, self.target_class, verbose, predict_fn, self.disagreement, base_counterfactuals, self.labels)
        self.is_counterfactual_valid = self.utils.is_counterfactual_valid
        self.print_results = self.utils.print_results
        self.generate_counterfactuals = GeneticAlgorithm(num_generations, early_stopping, predict_fn, self.evaluation, self.selection, self.crossover, self.mutate, verbose, self.target_class, max_time, self.utils, self.replacement, self.population, base_counterfactuals, data_instance).generate_counterfactuals

    def __str__(self):
        attributes_str = [
            f"categorical_features: {self.categorical_features}",
            f"immutable_features: {self.immutable_features}",
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
    
    def weighted_random_choice(self, options, weights):
        return random.choices(options, weights, k=1)[0]

    def initialise_population(self):
        random.seed(self.seed)
        population = []
        options = [self.generate_random_instance, self.nearest_unlike_neighbors, self.randomly_sample_counterfactual]
        weights = [0.5, 0.2, 0.3]  # These should add up to 1

        for _ in range(int(self.population_size / 2)):
            choice = self.weighted_random_choice(options, weights)
            candidate = choice()
            population.append(candidate)

        return population
    
    def randomly_sample_counterfactual(self):
        return random.choice(self.base_counterfactuals)
    
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
                LOWER_BOUND = min(self.data[:, i]) - (min(self.data[:, i]) / 10)
                UPPER_BOUND = max(self.data[:, i]) + (max(self.data[:, i]) / 10)
                feature_ranges.append((LOWER_BOUND, UPPER_BOUND))

        return feature_ranges
    
    def generate_random_instance(self):
        candidate_instance = []

        for i, (min_val, max_val) in enumerate(self.feature_ranges):
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
    
    def get_categories(self, categorical_features):
        categories = {}
        for feature in categorical_features:
            options = np.unique(self.data[:,feature])
            categories[feature] = options

        return categories
    
    def nearest_unlike_neighbors(self):
        unlike_neighbors = []
        distances = []

        for i, instance in enumerate(self.data):
            if self.target_class != self.class_labels[i]:
                distance = self.disagreement.euclidean_distance(self.data_instance, instance)
                distances.append((distance, i))  # Store both distance and index

        # Use heapq to efficiently find the n smallest distances and their corresponding indices
        smallest_distances = heapq.nsmallest(int(self.population_size / 2), distances)

        # Get the actual instances for the smallest distances
        for distance, index in smallest_distances:
            neighbor = self.data[index]
            unlike_neighbors.append(neighbor)

        return random.choice(unlike_neighbors)