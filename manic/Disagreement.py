import numpy as np
from scipy.spatial.distance import cosine
from functools import lru_cache

class Disagreement:
    def __init__(self, disagreement_method, data_instance, base_counterfactuals, categorical_features, continuous_feature_ranges, predict_fn, predict_proba_fn, target_class):
        self.data_instance = data_instance
        self.base_counterfactuals = base_counterfactuals
        self.categorical_features = categorical_features
        self.continuous_feature_ranges = continuous_feature_ranges
        self.predict_fn = predict_fn
        self.predict_proba_fn = predict_proba_fn
        self.target_class = target_class
        self.calculate_disagreement = self.set_disagreement_method(disagreement_method)

    def euclidean_distance(self, instance1, instance2):
        return np.sqrt(np.sum((instance1 - instance2) ** 2))

    def calculate_cosine_distance(self, counterfactual1, counterfactual2):
        return cosine(counterfactual1, counterfactual2)

    @lru_cache(maxsize=None)
    def median_absolute_deviation(self, data):
        data_tuple = tuple(data)
        median = np.median(data)
        absolute_deviations = np.abs(np.array(data) - median)

        return np.median(absolute_deviations)

    def calculate_proximity(self, data_instance, counterfactual_instance):
        data_instance_tuple = tuple(data_instance)
        counterfactual_instance_tuple = tuple(counterfactual_instance)
        l1_norm_score = np.sum(np.abs(np.array(data_instance) - np.array(counterfactual_instance)))
        inverse_mad = 1 / self.median_absolute_deviation(data_instance_tuple)

        return l1_norm_score * inverse_mad

    def calculate_feature_overlap(self, candidate_instance, base_cf):
        cf1_changed_features = set(i for i, cf1_val in enumerate(candidate_instance) if cf1_val != self.data_instance[i])
        cf2_changed_features = set(i for i, cf2_val in enumerate(base_cf) if cf2_val != self.data_instance[i])

        union_changed_features = cf1_changed_features.union(cf2_changed_features)
        total_features = len(self.data_instance)
        agreement_score = abs(len(union_changed_features)) / total_features

        return 1 - agreement_score

    def set_disagreement_method(self, function):
        if function == "feature_overlap":
            return self.calculate_feature_overlap
        elif function == "cosine_distance":
            return self.calculate_cosine_distance
        elif function == "euclidean_distance":
            return self.euclidean_distance
        else:
            return self.calculate_cosine_distance
        
    def calculate_sparsity(self, cf):
        num_changes = 0

        for i in range(len(cf)):
            if(cf[i] != self.data_instance[i]):
                num_changes += 1

        sparsity = num_changes / len(cf)

        return sparsity, num_changes