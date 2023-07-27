import numpy as np

class Evaluation:
    def __init__(self, alpha, beta, predict_proba_fn, instance_probability, base_counterfactuals, disagreement, data_instance, theta):
        self.alpha = alpha
        self.beta = beta
        self.predict_proba_fn = predict_proba_fn
        self.instance_probability = instance_probability
        self.base_counterfactuals = base_counterfactuals
        self.disagreement = disagreement
        self.data_instance = data_instance
        self.theta = theta

    def calculate_base_cf_scores(self, population, base_cf):
        base_cf_scores = []

        for candidate_instance in population:
            agreement_score = self.disagreement.calculate_disagreement(candidate_instance, base_cf)
            base_cf_scores.append(agreement_score)

        return sum(base_cf_scores) / len(base_cf_scores)
    
    def misclassification_penalty(self, counterfactual):
        probability = self.predict_proba_fn(counterfactual)

        return np.dot(probability, self.instance_probability)
    
    def evaluate_population(self, population):
        combined_fitness_scores = []
        base_cf_scores = []

        for base_cf in self.base_counterfactuals:
            base_cf_scores.append(self.calculate_base_cf_scores(population, base_cf))

        for candidate_instance in population:
            avg_disagreement = sum(score for score in base_cf_scores) / len(base_cf_scores)
            proximity_score = self.disagreement.calculate_proximity(self.data_instance, candidate_instance)
            penalty = self.misclassification_penalty(candidate_instance)


            combined_fitness = (self.alpha * avg_disagreement) + (self.theta * penalty) #+ (self.beta * proximity_score) #
            combined_fitness_scores.append(combined_fitness)

        return combined_fitness_scores