import numpy as np
import concurrent.futures

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
        self.parallel = False

    def calculate_base_cf_scores(self, population, base_cf):
        base_cf_scores = []

        if(self.parallel):
          # Use ThreadPoolExecutor to compute agreement scores in parallel
          with concurrent.futures.ThreadPoolExecutor() as executor:
              agreement_scores = list(executor.map(self.calculate_agreement_score, population, [base_cf] * len(population)))

          base_cf_scores = agreement_scores  # Store the agreement scores in the base_cf_scores list
        else:
          for candidate_instance in population:
            agreement_score = self.calculate_agreement_score(candidate_instance, base_cf)
            base_cf_scores.append(agreement_score)
        return sum(base_cf_scores) / len(base_cf_scores)
    
    def calculate_agreement_score(self, candidate_instance, base_cf):
        agreement_score = self.disagreement.calculate_disagreement(candidate_instance, base_cf)
        
        return agreement_score

    def calculate_combined_fitness(self, candidate_instance, base_cf_scores):
      avg_disagreement = sum(score for score in base_cf_scores) / len(base_cf_scores)
      proximity_score = self.disagreement.calculate_proximity(self.data_instance, candidate_instance)
      penalty = self.misclassification_penalty(candidate_instance)

      combined_fitness = (self.alpha * avg_disagreement) + (self.theta * penalty) + (self.beta * proximity_score)
      return combined_fitness

    def misclassification_penalty(self, counterfactual):
        probability = self.predict_proba_fn(counterfactual)

        return np.dot(probability, self.instance_probability)
    
    def evaluate_population(self, population):
        combined_fitness_scores = []
        base_cf_scores = []

        for base_cf in self.base_counterfactuals:
            base_cf_scores.append(self.calculate_base_cf_scores(population, base_cf))

        if(self.parallel):
          with concurrent.futures.ThreadPoolExecutor() as executor:
          # Use the map method to run calculate_combined_fitness in parallel
          # This will automatically distribute the workload across multiple threads
            combined_fitness_scores = list(executor.map(self.calculate_combined_fitness, population, [base_cf_scores]*len(population)))
        else:
          for candidate_instance in population:
            fitness_score = self.calculate_combined_fitness(candidate_instance, base_cf_scores)
            combined_fitness_scores.append(fitness_score)

        return combined_fitness_scores