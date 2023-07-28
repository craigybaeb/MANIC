class Baseline:
    def __init__(self, disagreement, base_counterfactuals, data_instance):
        self.disagreement = disagreement
        self.base_counterfactuals = base_counterfactuals
        self.data_instance = data_instance

    def most_proximal_counterfactual(self):
        best_proximity = float('inf')
        best_counterfactual = None

        for counterfactual in self.base_counterfactuals:
            proximity = self.disagreement.calculate_proximity(self.data_instance, counterfactual)

            if(proximity < best_proximity):
                best_proximity = proximity
                best_counterfactual = counterfactual
        
        return best_counterfactual
    
    def most_proximal_counterfactual_with_disagreement(self):
        best_score = float('inf')
        best_counterfactual = None

        for counterfactual in self.base_counterfactuals:
            proximity = self.disagreement.calculate_proximity(self.data_instance, counterfactual)
            disagreement = self.disagreement.calculate_disagreement(self.data_instance, counterfactual)
            score = proximity + disagreement

            if(score < best_score):
                best_score = score
                best_counterfactual = counterfactual
        
        return best_counterfactual
    