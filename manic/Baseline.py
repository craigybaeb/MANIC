class Baseline:
    """
    Baseline class for selecting counterfactuals from a set of base counterfactuals.

    @param disagreement: The Disagreement object used for calculating disagreement measures.
    @type disagreement: Disagreement

    @param base_counterfactuals: The base counterfactual instances.
    @type base_counterfactuals: list of list of int

    @param data_instance: The original data instance for which counterfactuals are being generated.
    @type data_instance: list of int
    """
    def __init__(self, disagreement, base_counterfactuals, data_instance):
        self.disagreement = disagreement
        self.base_counterfactuals = base_counterfactuals
        self.data_instance = data_instance
    
    def __str__(self):
        """
        Return a string representation of the Baseline object.

        @return: String representation of the Baseline object.
        @rtype: str
        """
        return f"Baseline Object:\n" \
               f"Disagreement: {self.disagreement}\n" \
               f"Base Counterfactuals: {self.base_counterfactuals}\n" \
               f"Data Instance: {self.data_instance}"

    def to_string(self):
        """
        Convert the Baseline object to a string.

        @return: String representation of the Baseline object.
        @rtype: str
        """
        return self.__str__()

    def most_proximal_counterfactual(self):
        """
        Find the most proximal counterfactual instance based on proximity score.

        @return: The most proximal counterfactual instance.
        @rtype: list of int
        """
        best_proximity = float('inf')
        best_counterfactual = None

        for counterfactual in self.base_counterfactuals:
            proximity = self.disagreement.calculate_proximity(self.data_instance, counterfactual)

            if proximity < best_proximity:
                best_proximity = proximity
                best_counterfactual = counterfactual
        
        return best_counterfactual
    
    def most_proximal_counterfactual_with_disagreement(self):
        """
        Find the most proximal counterfactual instance based on a combination of proximity and disagreement scores.

        @return: The most proximal counterfactual instance.
        @rtype: list of int
        """
        best_score = float('inf')
        best_counterfactual = None

        for counterfactual in self.base_counterfactuals:
            proximity = self.disagreement.calculate_proximity(self.data_instance, counterfactual)
            disagreement = self.disagreement.calculate_disagreement(self.data_instance, counterfactual)
            score = proximity + disagreement

            if score < best_score:
                best_score = score
                best_counterfactual = counterfactual
        
        return best_counterfactual