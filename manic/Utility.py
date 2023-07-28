import decimal
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Utility:
    def __init__(self, data_instance, categories, immutable_features, target_class, verbose, predict_fn, disagreement, base_counterfactuals, evaluation, labels):
        self.data_instance = data_instance
        self.categories = categories
        self.immutable_features = immutable_features
        self.target_class = target_class
        self.verbose = verbose
        self.predict_fn = predict_fn
        self.disagreement = disagreement
        self.base_counterfactuals = base_counterfactuals
        self.evaluation = evaluation
        self.labels = labels

    def format_counterfactual(self, counterfactual):
      formatted_counterfactual = []
      for i in range(len(counterfactual)):
        if(i in self.categories):
          formatted_counterfactual.append(round(counterfactual[i]))
        else:
          decimal_feature = decimal.Decimal(self.data_instance[i])
          decimal_places = decimal_feature.as_tuple().exponent * -1

          if(decimal_places == 0):
            formatted_counterfactual.append(int(counterfactual[i]))
          else:
            formatted_counterfactual.append(round(counterfactual[i], decimal_places))

      return formatted_counterfactual
    
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
    
    def print_results(self, best_counterfactual, best_fitness, num_generations, generation_found, time_taken, time_found, cpu_cycles):
        print("\n------ Counterfactual Generation Results ------")
        if best_counterfactual is not None:
            proximity_score = self.disagreement.calculate_proximity(self.data_instance, best_counterfactual, True)
            sparsity_score, number_of_changes = self.disagreement.calculate_sparsity(best_counterfactual)
            agreement_score = self.evaluation.calculate_base_cf_scores([best_counterfactual], self.base_counterfactuals[0])
            print(f"{np.array2string(np.array(best_counterfactual), separator=', ')}: Best Counterfactual 👑")
            print(f"{np.array2string(np.array(self.data_instance), separator=', ')}: Instance Explained 🔍")
            for i, counterfactual in enumerate(self.base_counterfactuals):
                print(f"{np.array2string(counterfactual, separator=', ')}: {self.labels[i]}")
            print("Proximity from Data Instance:", proximity_score)
            print("Sparsity:", sparsity_score)
            print("Number of changes made to produce the counterfactual:", number_of_changes)
            print("Agreement Score against Base Counterfactuals:", agreement_score)
            print("Number of Generations:", num_generations)
            print(f"Counterfactual found after {generation_found + 1} generations")
            print("Fitness Score:", best_fitness)
            print(f"Time taken to find counterfactual: {time_found:.4f} seconds")
            print(f"Total time searched: {time_taken:.4f} seconds")
            print(f"Total CPU cycles ran: {cpu_cycles:.4f}")
        else:
            print("No valid counterfactual found within the specified number of generations.")
            print("Try increasing the number of generations or population size and/or altering alpha, beta and/or perturbation_fraction. As a last resort, you can also try changing the seed.")
        print("------ End of Results ------\n")



    def compute_disagreement_matrix(self,counterfactuals, agreement):
        n = len(counterfactuals)
        disagreement_matrix = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                disagreement_score = self.disagreement.calculate_disagreement(counterfactuals[i], counterfactuals[j])
                if(agreement):
                    disagreement_score = 1 - disagreement_score
                disagreement_matrix[i, j] = disagreement_score
                disagreement_matrix[j, i] = disagreement_score

        return disagreement_matrix
    
    def plot_agreement_heatmap(self, agreement=True):
        disagreement_matrix = self.compute_disagreement_matrix(self.base_counterfactuals, agreement)
        plt.figure(figsize=(len(self.labels), len(self.labels)))
        sns.heatmap(disagreement_matrix, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel('Counterfactual Index')
        plt.ylabel('Counterfactual Index')
        plt.title('Pairwise Agreement for the Diabetes Dataset')
        plt.show()