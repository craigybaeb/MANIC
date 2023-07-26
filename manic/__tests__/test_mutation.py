import unittest
from manic import Manic

class TestManicMutation(unittest.TestCase):
    def test_mutate(self):
        # Replace the inputs as needed
        data_instance = [...]
        base_counterfactuals = [...]
        categorical_features = [...]
        immutable_features = [...]
        feature_ranges = {...}
        data = [...]
        predict_fn = your_prediction_function

        population_size = 100
        num_generations = 50
        alpha = 0.5
        beta = 0.5
        perturbation_fraction = 0.1
        num_parents = 2
        seed = 42
        verbose = 0
        early_stopping = None
        max_time = None

        # Create the Manic instance
        manic_instance = Manic(
            data_instance,
            base_counterfactuals,
            categorical_features,
            immutable_features,
            feature_ranges,
            data,
            predict_fn,
            population_size=population_size,
            num_generations=num_generations,
            alpha=alpha,
            beta=beta,
            perturbation_fraction=perturbation_fraction,
            num_parents=num_parents,
            seed=seed,
            verbose=verbose,
            early_stopping=early_stopping,
            max_time=max_time
        )

        # Initialize the population
        population = manic_instance.initialize_population()

        # Perform mutation
        mutated_population = manic_instance.mutate(population)

        # Assert that the mutated population has the same size as the original population
        self.assertEqual(len(mutated_population), len(population))

        # Assert that each candidate instance in the mutated population contains valid values within the specified ranges
        for candidate_instance in mutated_population:
            for i in range(len(candidate_instance)):
                lower_bound, upper_bound = feature_ranges[i]
                self.assertGreaterEqual(candidate_instance[i], lower_bound)
                self.assertLessEqual(candidate_instance[i], upper_bound)

if __name__ == '__main__':
    unittest.main()
