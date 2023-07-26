import unittest
from manic import Manic

class TestManicCrossover(unittest.TestCase):

    def test_crossover(self):
        # Create a Manic instance
        data_instance = [1, 0.5, 'category_a']
        base_counterfactuals = [[1, 0.4, 'category_b'], [1, 0.6, 'category_c']]
        categorical_features = [2]
        immutable_features = [0]
        feature_ranges = {1: (0, 1), 2: ['category_a', 'category_b', 'category_c']}
        data = [[1, 0.5, 'category_a'], [2, 0.7, 'category_b'], [1, 0.3, 'category_c']]
        predict_fn = lambda x: 0

        manic_instance = Manic(
            data_instance=data_instance,
            base_counterfactuals=base_counterfactuals,
            categorical_features=categorical_features,
            immutable_features=immutable_features,
            feature_ranges=feature_ranges,
            data=data,
            predict_fn=predict_fn,
            population_size=100,
            num_generations=50,
            alpha=0.5,
            beta=0.5,
            perturbation_fraction=0.1,
            num_parents=2,
            seed=42,
            verbose=0,
            early_stopping=None,
            max_time=None
        )

        # Initialize a population
        population = manic_instance.initialize_population()

        # Test the crossover function
        offspring = manic_instance.crossover(population)

        # Assert that the number of offspring is the same as the population size
        self.assertEqual(len(offspring), manic_instance.population_size)

        # Assert that each offspring is a valid instance (same length as the data instance)
        for child in offspring:
            self.assertEqual(len(child), len(data_instance))

            # Assert that the features are within the valid range
            for i, (min_val, max_val) in enumerate(manic_instance.feature_ranges):
                self.assertGreaterEqual(child[i], min_val)
                self.assertLessEqual(child[i], max_val)

if __name__ == '__main__':
    unittest.main()
