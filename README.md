# Manic: A Genetic Algorithm-based Metaheuristic Approach for Nature-Inspired Aggregation of Counterfactuals

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Docker Instructions](#docker-instructions)
- [Citation](#citation)
- [Author](#author)
- [Contact](#contact)
- [License](#license)

## Introduction

Manic is a Python package that provides a genetic algorithm-based metaheuristic approach for aggregating counterfactual explanations. It implements a nature-inspired optimization technique to generate counterfactuals that explain the disagreement between different explainers. The goal is to find counterfactual instances that are both diverse and informative to enhance the interpretability of machine learning models.

## Installation

To install Manic, use the following pip3 command:

```bash
pip3 install manic-xai
```

## Usage

You can use Manic in your Python code as follows:

```python
from manic import Manic

# Define your data_instance, base_counterfactuals, categorical_features, immutable_features, feature_ranges, data, and predict_fn

manic_instance = Manic(data_instance, base_counterfactuals, categorical_features, immutable_features, feature_ranges, data, predict_fn)

# Generate counterfactuals
counterfactuals = manic_instance.generate_counterfactuals()
```

## Docker Instructions

To run Manic using Docker, follow these steps:

1. Build the Docker image:

```bash
docker build -t manic .
```

2. Run the Docker container

```bash
docker run -v /path/to/your/data:/data -it manic python3 your_script.py
```


We appreciate your support and acknowledgment of our work.


## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributions
Contributions to the Manic package are always welcome. If you find any issues or have ideas for improvements, please feel free to open an issue or submit a pull request on the GitHub repository. Together, we can make Manic better for everyone.

## Changelog
For updates and a history of changes to the Manic package, please refer to the Changelog.
