from setuptools import setup, find_packages

setup(
    name='manic-xai',
    version='0.1.0',
    description='Genetic Algorithm for Generating Metacounterfactual Explanations',
    authors= [
        { name="Craig Pirie", email="c.pirie11@rgu.ac.uk" }
    ],
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.2',
        'scikit-learn>=0.24.2'
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
