import numpy as np
from scipy.spatial.distance import pdist, squareform

class ContinuousTimeRNN:
    """
    Continuous-time recurrent neural network with Euler integration.
    
    Supports multiple connectivity types:
    - 'gaussian': Random Gaussian weights with controllable strength
    - 'erdos_renyi': Erdős-Rényi random graph with K connections per neuron
    - 'geometric': Geometric random graph
    - 'geometric_inhibitory': Geometric graph with inhibitory neurons
    
    All eigenvalues are constrained to be negative, with the largest
    eigenvalue slightly below zero for stability.
    "