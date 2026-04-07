import numpy as np
from scipy.spatial.distance import pdist, squareform

class ContinuousTimeRNN:
    """
    Continuous-time recurrent neural network with Euler integration.
    
    Supports multiple connectivity types:
    - 'gaussian': Random Gaussian weights with controllable strength
    - 'erdos_renyi': Erdős-Rényi random graph with K connections per neuron
    - 'geometric': Geometric random graph with exponential decay rule (2D or 3D)
    - 'geometric_inhibitory': Geometric graph with inhibitory neurons and exponential decay
    
    All eigenvalues are constrained to be negative, with the largest
    eigenvalue slightly below zero for stability.
    """
    
    def __init__(self, num_neurons, connectivity_type='gaussian', 
                 decay_rate=0.1, strength=1.0, K=10, sparsity=0.1, 
                 dimension=2, decay_constant=1.0, seed=None):
        """
        Initialize the RNN.
        
        Parameters:
        -----------
        num_neurons : int
            Number of neurons in the network
        connectivity_type : str
            Type of connectivity ('gaussian', 'erdos_renyi', 'geometric', 'geometric_inhibitory')
        decay_rate : float
            Self-decay term for each neuron
        strength : float
            Strength of Gaussian weights (standard deviation scaling for 'gaussian' connectivity)
        K : int
            Average number of connections per neuron for 'erdos_renyi' connectivity
        sparsity : float
            Sparsity parameter (for geometric graphs)
        dimension : int
            Dimensionality for geometric graphs (2 or 3)
        decay_constant : float
            Time constant for exponential decay rule in geometric connectivity
        seed : int, optional
            Random seed for reproducibility
        """
        self.num_neurons = num_neurons
        self.connectivity_type = connectivity_type
        self.decay_rate = decay_rate
        self.strength = strength
        self.K = K
        self.sparsity = sparsity
        self.dimension = dimension
        self.decay_constant = decay_constant
        
        if seed is not None:
            np.random.seed(seed)
        
        self.weights = self._initialize_weights()
        self.state = np.zeros(num_neurons)
        self._validate_eigenvalues()
    
    def _initialize_weights(self):
        """Initialize weight matrix based on connectivity type."""
        
        if self.connectivity_type == 'gaussian':
            # Gaussian connectivity with controllable strength
            weights = np.random.normal(0, self.strength / np.sqrt(self.num_neurons), 
                                      (self.num_neurons, self.num_neurons))
        
        elif self.connectivity_type == 'erdos_renyi':
            # Erdős-Rényi: each neuron has K connections on average
            connection_prob = self.K / self.num_neurons
            mask = np.random.rand(self.num_neurons, self.num_neurons) < connection_prob
            weights = np.zeros((self.num_neurons, self.num_neurons))
            weights[mask] = np.random.normal(0, 1 / np.sqrt(self.K), np.sum(mask))
        
        elif self.connectivity_type == 'geometric':
            weights = self._geometric_graph()
        
        elif self.connectivity_type == 'geometric_inhibitory':
            weights = self._geometric_graph_inhibitory()
        
        else:
            raise ValueError(f"Unknown connectivity type: {self.connectivity_type}")
        
        return weights
    
    def _geometric_graph(self):
        """Create a geometric random graph with exponential decay rule."""
        # Place neurons randomly in a unit hypercube (2D or 3D)
        if self.dimension == 2:
            positions = np.random.rand(self.num_neurons, 2)
        elif self.dimension == 3:
            positions = np.random.rand(self.num_neurons, 3)
        else:
            raise ValueError(f"Dimension must be 2 or 3, got {self.dimension}")
        
        # Compute pairwise Euclidean distances
        distances = squareform(pdist(positions, metric='euclidean'))
        
        # Create connections based on distance threshold
        threshold = np.percentile(distances[distances > 0], self.sparsity * 100)
        mask = (distances < threshold) & (distances > 0)
        
        # Initialize weights with random values
        weights = np.zeros((self.num_neurons, self.num_neurons))
        weights[mask] = np.random.normal(0, 1 / np.sqrt(self.num_neurons * self.sparsity), 
                                        np.sum(mask))
        
        # Apply exponential decay rule based on distance
        # w_ij = w_ij * exp(-d_ij / decay_constant)
        decay_factor = np.exp(-distances / self.decay_constant)
        weights = weights * decay_factor
        
        return weights
    
    def _geometric_graph_inhibitory(self):
        """Create a geometric random graph with inhibitory neurons and exponential decay."""
        # Place neurons randomly in a unit hypercube (2D or 3D)
        if self.dimension == 2:
            positions = np.random.rand(self.num_neurons, 2)
        elif self.dimension == 3:
            positions = np.random.rand(self.num_neurons, 3)
        else:
            raise ValueError(f"Dimension must be 2 or 3, got {self.dimension}")
        
        # Compute pairwise Euclidean distances
        distances = squareform(pdist(positions, metric='euclidean'))
        
        # 20% of neurons are inhibitory
        num_inhibitory = max(1, int(0.2 * self.num_neurons))
        inhibitory_neurons = np.random.choice(self.num_neurons, num_inhibitory, replace=False)
        excitatory_neurons = np.array([i for i in range(self.num_neurons) if i not in inhibitory_neurons])
        
        # Create connections based on distance threshold
        threshold = np.percentile(distances[distances > 0], self.sparsity * 100)
        mask = (distances < threshold) & (distances > 0)
        
        # Initialize weights with random values
        weights = np.zeros((self.num_neurons, self.num_neurons))
        weights[mask] = np.random.normal(0, 1 / np.sqrt(self.num_neurons * self.sparsity), 
                                        np.sum(mask))
        
        # Apply exponential decay rule based on distance
        decay_factor = np.exp(-distances / self.decay_constant)
        weights = weights * decay_factor
        
        # Make connections from inhibitory neurons negative
        weights[inhibitory_neurons, :] *= -1
        
        # Apply exponential decay specifically for excitatory connections (from excitatory neurons)
        # to maintain stability of inhibitory inputs
        for i in excitatory_neurons:
            weights[i, :] = weights[i, :] * decay_factor[i, :]
        
        return weights
    
    def _validate_eigenvalues(self):
        """Ensure all eigenvalues are negative with max slightly below zero."""
        eigenvalues = np.linalg.eigvals(self.weights)
        max_eigenvalue = np.max(np.real(eigenvalues))
        
        if max_eigenvalue >= -0.001:  # Small tolerance
            # Scale down weights to push max eigenvalue below zero
            scaling_factor = -0.01 / (max_eigenvalue + 1e-6)
            self.weights *= scaling_factor
    
    def step(self, dt):
        """
        Perform one Euler integration step.
        
        Parameters:
        -----------
        dt : float
            Integration time step
        """
        # dx/dt = -decay_rate * x + W @ x
        dynamics = -self.decay_rate * self.state + np.dot(self.weights, self.state)
        self.state += dynamics * dt
    
    def simulate(self, duration, dt=0.01, initial_state=None):
        """
        Simulate network dynamics.
        
        Parameters:
        -----------
        duration : float
            Total simulation time
        dt : float
            Integration time step (default: 0.01)
        initial_state : array-like, optional
            Initial state vector (default: zeros)
        
        Returns:
        --------
        states : ndarray
            Array of shape (num_steps, num_neurons) containing state history
        """
        if initial_state is not None:
            self.state = np.array(initial_state)
        else:
            self.state = np.zeros(self.num_neurons)
        
        num_steps = int(duration / dt)
        states = np.zeros((num_steps, self.num_neurons))
        
        for step in range(num_steps):
            self.step(dt)
            states[step] = self.state.copy()
        
        return states
    
    def get_eigenvalues(self):
        """Get eigenvalues of the weight matrix."""
        return np.linalg.eigvals(self.weights)
    
    def get_max_eigenvalue(self):
        """Get the largest (most positive) eigenvalue."""
        eigenvalues = self.get_eigenvalues()
        return np.max(np.real(eigenvalues))


# Example usage:
if __name__ == '__main__':
    # Create an RNN with Gaussian connectivity and strength parameter
    rnn_gaussian = ContinuousTimeRNN(num_neurons=100, connectivity_type='gaussian', 
                                     decay_rate=0.1, strength=0.5, seed=42)
    
    print("Gaussian RNN (strength=0.5):")
    print(f"  Max eigenvalue: {rnn_gaussian.get_max_eigenvalue():.6f}")
    print(f"  All eigenvalues < 0: {np.all(np.real(rnn_gaussian.get_eigenvalues()) < 0)}")
    
    # Create an RNN with Erdős-Rényi connectivity and K parameter
    rnn_er = ContinuousTimeRNN(num_neurons=100, connectivity_type='erdos_renyi', 
                               decay_rate=0.1, K=15, seed=42)
    
    print("\nErdős-Rényi RNN (K=15):")
    print(f"  Max eigenvalue: {rnn_er.get_max_eigenvalue():.6f}")
    print(f"  All eigenvalues < 0: {np.all(np.real(rnn_er.get_eigenvalues()) < 0)}")
    
    # Create an RNN with 2D geometric connectivity and exponential decay
    rnn_geom_2d = ContinuousTimeRNN(num_neurons=100, connectivity_type='geometric', 
                                    decay_rate=0.1, dimension=2, decay_constant=0.5, seed=42)
    
    print("\nGeometric RNN (2D, decay_constant=0.5):")
    print(f"  Max eigenvalue: {rnn_geom_2d.get_max_eigenvalue():.6f}")
    print(f"  All eigenvalues < 0: {np.all(np.real(rnn_geom_2d.get_eigenvalues()) < 0)}")
    
    # Create an RNN with 3D geometric connectivity with inhibitory neurons
    rnn_geom_3d_inh = ContinuousTimeRNN(num_neurons=100, connectivity_type='geometric_inhibitory', 
                                        decay_rate=0.1, dimension=3, decay_constant=0.5, seed=42)
    
    print("\nGeometric Inhibitory RNN (3D, decay_constant=0.5):")
    print(f"  Max eigenvalue: {rnn_geom_3d_inh.get_max_eigenvalue():.6f}")
    print(f"  All eigenvalues < 0: {np.all(np.real(rnn_geom_3d_inh.get_eigenvalues()) < 0)}")
    
    # Simulate with random initial perturbation
    initial_state = np.random.normal(0, 0.1, 100)
    states = rnn_gaussian.simulate(duration=10, dt=0.01, initial_state=initial_state)
    
    print(f"\nSimulated {states.shape[0]} time steps")
    print(f"Final state norm: {np.linalg.norm(rnn_gaussian.state):.6f}")
