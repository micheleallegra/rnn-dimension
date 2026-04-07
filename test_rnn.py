# test_rnn.py

import numpy as np
from rnn_dimension import ContinuousTimeRNN  # Adjust the import based on your package structure

def test_rnn():
    connectivity_types = ['full', 'diagonal', 'sparse']  # Example connectivity types
    for connectivity in connectivity_types:
        rnn = ContinuousTimeRNN(connectivity=connectivity)
        # Example input - Adjust size based on your specific use case
        input_data = np.random.rand(10, 5)  # 10 time steps, 5 features
        output = rnn.forward(input_data)
        print(f"Connectivity Type: {connectivity}, Output Shape: {output.shape}")

if __name__ == "__main__":
    test_rnn()
