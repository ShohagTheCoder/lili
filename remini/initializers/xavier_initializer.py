import numpy as np

class XavierInitializer:
    def __init__(self):
        pass
    
    def __call__(self, shape):
        # Xavier initialization (uniform distribution)
        fan_in = shape[0]  # Number of input neurons (previous layer)
        fan_out = shape[1]  # Number of output neurons (next layer)
        limit = np.sqrt(6 / (fan_in + fan_out))  # Xavier bound
        return np.random.uniform(-limit, limit, size=shape)
