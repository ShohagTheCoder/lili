import numpy as np
from remini.tensor.tensor import Tensor

class Dense:
    def __init__(self, in_features, out_features):
        self.weight = Tensor(np.random.randn(in_features, out_features) * 0.01)
        self.bias = Tensor(np.zeros((1, out_features)))
        self._input = None  # Save input for backward

    def __call__(self, x):
        self._input = x  # Save input
        return (x @ self.weight) + self.bias

    def parameters(self):
        return [self.weight, self.bias]

    def backward(self, grad_output):
        # grad_output: gradient from next layer
        
        # Compute gradients for weights and bias
        self.weight.grad = self._input.data.T @ grad_output
        self.bias.grad = np.sum(grad_output, axis=0, keepdims=True)

        # Compute gradient to pass to previous layer
        grad_input = grad_output @ self.weight.data.T
        return grad_input
