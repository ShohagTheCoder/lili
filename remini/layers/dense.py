import numpy as np
from remini.initializers.xavier_initializer import XavierInitializer
from remini.tensor import Tensor

class Dense:
    def __init__(self, in_features, out_features, weight_initializer=None, bias_initializer='zeros'):
        self.in_features = in_features
        self.out_features = out_features
        self.input = None
        
        # Initialize weight and bias
        self.weights = self._initialize_weights(in_features, out_features, weight_initializer)
        self.bias = self._initialize_bias(out_features, bias_initializer)

    def _initialize_weights(self, in_features, out_features, weight_initializer):
        if weight_initializer is None:
            # Default: Xavier initialization (if no initializer provided)
            weight_initializer = XavierInitializer()

        # Call the weight initializer (either custom or Xavier)
        return Tensor(weight_initializer((in_features, out_features)), requires_grad=True)

    def _initialize_bias(self, out_features, bias_initializer):
        if bias_initializer == 'zeros':
            return Tensor(np.zeros(out_features), requires_grad=True)
        elif bias_initializer == 'ones':
            return Tensor(np.ones(out_features), requires_grad=True)
        elif bias_initializer == 'small':
            return Tensor(np.full(out_features, 1e-4), requires_grad=True)
        else:
            raise ValueError("Unsupported bias_initializer")

    def forward(self, input_tensor):
        # Forward pass: input -> linear transformation -> output
        self.input = input_tensor
        return input_tensor @ self.weights + self.bias

    def backward(self, grad_output):
        # Backward pass: compute gradients with respect to weights and bias
        grad_input = grad_output @ self.weights.T
        grad_weights = self.input.T @ grad_output
        grad_bias = grad_output.sum(axis=0)

        # Update the gradients in the tensors
        self.weights.grad = grad_weights
        self.bias.grad = grad_bias

        return grad_input
