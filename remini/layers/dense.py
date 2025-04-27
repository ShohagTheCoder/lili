import numpy as np
from remini.tensor.tensor import Tensor

# Dense (Linear) layer implementation
class Dense:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.input = None
        
        # Weights and bias initialization (Xavier Initialization for weights)
        self.weights = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x):
        # Save input for backward pass
        self.input = x
        # Forward pass includes bias addition
        return x @ self.weights.data + self.bias.data  # Add bias to the output

    def backward(self, grad_output):
        """Backward pass to compute gradients"""
        # Gradient of weights
        self.weights.grad = np.dot(self.input.data.T, grad_output.data)
        # Gradient of bias
        self.bias.grad = np.sum(grad_output.data, axis=0)
        # Gradient of input
        grad_input = np.dot(grad_output.data, self.weights.data.T)
        
        return Tensor(grad_input, requires_grad=True)  # Return gradient w.r.t input

    def __repr__(self):
        return f"Dense(in_features={self.in_features}, out_features={self.out_features})"
