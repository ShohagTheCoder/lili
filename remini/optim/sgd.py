import numpy as np

class SGD:
    def __init__(self, params, lr=0.01):
        self.lr = lr
        self.params = params

    def step(self):
        # Update parameters using the gradients
        for param in self.params:
            if param.grad is not None:  # Check if gradient exists
                param.data -= self.lr * param.grad

    def zero_grad(self):
        # Zero out the gradients
        for param in self.params:
            if param.grad is not None:
                param.grad.fill(0)  # Reset the gradient to zero, ensuring the same shape
