import numpy as np
from remini.utils import unbroadcast

# Main tensor class to handle tensor operations
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.size = np.prod(self.shape)

        # Set backward function and previous tensor for autograd
        self._backward = lambda : None
        self._prev = []
        self._op = '' # Operation type (e.g., 'add', 'mul', etc.)

    # Method to perform addition operation
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        # Define the backward function for addition
        if result.requires_grad:
            def _backward():
                grad = result.grad
                #backprop for self
                if self.requires_grad:
                    if grad.shape != self.shape:
                        self.grad += unbroadcast(grad, self.shape)
                    else:
                        self.grad += grad

                #backprop for other
                if other.requires_grad:
                    if grad.shape != other.shape:
                        other.grad += unbroadcast(grad, other.shape)
                    else:
                        other.grad += grad

            # Define the backward function and previous tensors
            result._backward = _backward
            result._prev = [self, other]
            result._op = "+"

        # Return the result
        return result
    
    def backward(self):
        # Only proceed if tensor requires gradients
        if not self.requires_grad:
            return

        # If no gradient is passed, initialize it to ones (typically for loss)
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # Call the operation-specific backward function
        self._backward()

        # Propagate the gradients to previous tensors
        for prev in self._prev:
            if prev.requires_grad:
                prev.backward()
    
    # Reset the gradient to zero
    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    # Represent the tensor as a string for easy debugging
    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"