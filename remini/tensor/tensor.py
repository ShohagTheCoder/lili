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

    # Method to perform addition operation\
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        # Check for empty tensor
        if self.data.size == 0 or other.data.size == 0:
            print("Warning: Adding an empty tensor.")
            result = Tensor(self.data if self.data.size != 0 else other.data, requires_grad=self.requires_grad or other.requires_grad)
            return result
        
        # Result of addition
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
    
    # Method to perform multiplication operation
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        # Check for empty tensor
        if self.data.size == 0 or other.data.size == 0:
            print("Warning: Multiplying with an empty tensor.")
            result = Tensor(self.data if self.data.size != 0 else other.data, requires_grad=self.requires_grad or other.requires_grad)
            return result
        
        # Result of multiplication
        result = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        if result.requires_grad:
            def _backward():
                grad = result.grad

                # Backpropagation for self
                if self.requires_grad:
                    if grad.shape != self.shape:
                        self.grad += unbroadcast(grad * other.data, self.shape)
                    else:
                        self.grad += grad * other.data

                # Backpropagation for other
                if other.requires_grad:
                    if grad.shape != other.shape:
                        other.grad += unbroadcast(grad * self.data, other.shape)
                    else:
                        other.grad += grad * self.data
                
            result._backward = _backward
            result._prev = [self, other]
            result._op = "*"

        # Return the result
        return result
    
    # Method to perform matrix multiplication operation
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        # Check for empty tensor
        if self.data.size == 0 or other.data.size == 0:
            print("Warning: Matrix multiplying with an empty tensor.")
            result = Tensor(self.data if self.data.size != 0 else other.data, requires_grad=self.requires_grad or other.requires_grad)
            return result
        
        # Check shape compatibility for matrix multiplication
        if self.shape[-1] != other.shape[0]:
            print(f"Shapes {self.shape} and {other.shape} are not aligned for matrix multiplication.")
            return Tensor(np.empty((self.shape[0], other.shape[1])), requires_grad=self.requires_grad or other.requires_grad)
        
        # Matrix multiplication result
        result = Tensor(np.matmul(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)

        if result.requires_grad:
            def _backward():
                grad = result.grad

                # Backpropagation for self
                if self.requires_grad:
                    if grad.shape != self.shape:
                        self.grad += unbroadcast(np.matmul(grad, other.data.T), self.shape)
                    else:
                        self.grad += np.matmul(grad, other.data.T)
                
                # Backpropagation for other
                if other.requires_grad:
                    if grad.shape != other.shape:
                        other.grad += unbroadcast(np.matmul(self.data.T, grad), other.shape)
                    else:
                        other.grad += np.matmul(self.data.T, grad)
                
            result._backward = _backward
            result._prev = [self, other]
            result._op = "@"
        
        # Return the result
        return result
    
    # Method to perform subtraction operation
    def __sub__(self, other):
        if isinstance(other, Tensor):
            return self.__add__(other.__neg__())
        
    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return other.__add__(self.__neg__())
    
    # Additional methods for tensor operations
    def add(self, other):
        return self.__add__(other)
    def matmul(self, other):
        return self.__matmul__(other)
    def neg(self):
        return self.__neg__()

    # Backward propagation method to compute gradients
    def backward(self, grad=None):
        self.grad = np.ones_like(self.data)

        # Visited set to keep track of visited tensors
        visited = set()
        topo = []

        # Build the topological order of tensors for backward pass
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)
        # Call the build_topo function for the current tensor
        build_topo(self)

        # Perform the backward pass in reverse topological order
        for t in reversed(topo):
            t._backward()
            
    # Neg method for negation
    def __neg__(self):
        return Tensor(-self.data, requires_grad=self.requires_grad)

    # ReLU activation function
    def relu(self):
        # Apply ReLU element-wise (max(0, x))
        result = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        
        if result.requires_grad:
            def _backward():
                # ReLU derivative: 1 where self.data > 0, else 0
                if self.grad is not None:
                    self.grad += result.grad * (self.data > 0)  # Accumulate gradients correctly
                else:
                    self.grad = result.grad * (self.data > 0)  # Initialize gradient for first backward pass
                
                # Debugging statements
                print("Tensor (self):", self)
                print("self.grad (after backward pass):", self.grad)
                print("result.grad:", result.grad)  # The gradient of the output tensor

            result._backward = _backward
            result._prev = [self]
            result._op = "ReLU"
        
        return result
    
    # Leaky ReLU activation function
    def leaky_relu(self, negative_slope=0.01):
        # Apply Leaky ReLU element-wise
        result = Tensor(np.where(self.data > 0, self.data, negative_slope * self.data), requires_grad=self.requires_grad)
        
        if result.requires_grad:
            def _backward():
                if self.grad is not None:
                    # Leaky ReLU derivative: 1 where x > 0 else negative_slope
                    self.grad += result.grad * np.where(self.data > 0, 1.0, negative_slope)
                else:
                    self.grad = result.grad * np.where(self.data > 0, 1.0, negative_slope)
                
                # Debugging statements
                print("Tensor (self):", self)
                print("self.grad (after backward pass):", self.grad)
                print("result.grad:", result.grad)  # The gradient of the output tensor

            result._backward = _backward
            result._prev = [self]
            result._op = "LeakyReLU"
        
        return result

    # Reset the gradient to zero
    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
            
    @property
    def T(self):
        result = Tensor(self.data.T, requires_grad=self.requires_grad)

        if result.requires_grad:
            def _backward():
                if self.grad is not None:
                    self.grad += result.grad.T
                else:
                    self.grad = result.grad.T

            result._backward = _backward
            result._prev = [self]
            result._op = "Transpose"
        
        return result
    
    def sum(self, axis=None, keepdims=False):
        """Sum of tensor elements over a given axis."""
        result = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward():
                grad = result.grad
                if axis is None:
                    grad = np.full_like(self.data, grad)
                else:
                    grad = np.expand_dims(grad, axis) if not keepdims else grad
                    grad = np.broadcast_to(grad, self.data.shape)

                if self.grad is not None:
                    self.grad += grad
                else:
                    self.grad = grad

            result._backward = _backward
            result._prev = [self]
            result._op = "Sum"

        return result
    
    # Transpose the tensor
    def transpose(self):
        return Tensor(np.transpose(self.data), requires_grad=self.requires_grad)

    # Reshape the tensor
    def reshape(self, new_shape):
        return Tensor(self.data.reshape(new_shape), requires_grad=self.requires_grad)

    # Represent the tensor as a string for easy debugging
    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"