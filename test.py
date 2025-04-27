import numpy as np

from remini.tensor.tensor import Tensor

# Test Tensor class
# Define two tensors
t1 = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
t2 = Tensor([[5, 6], [7, 8], [8, 9]], requires_grad=True)

# Test addition
t_add = t1 + t2
expected_add = t1.data + t2.data
print(f"Addition Result: \n{t_add}")
print(f"Expected Addition Result: \n{expected_add}")
t_add.backward()  # Compute gradients
print(f"Expected Gradients: \n{t_add.grad}")

# Test multiplication
t_mul = t1 * t2
expected_mul = t1.data * t2.data
print(f"Multiplication Result: \n{t_mul}")
print(f"Expected Multiplication Result: \n{expected_mul}")
t_mul.backward()  # Compute gradients
print(f"Expected Gradients: \n{t_mul.grad}")

# Test matrix multiplication
t_matmul = t1 @ t2
expected_matmul = np.matmul(t1.data, t2.data)
print(f"Matrix Multiplication Result: \n{t_matmul}")
print(f"Expected Matrix Multiplication Result: \n{expected_matmul}")
t_matmul.backward()  # Compute gradients
print(f"Expected Gradients: \n{t_matmul.grad}")

# Test scalar addition
scalar = 2
t_scalar_add = t1 + scalar
expected_scalar_add = t1.data + scalar
print(f"Scalar Addition Result: \n{t_scalar_add}")
print(f"Expected Scalar Addition Result: \n{expected_scalar_add}")
t_scalar_add.backward()  # Compute gradients
print(f"Expected Gradients: \n{t_scalar_add.grad}")

# Test ReLU activation
t_relu = t1.relu()
expected_relu = np.maximum(0, t1.data)
print(f"ReLU Activation Result: \n{t_relu}")
print(f"Expected ReLU Activation Result: \n{expected_relu}")
t_relu.backward()  # Compute gradients
print(f"Expected Gradients for ReLU: \n{t_relu.grad}")

# Check the gradient after ReLU
print(f"t1 Gradient after ReLU backward: \n{t1.grad}")

# Test reshaping
t_reshaped = t1.reshape((-1,))
expected_reshaped = t1.data.reshape((-1,))
print(f"Reshaped Tensor: \n{t_reshaped}")
print(f"Expected Reshaped Tensor: \n{expected_reshaped}")

# Test transpose
t_transposed = t1.transpose()
expected_transposed = t1.data.T
print(f"Transposed Tensor: \n{t_transposed}")
print(f"Expected Transposed Tensor: \n{expected_transposed}")

# Reset gradients
t1.zero_grad()
print(f"t1 Gradient after zero_grad: \n{t1.grad}")
