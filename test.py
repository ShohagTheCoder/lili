import numpy as np
from remini.tensor import Tensor  # Assuming the Tensor class is in a file named tensor.py

# Test 1: Basic Matrix Multiplication (2x2 matrices)
def test_basic_matmul():
    a = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True)
    b = Tensor(np.array([[5, 6], [7, 8]]), requires_grad=True)

    c = a.matmul(b)
    print("Result of matrix multiplication (2x2 matrices):")
    print(c.data)  # Expected: [[19, 22], [43, 50]]
    print("Expected: [[19, 22], [43, 50]]")

    c.backward()
    print("Gradient of a:", a.grad)  # Expected: [[12, 14], [16, 18]]
    print("Expected: [[12, 14], [16, 18]]")
    print("Gradient of b:", b.grad)  # Expected: [[4, 6], [4, 6]]
    print("Expected: [[4, 6], [4, 6]]")

# Test 2: Multiplying a Row Vector with a Column Vector (1xN) * (Nx1)
def test_row_col_matmul():
    a = Tensor(np.array([[1, 2, 3]]), requires_grad=True)  # 1x3 row vector
    b = Tensor(np.array([[4], [5], [6]]), requires_grad=True)  # 3x1 column vector

    c = a.matmul(b)
    print("Result of row vector x column vector multiplication:")
    print(c.data)  # Expected: [[32]] (1x1 scalar result)
    print("Expected: [[32]]")

    c.backward()
    print("Gradient of a:", a.grad)  # Expected: [[4, 5, 6]]
    print("Expected: [[4, 5, 6]]")
    print("Gradient of b:", b.grad)  # Expected: [[1], [2], [3]]
    print("Expected: [[1], [2], [3]]")

# Test 3: Multiplying a Scalar with a Matrix (1x1 matrix multiplication)
def test_scalar_matmul():
    a = Tensor(np.array([[2]]), requires_grad=True)  # Scalar
    b = Tensor(np.array([[5, 6], [7, 8]]), requires_grad=True)  # 2x2 matrix

    c = a.matmul(b)
    print("Result of scalar x matrix multiplication:")
    print(c.data)  # Expected: [[10, 12], [14, 16]]
    print("Expected: [[10, 12], [14, 16]]")

    c.backward()
    print("Gradient of a:", a.grad)  # Expected: [[22]]
    print("Expected: [[22]]")
    print("Gradient of b:", b.grad)  # Expected: [[2], [2]]
    print("Expected: [[2], [2]]")

# Run the tests
if __name__ == "__main__":
    test_basic_matmul()
    print("\n--------------------------------\n")
    test_row_col_matmul()
    print("\n--------------------------------\n")
    test_scalar_matmul()
