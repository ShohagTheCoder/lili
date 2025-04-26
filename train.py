from remini import Tensor

# 1. Basic Test: Adding two tensors
a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor([4, 5, 6], requires_grad=True)
c = a + b
c.backward()
print("Basic Test")
print("a.grad:", a.grad)  # Should print [1. 1. 1.]
print("b.grad:", b.grad)  # Should print [1. 1. 1.]
print("c.grad:", c.grad)  # Should print [1. 1. 1.]

# 2. Addition with Scalar
a = Tensor([1, 2, 3], requires_grad=True)
c = a + 5
c.backward()
print("\nAddition with Scalar Test")
print("a.grad:", a.grad)  # Should print [1. 1. 1.]

# 3. Addition with Non-Grad Tensor
a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor([4, 5, 6], requires_grad=False)
c = a + b
c.backward()
print("\nAddition with Non-Grad Tensor Test")
print("a.grad:", a.grad)  # Should print [1. 1. 1.]
print("b.grad:", b.grad)  # Should print [0. 0. 0.]

# 4. Multiple Additions
a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor([4, 5, 6], requires_grad=True)
c = Tensor([7, 8, 9], requires_grad=True)
d = a + b + c
d.backward()
print("\nMultiple Additions Test")
print("a.grad:", a.grad)  # Should print [1. 1. 1.]
print("b.grad:", b.grad)  # Should print [1. 1. 1.]
print("c.grad:", c.grad)  # Should print [1. 1. 1.]

# 5. Addition with Zero Tensor
a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor([0, 0, 0], requires_grad=True)
c = a + b
c.backward()
print("\nAddition with Zero Tensor Test")
print("a.grad:", a.grad)  # Should print [1. 1. 1.]
print("b.grad:", b.grad)  # Should print [1. 1. 1.]

# 6. Addition with Different Shapes (Broadcasting)
a = Tensor([[1, 2], [3, 4]], requires_grad=True)
b = Tensor([10, 20], requires_grad=True)
c = a + b
c.backward()
print("\nAddition with Broadcasting Test")
print("a.grad:", a.grad)  # Should print [[1. 1.] [1. 1.]]
print("b.grad:", b.grad)  # Should print [2. 2.]

# 7. Addition with Same Shape, No Grad
a = Tensor([1, 2, 3], requires_grad=False)
b = Tensor([4, 5, 6], requires_grad=False)
c = a + b
print("\nAddition with No Grad Test")
print("c.data:", c.data)  # Should print [5 7 9]

# 8. Adding Empty Tensor
a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor([], requires_grad=True)
c = a + b
print("\nAdding Empty Tensor Test")
print("c.data:", c.data)  # Should print an empty tensor or handle it gracefully

# 9. Backward Pass with Multiple Tensors
a = Tensor([2, 3], requires_grad=True)
b = Tensor([4, 5], requires_grad=True)
c = Tensor([6, 7], requires_grad=True)
d = a + b
e = d + c
e.backward()
print("\nBackward Pass with Multiple Tensors Test")
print("a.grad:", a.grad)  # Should print [1. 1.]
print("b.grad:", b.grad)  # Should print [1. 1.]
print("c.grad:", c.grad)  # Should print [1. 1.]

# 10. Scalar Addition Backward Test
a = Tensor([1, 2, 3], requires_grad=True)
c = a + 1
c.backward()
print("\nScalar Addition Backward Test")
print("a.grad:", a.grad)  # Should print [1. 1. 1.]

# 11. Test zero_grad method
a = Tensor([10, 20, 30], requires_grad=True)
b = Tensor([1, 1, 1], requires_grad=True)
c = a + b
c.backward()
print("\nBefore zero_grad:")
print("a.grad:", a.grad)  # Should print [1. 1. 1.]
print("b.grad:", b.grad)  # Should print [1. 1. 1.]

a.zero_grad()
b.zero_grad()
print("\nAfter zero_grad:")
print("a.grad:", a.grad)  # Should print [0. 0. 0.]
print("b.grad:", b.grad)  # Should print [0. 0. 0.]

# 12. Non-Grad Tensor Test
a = Tensor([10, 20, 30], requires_grad=True)
b = Tensor([5, 5, 5], requires_grad=False)
c = a + b
c.backward()
print("\nNon-Grad Tensor Test")
print("a.grad:", a.grad)  # Should print [1. 1. 1.]
print("b.grad:", b.grad)  # Should print [0. 0. 0.]

# 13. Dimension Mismatch Test (Should raise an error)
try:
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([[4, 5], [6, 7]], requires_grad=True)
    c = a + b  # This should raise an error because the shapes are incompatible
    c.backward()
except ValueError as e:
    print("\nDimension Mismatch Test")
    print("Error:", e)  # Expecting a broadcasting error
