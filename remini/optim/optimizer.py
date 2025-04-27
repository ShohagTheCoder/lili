# import numpy as np
# from remini.tensor.tensor import Tensor

# class Optimizer:
#     def __init__(self, params, lr=0.001):
#         self.lr = lr
#         self.params = params
        
#     def step(self):
#         raise NotImplementedError("Optimizer step method not implemented.")
    
#     def zero_grad(self):
#         for param in self.params:
#             if param.grad is not None:
#                 param.grad = np.zeros_like(self.grad)