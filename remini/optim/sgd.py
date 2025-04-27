import numpy as np

class SGD:
    def __init__(self, params, lr=0.01):
        self.lr = lr
        self.params = params

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)