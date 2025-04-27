import numpy as np


class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(param.data) for param in params]
        self.v = [np.zeros_like(param.data) for param in params]
        self.t = 0

    def zero_grad(self):
        # Ensure that param.grad.data is a NumPy array and then fill it with zeros
        for param in self.params:
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is not None:
                grad = param.grad.data
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
