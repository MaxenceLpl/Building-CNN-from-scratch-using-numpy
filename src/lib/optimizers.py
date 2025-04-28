import numpy as np

class Optimizer:
    def __call__(self, param, grad, key):
        raise NotImplementedError("Optimizer must implement the call method.")

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def __call__(self, param, grad, key=None):
        return param - self.learning_rate * grad

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # 1er moment (moyenne)
        self.v = {}  # 2e moment (carré des gradients)
        self.t = {}  # timestep

    def __call__(self, param, grad, key):
        if key not in self.m:
            self.m[key] = np.zeros_like(grad)
            self.v[key] = np.zeros_like(grad)
            self.t[key] = 0

        self.t[key] += 1

        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[key] / (1 - self.beta1 ** self.t[key])
        v_hat = self.v[key] / (1 - self.beta2 ** self.t[key])

        return param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
