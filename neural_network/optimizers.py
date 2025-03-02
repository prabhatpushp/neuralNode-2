# optimizers.py

import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}

    def update(self, node, grad, param_name):
        self.t += 1

        if node not in self.m:
            self.m[node] = {}
            self.v[node] = {}

        if param_name not in self.m[node]:
            self.m[node][param_name] = np.zeros_like(grad)
            self.v[node][param_name] = np.zeros_like(grad)

        self.m[node][param_name] = self.beta1 * self.m[node][param_name] + (1 - self.beta1) * grad
        self.v[node][param_name] = self.beta2 * self.v[node][param_name] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[node][param_name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[node][param_name] / (1 - self.beta2 ** self.t)

        update_value = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return update_value
