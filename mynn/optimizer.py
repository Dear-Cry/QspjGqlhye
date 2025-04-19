from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocities = {}

        for idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                self.velocities[idx] = {}
                for key in layer.params.keys():
                    self.velocities[idx][key] = np.zeros_like(layer.params[key])
    
    def step(self):
        for idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                for key in layer.params.keys():
                    velocity = self.velocities[idx][key]
                    velocity = self.mu * velocity - self.init_lr * layer.grads[key]
                    if layer.weight_decay:
                        velocity -= self.init_lr * layer.weight_decay_lambda * layer.params[key]
                    layer.params[key] += velocity
                    self.velocities[idx][key] = velocity


class Adam(Optimizer):
    def __init__(self, init_lr, model, beta1=0.9, beta2=0.99, epsilon=1e-8):
        super().__init__(init_lr, model)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.ms = {}
        self.vs = {}

        for idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                self.ms[idx] = {}
                self.vs[idx] = {}
                for key in layer.params.keys():
                    self.ms[idx][key] = np.zeros_like(layer.params[key])
                    self.vs[idx][key] = np.zeros_like(layer.params[key])

    def step(self):
        self.t += 1
        for idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                for key in layer.params.keys():
                    m = self.ms[idx][key]
                    m = self.beta1 * m + (1 - self.beta1) * layer.grads[key]
                    v = self.vs[idx][key]
                    v = self.beta2 * v + (1 - self.beta2) * (layer.grads[key] ** 2)
                    self.ms[idx][key] = m
                    self.vs[idx][key] = v
                    m_unbias = m / (1 - self.beta1 ** self.t)
                    v_unbias = v / (1 - self.beta2 ** self.t)
                    if layer.weight_decay:
                        layer.params[key] -= self.init_lr * layer.weight_decay_lambda * layer.params[key]
                    layer.params[key] -= self.init_lr * m_unbias / (np.sqrt(v_unbias) + self.epsilon)
                    