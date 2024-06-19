import numpy as np

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.01):
        self.weights = np.random.rand(num_inputs + 1)
        self.learning_rate = learning_rate

    def linear(self, inputs):
        Z = inputs @ self.weights[1:].T + self.weights[0]
        return Z

    def Heaviside_step_fn(self, z):
        return 1 if z >= 0 else 0

    def predict(self, inputs):
        Z = self.linear(inputs)
        try:
            pred = [self.Heaviside_step_fn(z) for z in Z]
        except TypeError:
            return self.Heaviside_step_fn(Z)
        return pred

    def loss(self, prediction, target):
        return prediction - target

    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = self.loss(prediction, target)
        self.weights[1:] += self.learning_rate * error * inputs
        self.weights[0] += self.learning_rate * error

    def fit(self, X, y, num_epochs):
        for epoch in range(num_epochs):
            for inputs, target in zip(X, y):
                self.train(inputs, target)