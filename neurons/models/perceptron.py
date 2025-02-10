# Importing the libraries
import numpy as np

class Perceptron:
    def __init__(self, learning_rate = 0.01, epochs = 50, random_state = 1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        # Initialise a random state (seed) for reproducibility
        seed = np.random.RandomState(self.random_state)
        
        # Initialise the weights and bias using a normal distribution
        self.weights = seed.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        self.bias = np.float64(0)
        self.errors = []

        # Train the model
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights += update * xi
                self.bias += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self
    
    def net_input(self, X):
        # Calculate the dot product of the input features and the weights
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        # Return the class label after the unit step function
        return np.where(self.net_input(X) >= 0.0, 1, -1)