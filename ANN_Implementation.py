import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import numpy as np


class NeuralNet:
    def __init__(self, X: np.ndarray, y: np.ndarray, layer_sizes, alpha=0.1):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1  # Number of layers (excluding input layer)

        # Initialize weights and biases for each layer
        self.W = {}
        self.b = {}
        for l in range(1, self.L + 1):
            self.W[str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l - 1]) * 0.1
            self.b[str(l)] = np.random.randn(layer_sizes[l], 1) * 0.1

        # Cache for intermediate values in forward pass
        self.cache = {}

    @staticmethod
    def sigmoid(arr):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-arr))

    @staticmethod
    def sigmoid_derivative(arr):
        """Derivative of the sigmoid function."""
        return arr * (1 - arr)

    def forward(self, X):
        """Forward propagation through all layers."""
        A = X
        self.cache["A0"] = A  # Store the input layer activation
        for l in range(1, self.L + 1):
            Z = self.W[str(l)] @ A + self.b[str(l)]
            A = self.sigmoid(Z)
            self.cache[f"A{l}"] = A  # Cache activation
        return A

    def predict(self, X):
        return self.forward(X)

    def compute_cost(self, y_hat, y):
        """Calculate mean squared error as cost."""
        m = y.shape[1]
        return np.sum((y_hat - y) ** 2) / (2 * m)

    def backward(self, y_hat):
        """Backward propagation to compute gradients for weights and biases."""
        m = self.X.shape[1]
        grads = {}
        dA = (
            y_hat - self.y
        ) / m  # Initial gradient based on cost function (mean squared error)

        for l in reversed(range(1, self.L + 1)):
            dZ = dA * self.sigmoid_derivative(
                self.cache[f"A{l}"]
            )  # Element-wise gradient
            grads[f"dW{l}"] = dZ @ self.cache[f"A{l - 1}"].T
            grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True)
            if l > 1:  # Calculate dA for the previous layer if not at the input layer
                dA = self.W[str(l)].T @ dZ

        return grads

    def update_parameters(self, grads):
        """Update weights and biases using gradients."""
        for l in range(1, self.L + 1):
            self.W[str(l)] -= self.alpha * grads[f"dW{l}"]
            self.b[str(l)] -= self.alpha * grads[f"db{l}"]

    def train(self, epochs=1000):
        """Train the network for a set number of epochs."""
        for epoch in range(epochs):
            y_hat = self.forward(self.X)
            cost = self.compute_cost(y_hat, self.y)
            grads = self.backward(y_hat)
            self.update_parameters(grads)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost}")

    def predict(self, X):
        """Predict output for given input data."""
        return self.forward(X)

    def accuracy(self, y_hat, y):
        """Compute accuracy as percentage of correct predictions (rounded)."""
        predictions = np.round(y_hat)  # Rounding since outputs are continuous
        accuracy = np.mean(predictions == y) * 100
        return accuracy


# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

scaler = StandardScaler()
encoder = LabelEncoder()

X_scaled = scaler.fit_transform(X)
y = encoder.fit_transform(y)

y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, shuffle=True
)

X_train = X_train.T  # Transpose to shape (30, 142)
y_train = y_train.T  # Transpose to shape (1, 142)
X_test = X_test.T  # Transpose to shape (30, 142)
y_test = y_test.T  # Transpose to shape (1, 142)


# Model initialization
model = NeuralNet(X_train, y_train, layer_sizes=[30, 4, 1], alpha=0.01)

# Training
model.train(epochs=10000)

# Prediction and accuracy testing
y_pred = model.predict(X_test)
accuracy = model.accuracy(y_pred, y_test)

results = pd.DataFrame(
    {"Actuals": y_test.flatten(), "Predictions": np.round(y_pred.flatten())}
)

print(f"Accuracy: {accuracy}%")
