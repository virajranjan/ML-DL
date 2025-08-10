import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, method='gd', lr=0.01, epochs=100, batch_size=None):
        self.method = method
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.theta = None
        self.losses = []

    def _add_bias(self, X):
        """Add bias (column of 1s) to our features"""
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _mse_loss(self, X, y):
        pred = X @ self.theta
        return ((pred - y) ** 2).mean()

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)

        X_bias = self._add_bias(X)
        n_samples, n_features = X.shape

        if self.method == 'analytics':
            self.theta = np.linalg.inv(X_bias.T @ X_bias) @ (X_bias.T @ y)
            return {"theta": self.theta, "losses": []}

        # Random initialization
        self.theta = np.random.randn(n_features + 1, 1)
        self.losses = []

        if self.method == 'gd':
            return self.gradient_descent(X_bias, y)
        elif self.method == 'sgd':
            return self.SGD(X_bias, y)

    def predict(self, X):
        X_bias = self._add_bias(np.array(X, dtype=float))
        return X_bias @ self.theta

    def gradient_descent(self, X_bias, y):
        n_samples = X_bias.shape[0]
        for epoch in range(self.epochs):
            y_pred = X_bias @ self.theta
            error = y_pred - y
            gradient = (2 / n_samples) * (X_bias.T @ error)
            self.theta -= self.lr * gradient
            self.losses.append(self._mse_loss(X_bias, y))
        return {"theta": self.theta, "losses": self.losses}

    def SGD(self, X_bias, y):
        n_samples = X_bias.shape[0]
        batch_size = self.batch_size if self.batch_size else 1

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_bias[indices]
            y_shuffled = y[indices]
            epoch_loss = 0

            for i in range(0, n_samples, batch_size):
                xi = X_shuffled[i:i + batch_size]
                yi = y_shuffled[i:i + batch_size]
                error_i = (xi @ self.theta) - yi
                gradient_i = (2 / batch_size) * (xi.T @ error_i)
                self.theta -= self.lr * gradient_i
                epoch_loss += self._mse_loss(xi, yi)

            self.losses.append(epoch_loss / (n_samples / batch_size))

        return {"theta": self.theta, "losses": self.losses}


if __name__ == "__main__":
    n_samples = int(input("Enter number of samples: "))
    n_features = int(input("Enter number of features: "))

    X = np.random.randn(n_samples, n_features)
    true_m = 5.00
    true_b = 3.00
    y = (true_m * X + true_b) @ np.ones((n_features, 1))

    model = LinearRegression(method="gd", lr=0.01, epochs=1000)
    result = model.fit(X, y)

    print("\nLearned parameters (theta):")
    print(result["theta"])
    print("\nLoss history:")
    print(result["losses"])
