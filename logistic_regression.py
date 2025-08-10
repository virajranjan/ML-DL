import numpy as np

class Logistic_Regression:
    def __init__(self, lr=0.01, epochs=1000, batch_size=None, re_lambda=0.01):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.losses = []
        self.re_lambda = re_lambda
        self.theta = None

    def _add_bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, h, y):
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        l2_term = (self.re_lambda / (2 * len(y))) * np.sum(self.theta[1:] ** 2)
        return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h)) + l2_term

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)
        X_bias = self._add_bias(X)
        n_samples = X_bias.shape[0]

        self.theta = np.zeros((X_bias.shape[1], 1))

        for _ in range(self.epochs):
            z = X_bias @ self.theta
            h = self._sigmoid(z)

            gradient = (X_bias.T @ (h - y)) / n_samples
            gradient[1:] += (self.re_lambda / n_samples) * self.theta[1:]  # L2 term, exclude bias

            self.theta -= self.lr * gradient

            loss = self._loss(h, y)
            self.losses.append(loss)

        return {"theta": self.theta, "losses": self.losses}

    def predict_proba(self, X):
        X_bias = self._add_bias(X)
        return self._sigmoid(X_bias @ self.theta)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

## chat gpt
if __name__ == "__main__":
    # Simple AND gate dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    model = Logistic_Regression(lr=0.1, epochs=1000, re_lambda=0.5)
    result = model.fit(X, y)

    print("Parameters:", result["theta"].ravel())
    print("Predictions:", model.predict(X).ravel())
    print("Loss history (first 5):", result["losses"][:5])
