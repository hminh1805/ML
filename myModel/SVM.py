import numpy as np

import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = 0

    # -----------------------------
    # Train
    # -----------------------------
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Convert label: {0,1} -> {-1,1}
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.epochs):
            for i, x_i in enumerate(X):
                condition = y_[i] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    # Không vi phạm margin
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Vi phạm margin
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y_[i] * x_i)
                    self.b -= self.lr * (-y_[i])

            if epoch % 100 == 0:
                print(f"Epoch {epoch} completed")

    # -----------------------------
    # Predict
    # -----------------------------
    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)