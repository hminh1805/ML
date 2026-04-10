import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    # -----------------------------
    # Sigmoid
    # -----------------------------
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  # tránh overflow
        return 1 / (1 + np.exp(-z))

    # -----------------------------
    # Loss (Binary Cross Entropy)
    # -----------------------------
    def compute_loss(self, y, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # -----------------------------
    # Train
    # -----------------------------
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Đảm bảo y có dạng (n,1)
        y = y.reshape(-1, 1)

        # Khởi tạo
        self.w = np.zeros((n_features, 1))
        self.b = 0.0

        for epoch in range(self.epochs):
            # Forward
            z = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(z)

            # Loss
            loss = self.compute_loss(y, y_pred)

            # Gradient
            grad_w = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            grad_b = (1 / n_samples) * np.sum(y_pred - y)

            # Update
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            # In log mỗi 100 epoch
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

    # -----------------------------
    # Predict xác suất
    # -----------------------------
    def predict_proba(self, X):
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    # -----------------------------
    # Predict nhãn
    # -----------------------------
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int).flatten()