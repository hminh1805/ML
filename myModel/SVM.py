import numpy as np

class SVM:
    def __init__(self, learning_rate=0.1, lambda_param=0.01, epochs=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        best_loss = float('inf')
        best_w, best_b = self.w.copy(), self.b

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_[indices]

            lr_t = self.lr / (1 + self.lambda_param * epoch)

            for i, x_i in enumerate(X_shuffled):
                condition = y_shuffled[i] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    self.w -= lr_t * (2 * self.lambda_param * self.w)
                    
                else:
                    self.w -= lr_t * (2 * self.lambda_param * self.w - y_shuffled[i] * x_i)
                    self.b += lr_t * y_shuffled[i]

            loss = self._compute_loss(X, y_)
            if loss < best_loss:
                best_loss = loss
                best_w, best_b = self.w.copy(), self.b

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")
        
        self.w, self.b = best_w, best_b
        print(f"\nBest loss: {best_loss:.4f}")

    def _compute_loss(self, X, y_):
        """Hinge loss + L2 regularization"""
        margins = y_ * (np.dot(X, self.w.T) + self.b)
        hinge = np.maximum(0, 1 - margins)
        return self.lambda_param * np.dot(self.w, self.w) + np.mean(hinge)

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.where(approx >= 0, 1, 0)
    
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
