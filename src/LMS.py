import numpy as np

class LMS:
    def __init__(self, M=10, lr=None):
        self.M = M
        self.lr = lr
        self.b = np.zeros(self.M) 
        
    def fit_transform(self, f, d):
        n_time = f.shape[0]
        loss = []
        y = np.copy(f)

        for n in range(self.M - 1, n_time):
            f_window = f[n - self.M + 1:n + 1][::-1]
            y_n = np.dot(f_window, self.b)

            y[n] = y_n

            e = d[n] - y_n
            loss.append(e ** 2)

            self.b = self.b + self.lr * e * f_window

        return loss, y

    def transform(self, f):
        n_time = f.shape[0]
        y = np.copy(f)

        for n in range(self.M - 1, n_time):
            f_window = f[n - self.M + 1:n + 1][::-1]
            y_n = np.dot(f_window, self.b)

            y[n] = y_n

        return y
                
    def find_lr(self, f, safety=0.1):
        f = np.asarray(f, dtype=np.float64)

        X = np.lib.stride_tricks.sliding_window_view(f, self.M)
        X = X[:, ::-1]

        R = (X.T @ X) / len(X)
        lambda_max = np.linalg.eigvalsh(R)[-1]
        lr_max = 2.0 / lambda_max
        lr = safety * lr_max

        return lr, lr_max