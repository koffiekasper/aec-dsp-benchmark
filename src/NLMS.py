import numpy as np

class NLMS:
    def __init__(self, M = 1280, lr=0.5, eps=1e-3):
        self.M = M
        self.lr = lr
        self.eps = eps

    def fit_transform(self, f, d):
        self.b = np.zeros(self.M)
        n_time = min(f.shape[0], d.shape[0])
        f = f.astype(np.float64)[:n_time]
        d = d.astype(np.float64)[:n_time]
        
        y = np.zeros(n_time)
        
        for n in range(self.M - 1, n_time):
            f_window = f[n - self.M + 1:n + 1][::-1]
            y_n = np.dot(f_window, self.b)
            y[n] = y_n

            e = d[n] - y_n
            energy = np.dot(f_window, f_window) + self.eps

            self.b += (self.lr / energy) * e * f_window
             
        return d - y