import numpy as np

class LMS:
    def __init__(self, M = 1280):
        self.M = M

    def fit_transform(self, f, d, lr=None):
        b = np.zeros(self.M)
        n_time = min(f.shape[0], d.shape[0])
        f = f.astype(np.float64)[:n_time] / 32768.0
        d = d.astype(np.float64)[:n_time] / 32768.0
        
        if lr == None:
            lr, _ = self._find_lr(f) 

        y = np.zeros(n_time)
        
        for n in range(self.M - 1, n_time):
            f_window = f[n - self.M + 1:n + 1][::-1]
            y_n = np.dot(f_window, b)
            y[n] = y_n

            e = d[n] - y_n

            b = b + lr * e * f_window
        
        return d - y

    def _find_lr(self, f, safety=0.1):
        X = np.lib.stride_tricks.sliding_window_view(
            np.asarray(f, dtype=np.float64),
            self.M
        )

        max_energy = np.max(np.sum(X * X, axis=1))

        lr_max = 2.0 / max_energy
        return safety * lr_max, lr_max