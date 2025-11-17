import numpy as np

class PSO:
    def __init__(self, n_particles, dim, lb, ub, max_iter=100, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.dim = dim
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.X = np.random.uniform(self.lb, self.ub, (n_particles, dim))
        self.V = np.zeros((n_particles, dim))
        self.pbest = self.X.copy()
        self.pbest_val = np.full(n_particles, np.inf)
        self.gbest = None
        self.gbest_val = np.inf

    def optimize(self, objective):
        history = []
        for it in range(self.max_iter):
            for i in range(self.n_particles):
                val = objective(self.X[i])
                if val < self.pbest_val[i]:
                    self.pbest_val[i] = val
                    self.pbest[i] = self.X[i].copy()
                if val < self.gbest_val:
                    self.gbest_val = val
                    self.gbest = self.X[i].copy()
            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)
            cognitive = self.c1 * r1 * (self.pbest - self.X)
            social = self.c2 * r2 * (self.gbest - self.X)
            self.V = self.w * self.V + cognitive + social
            self.X = self.X + self.V
            self.X = np.minimum(np.maximum(self.X, self.lb), self.ub)
            history.append(self.gbest_val)
        return self.gbest, self.gbest_val, history
