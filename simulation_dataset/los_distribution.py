import numpy as np


class LoSDistribution:

    def __init__(self, mean, std, min, max):
        self.mu = mean
        self.sigma = std # mean and standard deviation (in mins)
        self.max = max
        self.min = min

    def sample(self):
        return max(self.min, min(np.random.normal(self.mu, self.sigma, 1)[0], self.max))

