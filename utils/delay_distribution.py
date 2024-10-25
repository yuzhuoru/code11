import numpy as np

class GamaDistribution:
    def __init__(self, max_delay=6):

        self.max_delay = max_delay

    def dis_sample(self):
        return np.random.choice([1, 2, 3, 4, 5, 6], p=[0.40438, 0.29753, 0.16418, 0.08053, 0.03703, 0.01635])

    def dis_probability(self):
        return np.array([0, 0.40438, 0.29753, 0.16418, 0.08053, 0.03703, 0.01635])

 
class UniformDistribution:
    def __init__(self, max_delay=9):

        self.max_delay = max_delay

    def dis_sample(self):
        return np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.11112,0.11111,0.11111,0.11111,0.11111,0.11111,0.11111,0.11111,0.11111])

    def dis_probability(self):
        return np.array([0,0.11112,0.11111,0.11111,0.11111,0.11111,0.11111,0.11111,0.11111,0.11111])
    

class DoubleGaussianDistribution:
    def __init__(self, max_delay=10):

        self.max_delay = max_delay

    def dis_sample(self):
        return np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], p=[0.03982, 0.12356, 0.15381, 0.13144, 0.10588, 0.13168, 0.15682, 0.11015, 0.04098, 0.00586])

    def dis_probability(self):
        return np.array([0, 0.03982, 0.12356, 0.15381, 0.13144, 0.10588, 0.13168, 0.15682, 0.11015, 0.04098, 0.00586])