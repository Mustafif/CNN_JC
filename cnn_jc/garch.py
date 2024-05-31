import random
from math import pow


class GarchGen: 
    def __init__(self, long_term_vol=1): 
        self.alpha = random.uniform(0, 1)
        self.beta = random.uniform(0, 1-self.alpha)
        self.gamma = 1 - self.alpha - self.beta
        long_term_vol = random.uniform(0, long_term_vol)
        self.omega = pow(long_term_vol, 2) * (self.gamma + 0.000001)