import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from scipy.optimize import minimize

alpha_range = [1e-7, 1e-5]
beta_range = [0.5, 0.7]
omega_range = [1e-7, 1e-5]
gamma_range = [400, 600]
lambda_range = [0.4, 0.6]
