import numpy as np
import pandas as pd

# import norm
from numpy.linalg import norm

true_data = pd.read_csv("true_params.csv")
initial_data = pd.read_csv("initial_params.csv")
calibrated_data = pd.read_csv("calibrated_params.csv")

error = norm(true_data["sigma"]-calibrated_data["sigma"], ord=2)
error2 = norm(true_data["sigma"]- initial_data["sigma"], ord=2)

print(error)
print(error2)

error3 = norm(true_data["V"]-calibrated_data["V"], ord=2)
error4 = norm(true_data["V"]-initial_data["V"], ord=2)

print(error3)
print(error4)
