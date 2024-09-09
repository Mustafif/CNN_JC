import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from scipy.optimize import minimize

# Generate sample data (replace this with your actual data)
np.random.seed(42)
returns = np.random.normal(0, 1, 1000)

# Define HN-GARCH model
def hn_garch_model(params, returns):
    omega, alpha, beta, gamma, lambda_ = params
    n = len(returns)
    h = np.zeros(n)
    h[0] = np.var(returns)
    for t in range(1, n):
        h[t] = omega + alpha * (returns[t-1] - gamma * np.sqrt(h[t-1]))**2 + beta * h[t-1]
    return h

# Define loss function (negative log-likelihood)
def neg_log_likelihood(params, returns):
    h = hn_garch_model(params, returns)
    return -np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * np.log(h) - 0.5 * returns**2 / h)

# Create and compile the neural network
def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(1,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(5, activation='softplus')  # Ensure positive HN-GARCH parameters
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Generate training data
def generate_training_data(n_samples=10000):
    X = np.random.uniform(0, 1, (n_samples, 1))
    y = np.zeros((n_samples, 5))
    for i in range(n_samples):
        # Generate parameters within typical ranges
        omega = np.random.uniform(1e-6, 1e-4)
        alpha = np.random.uniform(1e-6, 0.2)
        beta = np.random.uniform(0.7, 0.99)
        gamma = np.random.uniform(-0.5, 0.5)
        lambda_ = np.random.uniform(-0.5, 0.5)
        y[i] = [omega, alpha, beta, gamma, lambda_]
    return X, y

# Train the model
X_train, y_train = generate_training_data()
model = create_model()
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Use the trained model to predict HN-GARCH parameters
X_test = np.array([[0.5]])  # Example input
predicted_params = model.predict(X_test)[0]

print("Predicted HN-GARCH parameters:")
print(f"omega: {predicted_params[0]:.6f}")
print(f"alpha: {predicted_params[1]:.6f}")
print(f"beta: {predicted_params[2]:.6f}")
print(f"gamma: {predicted_params[3]:.6f}")
print(f"lambda: {predicted_params[4]:.6f}")

# Validate the results using traditional optimization
def objective(params):
    return neg_log_likelihood(params, returns)

initial_guess = [0.00005, 0.1, 0.8, 0, 0]
bounds = [(1e-6, 1e-4), (1e-6, 0.2), (0.7, 0.99), (-0.5, 0.5), (-0.5, 0.5)]
result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)

print("\nParameters estimated by traditional optimization:")
print(f"omega: {result.x[0]:.6f}")
print(f"alpha: {result.x[1]:.6f}")
print(f"beta: {result.x[2]:.6f}")
print(f"gamma: {result.x[3]:.6f}")
print(f"lambda: {result.x[4]:.6f}")
