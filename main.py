import numpy as np
import pandas as pd
from arch import arch_model

# Set random seed for reproducibility
np.random.seed(42)

# Define GARCH model parameters
omega = 0.1
alpha = 0.05
beta = 0.9

# Number of data points
n = 1000  

# Generate GARCH(1,1) process
garch_model = arch_model(None, vol='Garch', p=1, q=1, mean='Zero', dist='Normal')
simulated_data = garch_model.simulate([omega, alpha, beta], nobs=n)

# Extract simulated returns and conditional volatility
simulated_returns = simulated_data['data']
simulated_volatility = simulated_data['volatility']

from scipy.stats import norm

# Black-Scholes option pricing formula
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

# Generate option prices
strike_prices = np.linspace(80, 120, 10)
maturities = np.linspace(0.1, 1, 10)
risk_free_rate = 0.01
underlying_price = 100  # Assume the underlying asset price is 100

option_prices = []
for i in range(n):
    for K in strike_prices:
        for T in maturities:
            sigma = simulated_volatility[i]
            price = black_scholes_price(underlying_price, K, T, risk_free_rate, sigma)
            option_prices.append((underlying_price, K, T, sigma, price))

# Convert to DataFrame
option_prices_df = pd.DataFrame(option_prices, columns=['S', 'K', 'T', 'sigma', 'price'])

# Prepare features and target
features = option_prices_df[['S', 'K', 'T', 'sigma']]
target = option_prices_df['price']

# Split into training and validation sets
from sklearn.model_selection import train_test_split

features_train, features_val, target_train, target_val = train_test_split(features, target, test_size=0.2, random_state=42)

from keras.models import Sequential
from keras.layers import Dense

# Define the ANN architecture
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))  # input_dim should be 4 since we have 4 features: 'S', 'K', 'T', 'sigma'
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(features_train, target_train, epochs=100, batch_size=32, validation_data=(features_val, target_val))

from scipy.optimize import minimize

# Initial GARCH parameters for optimization
initial_params = [omega, alpha, beta]

# Define a function to update GARCH parameters and recalculate simulated volatility
def update_garch_parameters(params):
    garch_model = arch_model(simulated_returns, vol='Garch', p=1, q=1, mean='Zero')
    garch_fit = garch_model.fit(update_freq=0, starting_values=params)
    return garch_fit.conditional_volatility, garch_fit.loglikelihood

# Joint objective function with batching
def joint_objective(params):
    simulated_volatility, loglikelihood = update_garch_parameters(params)
    
    # Adjust features to include the updated volatility
    features_batch = features.copy()
    features_batch['sigma'] = simulated_volatility[:len(features)]
    
    # Split into smaller batches to handle memory issues
    batch_size = 10000  # Adjust batch size based on available memory
    total_error = 0
    
    for start_idx in range(0, len(features), batch_size):
        end_idx = min(start_idx + batch_size, len(features))
        
        features_sub = features_batch.iloc[start_idx:end_idx]
        target_sub = target.iloc[start_idx:end_idx]
        
        # Predict prices in batches
        predicted_prices_sub = model.predict(features_sub)
        
        # Calculate the error for the batch
        batch_error = np.mean((predicted_prices_sub.flatten() - target_sub.values) ** 2)
        total_error += batch_error
    
    # Calculate the average error across all batches
    option_price_error = total_error / (len(features) // batch_size + 1)
    
    # Combine the option price error with the negative log likelihood for GARCH model
    return option_price_error - loglikelihood

# Use scipy.optimize to minimize the joint objective function
result = minimize(joint_objective, initial_params, bounds=[(0, None), (0, 1), (0, 1)])
optimized_params = result.x

print(f'Optimized GARCH Parameters: {optimized_params}')


