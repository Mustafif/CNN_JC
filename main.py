import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate GARCH data
def generate_garch_data(n_samples, mean, std_dev, alpha, beta, omega):
    returns = np.zeros(n_samples)
    volatility = np.zeros(n_samples)
    volatility[0] = omega / (1 - alpha - beta)  # Initial volatility

    for t in range(1, n_samples):
        volatility[t] = omega + alpha * returns[t-1]**2 + beta * volatility[t-1]
        returns[t] = np.random.normal(mean, np.sqrt(volatility[t]))
    
    return returns, volatility

n_samples = 1000
mean = 0
std_dev = 1
alpha = 0.1
beta = 0.1
omega = 0.1

returns, volatility = generate_garch_data(n_samples, mean, std_dev, alpha, beta, omega)

# Convert volatility to binary classes (e.g., 0 for low volatility, 1 for high volatility)
threshold = np.median(volatility)
binary_volatility = (volatility > threshold).astype(int)

# Combine returns and binary_volatility
data = np.column_stack((returns, binary_volatility))
train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)

X_train = train_data[:, 0].reshape(-1, 1)
Y_train = train_data[:, 1]
X_test = test_data[:, 0].reshape(-1, 1)
Y_test = test_data[:, 1]

# Reshape data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
# Define the LSTM model with additional hidden layers
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(units=50, activation='relu', return_sequences=True))
model.add(LSTM(units=50, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=1)

# Make predictions
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy}')

cm = confusion_matrix(Y_test, Y_pred)
print(f'Confusion Matrix:\n{cm}')
