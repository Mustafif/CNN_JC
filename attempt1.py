import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.initializers import glorot_uniform

import matplotlib.pyplot as plt

# Generate GARCH data
def generate_garch_data(n_samples, mean, alpha, beta, omega):
    returns = np.zeros(n_samples)
    volatility = np.zeros(n_samples)
    volatility[0] = omega / (1 - alpha - beta)  # Initial volatility

    for t in range(1, n_samples):
        volatility[t] = omega + alpha * returns[t-1]**2 + beta * volatility[t-1]
        returns[t] = np.random.normal(mean, np.sqrt(volatility[t]))
    
    return returns, volatility

# Set parameters
n_samples = 1000
mean = 0
alpha = 0.1
beta = 0.1
omega = 0.1
layers = 8
units = 500
epochs = 200
activation = 'softplus'
kernel_init = glorot_uniform
dropout = 0.5
early_stop = 125 
lr_patience = 40 
reduce_lr = 0.5
reduce_lr_min = 0.000009
loss = 'binary_crossentropy'
optimizer = 'adam'

# Generate GARCH data
returns, volatility = generate_garch_data(n_samples, mean, alpha, beta, omega)

# Convert volatility to binary classes (e.g., 0 for low volatility, 1 for high volatility)
threshold = np.median(volatility)
binary_volatility = (volatility > threshold).astype(int)
scaler = StandardScaler()

returns_scaled = scaler.fit_transform(returns.reshape(-1, 1))

# Combine returns and binary_volatility
data = np.column_stack((returns_scaled, binary_volatility))
train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)

X_train = train_data[:, 0].reshape(-1, 1)
Y_train = train_data[:, 1].reshape(-1, 1)
X_test = test_data[:, 0].reshape(-1, 1)
Y_test = test_data[:, 1].reshape(-1, 1)

# Reshape data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Define the LSTM model with additional hidden layers
model = Sequential()

# Define a learning rate schedule function
def lr_schedule(epoch):
    initial_learning_rate = 0.01
    decay_rate = 0.1
    decay_steps = 100
    return initial_learning_rate * decay_rate ** (epoch // decay_steps)

# Add layers to the model
for i in range(layers):
    if i == 0: 
        model.add(Dense(units, input_shape=(X_train.shape[1],), kernel_initializer=kernel_init))
    else:
        model.add(Dense(units, kernel_initializer=kernel_init))
    model.add(Activation(activation))
    model.add(Dropout(dropout))
    model.add(Dense(Y_train.shape[1], kernel_initializer=kernel_init))

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'precision'])
lr_scheduler = LearningRateScheduler(lr_schedule)

# Define callbacks
callbacks = [lr_scheduler]
if early_stop is not None:
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    callbacks.append(early_stopping)
if reduce_lr is not None:
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr, patience=lr_patience,
                                  min_lr=reduce_lr_min, verbose=1)
    callbacks.append(reduce_lr)

# Train the model
model.fit(X_train, Y_train, epochs=epochs, batch_size=16, verbose=1, callbacks=callbacks)

# Make predictions
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy}')

cm = confusion_matrix(Y_test, Y_pred)
print(f'Confusion Matrix:\n{cm}')
