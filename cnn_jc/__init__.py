# The parameters we would want to calibrate are: 
# theta = (omega, alpha, beta)

# Will require data to be able to load options and returns data 
# When we have a NeuralNetwork class, we can use the model returned 
# from the `train_model` method, we can use `model.predict`. 

from garch import generate_garch_data
from cnn import NeuralNetwork, Data
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Generate GARCH data
n_samples = 1000
mean = 0
std_dev = 1
alpha = 0.1
beta = 0.1
omega = 0.1

returns, volatility = generate_garch_data(n_samples, mean, std_dev, alpha, beta, omega)

X_train, X_test = train_test_split(returns, test_size=0.2, random_state=0)
Y_train, Y_test = train_test_split(volatility, test_size=0.2, random_state=0)

# Reshape the input data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))


classifier = Sequential()

classifier.add(LSTM(units=6, activation='relu', input_shape=(X_train.shape[1], 1)))
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, Y_train, batch_size=10, epochs=100)

Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

print(Y_pred)

cm = confusion_matrix(Y_test, Y_pred)
print(cm)

accuracy_score(Y_test, Y_pred)
