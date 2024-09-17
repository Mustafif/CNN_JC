import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded data shape: {df.shape}")
    print(df.head())
    return df

def calculate_returns(prices):
    return np.log(prices[1:] / prices[:-1])

# Custom HN-GARCH negative log-likelihood loss function
@tf.function
def hn_garch_loss(y_true, y_pred):
    returns = y_true[:, 0]  # Assuming the first column of y_true contains returns
    omega, alpha, beta, gamma, lambda_ = tf.unstack(y_pred, axis=-1)
    
    T = tf.shape(returns)[0]
    h = tf.TensorArray(tf.float32, size=T)
    h = h.write(0, omega / (1 - alpha - beta))
    
    log_likelihood = tf.constant(0.0, dtype=tf.float32)
    
    for t in range(1, T):
        h_t = omega + alpha * tf.square(returns[t-1] - gamma * tf.sqrt(h.read(t-1))) + beta * h.read(t-1)
        h = h.write(t, h_t)
        log_likelihood += -0.5 * (tf.math.log(2 * np.pi) + tf.math.log(h_t) + tf.square(returns[t]) / h_t)
    
    return -log_likelihood  # Return negative log-likelihood as we want to minimize the loss

def create_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(5, activation='softplus')  # Ensure positive outputs for GARCH parameters
    ])
    model.compile(optimizer='adam', loss=hn_garch_loss)
    return model

def prepare_data(returns, lookback=20):
    X, y = [], []
    for i in range(lookback, len(returns)):
        X.append(returns[i-lookback:i])
        y.append(returns[i])
    return np.array(X), np.array(y).reshape(-1, 1)

def train_model(model, X, y, epochs=100, batch_size=32):
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    return history

def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    try:
        # Load data
        df = load_data('week.csv')
        
        # Calculate returns
        returns = calculate_returns(df['S'].values)
        
        # Prepare data
        X, y = prepare_data(returns)
        
        # Create and train the model
        model = create_model(X.shape[1])
        history = train_model(model, X, y)
        
        # Plot training history
        plot_training_history(history)
        
        # Predict HN-GARCH parameters
        predicted_params = model.predict(X[-1].reshape(1, -1))[0]
        
        print("Estimated HN-GARCH parameters:")
        param_names = ['omega', 'alpha', 'beta', 'gamma', 'lambda']
        for name, value in zip(param_names, predicted_params):
            print(f"{name}: {value:.6f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()