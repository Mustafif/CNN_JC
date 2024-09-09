import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras

def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data shape: {df.shape}")
        print(df.head())

        if df.empty:
            raise ValueError("The CSV file is empty.")

        # Calculate returns
        df['returns'] = df['S'].pct_change()

        # Drop the first row (NaN return) and reset index
        df = df.dropna().reset_index(drop=True)

        print(f"Preprocessed data shape: {df.shape}")
        print(df.head())

        return df
    except Exception as e:
        print(f"Error in load_and_preprocess_data: {str(e)}")
        raise

def create_features(df, lookback=5):
    try:
        features = []
        for i in range(lookback, len(df)):
            feature = df.iloc[i-lookback:i][['V', 'Sigma', 'S', 'returns']].values.flatten()
            features.append(feature)

        features = np.array(features)
        print(f"Created features shape: {features.shape}")

        if features.size == 0:
            raise ValueError("No features were created. Check if the dataframe has enough rows.")

        return features
    except Exception as e:
        print(f"Error in create_features: {str(e)}")
        raise

def create_and_train_model(X, y, epochs=100, batch_size=32):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)  # Predicting a single value (next return)
    ])

    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    return model, history

def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_predictions(df, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(df['S'], label='Actual Stock Price')
    plt.plot(df['S'].iloc[5:] * (1 + predictions), label='Predicted Stock Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def main():
    try:
        # Load and preprocess data
        df = load_and_preprocess_data('../data_gen/week.csv')

        # Create features and targets
        X = create_features(df)
        y = df['returns'].iloc[5:].values

        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Scale features and targets
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

        # Create and train the model
        model, history = create_and_train_model(X_train, y_train)

        # Evaluate the model
        mse = model.evaluate(X_test, y_test)
        print(f"Mean Squared Error on test set: {mse}")

        # Plot training history
        plot_training_history(history)

        # Make predictions on the entire dataset
        predictions_scaled = model.predict(X_scaled)
        predictions = scaler_y.inverse_transform(predictions_scaled).flatten()

        # Plot actual vs predicted stock prices
        plot_predictions(df, predictions)

        # Print some sample predictions
        print("\nSample Predictions (Return):")
        for i in range(5):
            actual = df['returns'].iloc[i+5]
            predicted = predictions[i]
            print(f"Actual: {actual:.4f}, Predicted: {predicted:.4f}")

    except Exception as e:
        print(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()
