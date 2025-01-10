import numpy as np
import pandas as pd

def calculate_loss(data_file):
    """
    Calculate various performance metrics between predictions and targets.

    Parameters:
    data_file (str): Path to the CSV file containing predictions and targets.

    Returns:
    dict: A dictionary containing detailed performance metrics.
    """
    # Load the CSV file
    data = pd.read_csv(data_file)
    # Extract predictions and targets as numpy arrays
    predictions = data['predictions'].to_numpy()
    targets = data['targets'].to_numpy()

    # Calculate various metrics
    absolute_errors = np.abs(predictions - targets)  # Absolute errors
    squared_errors = (predictions - targets) ** 2   # Squared errors

    mse = np.mean(squared_errors)  # Mean Squared Error
    mae = np.mean(absolute_errors)  # Mean Absolute Error

    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    # R^2 score
    r2 = 1 - (np.sum(squared_errors) / np.sum((targets - np.mean(targets)) ** 2))

    # Prepare detailed loss information
    loss_info = {
        'total_samples': len(predictions),
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R^2': r2,
        'min_error': np.min(absolute_errors),
        'max_error': np.max(absolute_errors),
        'std_error': np.std(absolute_errors)
    }

    return loss_info
