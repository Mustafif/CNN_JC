import numpy as np

def calculate_loss(data_file):
    """
    Calculate the mean absolute error (MAE) between predictions and targets.

    Parameters:
    data_file (str): Path to the text file containing predictions and targets

    Returns:
    float: Average loss (mean absolute error)
    dict: Detailed loss information
    """
    # Read the data from the file
    predictions = []
    targets = []

    with open(data_file, 'r') as file:
        # Skip the header
        next(file)

        for line in file:
            # Remove brackets and split
            pred_str, target_str = line.strip().split('],[')

            # Convert to float and remove any remaining brackets
            pred = float(pred_str.strip('[]'))
            target = float(target_str.strip('[]'))

            predictions.append(pred)
            targets.append(target)

    # Convert to numpy arrays for easier calculation
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate absolute error
    absolute_errors = np.abs(predictions - targets)

    # Calculate average loss (mean absolute error)
    mean_loss = np.mean(absolute_errors)

    # Prepare detailed loss information
    loss_info = {
        'total_samples': len(predictions),
        'mean_loss': mean_loss,
        'min_loss': np.min(absolute_errors),
        'max_loss': np.max(absolute_errors),
        'std_loss': np.std(absolute_errors)
    }

    return mean_loss, loss_info

# Example usage
if __name__ == '__main__':
    mean_loss, loss_details = calculate_loss('test_results.csv')
    print(f"Mean Absolute Error: {mean_loss}")
    print("Loss Details:")
    for key, value in loss_details.items():
        print(f"{key}: {value}")
