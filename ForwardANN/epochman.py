import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from dataset import dataset_train, dataset_test
from model import CaNNModel
from ANN2 import train_model, evaluate_model, train_val_split

target_scaler = dataset_train.target_scaler

def main():
    device = torch.device("cuda")
    print(f"Using device: {device}")
    params = json.load(open('params.json'))
    lr = params['lr']
    weight_decay = params['weight_decay']
    batch_size = params['batch_size']

    # Define the range of epochs to test
    epochs_list = [100, 500, 1000, 2000, 4000, 8000]

    # Prepare data loaders (same for all runs)
    train_sampler, val_sampler = train_val_split(dataset_train, val_size=0.2)
    test_sampler = RandomSampler(dataset_test)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)
    dropout_rate = 0.1

    # Collect all metrics
    metrics_data = []

    for epochs in epochs_list:
        print(f"\n=== Training with {epochs} epochs ===")
        model = CaNNModel(dropout_rate=dropout_rate).to(device)
        criterion = nn.HuberLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Train the model
        trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs)

        # Evaluate on train and test sets
        train_loss, train_pred, train_target = evaluate_model(trained_model, train_loader, criterion, device)
        test_loss, test_pred, test_target = evaluate_model(trained_model, test_loader, criterion, device)

        # Compute metrics
        train_metrics = compute_metrics(train_pred, train_target, target_scaler)
        test_metrics = compute_metrics(test_pred, test_target, target_scaler)

        # Store results
        metrics_data.append({
            'Epochs': epochs,
            'Train MAE': train_metrics['MAE'],
            'Train MSE': train_metrics['MSE'],
            'Train RMSE': train_metrics['RMSE'],
            'Train R2': train_metrics['R2'],
            'Test MAE': test_metrics['MAE'],
            'Test MSE': test_metrics['MSE'],
            'Test RMSE': test_metrics['RMSE'],
            'Test R2': test_metrics['R2'],
        })

    # Save metrics to Excel
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv('epochs_metrics.csv', index=False)
    print("Metrics saved to epochs_metrics.csv")

    # Generate plots
    plot_metrics(df_metrics)


from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def compute_metrics(predictions, targets, target_scaler):
    """Compute metrics from predictions and targets."""
    # Inverse transform using the scaler
    preds = target_scaler.inverse_transform(predictions.reshape(-1, 1))
    targets = target_scaler.inverse_transform(targets.reshape(-1, 1))

    mae = np.mean(np.abs(preds - targets))
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, preds)

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

def plot_metrics(df_metrics):
    """Plot training and test metrics vs epochs."""
    plt.figure(figsize=(15, 10))

    # MAE Plot
    plt.subplot(2, 2, 1)
    plt.plot(df_metrics['Epochs'], df_metrics['Train MAE'], label='Train')
    plt.plot(df_metrics['Epochs'], df_metrics['Test MAE'], label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('MAE vs Epochs')

    # MSE Plot
    plt.subplot(2, 2, 2)
    plt.plot(df_metrics['Epochs'], df_metrics['Train MSE'], label='Train')
    plt.plot(df_metrics['Epochs'], df_metrics['Test MSE'], label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE vs Epochs')

    # RMSE Plot
    plt.subplot(2, 2, 3)
    plt.plot(df_metrics['Epochs'], df_metrics['Train RMSE'], label='Train')
    plt.plot(df_metrics['Epochs'], df_metrics['Test RMSE'], label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('RMSE vs Epochs')

    # R² Plot
    plt.subplot(2, 2, 4)
    plt.plot(df_metrics['Epochs'], df_metrics['Train R2'], label='Train')
    plt.plot(df_metrics['Epochs'], df_metrics['Test R2'], label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.legend()
    plt.title('R² vs Epochs')

    plt.tight_layout()
    plt.savefig('metrics_vs_epochs.png')
    plt.show()

if __name__ == '__main__':
    main()
