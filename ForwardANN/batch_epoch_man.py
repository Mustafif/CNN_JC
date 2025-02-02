import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from dataset import dataset_train, dataset_test
from model import CaNNModel
from ANN2 import train_model, evaluate_model, train_val_split
from epochman import compute_metrics
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
target_scaler = dataset_train.target_scaler


def main():
    device = torch.device("cuda")
    print(f"Using device: {device}")
    params = json.load(open('params.json'))
    lr = params['lr']
    weight_decay = params['weight_decay']

    # Define hyperparameters to test
    batch_sizes = [64, 128, 256, 512, 1024]
    epochs_list = [100, 500, 1000, 2000, 4000, 8000]

    # Collect all metrics
    metrics_data = []

    for batch_size in batch_sizes:
        print(f"\n=== Testing Batch Size: {batch_size} ===")

        # Create data loaders for current batch size
        train_sampler, val_sampler = train_val_split(dataset_train, val_size=0.2)
        test_sampler = RandomSampler(dataset_test)

        train_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            dataset_test,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True
        )

        for epochs in epochs_list:
            print(f"\n--- Training with {epochs} epochs ---")
            model = CaNNModel(dropout_rate=0.1).to(device)
            criterion = nn.HuberLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

            # Train and evaluate
            trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs)
            _, train_pred, train_target = evaluate_model(trained_model, train_loader, criterion, device)
            _, test_pred, test_target = evaluate_model(trained_model, test_loader, criterion, device)

            # Calculate metrics
            train_metrics = compute_metrics(train_pred, train_target, target_scaler)
            test_metrics = compute_metrics(test_pred, test_target, target_scaler)

            # Store results
            metrics_data.append({
                'Batch Size': batch_size,
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

    # Save results
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv('hyperparameter_metrics.csv', index=False)
    print("\nMetrics saved to hyperparameter_metrics.csv")

    # Generate enhanced plots
    plot_metrics(df_metrics)

def plot_metrics(df_metrics):
    """Plot metrics for different batch sizes and epochs"""
    plt.figure(figsize=(15, 20))
    metrics = ['MAE', 'MSE', 'RMSE', 'R2']
    batch_sizes = df_metrics['Batch Size'].unique()

    for i, metric in enumerate(metrics, 1):
        plt.subplot(4, 1, i)
        for bs in batch_sizes:
            df_sub = df_metrics[df_metrics['Batch Size'] == bs]
            plt.plot(df_sub['Epochs'], df_sub[f'Test {metric}'],
                    marker='o', linestyle='-', label=f'BS={bs}')

        plt.title(f'Test {metric} Comparison')
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('batch_size_comparison.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
