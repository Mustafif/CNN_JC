from model import CaNNModel
from dataset import dataset_test, dataset_train
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
import pandas as pd
import numpy as np
import json
import os
import optuna
from ANN2 import train_model, evaluate_model, train_val_split
from loss import calculate_loss


# Global variables to store the best model and metrics
best_model_state = None
best_metrics = {}


def objective(trial):
    global best_model_state, best_metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define hyperparameter search space
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    epochs = trial.suggest_int("epochs", 50, 200)

    # Create samplers
    train_sampler, val_sampler = train_val_split(dataset_train, val_size=0.1)
    test_sampler = RandomSampler(dataset_test)

    # Create data loaders
    train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, sampler=test_sampler)

    # Model setup
    model = CaNNModel().to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs)

    # Evaluation
    train_loss, train_pred, train_target = evaluate_model(model, train_loader, criterion, device)
    test_loss, test_pred, test_target = evaluate_model(model, test_loader, criterion, device)

    # If this is the best trial, save the model state and metrics
    if trial.number == 0 or test_loss < trial.study.best_value:
        best_model_state = model.state_dict()  # Save model state_dict
        best_metrics = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_predictions": np.array(train_pred).flatten().tolist(),
            "train_targets": np.array(train_target).flatten().tolist(),
            "test_predictions": np.array(test_pred).flatten().tolist(),
            "test_targets": np.array(test_target).flatten().tolist(),
        }

    return test_loss


def main():
    global best_model_state, best_metrics

    # Create an Optuna study
    study = optuna.create_study(direction="minimize")  # Minimize test loss
    study.optimize(objective, n_trials=50)  # Adjust n_trials as needed

    # Display the best hyperparameters
    print("\nBest hyperparameters:")
    print(study.best_params)

    # Save best hyperparameters to JSON
    with open('params.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)

    # Save the best metrics and predictions
    train_df = pd.DataFrame({
        'predictions': best_metrics["train_predictions"],
        'targets': best_metrics["train_targets"]
    })
    train_df.to_csv('train_results.csv', index=False)

    test_df = pd.DataFrame({
        'predictions': best_metrics["test_predictions"],
        'targets': best_metrics["test_targets"]
    })
    test_df.to_csv('test_results.csv', index=False)

    print("In-sample (Training) Performance:")
    train_loss_details = calculate_loss('train_results.csv')
    for key, value in train_loss_details.items():
        print(f"{key}: {value}")

    print("\nOut-of-sample (Test) Performance:")
    test_loss_details = calculate_loss('test_results.csv')
    for key, value in test_loss_details.items():
        print(f"{key}: {value}")

    # Save metrics to JSON
    metrics = {
        "in_sample": train_loss_details,
        "out_of_sample": test_loss_details
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Save the best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaNNModel().to(device)
    model.load_state_dict(best_model_state)
    scripted_model = torch.jit.script(model)
    scripted_model.save("model.pt")
    print("Best model saved as 'model.pt'")


if __name__ == "__main__":
    main()
