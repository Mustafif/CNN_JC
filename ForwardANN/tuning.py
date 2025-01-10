from model import CaNNModel
from dataset import dataset_test, dataset_train
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from loss import calculate_loss
import time
import optuna

from ANN2 import train_model, evaluate_model, train_val_split

def objective(trial):
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Define hyperparameter search space
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
   # dropout = trial.suggest_categorical("dropout_rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    epochs = trial.suggest_int("epochs", 50, 200)  # You can limit epochs to speed up tuning

    # Create samplers
    train_sampler, val_sampler = train_val_split(dataset_train, val_size=0.1)
    test_sampler = RandomSampler(dataset_test)

    # Create data loaders
    train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, sampler=test_sampler)
    val_loader = DataLoader(
        dataset_train,  # Note: using same dataset, different sampler
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    # Model setup
    model = CaNNModel().to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs)

    # Evaluation
    test_loss, _, _ = evaluate_model(trained_model, test_loader, criterion, device)

    # Return the test loss as the metric to minimize
    return test_loss

def main():
    # Create an Optuna study
    study = optuna.create_study(direction="minimize")  # We minimize test loss

    # Optimize the study
    study.optimize(objective, n_trials=50)  # Adjust n_trials as needed

    # Display the best hyperparameters
    print("\nBest hyperparameters:")
    print(study.best_params)
# Save best hyperparameters to JSON
    import json
    with open('params.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)

    # Save the best hyperparameters and study results
    study_df = study.trials_dataframe()
    study_df.to_csv("optuna_results2.csv", index=False)


if __name__ == '__main__':
    main()
