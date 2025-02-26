from dataset import cleandataset, dataset_file
from model import CaNNModel, NetworkOfNetworks
# from dataset import dataset_test, dataset_train
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
import pandas as pd
from loss import calculate_loss
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from dataset import OptionDataset
# target_scaler = dataset_train.target_scaler

def train_model(model: torch.nn.Module, train_loader, val_loader, criterion, optimizer, device, epochs):
    """Train the model with improved learning rate scheduling and regularization"""
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'],
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    train_losses = []
    val_losses = []
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=5
    # )
    # best_val_loss = float('inf')
    # best_model_state = None
    # l1_lambda = 1e-4  # L1 regularization factor

    # for epoch in range(epochs):
    #     model.train()
    #     train_loss = 0

    #     for batch_X, batch_y in train_loader:
    #         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
    #         optimizer.zero_grad()
    #         output = model(batch_X.float())
    #         target = batch_y.float().view_as(output)
    #         # Add L1 regularization
    #         # l1_loss = 0
    #         # for param in model.parameters():
    #         #     l1_loss += torch.sum(torch.abs(param))
    #         loss = criterion(output, target) # + l1_lambda * l1_loss
    #         loss.backward()
    #         # Add gradient clipping
    #         # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    #         optimizer.step()
    #         scheduler.step()
    #         train_loss += loss.item()
    # Initialize GradScaler
    scaler = torch.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            # Mixed Precision Context
            with torch.amp.autocast(device_type="cuda"):  # Automatically uses FP16 precision where applicable
                output = model(batch_X.float())
                target = batch_y.float().view_as(output)
                loss = criterion(output, target)

            # Scales the loss and performs the backward pass
            scaler.scale(loss).backward()

            # Scales the gradients and steps the optimizer
            scaler.step(optimizer)

            # Updates the scaler to the next step
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X.float())
                target = batch_y.float().view_as(output)
                loss = criterion(output, target)
                val_loss += loss.item()
                # scheduler.step(val_loss)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print('Epoch {}: Train Loss {:.4f} Val Loss {:.4f}'.format(epoch + 1, avg_train_loss, avg_val_loss))

        # # Save best model state
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss

    return model, train_losses, val_losses



def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            # noise = torch.randn_like(X) * 0.05 # Add Gaussian noise to input
            output = model(X)
            # Reshape target to match output dimensions
            target = Y.float().view_as(output)
            loss = criterion(output, target)
            total_loss += loss.item()

            # Store predictions and targets as flattened arrays
            predictions.extend(output.cpu().numpy().flatten())
            targets.extend(Y.cpu().numpy().flatten())  # Original target shape is preserved here for correct metrics

    avg_loss = total_loss / len(data_loader)
    return avg_loss, np.array(predictions), np.array(targets)

# Split existing dataset into training and validation sets
def train_val_split(dataset, val_size=0.2, random_state=42):
    # Get indices of the full dataset
    indices = list(range(len(dataset)))

    # Split indices into train and validation
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_size,
        random_state=random_state
    )

    # Create samplers for train and validation
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler

def main(dataset_train, dataset_test, name):
    torch.set_float32_matmul_precision('high')
    num_workers = 6
    device = torch.device("cuda")
    print(f"Using device: {device}")
    params = json.load(open('params.json'))
    lr = params['lr']
    weight_decay = params['weight_decay']
    batch_size = params['batch_size']

    epochs = params['epochs']
    target_scaler = dataset_train.target_scaler

    # Create samplers
    train_sampler, val_sampler = train_val_split(dataset_train, val_size=0.2)
    test_sampler = RandomSampler(dataset_test)

    # Create data loaders
    train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(
        dataset_train,  # Using test dataset for validation
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(dataset_test, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True)
    dropout_rate = params['dropout_rate']  # Increased dropout rate for stronger regularization
    # Model setup
    model = CaNNModel(dropout_rate=dropout_rate).to(device)
    #model = torch.compile(model)
    criterion = nn.HuberLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,  # Increased weight decay for stronger L2 regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    # Training
    trained_model, tl, vl = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs)

    # Evaluation
    train_loss, train_pred, train_target = evaluate_model(trained_model, train_loader, criterion, device)
    test_loss, test_pred, test_target = evaluate_model(trained_model, test_loader, criterion, device)

    # Convert predictions and targets to plain floats
    train_pred = [float(x) for x in train_pred]
    train_target = [float(x) for x in train_target]

    test_pred = [float(x) for x in test_pred]
    test_target = [float(x) for x in test_target]

    # Save results
    train_df = pd.DataFrame({
        'predictions': train_pred,
        'targets': train_target
    })
    train_df.to_csv('train_results.csv', index=False)

    test_df = pd.DataFrame({
        'predictions': test_pred,
        'targets': test_target
    })
    test_df.to_csv('test_results.csv', index=False)

    # Calculate and print loss details
    print("In-sample (Training) Performance:")
    train_loss_details = calculate_loss('train_results.csv', target_scaler)
    for key, value in train_loss_details.items():
        print(f"{key}: {value}")

    print("\nOut-of-sample (Test) Performance:")
    test_loss_details = calculate_loss('test_results.csv', target_scaler)
    for key, value in test_loss_details.items():
        print(f"{key}: {value}")

    # Save metrics
    metrics = {
        "in_sample": train_loss_details,
        "out_of_sample": test_loss_details
    }
    if not os.path.exists('metrics.json'):
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f)
            scripted_model = torch.jit.script(trained_model)
            scripted_model.save("model.pt")
    else:
        with open('metrics.json', 'r') as f:
            data = json.load(f)
            if test_loss_details['MAE'] < data['out_of_sample']['MAE']:
                print("New model is better than the previous one. Saving new model...")
                with open('metrics.json', 'w') as f:
                    json.dump(metrics, f)
                    scripted_model = torch.jit.script(trained_model)
                    scripted_model.save("model.pt")
    save_model_checkpoint(trained_model, name, metrics, tl, vl)

from datetime import datetime

def save_model_checkpoint(trained_model, name, metrics, tl, vl):
    # Create base directory
    base_dir = "saved_models"
    os.makedirs(base_dir, exist_ok=True)

    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d")
    save_dir = os.path.join(base_dir, f"{name}_{timestamp}")

    # Prompt user for confirmation
    # user_input = input(f"Save model to folder '{save_dir}'? [y/n]: ").lower()


        # Create directory structure
    os.makedirs(save_dir, exist_ok=True)

        # Define paths
    metrics_path = os.path.join(save_dir, "metrics.json")
    model_path = os.path.join(save_dir, "model.pt")
    graph_path = os.path.join(save_dir, "learning_curve.png")

        # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

        # Save model
    scripted_model = torch.jit.script(trained_model)
    scripted_model.save(model_path)

        # Save learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(tl, label="Train Loss")
    plt.plot(vl, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(graph_path)
    plt.close()

    print("\nSaved successfully to:")
    print(f"📁 {save_dir}/")
    print("├── 📄 metrics.json")
    print("└── 🧠 model.pt\n")
    print("└── 📉 learning_curve.png\n")

class DS:
    def __init__(self, path, path2, name):
        self.path = path
        self.path2 = path2
        self.name = name
    def datasets(self):
        if self.path2 is None:
            # Load and clean the dataset
            df = cleandataset(dataset_file(self.path))

            # Split into train and test
            train_df, test_df = train_test_split(df)

            # Create OptionDataset for train
            ds_train = OptionDataset(train_df, is_train=True)

            # Create OptionDataset for test with shared scaler
            ds_test = OptionDataset(test_df, is_train=False, target_scaler=ds_train.target_scaler)

            return ds_train, ds_test
        else:
            # Load and clean both datasets
            train = cleandataset(dataset_file(self.path))
            test = cleandataset(dataset_file(self.path2))

            # Create OptionDataset for train
            ds_train = OptionDataset(train, is_train=True)

            # Create OptionDataset for test with shared scaler
            ds_test = OptionDataset(test, is_train=False, target_scaler=ds_train.target_scaler)

            return ds_train, ds_test



if __name__ == '__main__':
    datasets = [
        DS("train_dataset.csv", "test_dataset.csv", "stage1_HN"),
        DS("../data_gen/stage1b.csv", None, "stage1b_HN"),
        DS("../data_gen/stage2.csv", None, "stage2_HN"),
        DS("../data_gen/stage3.csv", None, "stage3_HN"),
        DS("../data_gen/Duan_Garch/stage3.csv", None, "stage3_Duan"),
        DS("../data_gen/GJR_Garch/stage3_gjr.csv", None, "stage3_GJR"),
        DS("../data_gen/Duan_Garch/stage1b.csv", None, "stage1b_Duan"),
        DS("../data_gen/Duan_Garch/stage2.csv", None, "stage2_Duan"),
        DS("../data_gen/GJR_Garch/stage1b.csv", None, "stage1b_GJR"),
        DS("../data_gen/GJR_Garch/stage2.csv", None, "stage2_GJR")
    ]
    for dataset in datasets:
        ds_train, ds_test = dataset.datasets()
        main(ds_train, ds_test, dataset.name)
