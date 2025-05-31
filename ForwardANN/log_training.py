from dataset import cleandataset, dataset_file
from model import CaNNModel
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

from log_dataset import OptionDataset

def train_model(model: torch.nn.Module, train_loader, val_loader, criterion, optimizer, device, epochs):
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

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            output = model(batch_X.float())
            target = batch_y.float().view_as(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X.float())
                target = batch_y.float().view_as(output)
                loss = criterion(output, target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print('Epoch {}: Train Loss {:.4f} Val Loss {:.4f}'.format(epoch + 1, avg_train_loss, avg_val_loss))

    return model, train_losses, val_losses

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            iv_pred = torch.exp(output)
            target = torch.exp(Y.float())
            loss = criterion(output, Y.float().view_as(output))
            total_loss += loss.item()
            predictions.extend(iv_pred.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy().flatten())

    avg_loss = total_loss / len(data_loader)
    return avg_loss, np.array(predictions), np.array(targets)

def train_val_split(dataset, val_size=0.2, random_state=42):
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=val_size, random_state=random_state)
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

    train_sampler, val_sampler = train_val_split(dataset_train, val_size=0.2)
    test_sampler = RandomSampler(dataset_test)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True)
    dropout_rate = params['dropout_rate']

    model = CaNNModel(dropout_rate=dropout_rate).to(device)
    criterion = nn.HuberLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    trained_model, tl, vl = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs)

    train_loss, train_pred, train_target = evaluate_model(trained_model, train_loader, criterion, device)
    test_loss, test_pred, test_target = evaluate_model(trained_model, test_loader, criterion, device)

    train_pred = [float(x) for x in train_pred]
    train_target = [float(x) for x in train_target]
    test_pred = [float(x) for x in test_pred]
    test_target = [float(x) for x in test_target]

    train_df = pd.DataFrame({ 'predictions': train_pred, 'targets': train_target })
    train_df.to_csv('train_results.csv', index=False)

    test_df = pd.DataFrame({ 'predictions': test_pred, 'targets': test_target })
    test_df.to_csv('test_results.csv', index=False)

    print("In-sample (Training) Performance:")
    train_loss_details = calculate_loss('train_results.csv', target_scaler)
    for key, value in train_loss_details.items():
        print(f"{key}: {value}")

    print("\nOut-of-sample (Test) Performance:")
    test_loss_details = calculate_loss('test_results.csv', target_scaler)
    for key, value in test_loss_details.items():
        print(f"{key}: {value}")

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
    base_dir = "saved_models"
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    save_dir = os.path.join(base_dir, f"{name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    metrics_path = os.path.join(save_dir, "metrics.json")
    model_path = os.path.join(save_dir, "model.pt")
    graph_path = os.path.join(save_dir, "learning_curve.png")

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    scripted_model = torch.jit.script(trained_model)
    scripted_model.save(model_path)

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
    print(f"\U0001F4C1 {save_dir}/")
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
            df = cleandataset(dataset_file(self.path))
            train_df, test_df = train_test_split(df)
            ds_train = OptionDataset(train_df, is_train=True, log_target=True)
            ds_test = OptionDataset(test_df, is_train=False, target_scaler=ds_train.target_scaler, log_target=True)
            return ds_train, ds_test
        else:
            train = cleandataset(dataset_file(self.path))
            test = cleandataset(dataset_file(self.path2))
            ds_train = OptionDataset(train, is_train=True, log_target=True)
            ds_test = OptionDataset(test, is_train=False, target_scaler=ds_train.target_scaler, log_target=True)
            return ds_train, ds_test

if __name__ == '__main__':
    datasets = [
        DS("stage2_both.csv", None, "stage2_both")
    ]
    for dataset in datasets:
        ds_train, ds_test = dataset.datasets()
        main(ds_train, ds_test, dataset.name)
