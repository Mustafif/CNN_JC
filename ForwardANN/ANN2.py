from model import CaNNModel
from dataset import dataset_test, dataset_train
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from loss import calculate_loss
import time
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split


def train_model(model: CaNNModel, train_loader, val_loader, criterion, optimizer, device, epochs):
    "Train the model"
    #scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-5)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        # start_time = time.time()

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X.float())
            target = batch_y.float().view_as(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step(train_loss / len(train_loader))

        # validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X , batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X.float())
                target = batch_y.float().view_as(output)
                loss = criterion(output, target)
                val_loss += loss.item()
        scheduler.step(val_loss / len(val_loader))
        # epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f} Val Loss {val_loss/len(val_loader):.4f}')
    return model

def evaluate_model(model: CaNNModel, data_loader, criterion, device):
    "Evaluate the model performance"
    model.eval()
    total_loss = 0
    predictions = []  # To store predictions
    targets = []      # To store true targets

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X.float())
            target = batch_y.float().view_as(outputs)
            loss = criterion(outputs, target)
            total_loss += loss.item()

            # Store predictions and targets
            predictions.extend(outputs.cpu().tolist())
            targets.extend(target.cpu().tolist())

    avg_loss = total_loss / len(data_loader)
    return avg_loss, predictions, targets

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

def main():
    device = torch.device("cuda")
    print(f"Using device: {device}")
    params = json.load(open('params.json'))
    lr = params['lr']
    weight_decay = params['weight_decay']
    batch_size = params['batch_size']

    # lr = 0.0001
    # weight_decay = 1e-5
    # batch_size = 64
    epochs = params['epochs']

    # Create samplers
    train_sampler, val_sampler = train_val_split(dataset_train, val_size=0.1)
    # train_sampler = RandomSampler(dataset_train)
    test_sampler = RandomSampler(dataset_test)

    # Create data loaders
    train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        dataset_train,  # Note: using same dataset, different sampler
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(dataset_test, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)
    dropout_rate = 0.0
    # Model setup
    model = CaNNModel(dropout_rate=dropout_rate).to(device)
    # model = CaNNModel(
    #     input_features=10,
    #     hidden_layers=[256, 256, 256, 256],
    #     dropout_rate=0.0,
    #     activation=nn.ReLU(),
    #     batch_norm=True,
    #     output_activation=nn.Softplus()
    # ).to(device)
    # criterion = nn.HuberLoss()
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs)

    # Evaluation
    train_loss, train_pred, train_target = evaluate_model(trained_model, train_loader, criterion, device)
    test_loss, test_pred, test_target = evaluate_model(trained_model, test_loader, criterion, device)

    train_df = pd.DataFrame({
        'predictions': np.array(train_pred).flatten(),
        'targets': np.array(train_target).flatten()
    })
    train_df.to_csv('train_results.csv', index=False)

    test_df = pd.DataFrame({
        'predictions': np.array(test_pred).flatten(),
        'targets': np.array(test_target).flatten()
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

if __name__ == '__main__':
    main()
