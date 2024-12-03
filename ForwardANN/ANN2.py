from model import CaNNModel
from dataset import dataset_test, dataset_train
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,RandomSampler
from torch.optim.lr_scheduler import StepLR
import pandas as pd

def train_model(model: CaNNModel, train_loader, criterion, optimizer, epochs=1000):
    "Train the model"
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X.float())
            target = batch_y.float().view_as(output)
            loss = criterion(output, target)
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += 0.005 * l1_norm
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

    return model

def evaluate_model(model: CaNNModel, data_loader, criterion):
    "Evaluate the model performance"
    model.eval()
    total_loss = 0
    predictions = []  # To store predictions
    targets = []      # To store true targets

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X.float())
            target = batch_y.float().view_as(outputs)
            loss = criterion(outputs, target)
            total_loss += loss.item()

            # Store predictions and targets
            predictions.extend(outputs.tolist())
            targets.extend(target.tolist())

    avg_loss = total_loss / len(data_loader)
    return avg_loss, predictions, targets


def main():
    # Create samplers
    #train_sampler = RandomSampler(dataset_train)
    #test_sampler = RandomSampler(dataset_test)

    # Create data loaders
    train_loader = DataLoader(dataset_train, batch_size=64)
    test_loader = DataLoader(dataset_test, batch_size=64)

    # Model setup
    model = CaNNModel()
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # training
    trained_model = train_model(model, train_loader, criterion, optimizer, epochs=1000)

    # Evaluation
    # In-sample (training) performance
    train_loss, train_pred, train_target = evaluate_model(trained_model, train_loader, criterion)
    print(f"In-sample (Training) Loss: {train_loss:.4f}")

    # Out-of-sample (testing) performance
    test_loss, test_pred, test_target = evaluate_model(trained_model, test_loader, criterion)
    print(f"Out-of-sample (Test) Loss: {test_loss:.4f}")

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


if __name__ == '__main__':
    main()
