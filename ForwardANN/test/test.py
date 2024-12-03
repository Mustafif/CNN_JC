import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# Assuming X and Y are your input features and target values
dataset = TensorDataset(X, Y)

# Split the dataset into training and test sets
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and test sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CaNNModel(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(CaNNModel, self).__init__()
        input_features = 10
        neurons = 200
        self.input_layer = nn.Linear(input_features, neurons)
        self.bn1 = nn.BatchNorm1d(neurons)
        self.hl1 = nn.Linear(neurons, neurons)
        self.bn2 = nn.BatchNorm1d(neurons)
        self.hl2 = nn.Linear(neurons, neurons)
        self.bn3 = nn.BatchNorm1d(neurons)
        self.hl3 = nn.Linear(neurons, neurons)
        self.bn4 = nn.BatchNorm1d(neurons)
        self.hl4 = nn.Linear(neurons, neurons)
        self.bn5 = nn.BatchNorm1d(neurons)
        self.output_layer = nn.Linear(neurons, 1)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.hl1(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.hl2(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.hl3(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        x = self.hl4(x)
        x = self.bn5(x)
        x = torch.relu(x)
        x = self.dropout4(x)
        x = self.output_layer(x)
        x = torch.nn.functional.softplus(x)
        return x

def train_model(model, train_loader, num_epochs=1000, learning_rate=0.01):
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch.float())
            loss = criterion(output, Y_batch.float().view(-1, 1))
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, Y_batch in data_loader:
            output = model(X_batch.float())
            predictions.append(output)
            actuals.append(Y_batch)

    predictions = torch.cat(predictions).numpy()
    actuals = torch.cat(actuals).numpy()
    return predictions, actuals

def plot_results(actuals, predictions, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, color='blue', label='Predicted vs Actual')
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'k--', lw=2, label='Ideal')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Train the model
model = CaNNModel().float()
trained_model = train_model(model, train_loader, num_epochs=1000, learning_rate=0.01)

# Evaluate the model on the training set (in-sample)
train_predictions, train_actuals = evaluate_model(trained_model, train_loader)

# Evaluate the model on the test set (out-of-sample)
test_predictions, test_actuals = evaluate_model(trained_model, test_loader)

# Plot in-sample results
plot_results(train_actuals, train_predictions, 'In-Sample Testing')

# Plot out-of-sample results
plot_results(test_actuals, test_predictions, 'Out-of-Sample Testing')
