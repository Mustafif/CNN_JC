import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np

class SimpleModel(nn.Module):
    def __init__(self, input_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, input_size=10):
        # Generate synthetic regression dataset
        self.X = torch.randn(num_samples, input_size)
        
        # Create a simple linear relationship with some noise
        self.y = torch.sum(self.X, dim=1).unsqueeze(1) + torch.randn(num_samples, 1) * 0.1
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_test_split(dataset, test_size=0.2, random_seed=42):
    """
    Pure PyTorch train-test split using indices
    
    Args:
        dataset (torch.utils.data.Dataset): Input dataset
        test_size (float): Proportion of test samples
        random_seed (int): Random seed for reproducibility
    
    Returns:
        tuple: Train and test indices
    """
    # Total number of samples
    dataset_size = len(dataset)
    
    # Create indices
    indices = list(range(dataset_size))
    
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Shuffle indices
    np.random.shuffle(indices)
    
    # Calculate split
    split = int(np.floor(test_size * dataset_size))
    
    # Split indices
    train_indices, test_indices = indices[split:], indices[:split]
    
    return train_indices, test_indices

def train_model(model, train_loader, criterion, optimizer, epochs=100):
    """Train the model"""
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Optional: Print average epoch loss
        # print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")
    return model

def evaluate_model(model, data_loader, criterion):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def main():
    # Create dataset
    full_dataset = SyntheticDataset()
    
    # Perform train-test split using pure PyTorch method
    train_indices, test_indices = train_test_split(full_dataset)
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Create data loaders with samplers
    train_loader = DataLoader(full_dataset, batch_size=32, sampler=train_sampler)
    test_loader = DataLoader(full_dataset, batch_size=32, sampler=test_sampler)
    
    # Model setup
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training
    trained_model = train_model(model, train_loader, criterion, optimizer)
    
    # Evaluation
    # In-sample (training) performance
    train_loss = evaluate_model(trained_model, train_loader, criterion)
    print(f"In-sample (Training) Loss: {train_loss:.4f}")
    
    # Out-of-sample (test) performance
    test_loss = evaluate_model(trained_model, test_loader, criterion)
    print(f"Out-of-sample (Test) Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
