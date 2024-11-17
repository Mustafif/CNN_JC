import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# First, create a custom dataset
class OptionDataset(Dataset):
    def __init__(self, data_dict, targets):
        """
        data_dict: Dictionary containing:
            - market_params (r, S0, T)
            - strike_prices (K array)
            - garch_params (alpha, beta, omega, gamma, lambda)
            - historical (price history)
            - call_prices
            - put_prices
        targets: Your target values
        """
        self.market_params = torch.FloatTensor(data_dict['market_params'])
        self.strike_prices = torch.FloatTensor(data_dict['strike_prices'])
        self.garch_params = torch.FloatTensor(data_dict['garch_params'])
        self.historical = torch.FloatTensor(data_dict['historical'])
        self.call_prices = torch.FloatTensor(data_dict['call_prices'])
        self.put_prices = torch.FloatTensor(data_dict['put_prices'])
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'market_params': self.market_params[idx],
            'strike_prices': self.strike_prices[idx],
            'garch_params': self.garch_params[idx],
            'historical': self.historical[idx],
            'call_prices': self.call_prices[idx],
            'put_prices': self.put_prices[idx],
            'target': self.targets[idx]
        }

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device='cuda'):
    """
    Complete training function with validation
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Learning rate scheduler (optional)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
    }

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []

        for batch in train_loader:
            # Move data to device
            market_params = batch['market_params'].to(device)
            strike_prices = batch['strike_prices'].to(device)
            garch_params = batch['garch_params'].to(device)
            historical = batch['historical'].to(device)
            call_prices = batch['call_prices'].to(device)
            put_prices = batch['put_prices'].to(device)
            targets = batch['target'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                market_params,
                strike_prices,
                garch_params,
                historical,
                call_prices,
                put_prices
            )

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()

            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                market_params = batch['market_params'].to(device)
                strike_prices = batch['strike_prices'].to(device)
                garch_params = batch['garch_params'].to(device)
                historical = batch['historical'].to(device)
                call_prices = batch['call_prices'].to(device)
                put_prices = batch['put_prices'].to(device)
                targets = batch['target'].to(device)

                # Forward pass
                outputs = model(
                    market_params,
                    strike_prices,
                    garch_params,
                    historical,
                    call_prices,
                    put_prices
                )

                # Calculate loss
                val_loss = criterion(outputs, targets)
                val_losses.append(val_loss.item())

        # Calculate average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()

        # Store losses
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # Print epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Val Loss: {avg_val_loss:.6f}')
        print('-' * 50)

    # Load best model
    model.load_state_dict(best_model)

    return model, history

# Usage example:
def prepare_data(your_data_dict, your_targets):
    """
    Prepare train and validation datasets
    """
    # Create dataset
    dataset = OptionDataset(your_data_dict, your_targets)

    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader

# Training execution
if __name__ == "__main__":
    # Initialize model
    hist_size = 100
    option_size = 50
    num_strikes = 20
    model = MixedInputNN(hist_size, option_size, num_strikes)

    # Prepare your data
    # train_loader, val_loader = prepare_data(your_data_dict, your_targets)

    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        learning_rate=0.001,
        device=device
    )

    # Save model
    torch.save(trained_model.state_dict(), 'option_model.pth')

    # Optional: Plot training history
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()
