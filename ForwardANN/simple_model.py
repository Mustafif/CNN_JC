import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVolatilityModel(nn.Module):
    """
    Simple 3-layer neural network for volatility prediction.
    Designed for small datasets with minimal overfitting.
    """
    def __init__(self, input_features=11, hidden_size=64, dropout_rate=0.1):
        super().__init__()

        # Simple 3-layer architecture
        self.layer1 = nn.Linear(input_features, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, 1)

        # Simple dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Simple batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Layer 1: Input -> Hidden
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2: Hidden -> Hidden/2
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3: Output with positive constraint for volatility
        x = self.layer3(x)
        x = F.softplus(x) + 1e-6  # Ensure positive volatility

        return x


class SimpleDataset(torch.utils.data.Dataset):
    """
    Simplified dataset using only the most important features
    """
    def __init__(self, dataframe, is_train=False, target_scaler=None):
        self.data = dataframe
        self.is_train = is_train

        # Use only the most important base features
        self.features = ["S0", "m", "r", "T", "corp", "alpha", "beta", "omega", "gamma", "lambda", "V"]

        # Simple target scaling
        if target_scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.target_scaler = StandardScaler()
            self.target_scaler.fit(self.data[["sigma"]])
        else:
            self.target_scaler = target_scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Extract basic features only
        X = torch.tensor(row[self.features].values, dtype=torch.float32)

        # Scale target
        target_value = row["sigma"]
        scaled_target = self.target_scaler.transform([[target_value]])[0, 0]
        Y = torch.tensor(scaled_target, dtype=torch.float32)

        return X, Y


def train_simple_model(model, train_loader, val_loader, device, epochs=100, lr=0.001):
    """
    Simple training loop without complex scheduling
    """
    # Simple optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=0.1)

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()

            # Simple gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output.squeeze(), batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f} Val Loss {avg_val_loss:.4f}')

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return model, train_losses, val_losses


def evaluate_simple_model(model, data_loader, device, target_scaler):
    """
    Simple evaluation function
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            output = model(X)

            # Convert back to original scale
            pred_scaled = output.cpu().numpy()
            target_scaled = Y.cpu().numpy()

            pred_original = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            target_original = target_scaler.inverse_transform(target_scaled.reshape(-1, 1)).flatten()

            predictions.extend(pred_original)
            targets.extend(target_original)

    return predictions, targets


def calculate_simple_metrics(predictions, targets):
    """
    Calculate basic metrics
    """
    import numpy as np

    predictions = np.array(predictions)
    targets = np.array(targets)

    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)

    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Mean Relative Error
    mre = np.mean(np.abs((predictions - targets) / (targets + 1e-8)))

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R^2': r2,
        'MRE': mre,
        'min_error': np.min(np.abs(predictions - targets)),
        'max_error': np.max(np.abs(predictions - targets)),
        'std_error': np.std(np.abs(predictions - targets))
    }


if __name__ == "__main__":
    # This can be imported and used in the main training script
    print("Simple model module loaded successfully!")
