import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import warnings
import pickle
from pathlib import Path
import copy

# Import your existing loss functions
from volatility_loss import create_volatility_loss
from loss import calculate_loss


class OptionDataset(Dataset):
    """PyTorch Dataset for option data with your specific format"""

    def __init__(self, features: np.ndarray, targets: np.ndarray, target_type: str = 'volatility'):
        """
        Initialize dataset

        Args:
            features: Input features (S0, m, r, T, corp, alpha, beta, omega, gamma, lambda)
            targets: Target values (sigma for volatility, V for price)
            target_type: 'volatility' or 'price'
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.target_type = target_type

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class VolatilityNet(nn.Module):
    """Neural network for predicting implied volatility"""

    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = [128, 256, 128, 64]):
        super(VolatilityNet, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.15)
            ])
            prev_dim = hidden_dim

        # Output layer for volatility (positive values)
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Softplus()  # Ensures positive volatility
        ])

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, x):
        return self.network(x)


class OptionPriceNet(nn.Module):
    """Neural network for directly predicting option prices"""

    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = [256, 512, 256, 128]):
        super(OptionPriceNet, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        # Output layer for option price (positive values)
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Softplus()  # Ensures positive prices
        ])

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, x):
        return self.network(x)


class NeuralOptionCalibrator:
    """Neural network-based option calibration system for your specific dataset"""

    def __init__(self, data_file: str, target_type: str = 'volatility'):
        """
        Initialize neural calibrator

        Args:
            data_file: Path to CSV file with your data format
            target_type: 'volatility' to predict sigma, 'price' to predict V
        """
        self.data_file = data_file
        self.target_type = target_type

        # Neural network components
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.trained = False

        # Training history
        self.training_history = {'train_loss': [], 'val_loss': []}
        self.detailed_loss_history = []

        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load and prepare data
        self._load_data()
        self._prepare_features()

    def _load_data(self):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(self.data_file)
            print(f"Loaded {len(self.data)} data points from {self.data_file}")

            # Display data info
            print(f"Data columns: {list(self.data.columns)}")
            print(f"Data shape: {self.data.shape}")

            # Check for missing values
            if self.data.isnull().any().any():
                print("Warning: Missing values detected. Removing rows with NaN values.")
                self.data = self.data.dropna()
                print(f"After removing NaN: {len(self.data)} data points")

        except Exception as e:
            raise ValueError(f"Error loading data from {self.data_file}: {str(e)}")

    def _prepare_features(self):
        """Prepare features and targets from the loaded data"""

        # Feature columns (all inputs except target)
        feature_columns = ['S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda']

        # Check if all required columns exist
        missing_cols = [col for col in feature_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Extract features
        self.features = self.data[feature_columns].values.astype(np.float32)

        # Extract targets based on target_type
        if self.target_type == 'volatility':
            if 'sigma' not in self.data.columns:
                raise ValueError("Column 'sigma' not found for volatility prediction")
            self.targets = self.data['sigma'].values.astype(np.float32).reshape(-1, 1)
        elif self.target_type == 'price':
            if 'V' not in self.data.columns:
                raise ValueError("Column 'V' not found for price prediction")
            self.targets = self.data['V'].values.astype(np.float32).reshape(-1, 1)
        else:
            raise ValueError("target_type must be 'volatility' or 'price'")

        # Remove invalid data points
        if self.target_type == 'volatility':
            # Remove extreme volatilities
            valid_mask = (self.targets.flatten() > 0.01) & (self.targets.flatten() < 5.0)
        else:
            # Remove negative or zero prices
            valid_mask = self.targets.flatten() > 0.001

        self.features = self.features[valid_mask]
        self.targets = self.targets[valid_mask]

        print(f"Prepared {len(self.features)} valid data points for {self.target_type} prediction")
        print(f"Feature shape: {self.features.shape}, Target shape: {self.targets.shape}")
        print(f"Target range: [{self.targets.min():.6f}, {self.targets.max():.6f}]")

    def create_model(self, hidden_dims: List[int] = None):
        """Create neural network model"""

        if hidden_dims is None:
            if self.target_type == 'volatility':
                hidden_dims = [128, 256, 128, 64]
            else:
                hidden_dims = [256, 512, 256, 128]

        input_dim = self.features.shape[1]

        if self.target_type == 'volatility':
            self.model = VolatilityNet(input_dim, hidden_dims)
        else:
            self.model = OptionPriceNet(input_dim, hidden_dims)

        self.model = self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Created {self.target_type} neural network:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

    def train(self,
              epochs: int = 1000,
              batch_size: int = 64,
              learning_rate: float = 0.001,
              validation_split: float = 0.2,
              early_stopping_patience: int = 50,
              hidden_dims: List[int] = None,
              loss_type: str = 'combined',
              use_scheduler: bool = True):
        """
        Train the neural network

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            validation_split: Fraction of data for validation
            early_stopping_patience: Patience for early stopping
            hidden_dims: Hidden layer dimensions
            loss_type: Type of loss function ('combined', 'adaptive', 'focal', 'weighted')
            use_scheduler: Whether to use learning rate scheduler
        """

        # Create model if not exists
        if self.model is None:
            self.create_model(hidden_dims)

        # Scale features and targets
        X_scaled = self.feature_scaler.fit_transform(self.features)
        y_scaled = self.target_scaler.fit_transform(self.targets)

        # Create dataset
        dataset = OptionDataset(X_scaled, y_scaled, self.target_type)

        # Train-validation split
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Loss function selection
        if self.target_type == 'volatility':
            if loss_type == 'combined':
                criterion = create_volatility_loss('combined',
                                               huber_weight=0.35,
                                               relative_weight=0.40,
                                               quantile_weight=0.15,
                                               constraint_weight=0.10)
            elif loss_type == 'adaptive':
                criterion = create_volatility_loss('adaptive',
                                               low_vol_threshold=0.12,
                                               high_vol_threshold=0.35)
            else:
                criterion = create_volatility_loss(loss_type)
        else:
            # For price prediction, use Huber loss
            criterion = nn.HuberLoss()

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Learning rate scheduler
        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=20, factor=0.5, verbose=True
            )

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        print(f"Starting {self.target_type} training for {epochs} epochs...")
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            epoch_detailed_losses = []

            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_features)

                # Calculate loss
                if hasattr(criterion, 'forward') and len(criterion.forward.__code__.co_varnames) > 2:
                    # Enhanced loss function returning tuple
                    loss, loss_details = criterion(outputs, batch_targets)
                    epoch_detailed_losses.append(loss_details)
                else:
                    # Standard loss function
                    loss = criterion(outputs, batch_targets)

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    outputs = self.model(batch_features)

                    if hasattr(criterion, 'forward') and len(criterion.forward.__code__.co_varnames) > 2:
                        loss, _ = criterion(outputs, batch_targets)
                    else:
                        loss = criterion(outputs, batch_targets)

                    val_loss += loss.item()

            # Average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            # Store history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)

            # Store detailed losses if available
            if epoch_detailed_losses:
                avg_detailed = {key: np.mean([d[key] for d in epoch_detailed_losses])
                              for key in epoch_detailed_losses[0].keys()}
                self.detailed_loss_history.append(avg_detailed)

            # Update learning rate
            if use_scheduler:
                scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if (epoch + 1) % 100 == 0 or epoch == 0:
                if epoch_detailed_losses:
                    latest_details = self.detailed_loss_history[-1]
                    print(f"Epoch [{epoch+1}/{epochs}] - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
                    print(f"  Huber: {latest_details.get('huber', 0):.4f}, Relative: {latest_details.get('relative', 0):.4f}, "
                          f"Quantile: {latest_details.get('quantile', 0):.4f}, Constraint: {latest_details.get('constraint', 0):.4f}")
                else:
                    print(f"Epoch [{epoch+1}/{epochs}] - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.trained = True
        print(f"Training completed! Best validation loss: {best_val_loss:.6f}")

    def predict(self, features: np.ndarray = None):
        """
        Make predictions

        Args:
            features: Input features. If None, uses training data

        Returns:
            predictions: Predicted values (unscaled)
        """

        if not self.trained:
            raise ValueError("Model must be trained before making predictions")

        if features is None:
            features = self.features

        # Scale features
        X_scaled = self.feature_scaler.transform(features)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy()

        # Inverse transform targets
        predictions = self.target_scaler.inverse_transform(predictions)

        return predictions.flatten()

    def evaluate(self, save_results: bool = True):
        """Evaluate model performance and optionally save results"""

        if not self.trained:
            raise ValueError("Model must be trained before evaluation")

        # Make predictions on all data
        predictions = self.predict()
        targets = self.targets.flatten()

        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)

        # Calculate relative error
        epsilon = 1e-8
        relative_errors = np.abs(predictions - targets) / (np.abs(targets) + epsilon)
        mre = np.mean(relative_errors)

        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MRE': mre,
            'Mean_Predicted': np.mean(predictions),
            'Mean_Target': np.mean(targets),
            'Min_Error': np.min(np.abs(predictions - targets)),
            'Max_Error': np.max(np.abs(predictions - targets)),
            'Std_Error': np.std(np.abs(predictions - targets))
        }

        # Save results if requested
        if save_results:
            results_df = pd.DataFrame({
                'predictions': predictions,
                'targets': targets
            })
            results_file = f'neural_{self.target_type}_results.csv'
            results_df.to_csv(results_file, index=False)
            print(f"Results saved to {results_file}")

            # Calculate loss using your existing function
            try:
                loss_info = calculate_loss(results_file, self.target_scaler)
                print("\nDetailed Loss Information:")
                for key, value in loss_info.items():
                    print(f"  {key}: {value}")
            except Exception as e:
                print(f"Could not calculate detailed loss: {e}")

        return metrics

    def plot_training_history(self):
        """Plot training and validation loss"""

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Training history
        axes[0].plot(self.training_history['train_loss'], label='Training Loss')
        axes[0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{self.target_type.capitalize()} Network Training History')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_yscale('log')

        # Detailed loss components (if available)
        if self.detailed_loss_history:
            for key in ['huber', 'relative', 'quantile', 'constraint']:
                if key in self.detailed_loss_history[0]:
                    values = [d[key] for d in self.detailed_loss_history]
                    axes[1].plot(values, label=key.capitalize())

            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss Component')
            axes[1].set_title('Detailed Loss Components')
            axes[1].legend()
            axes[1].grid(True)
            axes[1].set_yscale('log')
        else:
            axes[1].text(0.5, 0.5, 'No detailed loss history available',
                        ha='center', va='center', transform=axes[1].transAxes)

        plt.tight_layout()
        plt.show()

    def plot_predictions_vs_actual(self):
        """Plot predicted vs actual values"""

        if not self.trained:
            raise ValueError("Model must be trained before plotting")

        predictions = self.predict()
        actual = self.targets.flatten()

        plt.figure(figsize=(10, 8))
        plt.scatter(actual, predictions, alpha=0.6, s=20)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        plt.xlabel(f'Actual {self.target_type.capitalize()}')
        plt.ylabel(f'Predicted {self.target_type.capitalize()}')
        plt.title(f'Neural Network: Predicted vs Actual {self.target_type.capitalize()}')
        plt.grid(True, alpha=0.3)

        # Calculate R²
        r2 = r2_score(actual, predictions)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.show()

    def plot_residuals(self):
        """Plot residuals analysis"""

        if not self.trained:
            raise ValueError("Model must be trained before plotting")

        predictions = self.predict()
        actual = self.targets.flatten()
        residuals = predictions - actual

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Residuals vs Predicted
        axes[0, 0].scatter(predictions, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel(f'Predicted {self.target_type.capitalize()}')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)

        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=50, alpha=0.7)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].grid(True, alpha=0.3)

        # Residuals vs Features (using moneyness)
        if 'm' in self.data.columns:
            moneyness = self.data['m'].values[: len(residuals)]
            axes[1, 0].scatter(moneyness, residuals, alpha=0.6)
            axes[1, 0].axhline(y=0, color='r', linestyle='--')
            axes[1, 0].set_xlabel('Moneyness')
            axes[1, 0].set_ylabel('Residuals')
            axes[1, 0].set_title('Residuals vs Moneyness')
            axes[1, 0].grid(True, alpha=0.3)

        # Residuals vs Time to Expiration
        if 'T' in self.data.columns:
            time_to_exp = self.data['T'].values[:len(residuals)]
            axes[1, 1].scatter(time_to_exp, residuals, alpha=0.6)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Time to Expiration')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residuals vs Time to Expiration')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath: str):
        """Save the trained model and scalers"""

        if not self.trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            'model_state_dict': self.model.state_dict(),
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'target_type': self.target_type,
            'training_history': self.training_history,
            'detailed_loss_history': self.detailed_loss_history
        }

        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str, hidden_dims: List[int] = None):
        """Load a trained model"""

        model_data = torch.load(filepath, map_location=self.device)

        self.target_type = model_data['target_type']
        self.feature_scaler = model_data['feature_scaler']
        self.target_scaler = model_data['target_scaler']
        self.training_history = model_data['training_history']
        self.detailed_loss_history = model_data.get('detailed_loss_history', [])

        # Recreate model
        self.create_model(hidden_dims)
        self.model.load_state_dict(model_data['model_state_dict'])
        self.trained = True

        print(f"Model loaded from {filepath}")


def run_neural_calibration(data_file: str = 'impl_demo_improved.csv',
                          target_type: str = 'volatility',
                          epochs: int = 1000,
                          compare_both: bool = True):
    """
    Run neural network calibration on your data

    Args:
        data_file: Path to your data file
        target_type: 'volatility' or 'price'
        epochs: Number of training epochs
        compare_both: Whether to train both volatility and price models
    """

    print("=" * 60)
    print("NEURAL NETWORK OPTION CALIBRATION")
    print("=" * 60)

    results = {}

    # Single model training
    print(f"\nTraining {target_type} prediction model...")
    print("-" * 40)

    calibrator = NeuralOptionCalibrator(data_file, target_type)
    calibrator.train(epochs=epochs, batch_size=64, learning_rate=0.001,
                    loss_type='combined' if target_type == 'volatility' else 'huber')

    metrics = calibrator.evaluate()

    print(f"\n{target_type.capitalize()} Model Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    calibrator.plot_training_history()
    calibrator.plot_predictions_vs_actual()
    calibrator.plot_residuals()

    results[target_type] = {
        'calibrator': calibrator,
        'metrics': metrics
    }

    # Compare both models if requested
    if compare_both and target_type == 'volatility':
        print(f"\nTraining price prediction model for comparison...")
        print("-" * 40)

        price_calibrator = NeuralOptionCalibrator(data_file, 'price')
        price_calibrator.train(epochs=epochs, batch_size=64, learning_rate=0.001)

        price_metrics = price_calibrator.evaluate()

        print(f"\nPrice Model Results:")
        for key, value in price_metrics.items():
            print(f"  {key}: {value:.6f}")

        results['price'] = {
            'calibrator': price_calibrator,
            'metrics': price_metrics
        }

        # Comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Volatility model
        vol_pred = calibrator.predict()
        vol_actual = calibrator.targets.flatten()
        axes[0].scatter(vol_actual, vol_pred, alpha=0.6)
        axes[0].plot([vol_actual.min(), vol_actual.max()],
                    [vol_actual.min(), vol_actual.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Volatility')
        axes[0].set_ylabel('Predicted Volatility')
        axes[0].set_title(f"Volatility Model (R² = {metrics['R²']:.4f})")
        axes[0].grid(True, alpha=0.3)

        # Price model
        price_pred = price_calibrator.predict()
        price_actual = price_calibrator.targets.flatten()
        axes[1].scatter(price_actual, price_pred, alpha=0.6)
        axes[1].plot([price_actual.min(), price_actual.max()],
                    [price_actual.min(), price_actual.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual Price')
        axes[1].set_ylabel('Predicted Price')
        axes[1].set_title(f"Price Model (R² = {price_metrics['R²']:.4f})")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return results


if __name__ == "__main__":
    # Run calibration on your data
    results = run_neural_calibration(
        data_file='impl_demo_improved.csv',
        target_type='volatility',
        epochs=1000,
        compare_both=True
    )
