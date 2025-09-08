from dataset import cleandataset, dataset_file
from model import CaNNModel, NetworkOfNetworks
from volatility_loss import create_volatility_loss
# from dataset import dataset_test, dataset_train
import torch

from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
import pandas as pd
from loss import calculate_loss
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy
from datetime import datetime

from dataset import OptionDataset
# target_scaler = dataset_train.target_scaler

def train_model(model: torch.nn.Module, train_loader, val_loader, criterion, optimizer, device, epochs, patience=15, gradient_clip_val=0.5):
    """Train the model with improved learning rate scheduling and regularization"""
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'] * 2,  # Higher max_lr for better convergence
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Shorter warmup
        anneal_strategy='cos',
        div_factor=10,  # Better initial lr
        final_div_factor=100  # Better final lr
    )

    # Early stopping setup
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    train_losses = []
    val_losses = []
    detailed_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        epoch_detailed_losses = []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            # Mixed Precision Training
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    output = model(batch_X.float())
                    target = batch_y.float().view_as(output)

                    # Use enhanced loss function if available
                    if hasattr(criterion, 'forward') and len(criterion.forward.__code__.co_varnames) > 2:
                        loss, loss_details = criterion(output, target)
                        epoch_detailed_losses.append(loss_details)
                    else:
                        loss = criterion(output, target)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            else:
                output = model(batch_X.float())
                target = batch_y.float().view_as(output)

                # Use enhanced loss function if available
                if hasattr(criterion, 'forward') and len(criterion.forward.__code__.co_varnames) > 2:
                    loss, loss_details = criterion(output, target)
                    epoch_detailed_losses.append(loss_details)
                else:
                    loss = criterion(output, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_val)
                optimizer.step()
                scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Store detailed losses if available
        if epoch_detailed_losses:
            avg_detailed = {key: np.mean([d[key] for d in epoch_detailed_losses])
                          for key in epoch_detailed_losses[0].keys()}
            detailed_losses.append(avg_detailed)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        output = model(batch_X.float())
                        target = batch_y.float().view_as(output)
                        if hasattr(criterion, 'forward') and len(criterion.forward.__code__.co_varnames) > 2:
                            loss, _ = criterion(output, target)
                        else:
                            loss = criterion(output, target)
                else:
                    output = model(batch_X.float())
                    target = batch_y.float().view_as(output)
                    if hasattr(criterion, 'forward') and len(criterion.forward.__code__.co_varnames) > 2:
                        loss, _ = criterion(output, target)
                    else:
                        loss = criterion(output, target)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Print detailed loss information
        if detailed_losses:
            latest_details = detailed_losses[-1]
            print(f'Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f} Val Loss {avg_val_loss:.4f}')
            print(f'  Huber: {latest_details.get("huber", 0):.4f}, Relative: {latest_details.get("relative", 0):.4f}, '
                  f'Quantile: {latest_details.get("quantile", 0):.4f}, Constraint: {latest_details.get("constraint", 0):.4f}')
        else:
            print(f'Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f} Val Loss {avg_val_loss:.4f}')

        # Early stopping and best model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses



def evaluate_model(model, data_loader, criterion, device, use_tta=False):
    """Evaluate model with optional Test Time Augmentation"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)

            if use_tta:
                # Test Time Augmentation - multiple forward passes with slight noise
                tta_predictions = []
                for _ in range(5):
                    noise = torch.randn_like(X) * 0.01  # Small noise for TTA
                    augmented_X = X + noise
                    output = model(augmented_X)
                    tta_predictions.append(output)

                # Average TTA predictions
                output = torch.mean(torch.stack(tta_predictions), dim=0)
            else:
                output = model(X)

            # Reshape target to match output dimensions
            target = Y.float().view_as(output)

            # Handle both enhanced loss functions (returning tuple) and standard loss functions
            loss_result = criterion(output, target)
            if isinstance(loss_result, tuple):
                loss = loss_result[0]  # Extract actual loss from tuple
            else:
                loss = loss_result

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

    # Load params, check for temporary config first
    if os.path.exists('params_temp.json'):
        params = json.load(open('params_temp.json'))
        os.remove('params_temp.json')  # Clean up temp file
    else:
        params = json.load(open('params.json'))
    lr = params['lr']
    weight_decay = params['weight_decay']
    batch_size = params['batch_size']

    epochs = params['epochs']

    # Improved training parameters
    gradient_clip_val = 0.5  # Reduced for better stability
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

    # Model setup with improved architecture selection
    use_ensemble = params.get('use_ensemble', False)
    hidden_size = params.get('hidden_size', 256)  # Increased default size
    num_hidden_layers = params.get('num_hidden_layers', 8)  # Deeper network

    if use_ensemble:
        model = NetworkOfNetworks(
            num_child_networks=5,  # More child networks for better ensemble
            input_features=38,  # Updated based on dataset features
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            num_hidden_layers=num_hidden_layers
        ).to(device)
    else:
        model = CaNNModel(
            input_features=38,  # Updated based on dataset features
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            num_hidden_layers=num_hidden_layers
        ).to(device)

    # Compile model for faster training (PyTorch 2.0+) - commented out to avoid issues
    # try:
    #     if hasattr(torch, 'compile'):
    #         model = torch.compile(model, mode='default')
    # except Exception as e:
    #     print(f"Warning: Could not compile model: {e}")

    # Enhanced loss function selection with optimized parameters
    if use_ensemble:
        criterion = create_volatility_loss('adaptive',
                                         low_vol_threshold=0.12,
                                         high_vol_threshold=0.35)
    else:
        # Use combined loss with optimized weights for better volatility prediction
        criterion = create_volatility_loss('combined',
                                         huber_weight=0.35,      # Reduced for less sensitivity to outliers
                                         relative_weight=0.40,   # Increased for better relative accuracy
                                         quantile_weight=0.15,   # Reduced quantile emphasis
                                         constraint_weight=0.10, # Keep constraint weight
                                         huber_delta=0.05,       # Smaller delta for more precise loss
                                         quantile_alpha=0.15)    # Slightly higher alpha for better tail handling

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),  # Improved beta2 for better convergence
        eps=1e-8,
        amsgrad=True  # More stable optimization
    )
    # Training with early stopping
    patience = params.get('patience', 15)
    trained_model, tl, vl = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs, patience=patience, gradient_clip_val=gradient_clip_val)

    # Evaluation with Test Time Augmentation for better performance
    train_loss, train_pred, train_target = evaluate_model(trained_model, train_loader, criterion, device, use_tta=False)
    test_loss, test_pred, test_target = evaluate_model(trained_model, test_loader, criterion, device, use_tta=True)

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
            torch.save(trained_model.state_dict(), "model.pt")
    else:
        with open('metrics.json', 'r') as f:
            data = json.load(f)
            if test_loss_details['MAE'] < data['out_of_sample']['MAE']:
                print("New model is better than the previous one. Saving new model...")
                with open('metrics.json', 'w') as f:
                    json.dump(metrics, f)
                    torch.save(trained_model.state_dict(), "model.pt")
    save_model_checkpoint(trained_model, name, metrics, tl, vl)



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
    torch.save(trained_model.state_dict(), model_path)

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
    # Load base parameters
    base_params = json.load(open('params.json'))

    datasets = [
        # DS("train_dataset.csv", "test_dataset.csv", "stage1_HN"),
        # DS("../data_gen/stage1b.csv", None, "stage1b_HN"),
        # DS("../data_gen/stage2.csv", None, "stage2_HN"),
        # DS("../data_gen/stage3.csv", None, "stage3_HN"),
        # DS("../data_gen/Duan_Garch/stage3.csv", None, "stage3_Duan"),
        # DS("../data_gen/GJR_Garch/stage3_gjr.csv", None, "stage3_GJR"),
        # DS("../data_gen/Duan_Garch/stage1b.csv", None, "stage1b_Duan"),
        # DS("../data_gen/Duan_Garch/stage2.csv", None, "stage2_Duan"),
        # DS("../data_gen/GJR_Garch/stage1b.csv", None, "stage1b_GJR"),
        DS("impl_demo_improved.csv", None, "impl_single"),
        DS("impl_demo_improved.csv", None, "impl_ensemble"),
        DS("impl_demo_improved.csv", None, "impl_adaptive"),
        DS("impl_demo_improved.csv", None, "impl_focal")
    ]
    # Model configurations for different experiments
    model_configs = [
        {"use_ensemble": False, "hidden_size": 256, "num_hidden_layers": 8},
        {"use_ensemble": True, "hidden_size": 200, "num_hidden_layers": 6},
        {"use_ensemble": False, "hidden_size": 320, "num_hidden_layers": 10},
        {"use_ensemble": False, "hidden_size": 384, "num_hidden_layers": 6}
    ]

    for i, dataset in enumerate(datasets):
        # Create params copy and update with specific model configuration
        params = base_params.copy()
        if i < len(model_configs):
            params.update(model_configs[i])
            # Save updated params temporarily
            with open('params_temp.json', 'w') as f:
                json.dump(params, f)

        ds_train, ds_test = dataset.datasets()
        main(ds_train, ds_test, dataset.name)
