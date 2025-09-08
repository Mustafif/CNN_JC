import torch
import torch.nn as nn
import optuna
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
import logging
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

from dataset import cleandataset, dataset_file, OptionDataset
from model import CaNNModel, NetworkOfNetworks
from volatility_loss import create_volatility_loss
from loss import calculate_loss

class AdvancedTrainer:
    """Advanced trainer with hyperparameter optimization and ensemble methods"""

    def __init__(self,
                 data_path: str,
                 output_dir: str = "advanced_results",
                 device: str = "auto",
                 n_trials: int = 100,
                 cv_folds: int = 5,
                 ensemble_size: int = 5):

        self.data_path = data_path
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.ensemble_size = ensemble_size

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Load and prepare data
        self.load_data()

        # Results storage
        self.best_params = None
        self.best_score = float('inf')
        self.study_results = []
        self.ensemble_models = []

    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = os.path.join(self.output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """Load and prepare dataset"""
        self.logger.info(f"Loading data from {self.data_path}")

        # Load data
        df = cleandataset(dataset_file(self.data_path))

        # Split into train/test
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # Create datasets
        self.train_dataset = OptionDataset(train_df, is_train=True)
        self.test_dataset = OptionDataset(test_df, is_train=False,
                                        target_scaler=self.train_dataset.target_scaler)

        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        self.logger.info(f"Test samples: {len(self.test_dataset)}")

        # Get feature dimension
        sample_x, _ = self.train_dataset[0]
        self.input_features = sample_x.shape[0]
        self.logger.info(f"Input features: {self.input_features}")

    def objective(self, trial):
        """Optuna objective function for hyperparameter optimization"""

        # Suggest hyperparameters
        params = {
            'hidden_size': trial.suggest_categorical('hidden_size', [128, 200, 256, 320]),
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 4, 8),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.4),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'loss_type': trial.suggest_categorical('loss_type', ['combined', 'adaptive', 'focal']),
            'use_ensemble': trial.suggest_categorical('use_ensemble', [False, True]),
            'scheduler_type': trial.suggest_categorical('scheduler_type', ['onecycle', 'cosine', 'plateau'])
        }

        # Cross-validation
        cv_scores = []
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.train_dataset)):

            # Create data loaders
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_loader = DataLoader(
                self.train_dataset,
                batch_size=params['batch_size'],
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True
            )

            val_loader = DataLoader(
                self.train_dataset,
                batch_size=params['batch_size'],
                sampler=val_sampler,
                num_workers=4,
                pin_memory=True
            )

            # Create model
            if params['use_ensemble']:
                model = NetworkOfNetworks(
                    num_child_networks=3,
                    input_features=self.input_features,
                    hidden_size=params['hidden_size'],
                    dropout_rate=params['dropout_rate'],
                    num_hidden_layers=params['num_hidden_layers']
                ).to(self.device)
            else:
                model = CaNNModel(
                    input_features=self.input_features,
                    hidden_size=params['hidden_size'],
                    dropout_rate=params['dropout_rate'],
                    num_hidden_layers=params['num_hidden_layers']
                ).to(self.device)

            # Loss and optimizer
            criterion = create_volatility_loss(params['loss_type'])
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )

            # Scheduler
            if params['scheduler_type'] == 'onecycle':
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=params['lr'],
                    epochs=50,  # Reduced epochs for hyperparameter search
                    steps_per_epoch=len(train_loader)
                )
            elif params['scheduler_type'] == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=50
                )
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5
                )

            # Train model
            val_loss = self.train_fold(model, train_loader, val_loader,
                                     criterion, optimizer, scheduler,
                                     epochs=50, fold=fold)

            cv_scores.append(val_loss)

            # Pruning for unpromising trials
            trial.report(val_loss, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(cv_scores)

    def train_fold(self, model, train_loader, val_loader, criterion,
                   optimizer, scheduler, epochs=50, fold=0):
        """Train a single fold"""

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                output = model(batch_X.float())
                target = batch_y.float().view_as(output)

                if hasattr(criterion, 'forward') and len(criterion.forward.__code__.co_varnames) > 2:
                    loss, _ = criterion(output, target)
                else:
                    loss = criterion(output, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if hasattr(scheduler, 'step') and 'OneCycleLR' in str(type(scheduler)):
                    scheduler.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    output = model(batch_X.float())
                    target = batch_y.float().view_as(output)

                    if hasattr(criterion, 'forward') and len(criterion.forward.__code__.co_varnames) > 2:
                        loss, _ = criterion(output, target)
                    else:
                        loss = criterion(output, target)

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            # Scheduler step for non-OneCycleLR schedulers
            if 'ReduceLROnPlateau' in str(type(scheduler)):
                scheduler.step(avg_val_loss)
            elif 'CosineAnnealingLR' in str(type(scheduler)):
                scheduler.step()

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        return best_val_loss

    def optimize_hyperparameters(self):
        """Run hyperparameter optimization"""

        self.logger.info("Starting hyperparameter optimization...")

        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )

        study.optimize(self.objective, n_trials=self.n_trials)

        self.best_params = study.best_params
        self.best_score = study.best_value

        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best score: {self.best_score}")

        # Save optimization results
        self.save_optimization_results(study)

        return study

    def save_optimization_results(self, study):
        """Save hyperparameter optimization results"""

        # Save best parameters
        with open(os.path.join(self.output_dir, 'best_params.json'), 'w') as f:
            json.dump(self.best_params, f, indent=4)

        # Save study trials
        trials_df = study.trials_dataframe()
        trials_df.to_csv(os.path.join(self.output_dir, 'optimization_trials.csv'), index=False)

        # Create optimization plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Optimization history
        optuna.visualization.matplotlib.plot_optimization_history(study, ax=axes[0,0])
        axes[0,0].set_title('Optimization History')

        # Parameter importances
        try:
            optuna.visualization.matplotlib.plot_param_importances(study, ax=axes[0,1])
            axes[0,1].set_title('Parameter Importances')
        except:
            axes[0,1].text(0.5, 0.5, 'Parameter importance\nnot available',
                          ha='center', va='center', transform=axes[0,1].transAxes)

        # Parallel coordinate plot
        try:
            optuna.visualization.matplotlib.plot_parallel_coordinate(study, ax=axes[1,0])
            axes[1,0].set_title('Parallel Coordinate Plot')
        except:
            axes[1,0].text(0.5, 0.5, 'Parallel coordinate\nplot not available',
                          ha='center', va='center', transform=axes[1,0].transAxes)

        # Slice plot for learning rate
        try:
            optuna.visualization.matplotlib.plot_slice(study, params=['lr'], ax=axes[1,1])
            axes[1,1].set_title('Learning Rate Slice Plot')
        except:
            axes[1,1].text(0.5, 0.5, 'Slice plot\nnot available',
                          ha='center', va='center', transform=axes[1,1].transAxes)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'optimization_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def train_best_model(self, epochs=200):
        """Train the best model with full dataset"""

        if self.best_params is None:
            raise ValueError("Must run hyperparameter optimization first")

        self.logger.info("Training best model on full dataset...")

        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.best_params['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.best_params['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Create best model
        if self.best_params['use_ensemble']:
            model = NetworkOfNetworks(
                num_child_networks=3,
                input_features=self.input_features,
                hidden_size=self.best_params['hidden_size'],
                dropout_rate=self.best_params['dropout_rate'],
                num_hidden_layers=self.best_params['num_hidden_layers']
            ).to(self.device)
        else:
            model = CaNNModel(
                input_features=self.input_features,
                hidden_size=self.best_params['hidden_size'],
                dropout_rate=self.best_params['dropout_rate'],
                num_hidden_layers=self.best_params['num_hidden_layers']
            ).to(self.device)

        criterion = create_volatility_loss(self.best_params['loss_type'])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.best_params['lr'],
            weight_decay=self.best_params['weight_decay']
        )

        # Train with validation split
        train_indices = list(range(len(self.train_dataset)))
        train_idx, val_idx = train_test_split(train_indices, test_size=0.2, random_state=42)

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_val_loader = DataLoader(
            self.train_dataset,
            batch_size=self.best_params['batch_size'],
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            self.train_dataset,
            batch_size=self.best_params['batch_size'],
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )

        # Train model
        trained_model, train_losses, val_losses = self.train_model_full(
            model, train_val_loader, val_loader, criterion, optimizer, epochs
        )

        # Evaluate on test set
        test_results = self.evaluate_model(trained_model, test_loader, criterion)

        # Save model and results
        self.save_final_model(trained_model, train_losses, val_losses, test_results)

        return trained_model, test_results

    def train_model_full(self, model, train_loader, val_loader, criterion, optimizer, epochs):
        """Full model training with comprehensive logging"""

        # Setup scheduler
        if self.best_params['scheduler_type'] == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.best_params['lr'],
                epochs=epochs,
                steps_per_epoch=len(train_loader)
            )
        elif self.best_params['scheduler_type'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 25

        for epoch in range(epochs):
            # Training
            model.train()
            epoch_train_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                output = model(batch_X.float())
                target = batch_y.float().view_as(output)

                if hasattr(criterion, 'forward') and len(criterion.forward.__code__.co_varnames) > 2:
                    loss, _ = criterion(output, target)
                else:
                    loss = criterion(output, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if hasattr(scheduler, 'step') and 'OneCycleLR' in str(type(scheduler)):
                    scheduler.step()

                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    output = model(batch_X.float())
                    target = batch_y.float().view_as(output)

                    if hasattr(criterion, 'forward') and len(criterion.forward.__code__.co_varnames) > 2:
                        loss, _ = criterion(output, target)
                    else:
                        loss = criterion(output, target)

                    epoch_val_loss += loss.item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Scheduler step
            if 'ReduceLROnPlateau' in str(type(scheduler)):
                scheduler.step(avg_val_loss)
            elif 'CosineAnnealingLR' in str(type(scheduler)):
                scheduler.step()

            # Early stopping and best model saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Logging
            if epoch % 10 == 0:
                self.logger.info(f'Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}')

            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model, train_losses, val_losses

    def evaluate_model(self, model, data_loader, criterion):
        """Comprehensive model evaluation"""

        model.eval()
        predictions = []
        targets = []
        total_loss = 0

        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                output = model(batch_X.float())
                target = batch_y.float().view_as(output)

                if hasattr(criterion, 'forward') and len(criterion.forward.__code__.co_varnames) > 2:
                    loss, _ = criterion(output, target)
                else:
                    loss = criterion(output, target)

                total_loss += loss.item()
                predictions.extend(output.cpu().numpy().flatten())
                targets.extend(batch_y.cpu().numpy().flatten())

        # Convert to numpy arrays
        predictions = np.array(predictions)
        targets = np.array(targets)

        # Save predictions for detailed analysis
        results_df = pd.DataFrame({
            'predictions': predictions,
            'targets': targets
        })
        results_df.to_csv(os.path.join(self.output_dir, 'test_predictions.csv'), index=False)

        # Calculate detailed metrics
        metrics = calculate_loss(
            os.path.join(self.output_dir, 'test_predictions.csv'),
            self.test_dataset.target_scaler
        )

        return metrics

    def save_final_model(self, model, train_losses, val_losses, test_results):
        """Save final model and comprehensive results"""

        # Save model
        model_path = os.path.join(self.output_dir, 'best_model.pt')
        scripted_model = torch.jit.script(model)
        scripted_model.save(model_path)

        # Save training history
        history_df = pd.DataFrame({
            'epoch': range(len(train_losses)),
            'train_loss': train_losses,
            'val_loss': val_losses
        })
        history_df.to_csv(os.path.join(self.output_dir, 'training_history.csv'), index=False)

        # Save test results
        with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)

        # Create comprehensive plots
        self.create_final_plots(train_losses, val_losses, test_results)

        self.logger.info(f"Model and results saved to {self.output_dir}")
        self.logger.info(f"Test MAE: {test_results['MAE']:.4f}")
        self.logger.info(f"Test RMSE: {test_results['RMSE']:.4f}")
        self.logger.info(f"Test R²: {test_results['R^2']:.4f}")

    def create_final_plots(self, train_losses, val_losses, test_results):
        """Create comprehensive visualization plots"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Training history
        axes[0,0].plot(train_losses, label='Train Loss', alpha=0.8)
        axes[0,0].plot(val_losses, label='Validation Loss', alpha=0.8)
        axes[0,0].set_title('Training History')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Load predictions for plotting
        pred_df = pd.read_csv(os.path.join(self.output_dir, 'test_predictions.csv'))

        # Inverse transform for plotting
        scaler = self.test_dataset.target_scaler
        true_preds = scaler.inverse_transform(pred_df['predictions'].values.reshape(-1, 1)).flatten()
        true_targets = scaler.inverse_transform(pred_df['targets'].values.reshape(-1, 1)).flatten()

        # Predictions vs Actual
        axes[0,1].scatter(true_targets, true_preds, alpha=0.6, s=20)
        axes[0,1].plot([true_targets.min(), true_targets.max()],
                       [true_targets.min(), true_targets.max()], 'r--', lw=2)
        axes[0,1].set_title('Predictions vs Actual')
        axes[0,1].set_xlabel('Actual Volatility')
        axes[0,1].set_ylabel('Predicted Volatility')
        axes[0,1].grid(True, alpha=0.3)

        # Residuals
        residuals = true_preds - true_targets
        axes[0,2].scatter(true_targets, residuals, alpha=0.6, s=20)
        axes[0,2].axhline(y=0, color='r', linestyle='--')
        axes[0,2].set_title('Residuals')
        axes[0,2].set_xlabel('Actual Volatility')
        axes[0,2].set_ylabel('Residuals')
        axes[0,2].grid(True, alpha=0.3)

        # Residuals histogram
        axes[1,0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1,0].set_title('Residuals Distribution')
        axes[1,0].set_xlabel('Residuals')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].grid(True, alpha=0.3)

        # Error by volatility level
        error_abs = np.abs(residuals)
        axes[1,1].scatter(true_targets, error_abs, alpha=0.6, s=20)
        axes[1,1].set_title('Absolute Error by Volatility Level')
        axes[1,1].set_xlabel('Actual Volatility')
        axes[1,1].set_ylabel('Absolute Error')
        axes[1,1].grid(True, alpha=0.3)

        # Metrics summary
        metrics_text = f"""Test Results:
MAE: {test_results['MAE']:.4f}
RMSE: {test_results['RMSE']:.4f}
R²: {test_results['R^2']:.4f}
MRE: {test_results['MRE']:.4f}

Data Points: {test_results['total_samples']}
Mean Actual: {test_results['Mean of Targets']:.4f}
Mean Predicted: {test_results['Mean of Predicted']:.4f}"""

        axes[1,2].text(0.1, 0.5, metrics_text, transform=axes[1,2].transAxes,
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1,2].set_title('Performance Summary')
        axes[1,2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'final_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info("Final analysis plots saved")

    def run_complete_pipeline(self):
        """Run the complete advanced training pipeline"""

        self.logger.info("Starting complete advanced training pipeline...")

        # 1. Hyperparameter optimization
        study = self.optimize_hyperparameters()

        # 2. Train best model
        best_model, test_results = self.train_best_model()

        # 3. Create final summary
        self.create_final_summary(study, test_results)

        self.logger.info("Complete pipeline finished successfully!")

        return best_model, test_results

    def create_final_summary(self, study, test_results):
        """Create final summary report"""

        summary = {
            'optimization': {
                'n_trials': len(study.trials),
                'best_value': study.best_value,
                'best_params': study.best_params
            },
            'final_performance': test_results,
            'training_info': {
                'device': str(self.device),
                'input_features': self.input_features,
                'train_samples': len(self.train_dataset),
                'test_samples': len(self.test_dataset)
            }
        }

        # Save summary
        with open(os.path.join(self.output_dir, 'final_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

        # Print summary
        self.logger.info("=" * 50)
        self.logger.info("FINAL SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Optimization trials: {len(study.trials)}")
        self.logger.info(f"Best validation score: {study.best_value:.4f}")
        self.logger.info(f"Final test MAE: {test_results['MAE']:.4f}")
        self.logger.info(f"Final test R²: {test_results['R^2']:.4f}")
        self.logger.info("=" * 50)


def main():
    """Main function to run advanced training"""

    # Configuration
    config = {
        'data_path': 'impl_demo_improved.csv',
        'output_dir': f'advanced_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'n_trials': 50,  #
