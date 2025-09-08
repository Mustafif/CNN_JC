#!/usr/bin/env python3
"""
Neural Network Option Calibration Example

This script demonstrates how to use neural networks for option calibration
using your specific dataset format and custom loss functions.

Author: AI Assistant
Date: 2025-01-27
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import the neural calibration module
from option_neural_calibration import NeuralOptionCalibrator, run_neural_calibration


def setup_experiment_directory(experiment_name: str = "neural_calibration"):
    """Create directory for experiment outputs"""
    exp_dir = Path(f"experiments/{experiment_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def validate_data_file(data_file: str):
    """Validate the input data file"""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Load and check the data structure
    try:
        data = pd.read_csv(data_file)
        required_columns = ['S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda']
        missing_cols = [col for col in required_columns if col not in data.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        print(f"✓ Data file validated: {len(data)} rows, {len(data.columns)} columns")
        print(f"  Columns: {list(data.columns)}")

        # Check for target columns
        has_sigma = 'sigma' in data.columns
        has_V = 'V' in data.columns

        if not has_sigma and not has_V:
            raise ValueError("Data must contain either 'sigma' (volatility) or 'V' (price) column")

        print(f"  Targets available: {'sigma' if has_sigma else ''} {'V' if has_V else ''}")

        return data

    except Exception as e:
        raise ValueError(f"Error reading data file: {str(e)}")


def run_single_model_experiment(data_file: str, target_type: str, exp_dir: Path,
                               config: dict = None):
    """Run calibration for a single model type"""

    print(f"\n{'='*60}")
    print(f"TRAINING {target_type.upper()} PREDICTION MODEL")
    print(f"{'='*60}")

    # Default configuration
    default_config = {
        'epochs': 1000,
        'batch_size': 64,
        'learning_rate': 0.001,
        'validation_split': 0.2,
        'early_stopping_patience': 50,
        'loss_type': 'combined' if target_type == 'volatility' else 'huber',
        'hidden_dims': None
    }

    if config:
        default_config.update(config)

    # Initialize calibrator
    calibrator = NeuralOptionCalibrator(data_file, target_type)

    # Train the model
    print(f"\nTraining with configuration:")
    for key, value in default_config.items():
        print(f"  {key}: {value}")

    calibrator.train(**default_config)

    # Evaluate the model
    print(f"\nEvaluating {target_type} model...")
    metrics = calibrator.evaluate(save_results=True)

    print(f"\n{target_type.upper()} MODEL RESULTS:")
    print("-" * 40)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    # Save model
    model_path = exp_dir / f"neural_{target_type}_model.pth"
    calibrator.save_model(str(model_path))

    # Generate plots
    print(f"\nGenerating plots for {target_type} model...")

    plt.style.use('default')  # Ensure consistent styling

    # Training history
    calibrator.plot_training_history()
    plt.savefig(exp_dir / f"{target_type}_training_history.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Predictions vs actual
    calibrator.plot_predictions_vs_actual()
    plt.savefig(exp_dir / f"{target_type}_predictions_vs_actual.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Residuals analysis
    calibrator.plot_residuals()
    plt.savefig(exp_dir / f"{target_type}_residuals_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics to file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(exp_dir / f"{target_type}_metrics.csv", index=False)

    return calibrator, metrics


def run_comparison_experiment(data_file: str, exp_dir: Path, config: dict = None):
    """Run comparison between volatility and price models"""

    print(f"\n{'='*60}")
    print("NEURAL NETWORK MODEL COMPARISON")
    print(f"{'='*60}")

    results = {}

    # Check which targets are available
    data = pd.read_csv(data_file)
    has_sigma = 'sigma' in data.columns
    has_V = 'V' in data.columns

    # Train volatility model if sigma is available
    if has_sigma:
        vol_calibrator, vol_metrics = run_single_model_experiment(
            data_file, 'volatility', exp_dir, config
        )
        results['volatility'] = {
            'calibrator': vol_calibrator,
            'metrics': vol_metrics
        }

    # Train price model if V is available
    if has_V:
        price_calibrator, price_metrics = run_single_model_experiment(
            data_file, 'price', exp_dir, config
        )
        results['price'] = {
            'calibrator': price_calibrator,
            'metrics': price_metrics
        }

    # Generate comparison plots if both models were trained
    if len(results) == 2:
        print(f"\nGenerating comparison plots...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Model performance comparison
        vol_pred = results['volatility']['calibrator'].predict()
        vol_actual = results['volatility']['calibrator'].targets.flatten()
        vol_r2 = results['volatility']['metrics']['R²']

        price_pred = results['price']['calibrator'].predict()
        price_actual = results['price']['calibrator'].targets.flatten()
        price_r2 = results['price']['metrics']['R²']

        # Volatility predictions
        axes[0, 0].scatter(vol_actual, vol_pred, alpha=0.6, s=20)
        axes[0, 0].plot([vol_actual.min(), vol_actual.max()],
                       [vol_actual.min(), vol_actual.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Volatility')
        axes[0, 0].set_ylabel('Predicted Volatility')
        axes[0, 0].set_title(f'Volatility Model (R² = {vol_r2:.4f})')
        axes[0, 0].grid(True, alpha=0.3)

        # Price predictions
        axes[0, 1].scatter(price_actual, price_pred, alpha=0.6, s=20)
        axes[0, 1].plot([price_actual.min(), price_actual.max()],
                       [price_actual.min(), price_actual.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Price')
        axes[0, 1].set_ylabel('Predicted Price')
        axes[0, 1].set_title(f'Price Model (R² = {price_r2:.4f})')
        axes[0, 1].grid(True, alpha=0.3)

        # Training loss comparison
        vol_train_loss = results['volatility']['calibrator'].training_history['train_loss']
        vol_val_loss = results['volatility']['calibrator'].training_history['val_loss']
        price_train_loss = results['price']['calibrator'].training_history['train_loss']
        price_val_loss = results['price']['calibrator'].training_history['val_loss']

        axes[1, 0].plot(vol_train_loss, label='Vol Train', alpha=0.7)
        axes[1, 0].plot(vol_val_loss, label='Vol Val', alpha=0.7)
        axes[1, 0].plot(price_train_loss, label='Price Train', alpha=0.7)
        axes[1, 0].plot(price_val_loss, label='Price Val', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')

        # Metrics comparison
        metrics_comparison = pd.DataFrame({
            'Volatility': results['volatility']['metrics'],
            'Price': results['price']['metrics']
        })

        # Plot key metrics
        key_metrics = ['RMSE', 'MAE', 'R²', 'MRE']
        metrics_to_plot = metrics_comparison.loc[key_metrics]

        x_pos = np.arange(len(key_metrics))
        width = 0.35

        axes[1, 1].bar(x_pos - width/2, metrics_to_plot['Volatility'], width,
                      label='Volatility', alpha=0.7)
        axes[1, 1].bar(x_pos + width/2, metrics_to_plot['Price'], width,
                      label='Price', alpha=0.7)
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Model Performance Comparison')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(key_metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(exp_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Save comparison metrics
        metrics_comparison.to_csv(exp_dir / "comparison_metrics.csv")

    return results


def run_hyperparameter_study(data_file: str, target_type: str, exp_dir: Path):
    """Run a simple hyperparameter study"""

    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER STUDY - {target_type.upper()}")
    print(f"{'='*60}")

    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [0.001, 0.0005, 0.002],
        'batch_size': [32, 64, 128],
        'hidden_dims': [
            [64, 128, 64],
            [128, 256, 128, 64],
            [256, 512, 256, 128]
        ]
    }

    results = []

    for lr in param_grid['learning_rate']:
        for batch_size in param_grid['batch_size']:
            for hidden_dims in param_grid['hidden_dims']:

                print(f"\nTesting: LR={lr}, Batch={batch_size}, Dims={hidden_dims}")

                try:
                    config = {
                        'epochs': 300,  # Shorter for hyperparameter search
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'hidden_dims': hidden_dims,
                        'early_stopping_patience': 30
                    }

                    calibrator = NeuralOptionCalibrator(data_file, target_type)
                    calibrator.train(**config)
                    metrics = calibrator.evaluate(save_results=False)

                    result = {
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'hidden_dims': str(hidden_dims),
                        **metrics
                    }
                    results.append(result)

                    print(f"  R²: {metrics['R²']:.4f}, RMSE: {metrics['RMSE']:.6f}")

                except Exception as e:
                    print(f"  Failed: {str(e)}")
                    continue

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(exp_dir / f"hyperparameter_study_{target_type}.csv", index=False)

    # Find best configuration
    if len(results_df) > 0:
        best_idx = results_df['R²'].idxmax()
        best_config = results_df.iloc[best_idx]

        print(f"\nBest configuration for {target_type}:")
        print(f"  Learning Rate: {best_config['learning_rate']}")
        print(f"  Batch Size: {best_config['batch_size']}")
        print(f"  Hidden Dims: {best_config['hidden_dims']}")
        print(f"  R²: {best_config['R²']:.4f}")
        print(f"  RMSE: {best_config['RMSE']:.6f}")

        return best_config

    return None


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(description='Neural Network Option Calibration')
    parser.add_argument('--data-file', type=str, default='impl_demo_improved.csv',
                       help='Path to the data file')
    parser.add_argument('--target', type=str, choices=['volatility', 'price', 'both'],
                       default='both', help='Target to predict')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--experiment-name', type=str, default='neural_calibration',
                       help='Name for the experiment directory')
    parser.add_argument('--hyperparameter-study', action='store_true',
                       help='Run hyperparameter study')
    parser.add_argument('--gpu', action='store_true',
                       help='Force GPU usage if available')

    args = parser.parse_args()

    # Setup
    print("Neural Network Option Calibration")
    print("=" * 60)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        if args.gpu:
            print("  GPU usage requested")
    else:
        print("! CUDA not available - using CPU")

    # Validate data file
    print(f"\nValidating data file: {args.data_file}")
    data = validate_data_file(args.data_file)

    # Setup experiment directory
    exp_dir = setup_experiment_directory(args.experiment_name)
    print(f"✓ Experiment directory: {exp_dir}")

    # Training configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'validation_split': 0.2,
        'early_stopping_patience': 50
    }

    try:
        # Run hyperparameter study if requested
        if args.hyperparameter_study:
            if args.target in ['volatility', 'both'] and 'sigma' in data.columns:
                best_vol_config = run_hyperparameter_study(args.data_file, 'volatility', exp_dir)

            if args.target in ['price', 'both'] and 'V' in data.columns:
                best_price_config = run_hyperparameter_study(args.data_file, 'price', exp_dir)

        # Run main experiments
        if args.target == 'both':
            results = run_comparison_experiment(args.data_file, exp_dir, config)

        elif args.target == 'volatility':
            if 'sigma' not in data.columns:
                raise ValueError("Volatility target 'sigma' not found in data")
            vol_calibrator, vol_metrics = run_single_model_experiment(
                args.data_file, 'volatility', exp_dir, config
            )

        elif args.target == 'price':
            if 'V' not in data.columns:
                raise ValueError("Price target 'V' not found in data")
            price_calibrator, price_metrics = run_single_model_experiment(
                args.data_file, 'price', exp_dir, config
            )

        print(f"\n{'='*60}")
        print("CALIBRATION COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Results saved to: {exp_dir}")
        print("\nGenerated files:")
        for file in exp_dir.glob("*"):
            print(f"  - {file.name}")

    except Exception as e:
        print(f"\n❌ Calibration failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
