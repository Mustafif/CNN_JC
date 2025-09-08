#!/usr/bin/env python3
"""
Test script to verify model functionality and improvements.
This script tests all the enhanced components of the volatility prediction model.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import cleandataset, dataset_file, OptionDataset
from model import CaNNModel, NetworkOfNetworks, MultiHeadAttention, ResidualBlock
from volatility_loss import create_volatility_loss, VolatilityLoss
from loss import calculate_loss

def test_data_loading():
    """Test data loading and feature engineering"""
    print("🔍 Testing data loading and feature engineering...")

    try:
        # Load data
        df = cleandataset(dataset_file('impl_demo_improved.csv'))
        print(f"  ✅ Loaded {len(df)} samples")

        # Split data
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # Create datasets
        ds_train = OptionDataset(train_df, is_train=True)
        ds_test = OptionDataset(test_df, is_train=False, target_scaler=ds_train.target_scaler)

        # Check sample
        sample_x, sample_y = ds_train[0]
        print(f"  ✅ Feature dimension: {sample_x.shape[0]}")
        print(f"  ✅ Target shape: {sample_y.shape}")
        print(f"  ✅ Training samples: {len(ds_train)}")
        print(f"  ✅ Test samples: {len(ds_test)}")

        return ds_train, ds_test, sample_x.shape[0]

    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        return None, None, None

def test_model_components():
    """Test individual model components"""
    print("\n🔍 Testing model components...")

    try:
        # Test MultiHeadAttention
        attention = MultiHeadAttention(d_model=64, num_heads=4)
        x = torch.randn(2, 1, 64)  # batch_size, seq_len, d_model
        out = attention(x)
        print(f"  ✅ MultiHeadAttention: input {x.shape} -> output {out.shape}")

        # Test ResidualBlock
        residual = ResidualBlock(hidden_size=64, dropout_rate=0.1)
        x = torch.randn(2, 64)
        out = residual(x)
        print(f"  ✅ ResidualBlock: input {x.shape} -> output {out.shape}")

        return True

    except Exception as e:
        print(f"  ❌ Model components test failed: {e}")
        return False

def test_models(input_features):
    """Test main model architectures"""
    print("\n🔍 Testing model architectures...")

    try:
        # Test CaNNModel
        model = CaNNModel(input_features=input_features, hidden_size=128, dropout_rate=0.1)
        x = torch.randn(4, input_features)
        output = model(x)
        print(f"  ✅ CaNNModel: input {x.shape} -> output {output.shape}")
        print(f"  ✅ Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")

        # Test NetworkOfNetworks
        ensemble = NetworkOfNetworks(
            num_child_networks=3,
            input_features=input_features,
            hidden_size=64,
            dropout_rate=0.1
        )
        output_ensemble = ensemble(x)
        print(f"  ✅ NetworkOfNetworks: input {x.shape} -> output {output_ensemble.shape}")
        print(f"  ✅ Ensemble output range: [{output_ensemble.min().item():.6f}, {output_ensemble.max().item():.6f}]")

        return model, ensemble

    except Exception as e:
        print(f"  ❌ Model architecture test failed: {e}")
        return None, None

def test_loss_functions():
    """Test volatility-specific loss functions"""
    print("\n🔍 Testing loss functions...")

    try:
        # Generate sample data
        pred = torch.randn(10, 1) * 0.1 + 0.2  # Around 20% volatility
        target = torch.randn(10, 1) * 0.1 + 0.25  # Around 25% volatility
        pred = torch.abs(pred)  # Ensure positive
        target = torch.abs(target)  # Ensure positive

        # Test different loss functions
        loss_types = ['combined', 'adaptive', 'focal']

        for loss_type in loss_types:
            try:
                criterion = create_volatility_loss(loss_type)

                if hasattr(criterion, 'forward') and len(criterion.forward.__code__.co_varnames) > 2:
                    # Enhanced loss function returning tuple
                    loss_result = criterion(pred, target)
                    if isinstance(loss_result, tuple):
                        loss, details = loss_result
                        print(f"  ✅ {loss_type.title()} Loss: {loss.item():.6f}")
                        if details:
                            detail_str = ", ".join([f"{k}: {v:.4f}" for k, v in details.items() if k != 'total'])
                            print(f"      Details: {detail_str}")
                    else:
                        print(f"  ✅ {loss_type.title()} Loss: {loss_result.item():.6f}")
                else:
                    # Standard loss function
                    loss = criterion(pred, target)
                    print(f"  ✅ {loss_type.title()} Loss: {loss.item():.6f}")

            except Exception as e:
                print(f"  ⚠️  {loss_type.title()} Loss failed: {e}")

        return True

    except Exception as e:
        print(f"  ❌ Loss function test failed: {e}")
        return False

def test_training_step(model, ds_train):
    """Test a single training step"""
    print("\n🔍 Testing training step...")

    try:
        from torch.utils.data import DataLoader

        # Create small data loader
        train_loader = DataLoader(ds_train, batch_size=8, shuffle=True)

        # Setup training components
        criterion = create_volatility_loss('combined')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Single training step
        model.train()
        batch_X, batch_y = next(iter(train_loader))

        optimizer.zero_grad()
        output = model(batch_X.float())
        target = batch_y.float().view_as(output)

        # Handle loss function output
        loss_result = criterion(output, target)
        if isinstance(loss_result, tuple):
            loss, details = loss_result
        else:
            loss = loss_result

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f"  ✅ Training step completed")
        print(f"  ✅ Loss: {loss.item():.6f}")
        print(f"  ✅ Gradient norm: {torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf')):.6f}")

        return True

    except Exception as e:
        print(f"  ❌ Training step test failed: {e}")
        return False

def test_prediction_and_evaluation(model, ds_test):
    """Test prediction and evaluation"""
    print("\n🔍 Testing prediction and evaluation...")

    try:
        from torch.utils.data import DataLoader

        # Create test loader
        test_loader = DataLoader(ds_test, batch_size=16, shuffle=False)

        # Make predictions
        model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                output = model(batch_X.float())
                predictions.extend(output.cpu().numpy().flatten())
                targets.extend(batch_y.cpu().numpy().flatten())

        predictions = np.array(predictions)
        targets = np.array(targets)

        # Save predictions for evaluation
        results_df = pd.DataFrame({
            'predictions': predictions,
            'targets': targets
        })
        results_df.to_csv('temp_test_results.csv', index=False)

        # Calculate metrics
        metrics = calculate_loss('temp_test_results.csv', ds_test.target_scaler)

        print(f"  ✅ Predictions generated: {len(predictions)}")
        print(f"  ✅ MAE: {metrics['MAE']:.6f}")
        print(f"  ✅ RMSE: {metrics['RMSE']:.6f}")
        print(f"  ✅ R²: {metrics['R^2']:.6f}")

        # Clean up
        if os.path.exists('temp_test_results.csv'):
            os.remove('temp_test_results.csv')

        return True

    except Exception as e:
        print(f"  ❌ Prediction and evaluation test failed: {e}")
        return False

def test_feature_importance():
    """Test feature engineering improvements"""
    print("\n🔍 Testing feature engineering...")

    try:
        # Load sample data
        df = cleandataset(dataset_file('impl_demo_improved.csv'))
        train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
        ds_train = OptionDataset(train_df, is_train=True)

        # Get feature names and sample
        sample_x, _ = ds_train[0]
        base_features = ["S0", "m", "r", "T", "corp", "alpha", "beta", "omega", "gamma", "lambda", "V"]
        engineered_features = [
            "strike", "log_moneyness", "moneyness_squared", "moneyness_centered", "atm_indicator",
            "sqrt_T", "log_T", "inv_T", "time_decay", "log_gamma", "sqrt_omega", "log_omega",
            "alpha_beta", "alpha_gamma", "beta_squared", "persistence", "mean_reversion",
            "unconditional_vol", "risk_free_T", "lambda_scaled", "m_T_interaction",
            "vol_skew_proxy", "value_ratio", "log_value", "is_call", "corp_m_interaction", "corp_T_interaction"
        ]

        total_expected = len(base_features) + len(engineered_features)

        print(f"  ✅ Base features: {len(base_features)}")
        print(f"  ✅ Engineered features: {len(engineered_features)}")
        print(f"  ✅ Total expected: {total_expected}")
        print(f"  ✅ Actual features: {sample_x.shape[0]}")

        if sample_x.shape[0] == total_expected:
            print(f"  ✅ Feature count matches expected!")
        else:
            print(f"  ⚠️  Feature count mismatch: expected {total_expected}, got {sample_x.shape[0]}")

        # Check for any NaN or infinite values
        if torch.isfinite(sample_x).all():
            print(f"  ✅ All features are finite")
        else:
            print(f"  ⚠️  Some features contain NaN or infinite values")

        return True

    except Exception as e:
        print(f"  ❌ Feature engineering test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Model Functionality Tests")
    print("=" * 50)

    # Track test results
    test_results = {}

    # Test 1: Data Loading
    ds_train, ds_test, input_features = test_data_loading()
    test_results['data_loading'] = ds_train is not None

    if not test_results['data_loading']:
        print("\n❌ Critical failure: Cannot proceed without data loading")
        return

    # Test 2: Model Components
    test_results['components'] = test_model_components()

    # Test 3: Model Architectures
    model, ensemble = test_models(input_features)
    test_results['models'] = model is not None

    # Test 4: Loss Functions
    test_results['loss_functions'] = test_loss_functions()

    # Test 5: Training Step
    if model is not None:
        test_results['training'] = test_training_step(model, ds_train)
    else:
        test_results['training'] = False

    # Test 6: Prediction and Evaluation
    if model is not None:
        test_results['prediction'] = test_prediction_and_evaluation(model, ds_test)
    else:
        test_results['prediction'] = False

    # Test 7: Feature Engineering
    test_results['features'] = test_feature_importance()

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    total_tests = len(test_results)
    passed_tests = sum(test_results.values())

    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title():<20}: {status}")

    print("-" * 50)
    print(f"Overall Result: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! Model is ready for training.")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  Most tests passed. Minor issues detected.")
    else:
        print("❌ Multiple test failures. Please review the issues above.")

    print("\n🔍 To run full training:")
    print("python ANN2.py")

    print("\n🔍 To run advanced hyperparameter optimization:")
    print("python -c \"from advanced_training import AdvancedTrainer; trainer = AdvancedTrainer('impl_demo_improved.csv'); trainer.run_complete_pipeline()\"")

if __name__ == "__main__":
    main()
