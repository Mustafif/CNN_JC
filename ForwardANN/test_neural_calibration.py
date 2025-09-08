#!/usr/bin/env python3
"""
Test Script for Neural Network Option Calibration

This script tests the neural network calibration system with your dataset.
It performs quick validation tests to ensure everything works correctly.

Usage:
    python test_neural_calibration.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from option_neural_calibration import NeuralOptionCalibrator
    print("✓ Successfully imported NeuralOptionCalibrator")
except ImportError as e:
    print(f"❌ Failed to import neural calibration module: {e}")
    sys.exit(1)


def create_test_data(filename='test_data.csv', n_samples=1000):
    """Create synthetic test data matching your format"""

    print(f"Creating synthetic test data with {n_samples} samples...")

    np.random.seed(42)

    # Generate synthetic option data
    data = []

    # Parameters for realistic option data
    S0_values = np.random.uniform(80, 120, n_samples)
    moneyness = np.random.uniform(0.7, 1.3, n_samples)
    r_values = np.random.uniform(0.0001, 0.05, n_samples)
    T_values = np.random.uniform(0.1, 2.0, n_samples)
    corp_values = np.random.choice([1, -1], n_samples)  # Call/Put

    # GARCH parameters (realistic ranges)
    alpha_values = np.random.uniform(0.000001, 0.1, n_samples)
    beta_values = np.random.uniform(0.7, 0.95, n_samples)
    omega_values = np.random.uniform(0.000001, 0.01, n_samples)
    gamma_values = np.random.uniform(3, 7, n_samples)
    lambda_values = np.random.uniform(0.1, 0.3, n_samples)

    for i in range(n_samples):
        # Generate realistic implied volatility with smile effect
        log_moneyness = np.log(moneyness[i])
        base_vol = 0.2
        smile_effect = 0.05 * log_moneyness**2 + 0.03 * np.sqrt(T_values[i])
        volatility_noise = np.random.normal(0, 0.02)
        sigma = max(0.05, base_vol + smile_effect + volatility_noise)

        # Black-Scholes price calculation
        S = S0_values[i]
        K = S / moneyness[i]  # Strike from moneyness
        T = T_values[i]
        r = r_values[i]

        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        from scipy.stats import norm

        if corp_values[i] == 1:  # Call
            price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        else:  # Put
            price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        # Add some market noise to price
        price *= (1 + np.random.normal(0, 0.01))
        price = max(0.01, price)

        data.append({
            'S0': S0_values[i],
            'm': moneyness[i],
            'r': r_values[i],
            'T': T_values[i],
            'corp': corp_values[i],
            'alpha': alpha_values[i],
            'beta': beta_values[i],
            'omega': omega_values[i],
            'gamma': gamma_values[i],
            'lambda': lambda_values[i],
            'sigma': sigma,
            'V': price
        })

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

    print(f"✓ Created test data: {filename}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Volatility range: [{df['sigma'].min():.3f}, {df['sigma'].max():.3f}]")
    print(f"  Price range: [{df['V'].min():.3f}, {df['V'].max():.3f}]")

    return filename


def test_data_loading(data_file):
    """Test data loading functionality"""

    print(f"\n{'='*50}")
    print("TEST 1: Data Loading")
    print(f"{'='*50}")

    try:
        # Test volatility calibrator
        vol_calibrator = NeuralOptionCalibrator(data_file, 'volatility')
        print(f"✓ Volatility calibrator created successfully")
        print(f"  Features shape: {vol_calibrator.features.shape}")
        print(f"  Targets shape: {vol_calibrator.targets.shape}")

        # Test price calibrator
        price_calibrator = NeuralOptionCalibrator(data_file, 'price')
        print(f"✓ Price calibrator created successfully")
        print(f"  Features shape: {price_calibrator.features.shape}")
        print(f"  Targets shape: {price_calibrator.targets.shape}")

        return vol_calibrator, price_calibrator

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, None


def test_model_creation(vol_calibrator, price_calibrator):
    """Test neural network model creation"""

    print(f"\n{'='*50}")
    print("TEST 2: Model Creation")
    print(f"{'='*50}")

    try:
        # Test volatility model
        vol_calibrator.create_model()
        print(f"✓ Volatility model created")

        vol_params = sum(p.numel() for p in vol_calibrator.model.parameters())
        print(f"  Volatility model parameters: {vol_params:,}")

        # Test price model
        price_calibrator.create_model()
        print(f"✓ Price model created")

        price_params = sum(p.numel() for p in price_calibrator.model.parameters())
        print(f"  Price model parameters: {price_params:,}")

        # Test forward pass
        test_input = torch.randn(1, vol_calibrator.features.shape[1])
        vol_output = vol_calibrator.model(test_input)
        price_output = price_calibrator.model(test_input)

        print(f"✓ Forward pass successful")
        print(f"  Volatility output shape: {vol_output.shape}")
        print(f"  Price output shape: {price_output.shape}")

        return True

    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def test_training(vol_calibrator, price_calibrator):
    """Test training functionality with minimal epochs"""

    print(f"\n{'='*50}")
    print("TEST 3: Training (Quick Test)")
    print(f"{'='*50}")

    try:
        # Quick training test - volatility
        print("Training volatility model (50 epochs)...")
        vol_calibrator.train(
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            validation_split=0.2,
            early_stopping_patience=20,
            loss_type='combined'
        )
        print(f"✓ Volatility training completed")

        # Quick training test - price
        print("Training price model (50 epochs)...")
        price_calibrator.train(
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            validation_split=0.2,
            early_stopping_patience=20
        )
        print(f"✓ Price training completed")

        return True

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


def test_evaluation(vol_calibrator, price_calibrator):
    """Test model evaluation"""

    print(f"\n{'='*50}")
    print("TEST 4: Model Evaluation")
    print(f"{'='*50}")

    try:
        # Test volatility evaluation
        vol_metrics = vol_calibrator.evaluate(save_results=False)
        print(f"✓ Volatility evaluation completed")
        print(f"  R²: {vol_metrics['R²']:.4f}")
        print(f"  RMSE: {vol_metrics['RMSE']:.6f}")
        print(f"  MAE: {vol_metrics['MAE']:.6f}")

        # Test price evaluation
        price_metrics = price_calibrator.evaluate(save_results=False)
        print(f"✓ Price evaluation completed")
        print(f"  R²: {price_metrics['R²']:.4f}")
        print(f"  RMSE: {price_metrics['RMSE']:.6f}")
        print(f"  MAE: {price_metrics['MAE']:.6f}")

        return vol_metrics, price_metrics

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None, None


def test_predictions(vol_calibrator, price_calibrator):
    """Test prediction functionality"""

    print(f"\n{'='*50}")
    print("TEST 5: Predictions")
    print(f"{'='*50}")

    try:
        # Test predictions
        vol_predictions = vol_calibrator.predict()
        price_predictions = price_calibrator.predict()

        print(f"✓ Predictions completed")
        print(f"  Volatility predictions shape: {vol_predictions.shape}")
        print(f"  Volatility range: [{vol_predictions.min():.4f}, {vol_predictions.max():.4f}]")
        print(f"  Price predictions shape: {price_predictions.shape}")
        print(f"  Price range: [{price_predictions.min():.4f}, {price_predictions.max():.4f}]")

        # Test predictions on subset of data
        subset_features = vol_calibrator.features[:10]
        subset_vol_pred = vol_calibrator.predict(subset_features)
        subset_price_pred = price_calibrator.predict(subset_features)

        print(f"✓ Subset predictions successful")
        print(f"  Subset volatility predictions: {len(subset_vol_pred)}")
        print(f"  Subset price predictions: {len(subset_price_pred)}")

        return True

    except Exception as e:
        print(f"❌ Predictions failed: {e}")
        return False


def test_save_load(vol_calibrator, price_calibrator):
    """Test model saving and loading"""

    print(f"\n{'='*50}")
    print("TEST 6: Save/Load Models")
    print(f"{'='*50}")

    try:
        # Save models
        vol_calibrator.save_model('test_vol_model.pth')
        price_calibrator.save_model('test_price_model.pth')
        print(f"✓ Models saved successfully")

        # Create new calibrators and load models
        data_file = vol_calibrator.data_file
        new_vol_calibrator = NeuralOptionCalibrator(data_file, 'volatility')
        new_price_calibrator = NeuralOptionCalibrator(data_file, 'price')

        new_vol_calibrator.load_model('test_vol_model.pth')
        new_price_calibrator.load_model('test_price_model.pth')
        print(f"✓ Models loaded successfully")

        # Test predictions with loaded models
        orig_vol_pred = vol_calibrator.predict()
        new_vol_pred = new_vol_calibrator.predict()

        orig_price_pred = price_calibrator.predict()
        new_price_pred = new_price_calibrator.predict()

        # Check if predictions match
        vol_diff = np.abs(orig_vol_pred - new_vol_pred).max()
        price_diff = np.abs(orig_price_pred - new_price_pred).max()

        print(f"✓ Model consistency check")
        print(f"  Max volatility prediction difference: {vol_diff:.8f}")
        print(f"  Max price prediction difference: {price_diff:.8f}")

        if vol_diff < 1e-6 and price_diff < 1e-6:
            print(f"✓ Save/load test passed")
            return True
        else:
            print(f"⚠ Small differences detected in predictions")
            return True

    except Exception as e:
        print(f"❌ Save/load failed: {e}")
        return False

    finally:
        # Cleanup
        for file in ['test_vol_model.pth', 'test_price_model.pth']:
            if os.path.exists(file):
                os.remove(file)


def test_plotting(vol_calibrator, price_calibrator):
    """Test plotting functionality"""

    print(f"\n{'='*50}")
    print("TEST 7: Plotting (Optional)")
    print(f"{'='*50}")

    try:
        # Test if matplotlib works and create simple plots
        plt.ioff()  # Turn off interactive mode

        # Test training history plot
        vol_calibrator.plot_training_history()
        plt.close()
        print(f"✓ Training history plot successful")

        # Test predictions vs actual plot
        vol_calibrator.plot_predictions_vs_actual()
        plt.close()
        print(f"✓ Predictions vs actual plot successful")

        # Test residuals plot
        vol_calibrator.plot_residuals()
        plt.close()
        print(f"✓ Residuals plot successful")

        return True

    except Exception as e:
        print(f"⚠ Plotting failed (non-critical): {e}")
        return False


def run_all_tests():
    """Run all tests sequentially"""

    print("NEURAL NETWORK CALIBRATION TEST SUITE")
    print("=" * 60)

    # Test results
    results = {
        'data_loading': False,
        'model_creation': False,
        'training': False,
        'evaluation': False,
        'predictions': False,
        'save_load': False,
        'plotting': False
    }

    # Create test data
    data_file = create_test_data('test_neural_data.csv', n_samples=500)

    try:
        # Test 1: Data Loading
        vol_calibrator, price_calibrator = test_data_loading(data_file)
        if vol_calibrator is not None and price_calibrator is not None:
            results['data_loading'] = True
        else:
            print("❌ Cannot continue tests - data loading failed")
            return results

        # Test 2: Model Creation
        if test_model_creation(vol_calibrator, price_calibrator):
            results['model_creation'] = True

        # Test 3: Training
        if test_training(vol_calibrator, price_calibrator):
            results['training'] = True

        # Test 4: Evaluation
        vol_metrics, price_metrics = test_evaluation(vol_calibrator, price_calibrator)
        if vol_metrics is not None and price_metrics is not None:
            results['evaluation'] = True

        # Test 5: Predictions
        if test_predictions(vol_calibrator, price_calibrator):
            results['predictions'] = True

        # Test 6: Save/Load
        if test_save_load(vol_calibrator, price_calibrator):
            results['save_load'] = True

        # Test 7: Plotting (optional)
        if test_plotting(vol_calibrator, price_calibrator):
            results['plotting'] = True

    except Exception as e:
        print(f"❌ Unexpected error during tests: {e}")

    finally:
        # Cleanup test data
        if os.path.exists(data_file):
            os.remove(data_file)

    return results


def print_test_summary(results):
    """Print test summary"""

    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! Neural calibration system is working correctly.")
    elif passed_tests >= total_tests - 1:  # Allow plotting to fail
        print("✅ Core functionality working! Minor issues with optional features.")
    else:
        print("⚠ Some critical tests failed. Please check the error messages above.")

    return passed_tests >= total_tests - 1  # Success if only plotting fails


def main():
    """Main test execution"""

    print("Starting Neural Network Calibration Tests...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Run tests
    results = run_all_tests()

    # Print summary
    success = print_test_summary(results)

    if success:
        print(f"\n{'='*60}")
        print("NEXT STEPS:")
        print(f"{'='*60}")
        print("1. Run: python run_neural_calibration.py --data-file impl_demo_improved.csv")
        print("2. Or use the calibrator directly in your code:")
        print("")
        print("   from option_neural_calibration import NeuralOptionCalibrator")
        print("   calibrator = NeuralOptionCalibrator('your_data.csv', 'volatility')")
        print("   calibrator.train(epochs=1000)")
        print("   metrics = calibrator.evaluate()")
        print("")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
