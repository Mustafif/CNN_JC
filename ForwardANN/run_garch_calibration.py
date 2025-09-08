#!/usr/bin/env python3
"""
GARCH Calibration Script for Option Data
Runs both time series GARCH calibration and option surface calibration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from garch_calibration import GARCHCalibrator, GARCH11, GJRGARCH, EGARCH
from option_garch_calibration import OptionGARCHCalibrator, OptionData, BlackScholesModel
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath='impl_demo_improved.csv'):
    """Load and prepare option data for calibration"""
    print("Loading option data...")
    df = pd.read_csv(filepath)

    print(f"Loaded {len(df)} option records")
    print(f"Columns: {list(df.columns)}")
    print("\nData summary:")
    print(df.describe())

    # Extract unique values
    unique_S0 = df['S0'].unique()
    unique_r = df['r'].unique()

    print(f"\nUnique S0 values: {unique_S0}")
    print(f"Unique r values: {unique_r}")

    # Use most common values
    S0 = df['S0'].mode()[0]
    r = df['r'].mode()[0]

    # Prepare option data
    strikes = df['S0'] * df['m']  # K = S0 * moneyness
    maturities = df['T']
    option_types = df['corp']  # 1 for calls, -1 for puts
    option_values = df['V']

    print(f"\nUsing S0 = {S0:.2f}, r = {r:.6f}")
    print(f"Strike range: {strikes.min():.2f} - {strikes.max():.2f}")
    print(f"Maturity range: {maturities.min():.4f} - {maturities.max():.4f}")
    print(f"Option types: {np.unique(option_types)}")

    return df, S0, r, strikes, maturities, option_types, option_values

def run_time_series_garch_calibration(df):
    """Run traditional GARCH calibration on return series"""
    print("\n" + "="*60)
    print("TIME SERIES GARCH CALIBRATION")
    print("="*60)

    # Since we don't have return series, we'll simulate one based on the volatility data
    # This is a simplified approach - in practice you'd use actual return data

    # Extract volatility parameters from the data
    sigma_values = df['sigma'].values
    T_values = df['T'].values

    # Create a synthetic return series based on the GARCH parameters in the data
    # Use the GARCH parameters from the dataset
    alpha_mean = df['alpha'].mean()
    beta_mean = df['beta'].mean()
    omega_mean = df['omega'].mean()

    print(f"Average GARCH parameters from data:")
    print(f"  ω (omega): {omega_mean:.6f}")
    print(f"  α (alpha): {alpha_mean:.6f}")
    print(f"  β (beta): {beta_mean:.6f}")

    # Simulate a return series using these parameters
    np.random.seed(42)
    T = 1000
    returns = np.zeros(T)
    sigma2 = np.zeros(T)

    # Initial variance
    sigma2[0] = omega_mean / (1 - alpha_mean - beta_mean)

    for t in range(T):
        epsilon = np.random.normal(0, 1)
        returns[t] = np.sqrt(sigma2[t]) * epsilon
        if t < T - 1:
            sigma2[t+1] = omega_mean + alpha_mean * returns[t]**2 + beta_mean * sigma2[t]

    print(f"\nGenerated {T} synthetic returns for calibration")
    print(f"Return statistics:")
    print(f"  Mean: {np.mean(returns):.6f}")
    print(f"  Std: {np.std(returns):.6f}")
    print(f"  Min: {np.min(returns):.6f}")
    print(f"  Max: {np.max(returns):.6f}")

    # Create calibrator and add models
    calibrator = GARCHCalibrator()
    calibrator.add_model('GARCH(1,1)', GARCH11())
    calibrator.add_model('GJR-GARCH', GJRGARCH())
    calibrator.add_model('EGARCH', EGARCH())
    calibrator.add_model('GARCH(1,1)-t', GARCH11(distribution='t'))

    # Calibrate all models
    results = calibrator.calibrate_all(returns, maxiter=500)

    # Display results
    print("\nModel Comparison:")
    comparison = calibrator.compare_models()
    print(comparison)

    # Get best model
    best_name, best_result = calibrator.best_model('aic')
    print(f"\nBest model (AIC): {best_name}")
    print(calibrator.models[best_name].summary())

    # Plot volatility
    try:
        fig = calibrator.plot_volatility(returns)
        fig.savefig('garch_time_series_volatility.png', dpi=300, bbox_inches='tight')
        print("Time series volatility plot saved as 'garch_time_series_volatility.png'")
    except Exception as e:
        print(f"Could not create volatility plot: {e}")

    return calibrator, returns, best_name, best_result

def run_option_surface_calibration(S0, r, strikes, maturities, option_types, option_values):
    """Run GARCH option surface calibration"""
    print("\n" + "="*60)
    print("OPTION SURFACE GARCH CALIBRATION")
    print("="*60)

    # Create option data object
    option_data = OptionData(
        S0=S0,
        K=strikes.values,
        T=maturities.values,
        r=r,
        option_prices=option_values.values,
        option_types=option_types.values
    )

    print(f"Created option data with {len(option_values)} options")

    # Create calibrator
    calibrator = OptionGARCHCalibrator(option_data)

    print("\nCalculated implied volatilities:")
    print(f"  Min IV: {np.min(option_data.implied_vols):.4f}")
    print(f"  Max IV: {np.max(option_data.implied_vols):.4f}")
    print(f"  Mean IV: {np.mean(option_data.implied_vols):.4f}")

    # Calibrate with surface parameters
    print("\nCalibrating GARCH surface model...")
    calibrated_params = calibrator.calibrate(
        method='differential_evolution',
        include_surface_params=True,
        maxiter=100
    )

    if calibrated_params is not None:
        # Calculate pricing errors
        errors = calibrator.calculate_pricing_errors()
        print("\nPricing Error Metrics:")
        for metric, value in errors.items():
            print(f"  {metric}: {value:.6f}")

        # Create fitted surface
        surface = calibrator.create_fitted_surface(n_strikes=15, n_maturities=8)
        print(f"\nCreated volatility surface:")
        print(f"  Strike range: {surface.strikes.min():.2f} - {surface.strikes.max():.2f}")
        print(f"  Maturity range: {surface.maturities.min():.4f} - {surface.maturities.max():.4f}")

        # Plot results
        try:
            # Volatility surface plot
            fig1 = calibrator.plot_volatility_surface()
            fig1.savefig('garch_option_volatility_surface.png', dpi=300, bbox_inches='tight')
            print("Volatility surface plot saved as 'garch_option_volatility_surface.png'")

            # Volatility smile for different maturities
            available_maturities = np.unique(maturities.values)
            for i, T in enumerate(available_maturities[:3]):  # Plot first 3 maturities
                fig2 = calibrator.plot_implied_vol_smile(target_maturity=T)
                if fig2 is not None:
                    fig2.savefig(f'garch_vol_smile_T_{T:.2f}.png', dpi=300, bbox_inches='tight')
                    print(f"Volatility smile plot saved as 'garch_vol_smile_T_{T:.2f}.png'")

        except Exception as e:
            print(f"Could not create surface plots: {e}")

    else:
        print("Option surface calibration failed!")

    return calibrator, calibrated_params

def compare_with_dataset_parameters(df, calibrated_params_ts, calibrated_params_opt):
    """Compare calibrated parameters with those in the dataset"""
    print("\n" + "="*60)
    print("PARAMETER COMPARISON")
    print("="*60)

    # Dataset parameters (average values)
    dataset_params = {
        'omega': df['omega'].mean(),
        'alpha': df['alpha'].mean(),
        'beta': df['beta'].mean(),
        'gamma': df['gamma'].mean()
    }

    print("Dataset Average Parameters:")
    for param, value in dataset_params.items():
        print(f"  {param}: {value:.6f}")

    if calibrated_params_ts is not None:
        print(f"\nTime Series GARCH Parameters:")
        print(f"  omega: {calibrated_params_ts.params[0]:.6f}")
        print(f"  alpha: {calibrated_params_ts.params[1]:.6f}")
        print(f"  beta: {calibrated_params_ts.params[2]:.6f}")
        if len(calibrated_params_ts.params) > 3:
            print(f"  gamma: {calibrated_params_ts.params[3]:.6f}")

    if calibrated_params_opt is not None:
        print(f"\nOption Surface GARCH Parameters:")
        for param, value in calibrated_params_opt.items():
            print(f"  {param}: {value:.6f}")

    # Calculate differences
    if calibrated_params_ts is not None:
        print(f"\nTime Series vs Dataset Differences:")
        ts_params = calibrated_params_ts.params
        print(f"  Δω: {ts_params[0] - dataset_params['omega']:+.6f}")
        print(f"  Δα: {ts_params[1] - dataset_params['alpha']:+.6f}")
        print(f"  Δβ: {ts_params[2] - dataset_params['beta']:+.6f}")

    if calibrated_params_opt is not None:
        print(f"\nOption Surface vs Dataset Differences:")
        print(f"  Δω: {calibrated_params_opt['omega'] - dataset_params['omega']:+.6f}")
        print(f"  Δα: {calibrated_params_opt['alpha'] - dataset_params['alpha']:+.6f}")
        print(f"  Δβ: {calibrated_params_opt['beta'] - dataset_params['beta']:+.6f}")

def create_summary_report(df, time_series_results, option_results):
    """Create a summary report"""
    print("\n" + "="*60)
    print("GARCH CALIBRATION SUMMARY REPORT")
    print("="*60)

    print(f"\nDataset Information:")
    print(f"  Total options: {len(df)}")
    print(f"  Unique strikes: {df['m'].nunique()}")
    print(f"  Unique maturities: {df['T'].nunique()}")
    print(f"  Call options: {len(df[df['corp'] == 1])}")
    print(f"  Put options: {len(df[df['corp'] == -1])}")

    print(f"\nVolatility Statistics:")
    print(f"  Mean implied vol: {df['sigma'].mean():.4f}")
    print(f"  Std implied vol: {df['sigma'].std():.4f}")
    print(f"  Min implied vol: {df['sigma'].min():.4f}")
    print(f"  Max implied vol: {df['sigma'].max():.4f}")

    if time_series_results[1] is not None:
        best_name, best_result = time_series_results[2], time_series_results[3]
        print(f"\nBest Time Series Model: {best_name}")
        print(f"  Log-likelihood: {best_result.loglikelihood:.4f}")
        print(f"  AIC: {best_result.aic:.4f}")
        print(f"  Converged: {best_result.converged}")

    if option_results[1] is not None:
        errors = option_results[0].calculate_pricing_errors()
        print(f"\nOption Surface Calibration:")
        print(f"  RMSE: {errors['RMSE']:.6f}")
        print(f"  MAE: {errors['MAE']:.6f}")
        print(f"  MAPE: {errors['MAPE']:.2f}%")
        print(f"  R²: {errors['R_squared']:.4f}")

    print(f"\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)

    print("1. Time Series GARCH:")
    print("   - Use for modeling return volatility dynamics")
    print("   - Good for risk management and volatility forecasting")

    print("\n2. Option Surface GARCH:")
    print("   - Use for option pricing and hedging")
    print("   - Captures smile and term structure effects")

    print("\n3. Model Selection:")
    if time_series_results[1] is not None and option_results[1] is not None:
        print("   - Both calibrations successful")
        print("   - Choose based on intended application")
    elif time_series_results[1] is not None:
        print("   - Time series calibration successful")
        print("   - Option surface calibration needs refinement")
    elif option_results[1] is not None:
        print("   - Option surface calibration successful")
        print("   - Time series approach may need more data")
    else:
        print("   - Both calibrations need refinement")
        print("   - Consider data quality and model assumptions")

def main():
    """Main execution function"""
    print("GARCH Model Calibration Suite")
    print("="*60)
    print("This script performs both time series and option surface GARCH calibration")
    print()

    try:
        # Load data
        df, S0, r, strikes, maturities, option_types, option_values = load_and_prepare_data()

        # Run time series calibration
        time_series_results = run_time_series_garch_calibration(df)

        # Run option surface calibration
        option_results = run_option_surface_calibration(S0, r, strikes, maturities, option_types, option_values)

        # Compare parameters
        ts_params = time_series_results[3] if time_series_results[1] is not None else None
        opt_params = option_results[1] if option_results[1] is not None else None
        compare_with_dataset_parameters(df, ts_params, opt_params)

        # Create summary report
        create_summary_report(df, time_series_results, option_results)

        print(f"\n✅ GARCH calibration completed successfully!")
        print(f"📊 Check the generated plots for visual results")
        print(f"📈 Review the parameter comparisons above")

    except FileNotFoundError:
        print("❌ Error: Could not find 'impl_demo_improved.csv'")
        print("   Make sure the data file is in the current directory")

    except Exception as e:
        print(f"❌ Error during calibration: {e}")
        print("   Check your data format and try again")

if __name__ == "__main__":
    main()
