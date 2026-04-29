# Centralized Parameter Comparison System - Implementation Summary

## Overview
Created a centralized system for comparing GARCH model parameters (true, initial, and calibrated) using option pricing datasets. This system is modeled after `scaled_dataset.m` but specifically designed for parameter comparison studies.

## Files Created

### Core System Files
1. **`compare_dataset.m`** (Main Function)
   - Centralized function to generate comparison datasets
   - Similar interface to `scaled_dataset.m`
   - Supports multiple GARCH models (HN, Duan, GJR)
   - Uses shared randomness across all parameter sets for fair comparison
   - Built-in noise simulation capability
   - Batch processing for memory efficiency

2. **`compare_config.toml`** (Configuration File)
   - TOML format for easy editing and version control
   - Contains all configuration parameters:
     - Model selection (HN, Duan, GJR)
     - True, initial, and calibrated GARCH parameters
     - Scaling parameters (contracts per day, number of days)
     - Market parameters (risk-free rate, moneyness range, maturity range)
     - Willow tree parameters
     - Solver parameters
     - Noise configuration (level and seed)
     - Output settings
   - Pre-populated with actual values from `compare/init.json` and `compare/results_best1bin.json`

3. **`read_toml_config.py`** (Helper Script)
   - Python script to parse TOML files and output JSON
   - Enables MATLAB to read TOML configuration files
   - Simple command-line interface

4. **`rel_err_results.m`** (Updated Analysis Function)
   - Enhanced version of `compare/rel_err_results.m`
   - Flexible interface with name-value parameters
   - Computes mean relative errors by maturity and moneyness categories
   - Generates improvement metrics (calibrated vs initial)
   - Optional plotting and saving capabilities
   - Filters outliers automatically

### Example and Documentation Files
5. **`COMPARE_README.md`** (Comprehensive Documentation)
   - Complete usage guide
   - Configuration file reference
   - Multiple examples with different scenarios
   - Troubleshooting section
   - Advantages over the old system

6. **`example_compare_usage.m`** (Usage Examples)
   - 5 different usage examples
   - Shows basic usage, custom configs, and noise studies
   - Commented out by default (ready to uncomment and run)

7. **`compare_config_duan_example.toml`** (Example Config)
   - Example configuration for Duan GARCH model
   - Shows 3% noise configuration
   - Demonstrates different output folder

8. **`test_compare_system.m`** (System Test)
   - Comprehensive system check
   - Tests Python/TOML availability
   - Verifies required MATLAB functions
   - Validates configuration file
   - Reports any issues

## Key Features

### 1. Centralized Configuration
- All parameters in a single TOML file
- Easy to version control and share
- Clear separation of concerns
- No need to edit MATLAB code

### 2. Fair Comparison
- **Same randomness** used for all parameter sets
- GARCH paths generated with identical Z matrix
- Contract-level randomness (strike/maturity) consistent across sets
- Ensures differences are only due to parameter values

### 3. Noise Simulation
- Built-in noise addition capability
- Configurable noise level (as % of option price)
- Separate random seed for noise
- Useful for robustness testing

### 4. Multi-Model Support
- Heston-Nandi (HN) GARCH
- Duan GARCH
- GJR-GARCH
- Easy to switch by changing one config parameter

### 5. Enhanced Analysis
- Automatic outlier filtering
- Categorization by maturity and moneyness
- Improvement metrics
- Optional visualization
- CSV export for further analysis

### 6. Memory Efficient
- Batch processing
- Tree caching
- Memory cleanup
- Scalable to large datasets

## Workflow

### Basic Workflow
```matlab
% 1. Edit configuration
% Edit compare_config.toml with your parameters

% 2. Generate datasets
compare_dataset('compare_config.toml')

% 3. Analyze results
rel_err_results()
```

### Advanced Workflow
```matlab
% Test system first
test_compare_system

% Generate with custom config
compare_dataset('my_custom_config.toml')

% Analyze with plots
rel_err_results('folder', 'my_output', 'plot', true)
```

## Advantages Over Old System

### Old System (`compare.m`)
- ❌ Parameters hardcoded in script
- ❌ Need to load JSON files manually
- ❌ No noise simulation
- ❌ Single model only
- ❌ Hard to compare different scenarios
- ❌ Limited analysis capabilities

### New System (`compare_dataset.m`)
- ✅ All parameters in TOML config
- ✅ Automatic config loading
- ✅ Built-in noise simulation
- ✅ Multi-model support (HN, Duan, GJR)
- ✅ Easy to run multiple scenarios
- ✅ Enhanced analysis with filtering and visualization

## Example Usage Scenarios

### Scenario 1: Basic Parameter Comparison
```matlab
% Use default config with actual calibrated values
compare_dataset('compare_config.toml')
rel_err_results()
```

### Scenario 2: Noise Sensitivity Study
```matlab
% Test with 0%, 1%, 3%, 5% noise
% Create 4 configs with different noise.level values
% Run each and compare results
```

### Scenario 3: Model Comparison
```matlab
% Create configs for HN, Duan, and GJR models
% Compare which model shows better calibration
```

### Scenario 4: Large-Scale Study
```matlab
% Increase contracts_per_day and num_days
% Generate large dataset for ML training
```

## Configuration Examples

### No Noise (Ideal Case)
```toml
[noise]
level = 0.0
```

### 2% Noise (Realistic)
```toml
[noise]
level = 0.02
seed = 42
```

### Large Dataset
```toml
[scaling]
contracts_per_day = 100
num_days = 50
batch_size = 200
```

### Different Model
```toml
[model]
type = "Duan"
```

## Dependencies

### Python
- Python 3.x
- `toml` package: `pip install toml`

### MATLAB Functions (Must be in path)
- `mcHN.m`, `mcDuan.m`, `mcGJR.m`
- `genhDelta.m`
- `TreeNodes_ht_HN.m`, `TreeNodes_ht_Duan.m`
- `TreeNodes_logSt_HN.m`, `TreeNodes_logSt_Duan.m`
- `Prob_Xt.m`
- `American.m`
- `impvol_improved.m`, `impVol_Duan.m`, `impVol_GJR.m`

## Output Files

### Generated by `compare_dataset.m`
- `{output_folder}/true_params.csv` - Dataset with true parameters
- `{output_folder}/initial_params.csv` - Dataset with initial parameters
- `{output_folder}/calibrated_params.csv` - Dataset with calibrated parameters
- `{output_folder}/assetprices_*.csv` - Asset price paths (optional)

### Generated by `rel_err_results.m`
- `{output_folder}/rel_err_results.csv` - Detailed results table
- `{output_folder}/rel_err_heatmaps.png` - Visualization (if plot=true)

## Testing

Run `test_compare_system.m` to verify:
1. Python 3 availability
2. TOML module installation
3. TOML reader script
4. Required MATLAB functions
5. Configuration file validity
6. Parameter values

## Migration from Old System

### Old Way
```matlab
% Edit compare.m directly to change parameters
% Manually load JSON files
% Run compare.m
% Run compare/rel_err_results.m
```

### New Way
```matlab
% Edit compare_config.toml (no code changes)
% Run compare_dataset('compare_config.toml')
% Run rel_err_results()
```

## Future Enhancements

Possible improvements:
1. Add support for more GARCH models
2. Parallel processing for faster generation
3. Interactive config editor GUI
4. Automatic parameter extraction from optimization results
5. More analysis metrics (e.g., VaR, Greeks)
6. Integration with neural network training pipeline

## Notes

- All three parameter sets use **identical randomness** for GARCH simulation
- Contract-level randomness (strikes/maturities) is also **synchronized**
- Noise is added **after** pricing if configured
- Results automatically **filter outliers** (sigma < 0.01 or V < 0.5)
- System is **backwards compatible** - old CSV files can still be analyzed with `rel_err_results()`

## Contact/Support

See `COMPARE_README.md` for detailed documentation and troubleshooting.
