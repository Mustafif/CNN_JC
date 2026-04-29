# Centralized Parameter Comparison System

## Overview

This system provides a centralized framework for comparing GARCH model parameters (true, initial, and calibrated) using option pricing datasets. It's similar to `scaled_dataset.m` but specifically designed for parameter comparison with consistent randomness across all parameter sets.

## Files

1. **`compare_config.toml`** - Configuration file with all parameters
2. **`compare_dataset.m`** - Main MATLAB function to generate comparison datasets
3. **`rel_err_results.m`** - Analysis function to compute relative errors
4. **`read_toml_config.py`** - Python helper to parse TOML files

## Quick Start

### 1. Configure Parameters

Edit `compare_config.toml` to set your parameters:

```toml
[model]
type = "HN"  # Options: "HN", "Duan", "GJR"

[parameters.true]
omega = 1.0e-6
alpha = 1.33e-6
beta = 0.8
gamma = 5.0
lambda = 0.2

[parameters.initial]
omega = 2.0e-6
alpha = 2.0e-6
beta = 0.75
gamma = 4.5
lambda = 0.15

[parameters.calibrated]
omega = 1.05e-6
alpha = 1.35e-6
beta = 0.79
gamma = 4.98
lambda = 0.19

[noise]
level = 0.0   # 0.0 = no noise, 0.01 = 1% noise, 0.05 = 5% noise
seed = 42
```

### 2. Generate Datasets

In MATLAB:
```matlab
% Use default config file (compare_config.toml)
compare_dataset('compare_config.toml')

% Or with custom config
compare_dataset('my_custom_config.toml')
```

This generates three datasets in the `compare/` folder:
- `true_params.csv` - Prices with true parameters
- `initial_params.csv` - Prices with initial parameters  
- `calibrated_params.csv` - Prices with calibrated parameters

### 3. Analyze Results

```matlab
% Basic analysis
rel_err_results()

% With plots
rel_err_results('plot', true)

% Custom folder
rel_err_results('folder', 'my_comparison')
```

## Configuration File Options

### Model Configuration
```toml
[model]
type = "HN"  # Model type: "HN" (Heston-Nandi), "Duan", or "GJR"
```

### Scaling Parameters
```toml
[scaling]
contracts_per_day = 10  # Number of option contracts per day
num_days = 10           # Number of trading days
batch_size = 100        # Batch size for processing
```

### Market Parameters
```toml
[market]
N = 1024                      # Number of time steps
r = 0.0001984                 # Risk-free rate (daily, e.g., 0.05/252)
S0 = 100.0                    # Initial stock price
moneyness_range = [0.8, 1.2]  # Moneyness range [min, max]
maturity_range = [0.1, 1.0]   # Maturity range in years [min, max]
```

### Willow Tree Parameters
```toml
[willow_tree]
m_h = 6
m_ht = 6
m_x = 30
gamma_h = 0.6
gamma_x = 0.8
```

### Solver Parameters
```toml
[solver]
itmax = 100   # Maximum iterations
tol = 1e-6    # Tolerance
```

### Noise Configuration
```toml
[noise]
level = 0.0  # Noise as fraction of option price (0.01 = 1%, 0.05 = 5%)
seed = 42    # Random seed (-1 for random)
```

### GARCH Parameters

Each parameter set (true, initial, calibrated) has:
```toml
[parameters.true]
omega = 1.0e-6
alpha = 1.33e-6
beta = 0.8
gamma = 5.0
lambda = 0.2
```

### Output Configuration
```toml
[output]
folder = "compare"            # Output folder
save_asset_prices = true      # Save asset price paths
save_logs = true              # Save detailed logs
```

## Examples

### Example 1: Basic Comparison (No Noise)

```toml
[model]
type = "HN"

[noise]
level = 0.0
```

```matlab
compare_dataset('compare_config.toml')
rel_err_results()
```

### Example 2: Comparison with 2% Noise

```toml
[noise]
level = 0.02  # 2% noise
seed = 42
```

```matlab
compare_dataset('compare_config.toml')
rel_err_results('plot', true)
```

### Example 3: Duan GARCH Model

```toml
[model]
type = "Duan"

[parameters.true]
omega = 2.0e-6
alpha = 1.5e-6
beta = 0.85
lambda = 0.25
# Note: Duan model doesn't use gamma
```

```matlab
compare_dataset('duan_config.toml')
rel_err_results('folder', 'compare')
```

### Example 4: Large Dataset

```toml
[scaling]
contracts_per_day = 100
num_days = 50
batch_size = 200
```

```matlab
compare_dataset('large_config.toml')
```

## Relative Error Analysis

The `rel_err_results()` function computes mean relative errors grouped by:

**Maturity Categories:**
- Short: < 30 days
- Medium: 30-180 days  
- Long: > 180 days

**Moneyness Categories:**
- Deep OTM: < 0.8
- OTM: 0.8-1.0
- ITM: 1.0-1.2
- Deep ITM: > 1.2

### Output Tables

1. **Initial vs True**: Mean relative error using initial parameters
2. **Calibrated vs True**: Mean relative error using calibrated parameters
3. **Improvement**: Percentage improvement from calibration

### Saved Files

- `rel_err_results.csv` - Detailed results table
- `rel_err_heatmaps.png` - Visualization (if `plot=true`)

## Advantages Over compare.m

1. **Configuration File**: All parameters in one TOML file (easy to version control)
2. **Centralized**: Single function like `scaled_dataset.m`
3. **Noise Support**: Built-in option to add realistic noise
4. **Model Support**: Easy to switch between HN, Duan, GJR models
5. **Reproducibility**: Fixed randomness ensures fair comparison
6. **Better Analysis**: Enhanced relative error computation with filtering

## Dependencies

- MATLAB with required functions:
  - `mcHN.m`, `mcDuan.m`, `mcGJR.m`
  - `TreeNodes_*.m`, `Prob_Xt.m`, `American.m`
  - `impvol_improved.m`, `impVol_Duan.m`, `impVol_GJR.m`
- Python 3 with `toml` package:
  ```bash
  pip install toml
  ```

## Troubleshooting

**Error: "Failed to read TOML config file"**
- Ensure `read_toml_config.py` is in the same folder
- Check Python 3 is installed: `python3 --version`
- Install toml package: `pip install toml`

**Error: "Unsupported model type"**
- Check `[model] type` is one of: "HN", "Duan", "GJR"
- Model names are case-insensitive

**Memory Issues**
- Reduce `contracts_per_day` or `num_days`
- Increase `batch_size` for better memory management

**Tree Construction Errors**
- Check GARCH parameters are valid (e.g., beta < 1)
- Ensure all required tree functions are in MATLAB path

## Notes

- All three parameter sets use the **same randomness** for fair comparison
- Contract-level randomness (strike/maturity) is also consistent across sets
- Noise (if enabled) is applied after pricing and uses separate random seed
- Results are filtered to remove outliers (sigma < 0.01 or V < 0.5)
