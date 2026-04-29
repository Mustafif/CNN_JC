# Configuration-Driven Dataset Generation Guide

## Overview

The new `scaled_dataset_optimized_configurable.m` script provides a flexible, config-file-based approach to dataset generation for GARCH option pricing models. It combines the performance optimizations of `scaled_dataset_optimized.m` with configurable options and **noise injection capability**.

## Key Features

- **TOML Configuration Files**: Easy-to-read and modify configuration
- **Noise Injection**: Add realistic market noise to simulated option prices
- **Command-Line Overrides**: Override config values from the command line
- **Multiple Models**: Support for HN (Heston-Nandi), Duan, and GJR GARCH models
- **Performance Optimization**: Parallel processing, memory optimization, optional GPU support
- **Reproducibility**: Configurable random seeds

## Quick Start

### Basic Usage

```matlab
% Use default config (scaled_dataset_config.toml)
scaled_dataset_optimized_configurable()

% Use a specific config file
scaled_dataset_optimized_configurable('config', 'my_config.toml')

% Override noise level via command line
scaled_dataset_optimized_configurable('config', 'my_config.toml', 'noise_level', 0.05)

% Multiple overrides
scaled_dataset_optimized_configurable('config', 'my_config.toml', ...
    'noise_level', 0.03, ...
    'contracts_per_day', 100)
```

## Configuration File Format

The configuration uses TOML format. Here's the structure:

### `[model]` Section
Specifies the GARCH model to use:
```toml
[model]
type = "HN"  # "HN", "Duan", or "GJR"
```

### `[scaling]` Section
Controls dataset size:
```toml
[scaling]
contracts_per_day = 50      # Contracts per trading day
num_days = 20               # Number of trading days
batch_size = 100            # Batch size for processing
num_param_sets = 5          # Number of parameter sets to generate
```

### `[market]` Section
Market and model parameters:
```toml
[market]
N = 1260                    # Time steps in GARCH simulation
r = 0.0001984              # Risk-free rate (daily, 0.05/252)
S0 = 100.0                 # Initial stock price
moneyness_range = [0.8, 1.2]  # Option moneyness (S0/K range)
maturity_days = [30, 60, 90, 180, 360]  # Maturity in days
```

### `[willow_tree]` Section
Willow tree discretization parameters:
```toml
[willow_tree]
m_h = 6        # Nodes for volatility tree
gamma_h = 0.6  # Volatility tree parameter
m_x = 30       # Nodes for log-stock tree
gamma_x = 0.8  # Log-stock tree parameter
```

### `[performance]` Section
Performance and resource settings:
```toml
[performance]
use_gpu = false       # Enable GPU acceleration
num_workers = 0       # Parallel workers (0 = auto-detect)
optimize_memory = false  # Memory vs speed tradeoff
```

### `[noise]` Section
**Noise injection parameters** (multiplicative noise on prices):
```toml
[noise]
level = 0.05   # Noise std dev as % of price (0.05 = 5% noise)
seed = 42      # Random seed for reproducibility (-1 = random)
```

The noise is applied as: `V_noisy = V_true * (1 + N(0, noise_level))`

where N(0, σ) is a normal distribution with mean 0 and standard deviation σ.

### `[output]` Section
Output configuration:
```toml
[output]
folder = ""              # Output folder ("" = auto timestamp-based)
filename = ""            # Output file basename ("" = dataset_<model>)
save_asset_prices = false  # Whether to save price paths
save_logs = true           # Whether to save detailed logs
```

## Example Configurations

### Example 1: HN Model with 5% Noise
```toml
[model]
type = "HN"

[scaling]
contracts_per_day = 50
num_days = 20
batch_size = 100
num_param_sets = 5

[noise]
level = 0.05
seed = 42

# ... other sections use defaults
```

### Example 2: Duan Model, No Noise, Custom Output
```toml
[model]
type = "Duan"

[scaling]
contracts_per_day = 100
num_days = 30
batch_size = 150
num_param_sets = 10

[noise]
level = 0.0    # No noise
seed = 42

[output]
folder = "duan_large_dataset"
filename = "duan_100x30"
```

### Example 3: GJR Model with 3% Noise and GPU
```toml
[model]
type = "GJR"

[scaling]
contracts_per_day = 75
num_days = 25
batch_size = 125
num_param_sets = 8

[performance]
use_gpu = true
num_workers = 4
optimize_memory = false

[noise]
level = 0.03
seed = 123
```

## Command-Line Parameter Overrides

Any configuration value can be overridden from the command line:

```matlab
% Override single parameter
scaled_dataset_optimized_configurable('config', 'my_config.toml', ...
    'noise_level', 0.02)

% Override multiple parameters
scaled_dataset_optimized_configurable('config', 'my_config.toml', ...
    'model_type', 'GJR', ...
    'noise_level', 0.025, ...
    'contracts_per_day', 50, ...
    'num_days', 15, ...
    'output_folder', 'custom_output_dir')
```

**Available Override Parameters:**
- `config` - Path to config file
- `model_type` - Model: 'HN', 'Duan', 'GJR'
- `noise_level` - Noise std dev (0.0-1.0)
- `contracts_per_day` - Contracts per day
- `num_days` - Number of trading days
- `output_folder` - Output folder path

## Noise Injection Details

### How Noise is Applied

1. **Calculate clean option price** from model (V_clean)
2. **Generate noise** from normal distribution N(0, σ) where σ = noise_level
3. **Apply multiplicative noise**: V_noisy = V_clean × (1 + noise)
4. **Calculate implied vol** from noisy price

This approach simulates market friction, bid-ask spreads, and model errors.

### Noise Levels

- **0.0** - No noise (perfect model)
- **0.01** - 1% noise (very clean market)
- **0.05** - 5% noise (typical market)
- **0.10** - 10% noise (noisy/illiquid market)

### Reproducibility

Set `seed` to a specific value (e.g., 42) for reproducible noise. Set `seed = -1` for random seeds.

## Output

The script generates:

1. **Dataset CSV**: Contains features and option prices
   - Columns: S0, m, r, T, corp, alpha, beta, omega, gamma, lambda, sigma, V
   - One row per option (call and put combined)

2. **Output Folder**: Named with pattern `generation_<model>_<timestamp>_cached/`
   - Or custom name if specified in config

3. **Summary Statistics**: Printed to console
   - Total contracts generated
   - Valid contracts after cleaning
   - Processing time and throughput
   - Whether noise was applied

## Performance Tips

### For Speed
- Reduce `num_param_sets` or `num_days` in config
- Set `num_workers` to match your CPU cores
- Use `batch_size` of 100-200

### For Memory
- Set `optimize_memory = true` in config
- Reduce `contracts_per_day`
- Use smaller `batch_size` (50-100)

### For GPU
- Set `use_gpu = true` if NVIDIA GPU available
- Monitor memory usage: GPU may require smaller batches

## Troubleshooting

### "Config file not found"
- Check file path is correct
- Use full path if not in current directory
- Default: `scaled_dataset_config.toml`

### "Invalid model type"
- Use exactly: "HN", "Duan", or "GJR" (case-insensitive)
- Check config `[model]` section

### "Could not start parallel pool"
- System may lack parallel computing toolbox
- Script falls back to serial processing automatically
- Set `num_workers = 0` in config (auto-detect)

### Low throughput
- Smaller dataset → smaller overhead per contract
- Increase `contracts_per_day` and `num_days`
- Check system load and available cores

## Comparison with Original

| Feature | Original | Configurable |
|---------|----------|--------------|
| Configuration | Command-line only | TOML file + CLI overrides |
| Noise support | ❌ No | ✅ Yes |
| Flexibility | Limited | High |
| Reproducibility | Command-line args | Config file |
| Batch processing | ✅ Yes | ✅ Yes |
| Parallel processing | ✅ Yes | ✅ Yes |

## Example Workflow

```matlab
% 1. Copy template and customize
% Copy scaled_dataset_config.toml to my_study_config.toml
% Edit my_study_config.toml with your parameters

% 2. Generate dataset with noise
scaled_dataset_optimized_configurable('config', 'my_study_config.toml')

% 3. Override for sensitivity analysis
% Same config but with different noise levels
scaled_dataset_optimized_configurable('config', 'my_study_config.toml', 'noise_level', 0.01)
scaled_dataset_optimized_configurable('config', 'my_study_config.toml', 'noise_level', 0.05)
scaled_dataset_optimized_configurable('config', 'my_study_config.toml', 'noise_level', 0.10)

% 4. Compare results across noise levels
% Load and analyze output CSVs from each run
```

## References

- **GARCH Models**: Heston-Nandi, Duan, GJR-GARCH variants
- **Willow Trees**: Discretization method for GARCH models
- **Implied Volatility**: Calculated from option prices using iterative solver
- **Noise Model**: Multiplicative log-normal noise

## Files

- `scaled_dataset_optimized_configurable.m` - Main script
- `scaled_dataset_config.toml` - Default config template
- `CONFIG_DRIVEN_DATAGEN_GUIDE.md` - This guide
