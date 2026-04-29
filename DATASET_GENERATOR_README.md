# Centralized GARCH Dataset Generator

## Overview

The `scaled_dataset.m` function provides a unified interface for generating option pricing datasets using three different GARCH models:

- **Heston-Nandi (HN) GARCH**
- **Duan GARCH**
- **GJR-GARCH**

## Location

```
/hdd/CNN_JC/scaled_dataset.m
```

## Usage

### Basic Syntax

```matlab
scaled_dataset(model_type)
scaled_dataset(model_type, 'Parameter', value, ...)
```

### Parameters

#### Required
- `model_type` - String specifying the GARCH model: `'HN'`, `'Duan'`, or `'GJR'`

#### Optional (Name-Value Pairs)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `ContractsPerDay` | Number of option contracts per trading day | `50` |
| `NumDays` | Number of trading days to simulate | `20` |
| `BatchSize` | Batch size for processing (memory management) | `100` |
| `NumParamSets` | Number of different GARCH parameter sets | `5` |
| `OutputFile` | Output CSV filename | `'dataset_<model>.csv'` |
| `MoneynessRange` | [min, max] moneyness range | `[0.8, 1.2]` |
| `MaturityDays` | Array of discrete maturity days | `[30, 60, 90, 180, 360]` |

## Examples

### Example 1: Basic HN-GARCH Dataset
```matlab
% Generate dataset with default parameters
scaled_dataset('HN')
% Output folder: generation_hn_20260331_012000/
%   - dataset_hn.csv (cleaned dataset)
%   - garch_params.csv (5 parameter sets)
%   - asset_prices.csv (price paths)
%   - metadata.txt (generation info)
```

### Example 2: Large Duan GARCH Dataset
```matlab
% Generate 30,000 contracts (100 per day × 30 days × 5 param sets)
scaled_dataset('Duan', 'ContractsPerDay', 100, 'NumDays', 30, 'NumParamSets', 5)
% Output folder: generation_duan_<timestamp>/
%   Contains cleaned dataset + params + prices + metadata
```

### Example 3: Custom GJR-GARCH Dataset
```matlab
% Custom parameters with specific maturities
scaled_dataset('GJR', ...
    'ContractsPerDay', 200, ...
    'NumDays', 50, ...
    'NumParamSets', 10, ...
    'OutputFile', 'my_gjr_options.csv', ...
    'MoneynessRange', [0.7, 1.3], ...
    'MaturityDays', [7, 14, 30, 60, 90, 180, 360])
% Includes weekly, bi-weekly, monthly, and longer maturities
```

### Example 4: Batch Processing for Memory Management
```matlab
% Large dataset with smaller batch size for systems with limited RAM
scaled_dataset('HN', ...
    'ContractsPerDay', 500, ...
    'NumDays', 100, ...
    'BatchSize', 50)  % Process 50 contracts at a time
```

## Output Structure

Each generation creates a timestamped folder containing:

```
generation_<model>_<timestamp>/
├── dataset_<model>.csv      # Main cleaned dataset
├── garch_params.csv          # GARCH parameters used
├── asset_prices.csv          # Generated asset price paths
└── metadata.txt              # Generation metadata and statistics
```

### Data Cleaning

The dataset is automatically cleaned before saving:
- **Removes** contracts with option value `V < 0.5` (too cheap to be meaningful)
- **Removes** contracts with maturity `T = 1/252` (exactly 1 day, unrealistic)
- **Removes** contracts with NaN or Inf values

Typical cleaning removes 5-15% of generated contracts.

## Dataset Format

The main CSV file (`dataset_<model>.csv`) contains the following columns:

| Column | Name | Description |
|--------|------|-------------|
| 1 | `S0` | Current spot price |
| 2 | `m` | Moneyness (K/S) |
| 3 | `r` | Risk-free rate (daily) |
| 4 | `T` | Time to maturity (years) |
| 5 | `corp` | Call/Put flag (1=Call, -1=Put) |
| 6 | `alpha` | GARCH alpha parameter |
| 7 | `beta` | GARCH beta parameter |
| 8 | `omega` | GARCH omega parameter |
| 9 | `gamma`/`theta` | Model-specific parameter* |
| 10 | `lambda` | GARCH lambda parameter |
| 11 | `sigma` | Implied volatility |
| 12 | `V` | Option price |

\* **Column 9 interpretation:**
- HN model: `gamma` (leverage parameter)
- Duan model: `theta` (risk premium adjustment)
- GJR model: `gamma` (always 0 in current implementation)

## Model-Specific Details

### Heston-Nandi GARCH
- **Parameter ranges:**
  - omega: [1e-7, 1e-6]
  - alpha: [1.15e-6, 1.36e-6]
  - beta: [0.75, 0.85]
  - gamma: [50, 150]
  - lambda: [0.4, 0.6]
- **Stationarity conditions:** 
  - `beta + alpha * gamma^2 < 1`
  - `alpha + beta < 1`
- **Files used:** `data_gen/genHNGarchParams.m`, `data_gen/mcHN.m`, `data_gen/impVol_HN.m`

### Duan GARCH
- **Parameter ranges:**
  - omega: [1e-7, 1e-6]
  - alpha: [1.15e-6, 1.36e-6]
  - beta: [0.75, 0.85]
  - theta: [0.25, 0.5]
  - lambda: [0.4, 0.6]
- **Stationarity conditions:**
  - `beta + alpha * theta^2 < 1`
  - `alpha + beta < 1`
- **Files used:** `data_gen/Duan_Garch/genDuanGarchParams.m`, `data_gen/Duan_Garch/mcDuan.m`, `data_gen/Duan_Garch/impVol_Duan.m`

### GJR-GARCH
- **Parameter ranges:**
  - omega: [1e-7, 1e-6]
  - alpha: [0.01, 0.15]
  - beta: [0.75, 0.85]
  - lambda: [0.01, 0.15] (leverage effect)
- **Stationarity condition:**
  - `beta + alpha + lambda/2 < 1`
- **Files used:** `data_gen/GJR_Garch/genGJRGarchParams.m`, `data_gen/GJR_Garch/mcGJR.m`, `data_gen/GJR_Garch/impVol_GJR.m`

## Technical Details

### Willow Tree Parameters (Fixed)
- `m_h = 6` - Number of possible values for h_t
- `m_x = 30` - Number of possible values for X_t
- `gamma_h = 0.6` - Willow tree parameter for h
- `gamma_x = 0.8` - Willow tree parameter for X

### Solver Parameters
- `N = 1260` - Number of time steps
- `tol = 1e-4` - Convergence tolerance
- `itmax = 60` - Maximum iterations for implied volatility solver

### Market Parameters
- `r = 0.05/252` - Daily risk-free rate (5% annual)
- `S0 = 100` - Initial stock price

## Performance Considerations

1. **Memory Usage:** For large datasets, reduce `BatchSize` to manage memory consumption
2. **Processing Time:** 
   - ~0.5-2 seconds per contract depending on model complexity
   - GJR-GARCH is typically fastest
   - Duan GARCH has more complex moment calculations
3. **Parallelization:** Currently sequential; could be parallelized for large-scale generation

## Troubleshooting

### Common Issues

1. **"Function not found" error**
   - Ensure you're running from the root `/hdd/CNN_JC/` directory
   - The function automatically adds necessary paths

2. **Pricing errors**
   - Some contracts may fail to price (typically <5%)
   - Failed contracts are replaced with placeholder values (sigma=0.2, V=0)
   - Check the summary statistics at the end of generation

3. **Memory issues**
   - Reduce `BatchSize` parameter
   - Reduce `ContractsPerDay` or `NumDays`
   - Process in multiple smaller runs

## Dependencies

The function requires the following files to be present:

### Common Files (in `data_gen/`)
- `American.m` - American option pricing
- `impvol.m` - Implied volatility solver

### HN-GARCH Specific
- `genHNGarchParams.m`
- `mcHN.m`
- `impVol_HN.m`
- `TreeNodes_ht_HN.m`
- `TreeNodes_logSt_HN.m`
- `Prob_Xt.m`
- `genhDelta.m`

### Duan GARCH Specific (in `data_gen/Duan_Garch/`)
- `genDuanGarchParams.m`
- `mcDuan.m`
- `impVol_Duan.m`
- `TreeNodes_ht_D.m`
- `TreeNodes_logSt_D.m`
- `Probility_Xt2.m`

### GJR-GARCH Specific (in `data_gen/GJR_Garch/`)
- `genGJRGarchParams.m`
- `mcGJR.m`
- `impVol_GJR.m`
- `TreeNodes_ht_GJR.m`
- `TreeNodes_logSt_GJR.m`
- `Probility_Xt2.m`

## Version History

- **v1.0** (2026-03-31): Initial centralized implementation
  - Unified interface for HN, Duan, and GJR-GARCH models
  - Flexible parameter configuration
  - Batch processing for memory management
  - Comprehensive error handling and statistics

## Author

Generated from existing model-specific implementations in `/hdd/CNN_JC/data_gen/`
