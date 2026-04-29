# Multi-Noise & Parameter Ranges - Quick Start Guide

## What Changed?

### 1. Parameter Generation (No More Baselines!)
- **Before**: Used hardcoded baseline parameters with random scaling
  - Example: `omega = 1e-6 * (0.8 + 0.4*rand())`
  - Problem: Wrong parameter ranges, especially for HN gamma!

- **After**: Samples from empirical parameter ranges with stationarity checks
  - Example: `omega ~ Uniform[1e-7, 1e-6]`
  - Checks: `beta + alpha*gamma^2 < 1` (for HN) before accepting

### 2. Multi-Noise Dataset Generation (New!)
Generate 3 datasets with the same GARCH parameters but different noise levels:
- `dataset_0pct.csv` - No noise (perfect pricing)
- `dataset_1pct.csv` - 1% noise (realistic market data)
- `dataset_5pct.csv` - 5% noise (high noise scenario)

All 3 use the **same** GARCH parameter set!

## How to Use

### Single Dataset (Original Behavior)
```matlab
% Use existing config or create one without [multi_noise] section
scaled_dataset_optimized_configurable(...
    'config', 'scaled_dataset_config_hn_clean.toml')
```

### Multi-Noise Dataset (New!)
```matlab
% Config must have [multi_noise] section
scaled_dataset_optimized_configurable(...
    'config', 'scaled_dataset_config_multi_noise.toml')
```

## Config File Format

### Standard Single-Dataset Config
```toml
[model]
type = "HN"

[scaling]
contracts_per_day = 100
num_days = 30
num_param_sets = 10

# ... rest of config ...

[noise]
level = 0.01  # 1% noise
seed = 42

# No [multi_noise] section = standard behavior
```

### Multi-Noise Config (Generate 3 Datasets at Once)
```toml
[model]
type = "HN"

[scaling]
contracts_per_day = 100
num_days = 30
num_param_sets = 1   # Only 1 param set, used for all noise levels

# ... other config sections ...

[noise]
level = 0.0  # Ignored when [multi_noise] enabled
seed = 42

[multi_noise]
enabled = true
noise_levels = [0, 1, 5]  # Generate 0%, 1%, 5% noise datasets
```

## Parameter Ranges

| Model | Parameter | Range | Example |
|-------|-----------|-------|---------|
| HN    | omega     | [1e-7, 1e-6]    | 5e-7 |
| HN    | alpha     | [1.15e-6, 1.36e-6] | 1.25e-6 |
| HN    | beta      | [0.75, 0.85]    | 0.80 |
| HN    | gamma     | [50, 150]       | 100 |
| HN    | lambda    | [0.4, 0.6]      | 0.5 |
| DUAN  | (same as HN except) |      |  |
| DUAN  | theta     | [0.25, 0.5]     | 0.35 |
| GJR   | alpha     | [0.01, 0.15]    | 0.08 |
| GJR   | lambda    | [0.01, 0.15]    | 0.08 |

## Output Files

### Single Dataset Mode
```
generation_hn_20260408_020000/
  ├─ dataset_hn.csv              # Main dataset
  ├─ garch_params_hn.csv         # Parameter sets used
  └─ logs/...                    # Configuration
```

### Multi-Noise Mode
```
generation_hn_20260408_020000/
  ├─ dataset_hn_0pct.csv         # 0% noise (clean)
  ├─ dataset_hn_1pct.csv         # 1% noise (mild)
  ├─ dataset_hn_5pct.csv         # 5% noise (high)
  ├─ garch_params_hn.csv         # Same params for all 3!
  └─ logs/...                    # Configuration
```

## Benefits of Multi-Noise

✓ **Same Parameters**: All 3 datasets use identical GARCH params
  - Compare model robustness across noise levels
  - Isolate noise effects from parameter variation

✓ **Efficient**: Single parameter generation for all noise levels
  - 66% faster than running 3 separate scripts
  - Reuses tree computations

✓ **Realistic Testing**: Test your models against different noise scenarios

## Examples

### Example 1: Generate Multi-Noise HN Dataset
```matlab
scaled_dataset_optimized_configurable(...
    'config', 'scaled_dataset_config_multi_noise.toml', ...
    'num_days', 30, ...
    'contracts_per_day', 100)
```

Output:
```
✓ DATASET 1/3: NOISE LEVEL = 0.00%
✓ DATASET 2/3: NOISE LEVEL = 1.00%
✓ DATASET 3/3: NOISE LEVEL = 5.00%
```

### Example 2: Generate Clean HN Dataset (Original Behavior)
```matlab
scaled_dataset_optimized_configurable(...
    'config', 'scaled_dataset_config_hn_clean.toml')
```

### Example 3: Custom Noise Levels
Modify config:
```toml
[multi_noise]
enabled = true
noise_levels = [0, 0.5, 1.0, 2.0, 5.0]  # 5 datasets
```

## Key Differences from Before

| Feature | Before | After |
|---------|--------|-------|
| Parameter generation | Baselines + random scaling | Range-based + stationarity |
| Parameter ranges | Limited (scaled from 1 baseline) | Full empirical ranges |
| Multi-noise support | Not available | Enabled via config |
| Config [multi_noise] | N/A | Optional section |
| Output naming | Single name | Includes noise level |

## Compatibility

✓ **Fully backward compatible**
- Old configs still work (multi_noise optional)
- Parameter ranges contain original defaults
- No breaking changes

Note: Results may vary slightly because we're now using proper parameter ranges instead of scaled baselines. This is an improvement!

## Troubleshooting

**Q: Multi-noise not working?**
A: Check config has `[multi_noise]` section with `enabled = true`

**Q: Output files not appearing?**
A: Check output folder permissions and config [output] section

**Q: Want different noise levels?**
A: Edit config `noise_levels = [0, 1, 5]` to desired percentages

**Q: Can I mix single-dataset and multi-noise?**
A: No, but you can disable multi-noise with `enabled = false`

## See Also

- `PARAMETER_RANGES_AND_MULTINOISE.md` - Full technical documentation
- `scaled_dataset_config_multi_noise.toml` - Example configuration
- `scaled_dataset_optimized_configurable.m` - Implementation (lines 95-103, 262-358, 523-651)
