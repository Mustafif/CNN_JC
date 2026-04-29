# Configuration-Driven Dataset Generation - Complete Index

## What Was Done

Your `scaled_dataset_optimized.m` has been refactored into a **configuration-driven system** with **noise injection support**, similar to `compare_unified.m`. This allows you to:

1. **Use TOML config files** instead of command-line arguments
2. **Add realistic noise** to simulated option prices
3. **Override any parameter** from the command line
4. **Maintain reproducibility** with seed control
5. **Use preset configurations** for common scenarios

## New Files Created

### Core Implementation

| File | Purpose | Size |
|------|---------|------|
| `scaled_dataset_optimized_configurable.m` | Main refactored script with noise support | 21KB |
| `scaled_dataset_config.toml` | Default configuration template | 1.7KB |

### Preset Configurations

| File | Use Case | Model | Noise |
|------|----------|-------|-------|
| `scaled_dataset_config_hn_clean.toml` | Clean model data, testing accuracy | HN | 0% |
| `scaled_dataset_config_duan_3pct.toml` | Typical market conditions | Duan | 3% |
| `scaled_dataset_config_gjr_5pct.toml` | Market friction, NN training | GJR | 5% |

### Documentation

| File | Content |
|------|---------|
| `CONFIG_DRIVEN_DATAGEN_GUIDE.md` | Comprehensive user guide (8.8KB) |
| `REFACTORING_SUMMARY.md` | Technical details and improvements (7.7KB) |
| `usage_examples_configurable.m` | 16 practical examples (6.0KB) |
| `CONFIGURABLE_DATAGEN_INDEX.md` | This file |

## Quick Start

### 1. Basic Usage
```matlab
% Use default config
scaled_dataset_optimized_configurable()

% Use a preset
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config_hn_clean.toml')
```

### 2. Add Noise
```matlab
% Override noise level
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config.toml', ...
    'noise_level', 0.05)
```

### 3. Multiple Runs
```matlab
% Sensitivity analysis
for noise = [0.01, 0.03, 0.05, 0.10]
    scaled_dataset_optimized_configurable('config', 'scaled_dataset_config.toml', ...
        'noise_level', noise)
end
```

## Key Features

### ✅ Configuration Files
- TOML format (human-readable)
- Full control over all parameters
- Comments explaining each option
- Easy to version control

### ✅ Noise Injection (NEW)
- **Multiplicative model**: `V_noisy = V_true × (1 + N(0, σ))`
- Configurable noise level (0-100%+)
- Reproducible with seed control
- Simulates market friction and bid-ask spreads

### ✅ Command-Line Overrides
```matlab
scaled_dataset_optimized_configurable('config', 'base.toml', ...
    'model_type', 'Duan', ...           % Override model
    'noise_level', 0.03, ...             % Override noise
    'contracts_per_day', 75, ...         % Override scaling
    'num_days', 20, ...                  % Override days
    'output_folder', 'custom_output')    % Override output
```

### ✅ Performance Features
- All optimizations from original maintained
- Parallel processing
- Optional GPU support
- Memory optimization option

### ✅ Reproducibility
- Config files document each run
- Fixed random seeds for reproducibility
- Version control friendly

## Configuration Structure

### Example: Basic TOML Config
```toml
[model]
type = "HN"

[scaling]
contracts_per_day = 50
num_days = 20
num_param_sets = 5

[noise]
level = 0.05      # 5% noise
seed = 42         # Reproducible

[output]
folder = ""       # Auto timestamp
filename = ""     # Auto name
```

### Available Sections
- `[model]` - Model selection
- `[scaling]` - Dataset size parameters
- `[market]` - Market and model parameters
- `[willow_tree]` - Tree discretization
- `[performance]` - Performance optimization
- `[noise]` - **NEW** Noise injection
- `[output]` - Output configuration

## Noise Levels Guide

| Level | Interpretation |
|-------|-----------------|
| 0.00 | Perfect model data |
| 0.01 | 1% noise - very clean |
| 0.05 | 5% noise - typical market |
| 0.10 | 10% noise - noisy market |
| 0.20+ | 20%+ noise - highly noisy |

## Common Use Cases

### Use Case 1: Test Model Accuracy
```matlab
% Clean data, no noise
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config_hn_clean.toml')
```

### Use Case 2: Train Neural Network
```matlab
% Larger dataset with realistic noise
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config_gjr_5pct.toml')
```

### Use Case 3: Calibration Robustness
```matlab
% Duan model with 3% noise
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config_duan_3pct.toml')
```

### Use Case 4: Sensitivity Analysis
```matlab
% Vary noise level
for noise = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
    scaled_dataset_optimized_configurable('config', 'base.toml', ...
        'noise_level', noise)
end
```

## Output

Each run generates:
- **CSV dataset** with columns: S0, m, r, T, corp, alpha, beta, omega, gamma, lambda, sigma, V
- **Timestamp folder** with auto-generated name (or custom)
- **Console summary** with performance metrics

### Example Output
```
PERFORMANCE SUMMARY
Total contracts generated: 20
Valid contracts: 12
Total processing time: 0.16 minutes
Throughput: 1.3 contracts/second
Output file: generation_hn_20260403_111403_cached/dataset_hn.csv
Noise applied: YES
```

## File Locations

All new files are in: `/home/mustafif/CNN_JC/data_gen/`

```
data_gen/
├── scaled_dataset_optimized_configurable.m      [Main script]
├── scaled_dataset_config.toml                   [Default config]
├── scaled_dataset_config_hn_clean.toml         [Preset: HN, no noise]
├── scaled_dataset_config_duan_3pct.toml        [Preset: Duan, 3% noise]
├── scaled_dataset_config_gjr_5pct.toml         [Preset: GJR, 5% noise]
├── CONFIG_DRIVEN_DATAGEN_GUIDE.md              [Full guide]
├── REFACTORING_SUMMARY.md                      [Technical details]
├── usage_examples_configurable.m               [16 examples]
└── CONFIGURABLE_DATAGEN_INDEX.md               [This file]
```

## Documentation Map

### For Quick Start
→ Read `usage_examples_configurable.m` (16 practical examples)

### For Complete Understanding
→ Read `CONFIG_DRIVEN_DATAGEN_GUIDE.md` (full user guide)

### For Technical Details
→ Read `REFACTORING_SUMMARY.md` (technical improvements)

### For Implementation
→ See `scaled_dataset_optimized_configurable.m` (annotated source)

## Examples of Config Customization

### Example 1: Larger Dataset
Edit `scaled_dataset_config.toml`:
```toml
[scaling]
contracts_per_day = 200    # Was 50
num_days = 50              # Was 20
num_param_sets = 20        # Was 5
```

### Example 2: Different Models
```matlab
% HN
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config.toml', ...
    'model_type', 'HN')

% Duan
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config.toml', ...
    'model_type', 'Duan')

% GJR
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config.toml', ...
    'model_type', 'GJR')
```

### Example 3: Custom Noise Study
```matlab
% Create custom config for your study
output_folder = 'noise_sensitivity_study';
noise_levels = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05];

for noise = noise_levels
    folder_name = sprintf('%s/noise_%.0f_percent', output_folder, noise*100);
    scaled_dataset_optimized_configurable('config', 'base.toml', ...
        'noise_level', noise, ...
        'output_folder', folder_name);
end
```

## Troubleshooting

### Q: Config file not found
**A**: Use full path or check file is in current directory

### Q: How do I change model?
**A**: Use `'model_type', 'Duan'` parameter or edit config file

### Q: How do I disable noise?
**A**: Set `noise_level = 0.0` in config or `'noise_level', 0.0` in call

### Q: How do I make it reproducible?
**A**: Set same seed in config: `seed = 42` (not -1)

### Q: Can I use GPU?
**A**: Set in config: `use_gpu = true` (needs NVIDIA GPU)

## Original vs New

| Feature | Original | New |
|---------|----------|-----|
| Configuration | CLI only | TOML + CLI overrides |
| Noise support | ❌ No | ✅ Yes |
| Preset configs | ❌ No | ✅ 3 presets |
| Reproducibility | Difficult | ✅ Config files |
| Documentation | Minimal | ✅ Comprehensive |
| Examples | None | ✅ 16 examples |

## Next Steps

1. **Try it out**:
   ```matlab
   scaled_dataset_optimized_configurable()
   ```

2. **Explore presets**:
   ```matlab
   scaled_dataset_optimized_configurable('config', 'scaled_dataset_config_hn_clean.toml')
   ```

3. **Run examples**:
   Open `usage_examples_configurable.m` to see 16 different use cases

4. **Customize**:
   Copy and edit a TOML config for your specific needs

5. **Automate**:
   Create batch scripts using examples from `usage_examples_configurable.m`

## References

- **Main Guide**: `CONFIG_DRIVEN_DATAGEN_GUIDE.md`
- **Technical Details**: `REFACTORING_SUMMARY.md`
- **Working Examples**: `usage_examples_configurable.m`
- **Implementation**: `scaled_dataset_optimized_configurable.m`

---

**Last Updated**: April 3, 2026
**Status**: ✅ Complete and Tested
**Backward Compatible**: ✅ Yes (original script unchanged)
