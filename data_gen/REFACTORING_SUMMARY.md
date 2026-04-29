# Refactoring Summary: Configuration-Driven Dataset Generation

## Changes Made

### Overview
Refactored `scaled_dataset_optimized.m` into a new configuration-driven system similar to `compare_unified.m`, adding **noise injection capability** to the data generation pipeline.

## New Files Created

### 1. **scaled_dataset_optimized_configurable.m** (Main Script)
- Configuration-file-based approach using TOML files
- Full feature parity with `scaled_dataset_optimized.m`
- **NEW**: Noise injection support (multiplicative noise on prices)
- Command-line parameter overrides
- Parallel processing with optional GPU support
- Automatic timestamp-based folder generation
- Support for HN, Duan, and GJR GARCH models

**Key Features:**
- Reads configuration from TOML files
- Overridable parameters via command line
- Noise levels from 0% (clean) to 100%+ (very noisy)
- Reproducible noise generation with seed control
- Performance metrics and throughput reporting

### 2. **scaled_dataset_config.toml** (Default Config Template)
- Complete TOML configuration template
- Sections: model, scaling, market, willow_tree, performance, noise, output
- Default values for HN model with 50 contracts/day, 20 days
- No noise by default (level = 0.0)
- Comments explaining each parameter

### 3. **Configuration Presets** (Example Configs)

#### scaled_dataset_config_hn_clean.toml
- HN model with clean data (no noise)
- Larger dataset: 100 contracts/day, 30 days, 10 parameter sets
- For testing model accuracy and calibration

#### scaled_dataset_config_duan_3pct.toml
- Duan model with 3% noise (typical market)
- 40 contracts/day, 15 days, 6 parameter sets
- Useful for calibration robustness testing

#### scaled_dataset_config_gjr_5pct.toml
- GJR model with 5% noise (simulates market frictions)
- Larger dataset: 60 contracts/day, 25 days, 8 parameter sets
- Suitable for neural network training

### 4. **CONFIG_DRIVEN_DATAGEN_GUIDE.md** (Comprehensive Documentation)
- Complete user guide for the new system
- TOML configuration format explanation
- Noise injection details and formulas
- Command-line override examples
- Performance tips and troubleshooting
- Comparison with original system

### 5. **usage_examples_configurable.m** (Usage Examples)
- 16 practical examples showing different use cases
- Batch processing templates
- Sensitivity analysis examples
- Parameter sweep patterns
- Comparison across models
- Tips and best practices

## Key Improvements

### 1. Configuration Management
| Aspect | Before | After |
|--------|--------|-------|
| Configuration | Command-line only | **TOML file + CLI overrides** |
| Flexibility | Limited, hard to reproduce | **High, config file as documentation** |
| Defaults | No template | **Template with presets** |
| Multi-run setup | Tedious | **Simple config modification** |

### 2. Noise Injection Feature (NEW)
**What It Does:**
- Adds realistic market noise to simulated option prices
- Multiplicative noise model: `V_noisy = V_true × (1 + N(0, σ))`
- Where σ = noise_level (0.01 = 1%, 0.05 = 5%, etc.)

**Why It Matters:**
- Simulates market friction and bid-ask spreads
- Tests model robustness to real-world imperfections
- Useful for validating calibration and pricing accuracy
- Enables sensitivity analysis across noise levels

**Parameters:**
```toml
[noise]
level = 0.05    # 5% noise
seed = 42       # Reproducible randomness
```

### 3. Command-Line Overrides
Full flexibility to override any config parameter without editing files:

```matlab
scaled_dataset_optimized_configurable('config', 'base_config.toml', ...
    'noise_level', 0.03, ...
    'contracts_per_day', 75, ...
    'model_type', 'Duan')
```

### 4. Better Organization
- Config files document the intent of each run
- Multiple presets for common use cases
- Examples for batch processing and sensitivity analysis
- Clear separation of code and configuration

## Technical Details

### Noise Implementation
```matlab
% In process_contracts_with_noise function:
if NOISE_LEVEL > 0
    noise_c = 1 + NOISE_LEVEL * randn(noise_rng);
    V_C = V_C * noise_c;  % Multiplicative noise on call price
end
```

### Reproducibility
- Fixed random seed (default 42) ensures reproducible noise
- Same config + same seed = identical results
- Use seed = -1 for random seeds when needed

### TOML Parser
- Custom MATLAB implementation for parsing TOML files
- Supports sections, subsections, arrays, and basic types
- No external dependencies required
- Sufficient for this use case

## Usage Examples

### Quick Start
```matlab
% Use default config
scaled_dataset_optimized_configurable()

% Use specific preset
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config_duan_3pct.toml')

% Override noise level
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config.toml', 'noise_level', 0.10)
```

### Batch Processing
```matlab
configs = {
    'scaled_dataset_config_hn_clean.toml';
    'scaled_dataset_config_duan_3pct.toml';
    'scaled_dataset_config_gjr_5pct.toml'
};
for i = 1:length(configs)
    scaled_dataset_optimized_configurable('config', configs{i});
end
```

### Sensitivity Analysis
```matlab
noise_levels = [0.01, 0.03, 0.05, 0.10];
for noise = noise_levels
    scaled_dataset_optimized_configurable('config', 'base_config.toml', ...
        'noise_level', noise)
end
```

## Testing & Verification

### Tests Run
✅ Basic configuration loading  
✅ TOML parsing (all data types)  
✅ Command-line parameter overrides  
✅ Noise injection (0% and 5%)  
✅ Dataset generation with parallel processing  
✅ Output file creation and validation  
✅ Performance metrics reporting  

### Test Results
- Configuration loading: **PASS**
- Noise application: **PASS** (shows "Noise applied: YES" when noise_level > 0)
- Parameter overrides: **PASS** (CLI overrides take precedence)
- Dataset generation: **PASS** (valid CSV output with proper headers)
- Performance: **PASS** (parallel speedup observed)

## Backward Compatibility

The original `scaled_dataset_optimized.m` remains unchanged. Users can:
- Continue using the original for existing workflows
- Migrate to the new configurable version gradually
- Mix both approaches in the same codebase

## Documentation Files

1. **CONFIG_DRIVEN_DATAGEN_GUIDE.md** - Full user guide
2. **usage_examples_configurable.m** - 16 practical examples
3. **scaled_dataset_config.toml** - Default template with comments
4. **scaled_dataset_config_*.toml** - 3 preset configurations
5. This summary document

## Next Steps (Optional Enhancements)

Future improvements could include:
- Configuration validation schema (JSON Schema)
- GUI for config generation
- Integration with data pipeline (automated retraining)
- Performance profiling dashboard
- Distributed processing support

## Migration Guide

### For Existing Users
1. Copy `scaled_dataset_config.toml` as a template
2. Edit with your desired parameters
3. Replace command: `scaled_dataset_optimized_configurable('config', 'my_config.toml')`
4. Benefit: Reproducible, documented runs

### For New Users
1. Start with a preset config (e.g., `scaled_dataset_config_hn_clean.toml`)
2. Copy and modify for your use case
3. Use command-line overrides for variations
4. Save successful configs for future use

## Summary

This refactoring provides a **professional, maintainable approach** to dataset generation with:
- **Flexibility**: Config-driven with CLI overrides
- **Reproducibility**: TOML files document each run
- **Noise Support**: NEW feature for realistic data
- **Organization**: Multiple presets for common use cases
- **Documentation**: Comprehensive guide and examples

The new system maintains all performance optimizations from the original while adding significant usability and research capabilities.
