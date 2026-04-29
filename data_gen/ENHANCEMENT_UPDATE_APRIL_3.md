# Enhancement Update: Parameter & Asset File Saving

**Date**: April 3, 2026  
**Status**: ✅ Complete and Tested  
**Changes**: Added automatic saving of GARCH parameters and asset price files

## What Was Added

### 1. Automatic GARCH Parameter Saving
Each time you run the dataset generator, it now saves a CSV file containing all GARCH parameters used:

**File**: `garch_parameters_<model>.csv`

**Example Content**:
```
set_id,omega,alpha,beta,gamma,lambda
1,8.60073397867553e-07,1.14389761916385e-06,0.79917013629552,4.75468990766645,0.227888112958222
2,1.16443889145267e-06,1.54870372563205e-06,0.790707897692583,4.63099180679225,0.205471532223634
```

### 2. Automatic Asset Price File Saving
Stock price paths from GARCH simulations are automatically saved:

**Files**: `asset_prices_set_1.csv`, `asset_prices_set_2.csv`, etc. (one per parameter set)

**Example Content**:
```
100
99.8852793583345
99.765563000669
99.3369844002107
99.4102690060806
...
```

## Configuration

In any config file's `[output]` section:

```toml
[output]
# Save asset price files from GARCH simulations
save_asset_prices = true    # Enable (default)
# save_asset_prices = false  # Disable if desired
```

**Default Behavior**: Enabled (asset files are saved automatically)

## Output Folder Contents

After running the generator, your output folder will contain:

```
generation_hn_20260403_205931_cached/
├── dataset_hn.csv                    ← Option prices (always saved)
├── garch_parameters_hn.csv           ← NEW: Parameter sets
├── asset_prices_set_1.csv            ← NEW: Stock prices, set 1
├── asset_prices_set_2.csv            ← NEW: Stock prices, set 2
├── asset_prices_set_3.csv            ← NEW: Stock prices, set 3
└── asset_prices_set_N.csv            ← NEW: Stock prices, set N
```

## Usage Examples

### Load and Analyze Parameters
```matlab
% Load parameter sets
params = readtable('generation_hn_*/garch_parameters_hn.csv');

% Analyze distributions
figure;
subplot(2,2,1); histogram(params.omega); title('omega');
subplot(2,2,2); histogram(params.alpha); title('alpha');
subplot(2,2,3); histogram(params.beta);  title('beta');
subplot(2,2,4); histogram(params.lambda); title('lambda');
```

### Visualize Asset Price Paths
```matlab
% Load asset prices
S_paths = readmatrix('generation_hn_*/asset_prices_set_1.csv');

% Plot
figure;
plot(S_paths);
xlabel('Trading Days');
ylabel('Stock Price');
title('GARCH-Generated Stock Price Path');
```

### Reproduce Experiments
```matlab
% Load original parameters
original_params = readtable('experiment_1/garch_parameters_hn.csv');

% Use same parameters for new simulations
for i = 1:height(original_params)
    omega = original_params.omega(i);
    alpha = original_params.alpha(i);
    beta = original_params.beta(i);
    gamma = original_params.gamma(i);
    lambda = original_params.lambda(i);
    % ... re-run with same params
end
```

## Files Modified

### 1. scaled_dataset_optimized_configurable.m
- **Added**: `all_S_paths` cell array to store asset paths
- **Added**: `save_param_sets()` function to save parameters
- **Added**: `save_asset_files()` function to save asset prices
- **Modified**: Parallel loop to capture S_paths
- **Modified**: Results saving section to write new files

### 2. scaled_dataset_config.toml
- **Updated**: Comments for `save_asset_prices` option
- **Default**: `save_asset_prices = true`

### 3. Preset Configurations
Updated all preset configs to save asset files:
- `scaled_dataset_config_hn_clean.toml`
- `scaled_dataset_config_duan_3pct.toml`
- `scaled_dataset_config_gjr_5pct.toml`

## Backward Compatibility

✅ **No breaking changes**
- Existing scripts continue to work
- All previous functionality preserved
- Asset file saving is optional (configurable)

## Performance Impact

- **Minimal**: Asset files are saved after main processing completes
- **I/O only**: No computational overhead
- **Storage**: ~1KB per asset file (varies with N and M)

## Examples

### Example 1: HN Model with 5 Parameter Sets
```matlab
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config_hn_clean.toml')
```

**Generated files**:
- `garch_parameters_hn.csv` (1 file)
- `asset_prices_set_1.csv` through `asset_prices_set_5.csv` (5 files)

### Example 2: Disable Asset File Saving
```matlab
% Create config with save_asset_prices = false
scaled_dataset_optimized_configurable('config', 'my_config.toml')
```

**Generated files**:
- `garch_parameters_hn.csv` (still saved)
- No asset_prices_set_*.csv files

### Example 3: Batch Processing with Parameter Tracking
```matlab
configs = {
    'scaled_dataset_config_hn_clean.toml',
    'scaled_dataset_config_duan_3pct.toml'
};

for i = 1:length(configs)
    scaled_dataset_optimized_configurable('config', configs{i})
    
    % Load and check parameters
    params = readtable(sprintf(...
        'generation_**/garch_parameters_%s.csv', model));
    fprintf('Generated %d parameter sets\n', height(params));
end
```

## Testing

✅ **Tested with**:
- 2 parameter sets
- 1 trading day
- 2 contracts per day
- N=50 time steps

✅ **Results**:
- ✓ Parameters saved correctly (2 rows)
- ✓ Asset files created (51 rows × 1 column each)
- ✓ All data validated
- ✓ CSV format correct

## Documentation

New guide created:
- `PARAMETER_AND_ASSET_SAVE_GUIDE.md` - Detailed documentation

## Summary

✅ GARCH parameters now automatically saved  
✅ Asset price files now automatically saved  
✅ Configurable behavior (can disable if needed)  
✅ No breaking changes to existing workflows  
✅ Clear, descriptive file naming  
✅ Ready for reproducibility and analysis  

---

**Note**: If you have multiple parameter sets, each will have its own asset price file (e.g., `asset_prices_set_1.csv`, `asset_prices_set_2.csv`, etc.) for easy identification.
