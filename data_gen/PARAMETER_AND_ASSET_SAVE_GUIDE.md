# Parameter Sets and Asset Files Enhancement

## Changes Made

Updated `scaled_dataset_optimized_configurable.m` to automatically save:

1. **GARCH Parameter Sets** - Parameters used for each dataset generation
2. **Asset Price Files** - Stock price paths from GARCH simulations

## New Files Saved

### GARCH Parameters File
- **Filename**: `garch_parameters_<model>.csv`
- **Format**: CSV with headers
- **Columns**: set_id, omega, alpha, beta, gamma, lambda
- **Content**: One row per parameter set with all GARCH parameters used

Example:
```
set_id,omega,alpha,beta,gamma,lambda
1,8.60073397867553e-07,1.14389761916385e-06,0.79917013629552,4.75468990766645,0.227888112958222
2,1.16443889145267e-06,1.54870372563205e-06,0.790707897692583,4.63099180679225,0.205471532223634
```

### Asset Price Files
- **Filename**: `asset_prices_set_<N>.csv` (one per parameter set)
- **Format**: CSV with one column (stock prices)
- **Content**: S_paths from GARCH simulation (rows = time steps, cols = paths)
- **Example**: `asset_prices_set_1.csv` contains S_paths for parameter set 1

Example:
```
100
99.8852793583345
99.765563000669
99.3369844002107
99.4102690060806
...
```

## Configuration

### Enable/Disable Asset File Saving

In the config file `[output]` section:

```toml
[output]
# Whether to save asset price paths (S_paths from GARCH simulations)
save_asset_prices = true    # Set to false to skip saving
```

### Default Behavior
- **Current**: `save_asset_prices = true` (enabled by default)
- Asset files are always saved unless explicitly disabled in config

## Output Folder Structure

After running the script, the output folder will contain:

```
generation_hn_20260403_205931_cached/
├── dataset_hn.csv                    # Main dataset (options)
├── garch_parameters_hn.csv           # Parameter sets
├── asset_prices_set_1.csv            # Asset prices for set 1
├── asset_prices_set_2.csv            # Asset prices for set 2
└── ...
```

## Use Cases

### 1. Analyze Parameter Diversity
```matlab
% Load and visualize generated parameters
params = readtable('generation_hn_20260403_*/garch_parameters_hn.csv');
histogram(params.alpha);
histogram(params.beta);
```

### 2. Plot Asset Paths
```matlab
% Load and plot asset price paths
S_paths_1 = readmatrix('generation_hn_20260403_*/asset_prices_set_1.csv');
plot(S_paths_1);
title('GARCH-Generated Stock Price Paths (Set 1)');
```

### 3. Reproduce Data Generation
```matlab
% Load parameters for reproducibility
params = readtable('garch_parameters_hn.csv');
% Can re-use these parameters for new simulations
```

### 4. Validate Calibration
```matlab
% Compare generated parameters with calibrated values
true_params = readtable('true_params.csv');
generated_params = readtable('garch_parameters_hn.csv');
% Analyze differences
```

## Integration with Existing Workflow

The changes are backward compatible:
- Existing configurations continue to work
- Asset files are saved automatically
- No changes needed to existing scripts

## Command-Line Control

To disable asset file saving at runtime:

```matlab
% Modify config file on the fly
config.output.save_asset_prices = false;
scaled_dataset_optimized_configurable('config', 'my_config.toml')
```

Or create a config with `save_asset_prices = false` in the `[output]` section.

## Files Updated

1. **scaled_dataset_optimized_configurable.m**
   - Added `all_S_paths` to store asset paths
   - Added `save_param_sets()` function
   - Added `save_asset_files()` function
   - Integrated saving in main workflow

2. **scaled_dataset_config.toml**
   - Updated comments for `save_asset_prices`
   - Default: `save_asset_prices = true`

3. **Preset Configuration Files**
   - Updated all presets to `save_asset_prices = true`
   - `scaled_dataset_config_hn_clean.toml`
   - `scaled_dataset_config_duan_3pct.toml`
   - `scaled_dataset_config_gjr_5pct.toml`

## Technical Details

### Parameter Sets Saving
```matlab
function save_param_sets(output_folder, param_sets, model_type)
    % Extracts omega, alpha, beta, gamma, lambda from each parameter set
    % Saves as CSV with set_id column for identification
    % File: garch_parameters_<model_type>.csv
end
```

### Asset Files Saving
```matlab
function save_asset_files(output_folder, all_S_paths)
    % Saves each S_paths matrix as separate CSV
    % One file per parameter set
    % File: asset_prices_set_<N>.csv
end
```

## Example Workflow

```matlab
% 1. Generate dataset with saved parameters and asset files
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config_hn_clean.toml')

% 2. Load and inspect parameters
params = readtable('generation_hn_*/garch_parameters_hn.csv');
disp(params)

% 3. Load and analyze asset paths
S_paths_1 = readmatrix('generation_hn_*/asset_prices_set_1.csv');
plot(S_paths_1)

% 4. Load main dataset
data = readtable('generation_hn_*/dataset_hn.csv');
summary(data)
```

## Summary

✅ GARCH parameter sets now automatically saved
✅ Asset price files now automatically saved
✅ Configuration option to disable if needed
✅ Files clearly named and organized
✅ Backward compatible with existing workflows
✅ No breaking changes to existing scripts
