% USAGE EXAMPLES: scaled_dataset_optimized_configurable.m
% ========================================================

% EXAMPLE 1: Basic usage with default config
% Generate dataset using scaled_dataset_config.toml
scaled_dataset_optimized_configurable()

% EXAMPLE 2: Use specific config file
% Generate HN dataset with no noise (clean model data)
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config_hn_clean.toml')

% EXAMPLE 3: Duan model with 3% noise
% For testing calibration and model robustness
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config_duan_3pct.toml')

% EXAMPLE 4: GJR model with 5% noise
% Larger dataset for neural network training
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config_gjr_5pct.toml')

% EXAMPLE 5: Override noise level from config
% Start with config, but increase noise to 10%
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config.toml', ...
    'noise_level', 0.10)

% EXAMPLE 6: Override multiple parameters
% Use Duan model, set noise, and reduce dataset size
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config.toml', ...
    'model_type', 'Duan', ...
    'noise_level', 0.05, ...
    'contracts_per_day', 30, ...
    'num_days', 10)

% EXAMPLE 7: Custom output folder
% Generate dataset with custom output location
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config.toml', ...
    'output_folder', 'my_custom_dataset_folder')

% EXAMPLE 8: Sensitivity analysis - multiple noise levels
% Generate same dataset with different noise levels for comparison
fprintf('Generating datasets with varying noise levels...\n');
noise_levels = [0.01, 0.03, 0.05, 0.10];
for noise = noise_levels
    fprintf('\n--- Generating with %.1f%% noise ---\n', noise*100);
    scaled_dataset_optimized_configurable('config', 'scaled_dataset_config_hn_clean.toml', ...
        'noise_level', noise, ...
        'output_folder', sprintf('hn_noise_%d_pct', round(noise*100)))
end

% EXAMPLE 9: Compare models with same parameters
% Generate HN, Duan, GJR with identical settings and 5% noise
fprintf('Comparing models...\n');
models = {'HN', 'Duan', 'GJR'};
for i = 1:length(models)
    fprintf('\n--- Generating %s model ---\n', models{i});
    scaled_dataset_optimized_configurable('config', 'scaled_dataset_config.toml', ...
        'model_type', models{i}, ...
        'noise_level', 0.05, ...
        'output_folder', sprintf('comparison_%s', models{i}))
end

% EXAMPLE 10: Quick test run - small dataset
% For testing pipeline without long computation time
scaled_dataset_optimized_configurable('config', 'test_config_small.toml')

% EXAMPLE 11: Large dataset for training
% Override config for bigger dataset
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config.toml', ...
    'contracts_per_day', 200, ...
    'num_days', 50, ...
    'num_param_sets', 20)

% EXAMPLE 12: Production run with all optimization options
% Use GPU, parallel processing, and memory optimization
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config.toml', ...
    'noise_level', 0.04, ...
    'contracts_per_day', 150, ...
    'num_days', 40)

% EXAMPLE 13: Batch processing script
% Generate multiple datasets in sequence
fprintf('Starting batch dataset generation...\n');
configs = {
    'scaled_dataset_config_hn_clean.toml';
    'scaled_dataset_config_duan_3pct.toml';
    'scaled_dataset_config_gjr_5pct.toml'
};

for i = 1:length(configs)
    fprintf('\n[%d/%d] Processing: %s\n', i, length(configs), configs{i});
    try
        scaled_dataset_optimized_configurable('config', configs{i});
        fprintf('[SUCCESS] Completed: %s\n', configs{i});
    catch ME
        fprintf('[ERROR] Failed: %s\n', configs{i});
        fprintf('Error: %s\n', ME.message);
    end
end

% EXAMPLE 14: Variable parameter sweep
% Sweep across parameter sets and noise levels
fprintf('Parameter sweep: contracts_per_day and noise_level\n');
cpd_values = [30, 50, 100];
noise_values = [0.0, 0.05, 0.10];

for cpd = cpd_values
    for noise = noise_values
        fprintf('\n--- cpd=%d, noise=%.1f%% ---\n', cpd, noise*100);
        scaled_dataset_optimized_configurable('config', 'scaled_dataset_config.toml', ...
            'contracts_per_day', cpd, ...
            'noise_level', noise, ...
            'output_folder', sprintf('sweep_cpd%d_noise%d', cpd, round(noise*100)))
    end
end

% EXAMPLE 15: Load and analyze generated dataset
% (After running one of the above examples)
output_file = 'generation_hn_20260403_111328_cached/dataset_hn.csv';
if isfile(output_file)
    data = readtable(output_file);
    fprintf('\nDataset loaded successfully:\n');
    fprintf('Rows: %d\n', height(data));
    fprintf('Columns: %d\n', width(data));
    fprintf('Variables: %s\n', strjoin(data.Properties.VariableNames, ', '));
    
    % Summary statistics
    fprintf('\nSummary statistics:\n');
    fprintf('Option prices (V): min=%.4f, max=%.4f, mean=%.4f\n', ...
        min(data.V), max(data.V), mean(data.V));
    fprintf('Implied volatility (sigma): min=%.4f, max=%.4f, mean=%.4f\n', ...
        min(data.sigma), max(data.sigma), mean(data.sigma));
end

% EXAMPLE 16: Memory-optimized small dataset
% For resource-limited systems
scaled_dataset_optimized_configurable('config', 'scaled_dataset_config.toml', ...
    'contracts_per_day', 20, ...
    'num_days', 5, ...
    'num_param_sets', 2)

% ==============================================================
% KEY TIPS:
%
% 1. Always check if output folder exists before running
%
% 2. Noise level interpretation:
%    - 0.00 = clean model data
%    - 0.01 = 1% noise (very clean)
%    - 0.05 = 5% noise (typical market)
%    - 0.10 = 10% noise (noisy/illiquid)
%
% 3. For batch processing, use try-catch to continue on errors
%
% 4. Set num_param_sets > 1 for diversity in training data
%
% 5. Use same seed in config for reproducibility
%
% 6. Command-line overrides take precedence over config file
%
% 7. Output folder defaults to: generation_<model>_<timestamp>_cached
%
% ==============================================================
