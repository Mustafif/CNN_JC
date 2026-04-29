function scaled_dataset_optimized_configurable(varargin)
% SCALED_DATASET_OPTIMIZED_CONFIGURABLE Highly optimized dataset generator with config file support
%
% USAGE:
%   scaled_dataset_optimized_configurable()
%   scaled_dataset_optimized_configurable('config', 'scaled_dataset_config.toml')
%   scaled_dataset_optimized_configurable('config', 'my_config.toml', 'noise_level', 0.05)
%
% This script:
%   1. Reads configuration from a TOML file
%   2. Generates datasets with optional noise
%   3. Parallelizes parameter set processing
%   4. Caches tree data for performance
%
% Optional Name-Value Pairs:
%   'config'          - Path to TOML config file (default: 'scaled_dataset_config.toml')
%   'model_type'      - Override model type from config (default: uses config value)
%   'noise_level'     - Override noise level from config
%   'contracts_per_day' - Override contracts per day
%   'num_days'        - Override number of days
%   'output_folder'   - Override output folder path

    fprintf('=================================================================\n');
    fprintf('  OPTIMIZED GARCH OPTION PRICING DATASET GENERATOR\n');
    fprintf('  (Config-based with Noise Support)\n');
    fprintf('=================================================================\n\n');

    %% PARSE INPUT ARGUMENTS
    p = inputParser;
    addParameter(p, 'config', 'scaled_dataset_config_multi_noise.toml', @ischar);
    addParameter(p, 'model_type', '', @ischar);
    addParameter(p, 'noise_level', [], @(x) isempty(x) || (isnumeric(x) && x >= 0));
    addParameter(p, 'contracts_per_day', [], @(x) isempty(x) || (isnumeric(x) && x > 0));
    addParameter(p, 'num_days', [], @(x) isempty(x) || (isnumeric(x) && x > 0));
    addParameter(p, 'output_folder', '', @ischar);
    
    parse(p, varargin{:});
    
    config_file = p.Results.config;
    
    %% LOAD CONFIG FILE
    fprintf('Loading configuration from: %s\n', config_file);
    if ~exist(config_file, 'file')
        error('Config file not found: %s', config_file);
    end
    
    config = read_toml_config_matlab(config_file);
    
    %% EXTRACT CONFIG VALUES (with command-line overrides)
    model_type = upper(p.Results.model_type);
    if isempty(model_type)
        model_type = upper(config.model.type);
    end
    
    if ~ismember(model_type, {'HN', 'DUAN', 'GJR'})
        error('Invalid model type: %s', model_type);
    end
    
    CONTRACTS_PER_DAY = p.Results.contracts_per_day;
    if isempty(CONTRACTS_PER_DAY)
        CONTRACTS_PER_DAY = config.scaling.contracts_per_day;
    end
    
    NUM_DAYS = p.Results.num_days;
    if isempty(NUM_DAYS)
        NUM_DAYS = config.scaling.num_days;
    end
    
    % Batch size: use auto-tune if enabled, otherwise use config value
    BATCH_SIZE_AUTO = false;
    if isfield(config, 'performance') && isfield(config.performance, 'batch_size_auto')
        BATCH_SIZE_AUTO = config.performance.batch_size_auto;
    end
    
    if BATCH_SIZE_AUTO
        % Auto-tune based on system memory
        mem_info = memory();
        available_mem_gb = mem_info.MemAvailableAllArrays / 1e9;
        % Target: 500MB per batch, estimate ~0.5KB per contract
        estimated_contracts_per_500mb = 500 / 0.0005;
        BATCH_SIZE = max(100, min(1000, round(estimated_contracts_per_500mb)));
    else
        % Use config value
        BATCH_SIZE = config.scaling.batch_size;
    end
    
    NUM_PARAM_SETS = config.scaling.num_param_sets;
    
    % Market parameters
    N = config.market.N;
    r = config.market.r;
    S0 = config.market.S0;
    moneyness_range = config.market.moneyness_range;
    maturity_days = config.market.maturity_days;
    
    % Willow tree parameters
    m_h = config.willow_tree.m_h;
    m_x = config.willow_tree.m_x;
    gamma_h = config.willow_tree.gamma_h;
    gamma_x = config.willow_tree.gamma_x;
    
    % Performance parameters
    USE_GPU = config.performance.use_gpu;
    NUM_WORKERS = config.performance.num_workers;
    OPTIMIZE_MEMORY = config.performance.optimize_memory;
    
    % Noise parameters
    NOISE_LEVEL = p.Results.noise_level;
    if isempty(NOISE_LEVEL)
        NOISE_LEVEL = config.noise.level;
    end
    NOISE_SEED = config.noise.seed;
    
    % Multi-noise generation parameters
    MULTI_NOISE_ENABLED = false;
    MULTI_NOISE_LEVELS = [0, 0.01, 0.05];  % Default: 0%, 1%, 5%
    if isfield(config, 'multi_noise') && ~isempty(config.multi_noise)
        if isfield(config.multi_noise, 'enabled')
            MULTI_NOISE_ENABLED = config.multi_noise.enabled;
        end
        if isfield(config.multi_noise, 'noise_levels')
            MULTI_NOISE_LEVELS = config.multi_noise.noise_levels;
        end
    end
    
    % Output parameters
    output_folder = p.Results.output_folder;
    if isempty(output_folder)
        if isempty(config.output.folder) || strcmp(config.output.folder, '')
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            output_folder = sprintf('generation_%s_%s_cached', lower(model_type), timestamp);
        else
            output_folder = config.output.folder;
        end
    end
    
    % Batching & parallelization parameters
    ENABLE_NESTED_PARFOR = false;  % Enable nested parallelization for contracts
    if isfield(config, 'performance') && isfield(config.performance, 'enable_nested_parfor')
        ENABLE_NESTED_PARFOR = config.performance.enable_nested_parfor;
    end
    
    output_filename = config.output.filename;
    if isempty(output_filename)
        output_filename = sprintf('dataset_%s', lower(model_type));
    end
    
    %% DISPLAY CONFIGURATION
    fprintf('\nCONFIGURATION SUMMARY:\n');
    fprintf('  Model: %s GARCH\n', model_type);
    fprintf('  Config file: %s\n', config_file);
    fprintf('\nSCALING:\n');
    fprintf('  Contracts per day: %d\n', CONTRACTS_PER_DAY);
    fprintf('  Number of days: %d\n', NUM_DAYS);
    fprintf('  Total contracts: %d (x2 for calls/puts)\n', CONTRACTS_PER_DAY * NUM_DAYS);
    fprintf('  Batch size: %d\n', BATCH_SIZE);
    fprintf('  Parameter sets: %d\n', NUM_PARAM_SETS);
    
    fprintf('\nMARKET PARAMETERS:\n');
    fprintf('  Stock price (S0): %.2f\n', S0);
    fprintf('  Risk-free rate (daily): %.6f\n', r);
    fprintf('  Time steps (N): %d\n', N);
    fprintf('  Moneyness range: [%.2f, %.2f]\n', moneyness_range(1), moneyness_range(2));
    fprintf('  Maturity days: %s\n', mat2str(maturity_days));
    
    fprintf('\nNOISE CONFIGURATION:\n');
    if MULTI_NOISE_ENABLED
        fprintf('  Multi-noise generation: ENABLED\n');
        fprintf('  Noise levels: %s\n', mat2str(MULTI_NOISE_LEVELS * 100));
        fprintf('  Will generate %d datasets\n', length(MULTI_NOISE_LEVELS));
    else
        fprintf('  Multi-noise generation: DISABLED\n');
    end
    fprintf('  Single-run noise level: %.4f (%.2f%%)\n', NOISE_LEVEL, NOISE_LEVEL * 100);
    fprintf('  Noise seed: %d\n', NOISE_SEED);
    
    fprintf('\nPERFORMANCE:\n');
    if USE_GPU
        gpu_str = 'ON';
    else
        gpu_str = 'OFF';
    end
    fprintf('  GPU acceleration: %s\n', gpu_str);
    
    if OPTIMIZE_MEMORY
        mem_str = 'ON';
    else
        mem_str = 'OFF';
    end
    fprintf('  Memory optimization: %s\n', mem_str);
    fprintf('  Batch size: %d contracts\n', BATCH_SIZE);
    
    % Recommend batch size based on configuration
    estimated_contracts_per_param = CONTRACTS_PER_DAY * NUM_DAYS;
    recommended_batch_size = max(100, round(estimated_contracts_per_param / 5));
    if abs(BATCH_SIZE - recommended_batch_size) > 50
        fprintf('  ⚠ Note: Recommended batch size for this config: %d\n', recommended_batch_size);
    end
    
    if ENABLE_NESTED_PARFOR
        fprintf('  Nested parallelization: ENABLED (for contracts)\n');
    else
        fprintf('  Nested parallelization: disabled\n');
    end
    fprintf('  Output folder: %s\n\n', output_folder);
    
    %% CHECK AND CONFIGURE PARALLEL POOL
    fprintf('PERFORMANCE CONFIGURATION:\n');
    
    pool_obj = gcp('nocreate');
    if isempty(pool_obj)
        if NUM_WORKERS == 0
            NUM_WORKERS = feature('numcores');
        end
        fprintf('  Starting parallel pool with %d workers...\n', NUM_WORKERS);
        try
            parpool(NUM_WORKERS);
            fprintf('  ✓ Parallel pool started\n');
        catch
            fprintf('  ⚠ Could not start parallel pool, using serial processing\n');
        end
    else
        fprintf('  ✓ Using existing parallel pool (%d workers)\n', pool_obj.NumWorkers);
    end
    
    % Check GPU availability
    if USE_GPU
        try
            % Try NVIDIA GPU (CUDA)
            gpu_device = gpuDevice();
            gpu_type = 'NVIDIA (CUDA)';
            fprintf('  ✓ GPU available: %s - %s (%.1f GB)\n', gpu_device.Name, gpu_type, gpu_device.AvailableMemory/1e9);
        catch nvidia_error
            % NVIDIA GPU not available, try to detect AMD GPU
            try
                % For AMD GPUs, check if we can still use parallel computing
                % without explicit GPU arrays. AMD support in MATLAB is limited,
                % but we can try ROCm or fall back to optimized CPU parallel processing
                fprintf('  ⚠ NVIDIA GPU (CUDA) not found\n');
                
                % Check for AMD GPU using system command
                [status, gpu_info] = system('rocm-smi --showproductname 2>/dev/null || lspci 2>/dev/null | grep -i amd');
                
                if status == 0 && ~isempty(gpu_info)
                    fprintf('  ℹ AMD GPU detected: %s\n', strtrim(gpu_info(1:min(50, length(gpu_info)))));
                    fprintf('  ⚠ Note: MATLAB has limited native support for AMD GPUs\n');
                    fprintf('  ℹ Using optimized CPU-based parallel processing instead\n');
                    USE_GPU = false;  % Fall back to CPU but keep parallelization enabled
                else
                    fprintf('  ⚠ GPU requested but not available, using CPU\n');
                    USE_GPU = false;
                end
            catch
                fprintf('  ⚠ No GPU detected, using CPU\n');
                USE_GPU = false;
            end
        end
    else
        fprintf('  ℹ GPU acceleration: disabled\n');
    end
    
    % Create output folder
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    
    output_file = sprintf('%s/%s.csv', output_folder, output_filename);
    
    %% PRE-GENERATE ALL RANDOM DATA
    fprintf('\nPre-generating random data...\n');
    rng(12345);
    
    M = 1;
    numPoint = N + 1;
    Z = randn(numPoint + 1, M);
    
    % Pre-generate all moneyness values
    num_unique_contracts = CONTRACTS_PER_DAY * NUM_DAYS;
    all_moneyness = moneyness_range(1) + ...
                    (moneyness_range(2) - moneyness_range(1)) * rand(1, num_unique_contracts);
    
    % Use maturities as-is (in days) - matches daily risk-free rate 'r'
    % Variable name is T_years for legacy compatibility, but values are in days
    T_years = maturity_days;
    
    % Initialize noise random stream
    if NOISE_LEVEL > 0
        if NOISE_SEED >= 0
            noise_rng = RandStream('mt19937ar', 'Seed', NOISE_SEED);
        else
            noise_rng = RandStream('mt19937ar', 'Seed', randi([0, 2^31-1]));
        end
    end
    
    fprintf('✓ Random data generated\n\n');
    
    %% GENERATE PARAMETER SETS
    fprintf('Generating %d parameter sets...\n', NUM_PARAM_SETS);
    param_sets = generate_param_sets(model_type, NUM_PARAM_SETS);
    
    %% PROCESS DATASETS (with multi-noise loop if enabled)
    if MULTI_NOISE_ENABLED
        noise_levels_to_process = MULTI_NOISE_LEVELS;
    else
        noise_levels_to_process = NOISE_LEVEL;
    end
    
    for noise_idx = 1:length(noise_levels_to_process)
        current_noise_level = noise_levels_to_process(noise_idx);
        
        if MULTI_NOISE_ENABLED
            fprintf('\n=================================================================\n');
            fprintf('  DATASET %d/%d: NOISE LEVEL = %.2f%%\n', noise_idx, length(MULTI_NOISE_LEVELS), current_noise_level*100);
            fprintf('=================================================================\n\n');
        end
        
        %% PARALLEL PROCESSING OF PARAMETER SETS
        fprintf('Processing %d parameter sets in parallel...\n', NUM_PARAM_SETS);
        total_start = tic;
        
        all_datasets = cell(NUM_PARAM_SETS, 1);
        all_processing_times = zeros(NUM_PARAM_SETS, 1);
        all_S_paths = cell(NUM_PARAM_SETS, 1);
        
        parfor ps = 1:NUM_PARAM_SETS
            fprintf('  [Worker] Processing parameter set %d/%d...\n', ps, NUM_PARAM_SETS);
            
            params = param_sets(ps);
            ps_start = tic;
            
            % Process based on model type
            switch upper(model_type)
                case 'HN'
                    dataset = process_parameter_set_HN(params, M, N, S0, Z, r, NUM_DAYS, ...
                                                       m_h, gamma_h, m_x, gamma_x, ...
                                                       all_moneyness, T_years, BATCH_SIZE, ...
                                                       current_noise_level, NOISE_SEED, CONTRACTS_PER_DAY, all_S_paths, ps);
                case 'DUAN'
                    dataset = process_parameter_set_DUAN(params, M, N, S0, Z, r, NUM_DAYS, ...
                                                         m_h, gamma_h, m_x, gamma_x, ...
                                                         all_moneyness, T_years, BATCH_SIZE, ...
                                                         current_noise_level, NOISE_SEED, CONTRACTS_PER_DAY, all_S_paths, ps);
                case 'GJR'
                    dataset = process_parameter_set_GJR(params, M, N, S0, Z, r, NUM_DAYS, ...
                                                        m_h, gamma_h, m_x, gamma_x, ...
                                                        all_moneyness, T_years, BATCH_SIZE, ...
                                                        current_noise_level, NOISE_SEED, CONTRACTS_PER_DAY, all_S_paths, ps);
                otherwise
                    error('Unknown model type: %s', model_type);
            end
            
            all_datasets{ps} = dataset;
            all_processing_times(ps) = toc(ps_start);
            
            fprintf('  [Worker] Parameter set %d complete: %.2f seconds\n', ps, all_processing_times(ps));
        end
        
        total_time = toc(total_start);
        
        fprintf('\n✓ All parameter sets processed in %.2f seconds\n', total_time);
        fprintf('  Average time per set: %.2f seconds\n', mean(all_processing_times));
        fprintf('  Parallel speedup: %.2fx (estimated)\n', sum(all_processing_times) / total_time);
        
        %% COMBINE AND SAVE RESULTS
        fprintf('Combining and saving results...\n');
        
        combined_dataset = horzcat(all_datasets{:});
        
        fprintf('Cleaning dataset...\n');
        original_size = size(combined_dataset, 2);
        valid_mask = (combined_dataset(12, :) >= 0.5) & ...
                     (abs(combined_dataset(4, :) - 1/252) > 1e-10) & ...
                     all(~isnan(combined_dataset), 1) & ...
                     all(~isinf(combined_dataset), 1);
        
        dataset_clean = combined_dataset(:, valid_mask);
        cleaned_size = size(dataset_clean, 2);
        
        fprintf('  Removed %d contracts (%.1f%%)\n', original_size - cleaned_size, ...
                100*(original_size - cleaned_size)/original_size);
        
        % Determine output filename based on multi-noise
        if MULTI_NOISE_ENABLED
            noise_percent = round(current_noise_level * 100);
            output_file_current = sprintf('%s/%s_%dpct.csv', output_folder, output_filename, noise_percent);
        else
            output_file_current = sprintf('%s/%s.csv', output_folder, output_filename);
        end
        
        % Save to CSV
        headers = {'S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'sigma', 'V'}';
        dataset_enhanced = [headers'; num2cell(dataset_clean')];
        writecell(dataset_enhanced, output_file_current);
        
        % Save GARCH parameter sets (for multi-noise, save for each iteration with noise level suffix)
        if MULTI_NOISE_ENABLED
            fprintf('Saving GARCH parameter sets for noise level %.2f%%...\n', current_noise_level*100);
            noise_percent = round(current_noise_level * 100);
            save_param_sets(output_folder, param_sets, model_type, noise_percent);
        elseif noise_idx == 1
            fprintf('Saving GARCH parameter sets...\n');
            save_param_sets(output_folder, param_sets, model_type);
        end
        
        % Save asset price files if enabled (for multi-noise, save for each iteration with noise level suffix)
        if config.output.save_asset_prices
            if MULTI_NOISE_ENABLED
                fprintf('Saving asset price files for noise level %.2f%%...\n', current_noise_level*100);
                noise_percent = round(current_noise_level * 100);
                save_asset_files(output_folder, all_S_paths, output_filename, noise_percent);
            elseif noise_idx == 1
                fprintf('Saving asset price files...\n');
                save_asset_files(output_folder, all_S_paths, output_filename);
            end
        end
        
        %% PERFORMANCE REPORT
        fprintf('\n=================================================================\n');
        if MULTI_NOISE_ENABLED
            fprintf('      NOISE LEVEL %.2f%% - PERFORMANCE SUMMARY\n', current_noise_level*100);
        else
            fprintf('            PERFORMANCE SUMMARY\n');
        end
        fprintf('=================================================================\n');
        fprintf('Total contracts generated: %d\n', original_size);
        fprintf('Valid contracts: %d\n', cleaned_size);
        fprintf('Total processing time: %.2f minutes\n', total_time / 60);
        fprintf('Throughput: %.1f contracts/second\n', cleaned_size / total_time);
        fprintf('Output file: %s (%.2f MB)\n', output_file_current, dir(output_file_current).bytes / 1e6);
        if current_noise_level > 0
            noise_str = 'YES';
        else
            noise_str = 'NO';
        end
        fprintf('Noise applied: %s\n', noise_str);
        fprintf('=================================================================\n');
    end

end

%% HELPER FUNCTIONS

function dataset = process_parameter_set_HN(params, M, N, S0, Z, r, NUM_DAYS, ...
                                           m_h, gamma_h, m_x, gamma_x, ...
                                           all_moneyness, T_years, BATCH_SIZE, ...
                                           NOISE_LEVEL, NOISE_SEED, CONTRACTS_PER_DAY, all_S_paths, ps)
    % Process parameter set for HN model
    alpha = params.alpha;
    beta = params.beta;
    omega = params.omega;
    gamma = params.gamma;
    lambda = params.lambda;
    
    % Generate GARCH paths using HN
    [S_paths, h0_paths] = mcHN(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);
    all_S_paths{ps} = S_paths;
    
    % Build trees and cache for HN
    c = gamma + lambda + 0.5;
    [hd, qhd] = genhDelta(h0_paths(1), beta, alpha, c, omega, m_h, gamma_h);
    [nodes_ht] = TreeNodes_ht_HN(m_h, hd, qhd, gamma_h, alpha, beta, c, omega, N+1);
    
    % Cache trees for all days
    tree_cache = cell(NUM_DAYS, 1);
    for day = 1:NUM_DAYS
        step = floor(size(S_paths, 1) / NUM_DAYS);
        S_current = S_paths(min(day * step, size(S_paths, 1)), 1);
        
        [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_HN(m_x, gamma_x, r, hd, qhd, S_current, alpha, beta, c, omega, N);
        [q_Xt, P_Xt, ~] = Prob_Xt(nodes_ht, qhd, nodes_Xt, S_current, r, alpha, beta, c, omega);
        nodes_S = exp(nodes_Xt);
        
        tree_cache{day} = struct('nodes_S', nodes_S, 'P_Xt', P_Xt, 'q_Xt', q_Xt, ...
                                 'S_current', S_current);
    end
    
    % Process contracts with noise
    dataset = process_contracts_with_noise(tree_cache, all_moneyness, T_years, r, N, m_x, gamma_x, ...
                                          alpha, beta, omega, gamma, lambda, ...
                                          CONTRACTS_PER_DAY, NUM_DAYS, BATCH_SIZE, ...
                                          NOISE_LEVEL, NOISE_SEED, false);
end

function dataset = process_parameter_set_DUAN(params, M, N, S0, Z, r, NUM_DAYS, ...
                                             m_h, gamma_h, m_x, gamma_x, ...
                                             all_moneyness, T_years, BATCH_SIZE, ...
                                             NOISE_LEVEL, NOISE_SEED, CONTRACTS_PER_DAY, all_S_paths, ps)
    % Process parameter set for DUAN model
    alpha = params.alpha;
    beta = params.beta;
    omega = params.omega;
    theta = params.theta;
    lambda = params.lambda;
    
    % Generate GARCH paths using DUAN
    addpath('Duan_Garch');
    [S_paths, h0_paths] = mcDuan(M, N, S0, Z, r, omega, alpha, beta, theta, lambda);
    all_S_paths{ps} = S_paths;
    
    % Build trees and cache for DUAN
    % Duan parameters: beta_0=omega, beta_1=beta, beta_2=alpha
    [hd, qhd] = genhDelta(h0_paths(1), beta, alpha, theta + lambda + 0.5, omega, m_h, gamma_h);
    
    % Get h tree nodes and moments
    [nodes_ht, qht, hmom1, hmom2, hmom3, hmom4_app] = TreeNodes_ht_D(m_h, h0_paths(1), gamma_h, omega, beta, alpha, theta, lambda, N+1);
    
    % Cache trees for all days
    tree_cache = cell(NUM_DAYS, 1);
    for day = 1:NUM_DAYS
        step = floor(size(S_paths, 1) / NUM_DAYS);
        S_current = S_paths(min(day * step, size(S_paths, 1)), 1);
        
        % Get X tree nodes
        [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_D(m_x, gamma_x, r, h0_paths(1), omega, beta, alpha, theta, lambda, N, hmom1, hmom2);
        nodes_Xt = nodes_Xt + log(S_current);
        
        % Get transition probabilities using Probility_Xt2
        [q_Xt, P_Xt, ~] = Probility_Xt2(nodes_ht, qht, nodes_Xt, S_current, r, omega, beta, alpha, theta, lambda);
        nodes_S = exp(nodes_Xt);
        
        tree_cache{day} = struct('nodes_S', nodes_S, 'P_Xt', P_Xt, 'q_Xt', q_Xt, ...
                                 'S_current', S_current);
    end
    
    % Process contracts with noise - use theta as gamma for dataset storage
    dataset = process_contracts_with_noise(tree_cache, all_moneyness, T_years, r, N, m_x, gamma_x, ...
                                          alpha, beta, omega, theta, lambda, ...
                                          CONTRACTS_PER_DAY, NUM_DAYS, BATCH_SIZE, ...
                                          NOISE_LEVEL, NOISE_SEED, false);
    rmpath('Duan_Garch');
end

function dataset = process_parameter_set_GJR(params, M, N, S0, Z, r, NUM_DAYS, ...
                                            m_h, gamma_h, m_x, gamma_x, ...
                                            all_moneyness, T_years, BATCH_SIZE, ...
                                            NOISE_LEVEL, NOISE_SEED, CONTRACTS_PER_DAY, all_S_paths, ps)
    % Process parameter set for GJR model
    alpha = params.alpha;
    beta = params.beta;
    omega = params.omega;
    lambda = params.lambda;
    
    % Generate GARCH paths using GJR
    addpath('GJR_Garch');
    [S_paths, h0_paths] = mcGJR(M, N, S0, Z, r, omega, alpha, beta, lambda);
    all_S_paths{ps} = S_paths;
    
    % Build trees and cache for GJR
    c = 0.5;  % For GJR, use a simple c value
    [hd, qhd] = genhDelta(h0_paths(1), beta, alpha, c, omega, m_h, gamma_h);
    [nodes_ht, p, hmom1, hmom2, hmom3, hmom4] = TreeNodes_ht_GJR(m_h, h0_paths(1), gamma_h, omega, beta, alpha, lambda, N+1);
    
    % Cache trees for all days
    tree_cache = cell(NUM_DAYS, 1);
    for day = 1:NUM_DAYS
        step = floor(size(S_paths, 1) / NUM_DAYS);
        S_current = S_paths(min(day * step, size(S_paths, 1)), 1);
        
        [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_GJR(m_x, m_h, gamma_x, r, omega, alpha, lambda, beta, h0_paths(1), N, hmom1, hmom2);
        % Use GJR's Probility_Xt: parameters are w (omega), beta, alpha, lambda
        [q_Xt, P_Xt, ~] = Probility_Xt(nodes_ht, qhd, nodes_Xt, S_current, r, omega, beta, alpha, lambda);
        nodes_S = exp(nodes_Xt);
        
        tree_cache{day} = struct('nodes_S', nodes_S, 'P_Xt', P_Xt, 'q_Xt', q_Xt, ...
                                 'S_current', S_current);
    end
    
    % Process contracts with noise - store 0 in gamma column (not used), lambda in lambda
    dataset = process_contracts_with_noise(tree_cache, all_moneyness, T_years, r, N, m_x, gamma_x, ...
                                          alpha, beta, omega, 0, lambda, ...
                                          CONTRACTS_PER_DAY, NUM_DAYS, BATCH_SIZE, ...
                                          NOISE_LEVEL, NOISE_SEED, false);
    rmpath('GJR_Garch');
end

function param_sets = generate_param_sets(model_type, num_sets)
    % Generate parameter sets for specified model type using proper parameter ranges
    % Instead of baselines, samples from empirically-derived ranges with stationarity checks
    
    param_sets = struct('alpha', {}, 'beta', {}, 'omega', {}, 'gamma', {}, 'theta', {}, 'lambda', {});
    
    if strcmpi(model_type, 'HN')
        % HN Model parameter ranges
        % Based on genHNGarchParams.m
        omegaRange = [1e-7, 1e-6];
        alphaRange = [1.15e-6, 1.36e-6];
        betaRange = [0.75, 0.85];
        gammaRange = [50, 150];
        lambdaRange = [0.4, 0.6];
        
        for i = 1:num_sets
            valid = false;
            attempts = 0;
            while ~valid && attempts < 1000
                omega = omegaRange(1) + (omegaRange(2) - omegaRange(1)) * rand();
                alpha = alphaRange(1) + (alphaRange(2) - alphaRange(1)) * rand();
                beta = betaRange(1) + (betaRange(2) - betaRange(1)) * rand();
                gamma = gammaRange(1) + (gammaRange(2) - gammaRange(1)) * rand();
                lambda = lambdaRange(1) + (lambdaRange(2) - lambdaRange(1)) * rand();
                
                % Stationarity: beta + alpha * gamma^2 < 1 and alpha + beta < 1
                if (beta + alpha * gamma^2 < 1) && (alpha + beta < 1)
                    param_sets(i).alpha = alpha;
                    param_sets(i).beta = beta;
                    param_sets(i).omega = omega;
                    param_sets(i).gamma = gamma;
                    param_sets(i).theta = 0;  % Not used in HN
                    param_sets(i).lambda = lambda;
                    valid = true;
                end
                attempts = attempts + 1;
            end
            if ~valid
                warning('Failed to generate valid HN parameters for set %d after 1000 attempts', i);
            end
        end
        
    elseif strcmpi(model_type, 'DUAN')
        % DUAN Model parameter ranges
        % Based on genDuanGarchParams.m
        omegaRange = [1e-7, 1e-6];
        alphaRange = [1.15e-6, 1.36e-6];
        betaRange = [0.75, 0.85];
        thetaRange = [0.25, 0.5];
        lambdaRange = [0.4, 0.6];
        
        for i = 1:num_sets
            valid = false;
            attempts = 0;
            while ~valid && attempts < 1000
                omega = omegaRange(1) + (omegaRange(2) - omegaRange(1)) * rand();
                alpha = alphaRange(1) + (alphaRange(2) - alphaRange(1)) * rand();
                beta = betaRange(1) + (betaRange(2) - betaRange(1)) * rand();
                theta = thetaRange(1) + (thetaRange(2) - thetaRange(1)) * rand();
                lambda = lambdaRange(1) + (lambdaRange(2) - lambdaRange(1)) * rand();
                
                % Stationarity: beta + alpha * theta^2 < 1 and alpha + beta < 1
                if (beta + alpha * theta^2 < 1) && (alpha + beta < 1)
                    param_sets(i).alpha = alpha;
                    param_sets(i).beta = beta;
                    param_sets(i).omega = omega;
                    param_sets(i).gamma = 0;  % Not used in DUAN
                    param_sets(i).theta = theta;
                    param_sets(i).lambda = lambda;
                    valid = true;
                end
                attempts = attempts + 1;
            end
            if ~valid
                warning('Failed to generate valid DUAN parameters for set %d after 1000 attempts', i);
            end
        end
        
    elseif strcmpi(model_type, 'GJR')
        % GJR Model parameter ranges
        % Based on genGJRGarchParams.m (current version)
        omegaRange = [1e-7, 1e-6];
        alphaRange = [0.01, 0.15];
        betaRange = [0.75, 0.85];
        lambdaRange = [0.01, 0.15];
        
        for i = 1:num_sets
            valid = false;
            attempts = 0;
            while ~valid && attempts < 1000
                omega = omegaRange(1) + (omegaRange(2) - omegaRange(1)) * rand();
                alpha = alphaRange(1) + (alphaRange(2) - alphaRange(1)) * rand();
                beta = betaRange(1) + (betaRange(2) - betaRange(1)) * rand();
                lambda = lambdaRange(1) + (lambdaRange(2) - lambdaRange(1)) * rand();
                
                % Stationarity: beta + alpha + lambda/2 < 1
                if (beta + alpha + (lambda/2) < 1) && (omega > 0) && ...
                   (alpha >= 0) && (beta >= 0) && (alpha + lambda >= 0)
                    param_sets(i).alpha = alpha;
                    param_sets(i).beta = beta;
                    param_sets(i).omega = omega;
                    param_sets(i).gamma = 0;  % Not used in GJR
                    param_sets(i).theta = 0;  % Not used in GJR
                    param_sets(i).lambda = lambda;
                    valid = true;
                end
                attempts = attempts + 1;
            end
            if ~valid
                warning('Failed to generate valid GJR parameters for set %d after 1000 attempts', i);
            end
        end
        
    else
        error('Unknown model type: %s', model_type);
    end
end

function dataset = process_contracts_with_noise(tree_cache, all_moneyness, T_years, r, N, m_x, gamma_x, ...
                                               alpha, beta, omega, gamma, lambda, ...
                                               CONTRACTS_PER_DAY, NUM_DAYS, BATCH_SIZE, ...
                                               NOISE_LEVEL, NOISE_SEED, ENABLE_NESTED_PARFOR)
    num_contracts = CONTRACTS_PER_DAY * NUM_DAYS;
    dataset = zeros(12, num_contracts * 2);
    dataset_idx = 1;
    
    % Initialize noise random stream if needed
    if NOISE_LEVEL > 0
        if NOISE_SEED >= 0
            noise_rng = RandStream('mt19937ar', 'Seed', NOISE_SEED);
        else
            noise_rng = RandStream('mt19937ar', 'Seed', randi([0, 2^31-1]));
        end
    end
    
    num_batches = ceil(num_contracts / BATCH_SIZE);
    batch_times = zeros(num_batches, 1);
    batch_valid_counts = zeros(num_batches, 1);
    
    for batch = 1:num_batches
        batch_start_time = tic;
        batch_start = (batch - 1) * BATCH_SIZE + 1;
        batch_end = min(batch * BATCH_SIZE, num_contracts);
        batch_size_actual = batch_end - batch_start + 1;
        
        batch_calls = zeros(12, batch_size_actual);
        batch_puts = zeros(12, batch_size_actual);
        
        % Optional nested parallelization for contracts within batch
        if ENABLE_NESTED_PARFOR && batch_size_actual > 10
            % Parallel processing of contracts in batch
            batch_calls_temp = zeros(12, batch_size_actual);
            batch_puts_temp = zeros(12, batch_size_actual);
            
            parfor i = 1:batch_size_actual
                contract_idx = batch_start + i - 1;
                day = ceil(contract_idx / CONTRACTS_PER_DAY);
                contract_in_day = mod(contract_idx - 1, CONTRACTS_PER_DAY) + 1;
                
                if day > NUM_DAYS
                    continue;
                end
                
                tree_data = tree_cache{day};
                S_current = tree_data.S_current;
                nodes_S = tree_data.nodes_S;
                P_Xt = tree_data.P_Xt;
                q_Xt = tree_data.q_Xt;
                
                moneyness = all_moneyness(contract_idx);
                K_strike = moneyness * S_current;
                T_maturity = T_years(mod(contract_in_day - 1, length(T_years)) + 1);
                
                try
                    % Price options
                    [V_C, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, 1);
                    [V_P, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, -1);
                    
                    if V_C < 0 || V_P < 0 || isnan(V_C) || isnan(V_P)
                        continue;
                    end
                    
                    % Apply noise to prices (multiplicative) - use shared seed
                    if NOISE_LEVEL > 0
                        rng_temp = RandStream('mt19937ar', 'Seed', NOISE_SEED + contract_idx);
                        noise_c = 1 + NOISE_LEVEL * randn(rng_temp);
                        noise_p = 1 + NOISE_LEVEL * randn(rng_temp);
                        V_C = V_C * noise_c;
                        V_P = V_P * noise_p;
                    end
                    
                    % Calculate implied volatility
                    [impl_c, ~, ~] = impvol_fzero(S_current, K_strike, T_maturity, r, V_C, 1, N, m_x, gamma_x);
                    [impl_p, ~, ~] = impvol_fzero(S_current, K_strike, T_maturity, r, V_P, -1, N, m_x, gamma_x);
                    
                    batch_calls_temp(:, i) = [S_current; moneyness; r; T_maturity; 1; ...
                                        alpha; beta; omega; gamma; lambda; impl_c; V_C];
                    batch_puts_temp(:, i) = [S_current; moneyness; r; T_maturity; -1; ...
                                       alpha; beta; omega; gamma; lambda; impl_p; V_P];
                catch
                    continue;
                end
            end
            batch_calls = batch_calls_temp;
            batch_puts = batch_puts_temp;
        else
            % Sequential processing (original code)
            for i = 1:batch_size_actual
                contract_idx = batch_start + i - 1;
                day = ceil(contract_idx / CONTRACTS_PER_DAY);
                contract_in_day = mod(contract_idx - 1, CONTRACTS_PER_DAY) + 1;
                
                if day > NUM_DAYS
                    continue;
                end
                
                tree_data = tree_cache{day};
                S_current = tree_data.S_current;
                nodes_S = tree_data.nodes_S;
                P_Xt = tree_data.P_Xt;
                q_Xt = tree_data.q_Xt;
                
                moneyness = all_moneyness(contract_idx);
                K_strike = moneyness * S_current;
                T_maturity = T_years(mod(contract_in_day - 1, length(T_years)) + 1);
                
                try
                    % Price options
                    [V_C, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, 1);
                    [V_P, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, -1);
                    
                    if V_C < 0 || V_P < 0 || isnan(V_C) || isnan(V_P)
                        continue;
                    end
                    
                    % Apply noise to prices (multiplicative)
                    if NOISE_LEVEL > 0
                        noise_c = 1 + NOISE_LEVEL * randn(noise_rng);
                        noise_p = 1 + NOISE_LEVEL * randn(noise_rng);
                        V_C = V_C * noise_c;
                        V_P = V_P * noise_p;
                    end
                    
                    % Calculate implied volatility
                    [impl_c, ~, ~] = impvol_fzero(S_current, K_strike, T_maturity, r, V_C, 1, N, m_x, gamma_x);
                    [impl_p, ~, ~] = impvol_fzero(S_current, K_strike, T_maturity, r, V_P, -1, N, m_x, gamma_x);
                    
                    batch_calls(:, i) = [S_current; moneyness; r; T_maturity; 1; ...
                                        alpha; beta; omega; gamma; lambda; impl_c; V_C];
                    batch_puts(:, i) = [S_current; moneyness; r; T_maturity; -1; ...
                                       alpha; beta; omega; gamma; lambda; impl_p; V_P];
                catch
                    continue;
                end
            end
        end
        
        batch_combined = [batch_calls, batch_puts];
        valid_batch = batch_combined(:, any(batch_combined, 1));
        
        n_valid = size(valid_batch, 2);
        dataset(:, dataset_idx:dataset_idx+n_valid-1) = valid_batch;
        dataset_idx = dataset_idx + n_valid;
        
        % Record batch performance
        batch_times(batch) = toc(batch_start_time);
        batch_valid_counts(batch) = n_valid;
        
        % Show progress every 5 batches or at end
        if mod(batch, max(1, round(num_batches/10))) == 0 || batch == num_batches
            elapsed = sum(batch_times(1:batch));
            completed_contracts = sum(batch_valid_counts(1:batch));
            throughput = completed_contracts / elapsed;
            fprintf('    Batch %d/%d: %d valid contracts, %.1f contracts/sec\n', ...
                    batch, num_batches, n_valid, throughput);
        end
    end
    
    dataset = dataset(:, 1:dataset_idx-1);
end

function config = read_toml_config_matlab(filename)
    % Simple TOML parser for MATLAB (subset of TOML features needed)
    
    fid = fopen(filename, 'r');
    if fid < 0
        error('Cannot open file: %s', filename);
    end
    
    config = struct();
    current_section = '';
    
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            % Remove leading/trailing whitespace
            line = strtrim(line);
            
            % Skip empty lines and comments
            if isempty(line) || line(1) == '#'
                continue;
            end
            
            % Parse section headers [section]
            if line(1) == '[' && line(end) == ']'
                section_name = line(2:end-1);
                section_parts = strsplit(section_name, '.');
                
                % Handle nested sections like [parameters.true]
                if length(section_parts) == 2
                    current_section = section_parts{1};
                    subsection = section_parts{2};
                    if ~isfield(config, current_section)
                        config.(current_section) = struct();
                    end
                    config.(current_section).(subsection) = struct();
                else
                    current_section = section_name;
                    config.(current_section) = struct();
                end
                continue;
            end
            
            % Parse key-value pairs
            if contains(line, '=')
                parts = strsplit(line, '=');
                if length(parts) >= 2
                    key = strtrim(parts{1});
                    value_str = strtrim(parts{2});
                    
                    % Remove comments
                    comment_idx = find(value_str == '#', 1);
                    if ~isempty(comment_idx)
                        value_str = strtrim(value_str(1:comment_idx-1));
                    end
                    
                    % Parse value
                    value = parse_toml_value(value_str);
                    
                    % Store in appropriate section
                    if ~isempty(current_section)
                        if contains(current_section, '.')
                            % Handle nested sections
                            parts = strsplit(current_section, '.');
                            section = parts{1};
                            subsection = parts{2};
                            if ~isfield(config, section)
                                config.(section) = struct();
                            end
                            if ~isfield(config.(section), subsection)
                                config.(section).(subsection) = struct();
                            end
                            config.(section).(subsection).(key) = value;
                        else
                            config.(current_section).(key) = value;
                        end
                    else
                        config.(key) = value;
                    end
                end
            end
        end
    end
    
    fclose(fid);
end

function value = parse_toml_value(str)
    % Parse TOML values
    str = strtrim(str);
    
    % String (quoted)
    if (str(1) == '"' && str(end) == '"')
        value = str(2:end-1);
        return;
    end
    if (str(1) == '''' && str(end) == '''')
        value = str(2:end-1);
        return;
    end
    
    % Boolean
    if strcmp(lower(str), 'true')
        value = true;
        return;
    elseif strcmp(lower(str), 'false')
        value = false;
        return;
    end
    
    % Array
    if str(1) == '[' && str(end) == ']'
        inner = str(2:end-1);
        elements = strsplit(inner, ',');
        value = [];
        for i = 1:length(elements)
            elem = strtrim(elements{i});
            num_val = str2double(elem);
            if ~isnan(num_val)
                value = [value, num_val];
            end
        end
        return;
    end
    
    % Number
    num_val = str2double(str);
    if ~isnan(num_val)
        value = num_val;
        return;
    end
    
    % Default to string
    value = str;
end

function save_param_sets(output_folder, param_sets, model_type, noise_percent)
    % Save GARCH parameter sets to CSV and JSON formats
    
    num_sets = length(param_sets);
    
    % Prepare parameter matrix
    params_matrix = zeros(num_sets, 5);
    for i = 1:num_sets
        params_matrix(i, 1) = param_sets(i).omega;
        params_matrix(i, 2) = param_sets(i).alpha;
        params_matrix(i, 3) = param_sets(i).beta;
        params_matrix(i, 4) = param_sets(i).gamma;
        params_matrix(i, 5) = param_sets(i).lambda;
    end
    
    % Save as CSV
    headers = {'set_id', 'omega', 'alpha', 'beta', 'gamma', 'lambda'}';
    set_ids = (1:num_sets)';
    csv_data = [headers'; num2cell([set_ids, params_matrix])];
    
    % Include noise level in filename if provided
    if nargin >= 4 && ~isempty(noise_percent)
        csv_file = sprintf('%s/garch_parameters_%s_%dpct.csv', output_folder, lower(model_type), noise_percent);
    else
        csv_file = sprintf('%s/garch_parameters_%s.csv', output_folder, lower(model_type));
    end
    
    writecell(csv_data, csv_file);
    
    fprintf('  ✓ Saved parameters: %s\n', csv_file);
end

function save_asset_files(output_folder, all_S_paths, output_filename, noise_percent)
    % Save asset price paths to a single CSV file per noise level
    % Combines all asset paths from all parameter sets into one file
    
    num_sets = length(all_S_paths);
    
    % Combine all S_paths into a single matrix
    % Stack them horizontally or vertically depending on structure
    % Assuming each S_paths is (M x N) where M = time steps, N = paths
    
    combined_S_paths = [];
    for ps = 1:num_sets
        S_paths = all_S_paths{ps};
        S_paths = double(S_paths);  % Ensure decimal format
        
        if ps == 1
            combined_S_paths = S_paths;
        else
            % Concatenate horizontally (more paths)
            combined_S_paths = [combined_S_paths, S_paths];
        end
    end
    
    % Determine filename
    if nargin >= 4 && ~isempty(noise_percent)
        % Multi-noise: include noise level in filename
        csv_file = sprintf('%s/%s_%dpct_asset_prices.csv', output_folder, output_filename, noise_percent);
    elseif nargin >= 3 && ~isempty(output_filename)
        % Single noise: use output filename prefix only
        csv_file = sprintf('%s/%s_asset_prices.csv', output_folder, output_filename);
    else
        % No prefix
        csv_file = sprintf('%s/asset_prices.csv', output_folder);
    end
    
    writematrix(combined_S_paths, csv_file);
    fprintf('  ✓ Saved asset prices: %s (%d x %d from %d parameter sets)\n', ...
            csv_file, size(combined_S_paths, 1), size(combined_S_paths, 2), num_sets);
end

function gpu_info = detect_gpu_type()
    % Detect GPU type and availability
    % Returns struct with gpu_type, is_available, device_name, memory_gb
    
    gpu_info = struct();
    gpu_info.gpu_type = 'NONE';
    gpu_info.is_available = false;
    gpu_info.device_name = 'CPU Only';
    gpu_info.memory_gb = 0;
    
    % Try NVIDIA CUDA GPU
    try
        nvidia_gpu = gpuDevice();
        gpu_info.gpu_type = 'NVIDIA_CUDA';
        gpu_info.is_available = true;
        gpu_info.device_name = nvidia_gpu.Name;
        gpu_info.memory_gb = double(nvidia_gpu.AvailableMemory) / 1e9;
        return;
    catch
        % NVIDIA not available, continue to check for AMD
    end
    
    % Try to detect AMD GPU
    try
        [status, output] = system('rocm-smi --showproductname 2>/dev/null');
        if status == 0 && ~isempty(output)
            gpu_info.gpu_type = 'AMD_ROCm';
            gpu_info.is_available = true;
            gpu_info.device_name = strtrim(output(1:min(80, length(output))));
            % Note: ROCm memory detection would require additional system queries
            return;
        end
    catch
        % ROCm not available
    end
    
    % Try generic GPU detection via lspci
    try
        [status, output] = system('lspci 2>/dev/null | grep -i "VGA\|3D"');
        if status == 0 && ~isempty(output)
            if ~isempty(strfind(output, 'AMD')) || ~isempty(strfind(output, 'ATI'))
                gpu_info.gpu_type = 'AMD_GENERIC';
                gpu_info.device_name = strtrim(output(1:min(80, length(output))));
            elseif ~isempty(strfind(output, 'NVIDIA'))
                gpu_info.gpu_type = 'NVIDIA_GENERIC';
                gpu_info.device_name = strtrim(output(1:min(80, length(output))));
            else
                gpu_info.gpu_type = 'GPU_UNKNOWN';
                gpu_info.device_name = strtrim(output(1:min(80, length(output))));
            end
        end
    catch
        % Generic detection failed
    end
end
