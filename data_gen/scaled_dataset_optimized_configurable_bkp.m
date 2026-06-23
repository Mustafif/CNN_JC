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
% Time convention: T_maturity is in trading DAYS throughout, matching daily r.
% No /252 conversion is applied anywhere in this file.
%
% Optional Name-Value Pairs:
%   'config'            - Path to TOML config file (default: 'scaled_dataset_config.toml')
%   'model_type'        - Override model type from config (default: uses config value)
%   'noise_level'       - Override noise level from config
%   'contracts_per_day' - Override contracts per day
%   'num_days'          - Override number of days
%   'output_folder'     - Override output folder path

    fprintf('=================================================================\n');
    fprintf('  OPTIMIZED GARCH OPTION PRICING DATASET GENERATOR\n');
    fprintf('  (Config-based with Noise Support)\n');
    fprintf('=================================================================\n\n');

    %% PARSE INPUT ARGUMENTS
    p = inputParser;
    addParameter(p, 'config', 'scaled_dataset_config.toml', @ischar);
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
        mem_info = memory();
        estimated_contracts_per_500mb = 500 / 0.0005;
        BATCH_SIZE = max(100, min(1000, round(estimated_contracts_per_500mb)));
    else
        BATCH_SIZE = config.scaling.batch_size;
    end

    NUM_PARAM_SETS = config.scaling.num_param_sets;

    % Market parameters
    N = config.market.N;
    r = config.market.r;
    S0 = config.market.S0;
    moneyness_range = config.market.moneyness_range;

    % maturity_days stays in days - no /252 conversion
    original_maturity_days = config.market.maturity_days;

    % Willow tree parameters
    m_h = config.willow_tree.m_h;
    m_x = config.willow_tree.m_x;
    gamma_h = config.willow_tree.gamma_h;
    gamma_x = config.willow_tree.gamma_x;

    % Performance parameters
    NUM_WORKERS = config.performance.num_workers;

    % Noise parameters
    NOISE_LEVEL = p.Results.noise_level;
    if isempty(NOISE_LEVEL)
        NOISE_LEVEL = config.noise.level;
    end
    NOISE_SEED = config.noise.seed;

    % Multi-noise generation parameters
    MULTI_NOISE_ENABLED = false;
    MULTI_NOISE_LEVELS = [0, 0.01, 0.05];
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

    ENABLE_NESTED_PARFOR = false;
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
    fprintf('  Maturity days: %s\n', mat2str(original_maturity_days));
    fprintf('  Time convention: DAILY (T stored in trading days, matches daily r)\n');

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
    fprintf('  Batch size: %d contracts\n', BATCH_SIZE);

    estimated_contracts_per_param = CONTRACTS_PER_DAY * NUM_DAYS;
    recommended_batch_size = max(100, round(estimated_contracts_per_param / 5));
    if abs(BATCH_SIZE - recommended_batch_size) > 50
        fprintf('  Note: Recommended batch size for this config: %d\n', recommended_batch_size);
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
            fprintf('  Parallel pool started\n');
        catch
            fprintf('  Could not start parallel pool, using serial processing\n');
        end
    else
        fprintf('  Using existing parallel pool (%d workers)\n', pool_obj.NumWorkers);
    end

    % Create output folder
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

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

    % Initialize noise random stream
    if NOISE_LEVEL > 0
        if NOISE_SEED >= 0
            noise_rng = RandStream('mt19937ar', 'Seed', NOISE_SEED); %#ok<NASGU>
        else
            noise_rng = RandStream('mt19937ar', 'Seed', randi([0, 2^31-1])); %#ok<NASGU>
        end
    end

    fprintf('Random data generated\n\n');

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

            switch upper(model_type)
                case 'HN'
                    [dataset, S_paths] = process_parameter_set_HN(params, M, N, S0, Z, r, NUM_DAYS, ...
                                                                   m_h, gamma_h, m_x, gamma_x, ...
                                                                   all_moneyness, original_maturity_days, ...
                                                                   BATCH_SIZE, current_noise_level, ...
                                                                   NOISE_SEED, CONTRACTS_PER_DAY, ...
                                                                   ENABLE_NESTED_PARFOR);
                case 'DUAN'
                    [dataset, S_paths] = process_parameter_set_DUAN(params, M, N, S0, Z, r, NUM_DAYS, ...
                                                                    m_h, gamma_h, m_x, gamma_x, ...
                                                                    all_moneyness, original_maturity_days, ...
                                                                    BATCH_SIZE, current_noise_level, ...
                                                                    NOISE_SEED, CONTRACTS_PER_DAY, ...
                                                                    ENABLE_NESTED_PARFOR);
                case 'GJR'
                    [dataset, S_paths] = process_parameter_set_GJR(params, M, N, S0, Z, r, NUM_DAYS, ...
                                                                   m_h, gamma_h, m_x, gamma_x, ...
                                                                   all_moneyness, original_maturity_days, ...
                                                                   BATCH_SIZE, current_noise_level, ...
                                                                   NOISE_SEED, CONTRACTS_PER_DAY, ...
                                                                   ENABLE_NESTED_PARFOR);
                otherwise
                    error('Unknown model type: %s', model_type);
            end

            all_datasets{ps} = dataset;
            all_S_paths{ps} = S_paths;
            all_processing_times(ps) = toc(ps_start);

            fprintf('  [Worker] Parameter set %d complete: %.2f seconds\n', ps, all_processing_times(ps));
        end

        total_time = toc(total_start);

        fprintf('\nAll parameter sets processed in %.2f seconds\n', total_time);
        fprintf('  Average time per set: %.2f seconds\n', mean(all_processing_times));
        fprintf('  Parallel speedup: %.2fx (estimated)\n', sum(all_processing_times) / total_time);

        %% COMBINE AND SAVE RESULTS
        fprintf('Combining and saving results...\n');

        combined_dataset = horzcat(all_datasets{:});

        fprintf('Cleaning dataset...\n');
        original_size = size(combined_dataset, 2);

        % T column (row 4) is now in days, so minimum meaningful maturity is 1 day
        valid_mask = (combined_dataset(12, :) >= 0.5) & ...
                     (combined_dataset(4, :) >= 1) & ...
                     all(~isnan(combined_dataset), 1) & ...
                     all(~isinf(combined_dataset), 1);

        dataset_clean = combined_dataset(:, valid_mask);
        cleaned_size = size(dataset_clean, 2);

        fprintf('  Removed %d contracts (%.1f%%)\n', original_size - cleaned_size, ...
                100*(original_size - cleaned_size)/original_size);

        if MULTI_NOISE_ENABLED
            noise_percent = round(current_noise_level * 100);
            output_file_current = sprintf('%s/%s_%dpct.csv', output_folder, output_filename, noise_percent);
        else
            output_file_current = sprintf('%s/%s.csv', output_folder, output_filename);
        end

        % T column is in trading days
        headers = {'S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'sigma', 'V'}';
        dataset_enhanced = [headers'; num2cell(dataset_clean')];
        writecell(dataset_enhanced, output_file_current);

        if MULTI_NOISE_ENABLED
            noise_percent = round(current_noise_level * 100);
            save_param_sets(output_folder, param_sets, model_type, noise_percent);
        elseif noise_idx == 1
            save_param_sets(output_folder, param_sets, model_type);
        end

        if config.output.save_asset_prices
            if MULTI_NOISE_ENABLED
                noise_percent = round(current_noise_level * 100);
                save_asset_files(output_folder, all_S_paths, output_filename, noise_percent);
            elseif noise_idx == 1
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
        if current_noise_level > 0; noise_str = 'YES'; else; noise_str = 'NO'; end
        fprintf('Noise applied: %s\n', noise_str);
        fprintf('=================================================================\n');
    end

end

%% -------------------------------------------------------------------------
%  MODEL-SPECIFIC PARAMETER SET PROCESSORS
%% -------------------------------------------------------------------------

function [dataset, S_paths] = process_parameter_set_HN(params, M, N, S0, Z, r, NUM_DAYS, ...
                                                        m_h, gamma_h, m_x, gamma_x, ...
                                                        all_moneyness, original_maturity_days, ...
                                                        BATCH_SIZE, NOISE_LEVEL, NOISE_SEED, ...
                                                        CONTRACTS_PER_DAY, ENABLE_NESTED_PARFOR)
    alpha  = params.alpha;
    beta   = params.beta;
    omega  = params.omega;
    gamma  = params.gamma;
    lambda = params.lambda;

    [S_paths, h0_paths] = mcHN(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);

    c = gamma + lambda + 0.5;
    [hd, qhd]     = genhDelta(h0_paths(1), beta, alpha, c, omega, m_h, gamma_h);
    [nodes_ht]    = TreeNodes_ht_HN(m_h, hd, qhd, gamma_h, alpha, beta, c, omega, N+1);

    tree_cache = cell(NUM_DAYS, 1);
    step = floor(size(S_paths, 1) / NUM_DAYS);
    for day = 1:NUM_DAYS
        S_current = S_paths(min(day * step, size(S_paths, 1)), 1);

        [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_HN(m_x, gamma_x, r, hd, qhd, S_current, alpha, beta, c, omega, N);
        [q_Xt, P_Xt, ~]        = Prob_Xt(nodes_ht, qhd, nodes_Xt, S_current, r, alpha, beta, c, omega);
        nodes_S = exp(nodes_Xt);

        tree_cache{day} = struct('nodes_S', nodes_S, 'P_Xt', P_Xt, 'q_Xt', q_Xt, ...
                                 'S_current', S_current, 'N_steps', N);
    end

    dataset = process_contracts_with_noise(tree_cache, all_moneyness, original_maturity_days, ...
                                           r, N, m_x, gamma_x, alpha, beta, omega, gamma, lambda, ...
                                           CONTRACTS_PER_DAY, NUM_DAYS, BATCH_SIZE, ...
                                           NOISE_LEVEL, NOISE_SEED, ENABLE_NESTED_PARFOR);
end

function [dataset, S_paths] = process_parameter_set_DUAN(params, M, N, S0, Z, r, NUM_DAYS, ...
                                                          m_h, gamma_h, m_x, gamma_x, ...
                                                          all_moneyness, original_maturity_days, ...
                                                          BATCH_SIZE, NOISE_LEVEL, NOISE_SEED, ...
                                                          CONTRACTS_PER_DAY, ENABLE_NESTED_PARFOR)
    alpha  = params.alpha;
    beta   = params.beta;
    omega  = params.omega;
    gamma  = params.gamma;
    lambda = params.lambda;

    addpath('Duan_Garch');
    [S_paths, h0_paths] = mcDuan(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);

    [nodes_ht, qht, hmom1, hmom2, ~, ~] = TreeNodes_ht_D(m_h, h0_paths(1), gamma_h, omega, beta, alpha, gamma, lambda, N+1);
    % X-tree base nodes depend only on h0 params, not S_current. Hoist out of day loop.
    [nodes_Xt_base, ~, ~, ~, ~] = TreeNodes_logSt_D(m_x, gamma_x, r, h0_paths(1), omega, beta, alpha, gamma, lambda, N, hmom1, hmom2);

    tree_cache = cell(NUM_DAYS, 1);
    step = floor(size(S_paths, 1) / NUM_DAYS);
    for day = 1:NUM_DAYS
        S_current = S_paths(min(day * step, size(S_paths, 1)), 1);

        nodes_Xt = nodes_Xt_base + log(S_current);

        [q_Xt, P_Xt, ~] = Probility_Xt2(nodes_ht, qht, nodes_Xt, S_current, r, omega, beta, alpha, gamma, lambda);
        nodes_S = exp(nodes_Xt);

        tree_cache{day} = struct('nodes_S', nodes_S, 'P_Xt', P_Xt, 'q_Xt', q_Xt, ...
                                 'S_current', S_current, 'N_steps', N);
    end

    dataset = process_contracts_with_noise(tree_cache, all_moneyness, original_maturity_days, ...
                                           r, N, m_x, gamma_x, alpha, beta, omega, gamma, lambda, ...
                                           CONTRACTS_PER_DAY, NUM_DAYS, BATCH_SIZE, ...
                                           NOISE_LEVEL, NOISE_SEED, ENABLE_NESTED_PARFOR);
    rmpath('Duan_Garch');
end

function [dataset, S_paths] = process_parameter_set_GJR(params, M, N, S0, Z, r, NUM_DAYS, ...
                                                         m_h, gamma_h, m_x, gamma_x, ...
                                                         all_moneyness, original_maturity_days, ...
                                                         BATCH_SIZE, NOISE_LEVEL, NOISE_SEED, ...
                                                         CONTRACTS_PER_DAY, ENABLE_NESTED_PARFOR)
    alpha  = params.alpha;
    beta   = params.beta;
    omega  = params.omega;
    lambda = params.lambda;

    addpath('GJR_Garch');
    [S_paths, h0_paths] = mcGJR(M, N, S0, Z, r, omega, alpha, beta, lambda);

    h0 = h0_paths(1);
    % h-tree and X-tree nodes depend only on h0, not on S_current.
    % Compute once here; inside the day loop we only shift by log(S_current).
    [nodes_ht, qht, hmom1, hmom2, ~, ~] = TreeNodes_ht_GJR(m_h, h0, gamma_h, omega, beta, alpha, lambda, N+1);
    [nodes_Xt_base, ~, ~, ~, ~]         = TreeNodes_logSt_GJR(m_x, m_h, gamma_x, r, omega, alpha, lambda, beta, h0, N, hmom1, hmom2);

    tree_cache = cell(NUM_DAYS, 1);
    step = floor(size(S_paths, 1) / NUM_DAYS);
    for day = 1:NUM_DAYS
        S_current = S_paths(min(day * step, size(S_paths, 1)), 1);

        nodes_Xt = nodes_Xt_base + log(S_current);

        [q_Xt, P_Xt, ~] = Probility_Xt2(nodes_ht, qht, nodes_Xt, S_current, r, omega, beta, alpha, lambda);
        nodes_S = exp(nodes_Xt);

        tree_cache{day} = struct('nodes_S', nodes_S, 'P_Xt', P_Xt, 'q_Xt', q_Xt, ...
                                 'S_current', S_current, 'N_steps', N);
    end

    % gamma = 0 for GJR (not a model parameter)
    dataset = process_contracts_with_noise(tree_cache, all_moneyness, original_maturity_days, ...
                                           r, N, m_x, gamma_x, alpha, beta, omega, 0, lambda, ...
                                           CONTRACTS_PER_DAY, NUM_DAYS, BATCH_SIZE, ...
                                           NOISE_LEVEL, NOISE_SEED, ENABLE_NESTED_PARFOR);
    rmpath('GJR_Garch');
end

%% -------------------------------------------------------------------------
%  CONTRACT PRICING LOOP
%% -------------------------------------------------------------------------

function dataset = process_contracts_with_noise(tree_cache, all_moneyness, original_maturity_days, ...
                                                r, N, m_x, gamma_x, ...
                                                alpha, beta, omega, gamma, lambda, ...
                                                CONTRACTS_PER_DAY, NUM_DAYS, BATCH_SIZE, ...
                                                NOISE_LEVEL, NOISE_SEED, ENABLE_NESTED_PARFOR)
% All maturities are in trading DAYS. T_maturity passed to American() and
% impvol_fzero() is in days, consistent with daily r.

    num_contracts = CONTRACTS_PER_DAY * NUM_DAYS;
    dataset = zeros(12, num_contracts * 2);
    dataset_idx = 1;

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
        batch_start      = (batch - 1) * BATCH_SIZE + 1;
        batch_end        = min(batch * BATCH_SIZE, num_contracts);
        batch_size_actual = batch_end - batch_start + 1;

        batch_calls = zeros(12, batch_size_actual);
        batch_puts  = zeros(12, batch_size_actual);

        if ENABLE_NESTED_PARFOR && batch_size_actual > 10
            batch_calls_temp = zeros(12, batch_size_actual);
            batch_puts_temp  = zeros(12, batch_size_actual);

            parfor i = 1:batch_size_actual
                contract_idx    = batch_start + i - 1;
                day             = ceil(contract_idx / CONTRACTS_PER_DAY);
                contract_in_day = mod(contract_idx - 1, CONTRACTS_PER_DAY) + 1;

                if day > NUM_DAYS; continue; end

                mat_idx        = mod(contract_in_day - 1, length(original_maturity_days)) + 1;
                base_maT  = original_maturity_days(mat_idx);
                days_elapsed   = day - 1;
                remaining_days = base_maT - days_elapsed;

                % Skip expired contracts
                if remaining_days < 1; continue; end

                % T_maturity in days - no /252
                T_maturity = remaining_days;

                tree_data = tree_cache{day};
                S_current = tree_data.S_current;
                nodes_S   = tree_data.nodes_S;
                P_Xt      = tree_data.P_Xt;
                q_Xt      = tree_data.q_Xt;
                N_iv      = get_tree_steps(tree_data, N);

                moneyness = all_moneyness(contract_idx);
                K_strike  = moneyness * S_current;

                try
                    [V_C, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike,  1);
                    [V_P, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, -1);

                    if V_C < 0 || V_P < 0 || isnan(V_C) || isnan(V_P); continue; end

                    if NOISE_LEVEL > 0
                        rng_temp = RandStream('mt19937ar', 'Seed', NOISE_SEED + contract_idx);
                        V_C = V_C * (1 + NOISE_LEVEL * randn(rng_temp));
                        V_P = V_P * (1 + NOISE_LEVEL * randn(rng_temp));
                    end

                    [impl_c, ~, ~] = impvol_fzero(S_current, K_strike, T_maturity, r, V_C,  1, N_iv, m_x, gamma_x);
                    [impl_p, ~, ~] = impvol_fzero(S_current, K_strike, T_maturity, r, V_P, -1, N_iv, m_x, gamma_x);

                    batch_calls_temp(:, i) = [S_current; moneyness; r; T_maturity;  1; alpha; beta; omega; gamma; lambda; impl_c; V_C];
                    batch_puts_temp(:, i)  = [S_current; moneyness; r; T_maturity; -1; alpha; beta; omega; gamma; lambda; impl_p; V_P];
                catch
                    continue;
                end
            end

            batch_calls = batch_calls_temp;
            batch_puts  = batch_puts_temp;

        else
            for i = 1:batch_size_actual
                contract_idx    = batch_start + i - 1;
                day             = ceil(contract_idx / CONTRACTS_PER_DAY);
                contract_in_day = mod(contract_idx - 1, CONTRACTS_PER_DAY) + 1;

                if day > NUM_DAYS; continue; end

                mat_idx        = mod(contract_in_day - 1, length(original_maturity_days)) + 1;
                base_maT  = original_maturity_days(mat_idx);
                days_elapsed   = day - 1;
                remaining_days = base_maT - days_elapsed;

                % Skip expired contracts
                if remaining_days < 1; continue; end

                % T_maturity in days - no /252
                T_maturity = remaining_days;

                tree_data = tree_cache{day};
                S_current = tree_data.S_current;
                nodes_S   = tree_data.nodes_S;
                P_Xt      = tree_data.P_Xt;
                q_Xt      = tree_data.q_Xt;
                N_iv      = get_tree_steps(tree_data, N);

                moneyness = all_moneyness(contract_idx);
                K_strike  = moneyness * S_current;

                try
                    [V_C, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike,  1);
                    [V_P, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, -1);

                    if V_C < 0 || V_P < 0 || isnan(V_C) || isnan(V_P); continue; end

                    if NOISE_LEVEL > 0
                        noise_c = 1 + NOISE_LEVEL * randn(noise_rng);
                        noise_p = 1 + NOISE_LEVEL * randn(noise_rng);
                        V_C = V_C * noise_c;
                        V_P = V_P * noise_p;
                    end

                    [impl_c, ~, ~] = impvol_fzero(S_current, K_strike, T_maturity, r, V_C,  1, N_iv, m_x, gamma_x);
                    [impl_p, ~, ~] = impvol_fzero(S_current, K_strike, T_maturity, r, V_P, -1, N_iv, m_x, gamma_x);

                    batch_calls(:, i) = [S_current; moneyness; r; T_maturity;  1; alpha; beta; omega; gamma; lambda; impl_c; V_C];
                    batch_puts(:, i)  = [S_current; moneyness; r; T_maturity; -1; alpha; beta; omega; gamma; lambda; impl_p; V_P];
                catch
                    continue;
                end
            end
        end

        batch_combined = [batch_calls, batch_puts];
        valid_batch    = batch_combined(:, any(batch_combined, 1));

        n_valid = size(valid_batch, 2);
        dataset(:, dataset_idx:dataset_idx+n_valid-1) = valid_batch;
        dataset_idx = dataset_idx + n_valid;

        batch_times(batch)        = toc(batch_start_time);
        batch_valid_counts(batch) = n_valid;

        if mod(batch, max(1, round(num_batches/10))) == 0 || batch == num_batches
            elapsed    = sum(batch_times(1:batch));
            completed  = sum(batch_valid_counts(1:batch));
            throughput = completed / elapsed;
            fprintf('    Batch %d/%d: %d valid contracts, %.1f contracts/sec\n', ...
                    batch, num_batches, n_valid, throughput);
        end
    end

    dataset = dataset(:, 1:dataset_idx-1);
end

%% -------------------------------------------------------------------------
%  PARAMETER GENERATION
%% -------------------------------------------------------------------------

function param_sets = generate_param_sets(model_type, num_sets)
    param_sets = struct('alpha', {}, 'beta', {}, 'omega', {}, 'gamma', {}, 'lambda', {});

    if strcmpi(model_type, 'HN')
        omegaRange  = [1e-7,    1e-6  ];
        alphaRange  = [1.15e-6, 1.36e-6];
        betaRange   = [0.75,    0.85  ];
        gammaRange  = [50,      150   ];
        lambdaRange = [0.4,     0.6   ];

        for i = 1:num_sets
            valid = false; attempts = 0;
            while ~valid && attempts < 1000
                omega  = omegaRange(1)  + diff(omegaRange)  * rand();
                alpha  = alphaRange(1)  + diff(alphaRange)  * rand();
                beta   = betaRange(1)   + diff(betaRange)   * rand();
                gamma  = gammaRange(1)  + diff(gammaRange)  * rand();
                lambda = lambdaRange(1) + diff(lambdaRange) * rand();
                if (beta + alpha * gamma^2 < 1) && (alpha + beta < 1)
                    param_sets(i).alpha = alpha; param_sets(i).beta = beta;
                    param_sets(i).omega = omega; param_sets(i).gamma = gamma;
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
        omegaRange  = [1e-7,    1e-6  ];
        alphaRange  = [1.15e-6, 1.36e-6];
        betaRange   = [0.75,    0.85  ];
        gammaRange  = [0.25,    0.5   ];
        lambdaRange = [0.4,     0.6   ];

        for i = 1:num_sets
            valid = false; attempts = 0;
            while ~valid && attempts < 1000
                omega  = omegaRange(1)  + diff(omegaRange)  * rand();
                alpha  = alphaRange(1)  + diff(alphaRange)  * rand();
                beta   = betaRange(1)   + diff(betaRange)   * rand();
                gamma  = gammaRange(1)  + diff(gammaRange)  * rand();
                lambda = lambdaRange(1) + diff(lambdaRange) * rand();
                if (beta + alpha * gamma^2 < 1) && (alpha + beta < 1)
                    param_sets(i).alpha = alpha; param_sets(i).beta = beta;
                    param_sets(i).omega = omega; param_sets(i).gamma = gamma;
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
        omegaRange  = [1e-7, 1e-6];
        alphaRange  = [0.01, 0.15];
        betaRange   = [0.75, 0.85];
        lambdaRange = [0.01, 0.15];

        for i = 1:num_sets
            valid = false; attempts = 0;
            while ~valid && attempts < 1000
                omega  = omegaRange(1)  + diff(omegaRange)  * rand();
                alpha  = alphaRange(1)  + diff(alphaRange)  * rand();
                beta   = betaRange(1)   + diff(betaRange)   * rand();
                lambda = lambdaRange(1) + diff(lambdaRange) * rand();
                if (beta + alpha + lambda/2 < 0.999) && (omega > 0) && ...
                   (alpha >= 0) && (beta >= 0) && (alpha + lambda >= 0)
                    param_sets(i).alpha = alpha; param_sets(i).beta = beta;
                    param_sets(i).omega = omega; param_sets(i).gamma = 0;
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

%% -------------------------------------------------------------------------
%  UTILITY FUNCTIONS
%% -------------------------------------------------------------------------

function N_steps = get_tree_steps(tree_data, default_steps)
    if isfield(tree_data, 'N_steps')
        N_steps = tree_data.N_steps;
    else
        N_steps = default_steps;
    end
end

function config = read_toml_config_matlab(filename)
    fid = fopen(filename, 'r');
    if fid < 0
        error('Cannot open file: %s', filename);
    end

    config = struct();
    current_section = '';

    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            line = strtrim(line);
            if isempty(line) || line(1) == '#'; continue; end

            if line(1) == '[' && line(end) == ']'
                section_name  = line(2:end-1);
                section_parts = strsplit(section_name, '.');
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

            if contains(line, '=')
                parts = strsplit(line, '=');
                if length(parts) >= 2
                    key       = strtrim(parts{1});
                    value_str = strtrim(strjoin(parts(2:end), '='));
                    comment_idx = find(value_str == '#', 1);
                    if ~isempty(comment_idx)
                        value_str = strtrim(value_str(1:comment_idx-1));
                    end
                    value = parse_toml_value(value_str);

                    if ~isempty(current_section)
                        if contains(current_section, '.')
                            sp      = strsplit(current_section, '.');
                            section = sp{1}; subsection = sp{2};
                            if ~isfield(config, section); config.(section) = struct(); end
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
    str = strtrim(str);
    if (str(1) == '"'  && str(end) == '"' ); value = str(2:end-1); return; end
    if (str(1) == '''' && str(end) == ''''); value = str(2:end-1); return; end
    if strcmp(lower(str), 'true');  value = true;  return; end
    if strcmp(lower(str), 'false'); value = false; return; end
    if str(1) == '[' && str(end) == ']'
        inner    = str(2:end-1);
        elements = strsplit(inner, ',');
        value = [];
        for i = 1:length(elements)
            num_val = str2double(strtrim(elements{i}));
            if ~isnan(num_val); value = [value, num_val]; end
        end
        return;
    end
    num_val = str2double(str);
    if ~isnan(num_val); value = num_val; return; end
    value = str;
end

function save_param_sets(output_folder, param_sets, model_type, noise_percent)
    num_sets = length(param_sets);
    params_matrix = zeros(num_sets, 5);
    for i = 1:num_sets
        params_matrix(i,:) = [param_sets(i).omega, param_sets(i).alpha, ...
                               param_sets(i).beta,  param_sets(i).gamma, param_sets(i).lambda];
    end
    headers  = {'set_id', 'omega', 'alpha', 'beta', 'gamma', 'lambda'}';
    set_ids  = (1:num_sets)';
    csv_data = [headers'; num2cell([set_ids, params_matrix])];
    if nargin >= 4 && ~isempty(noise_percent)
        csv_file = sprintf('%s/garch_parameters_%s_%dpct.csv', output_folder, lower(model_type), noise_percent);
    else
        csv_file = sprintf('%s/garch_parameters_%s.csv', output_folder, lower(model_type));
    end
    writecell(csv_data, csv_file);
    fprintf('  Saved parameters: %s\n', csv_file);
end

function save_asset_files(output_folder, all_S_paths, output_filename, noise_percent)
    nonempty_paths = all_S_paths(~cellfun('isempty', all_S_paths));
    if isempty(nonempty_paths); warning('No asset price paths to save'); return; end

    num_rows   = size(nonempty_paths{1}, 1);
    total_cols = sum(cellfun(@(x) size(x,2), nonempty_paths));
    combined_S_paths = zeros(num_rows, total_cols);
    next_col = 1;
    for ps = 1:length(all_S_paths)
        S = all_S_paths{ps};
        if isempty(S); continue; end
        S = double(S);
        nc = size(S, 2);
        combined_S_paths(:, next_col:next_col+nc-1) = S;
        next_col = next_col + nc;
    end

    if nargin >= 4 && ~isempty(noise_percent)
        csv_file = sprintf('%s/%s_%dpct_asset_prices.csv', output_folder, output_filename, noise_percent);
    elseif nargin >= 3 && ~isempty(output_filename)
        csv_file = sprintf('%s/%s_asset_prices.csv', output_folder, output_filename);
    else
        csv_file = sprintf('%s/asset_prices.csv', output_folder);
    end

    writematrix(combined_S_paths, csv_file);
    fprintf('  Saved asset prices: %s (%d x %d from %d parameter sets)\n', ...
            csv_file, size(combined_S_paths,1), size(combined_S_paths,2), length(all_S_paths));
end