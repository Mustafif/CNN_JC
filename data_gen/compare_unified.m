function compare_unified(model_type, varargin)
% compare_unified - Unified comparison script for GARCH option pricing models
%
% Usage:
%   compare_unified('heston-nandi')
%   compare_unified('gjr')
%   compare_unified('duan')
%   compare_unified('heston-nandi', 'contracts_per_day', 20, 'num_days', 15)
%
% This script:
%   1. Loads true, initial, and calibrated parameters from JSON files
%   2. Generates datasets for each parameter set using the specified model
%   3. Computes relative errors and displays results
%
% Parameters:
%   model_type - String: 'heston-nandi', 'gjr', or 'duan'
%
% Optional Name-Value Pairs:
%   'contracts_per_day' - Number of contracts per day (default: 10)
%   'num_days'          - Number of trading days (default: 10)
%   'batch_size'        - Batch size for processing (default: 100)
%   'json_folder'       - Folder containing JSON files (default: 'compare')
%   'true_json'         - True params JSON filename (default: 'true.json')
%   'initial_json'      - Initial params JSON filename (default: 'init.json')
%   'calibrated_json'   - Calibrated params JSON filename (default: 'results_best1bin.json')

    clear all; clc;
    
    % Parse inputs
    p = inputParser;
    addRequired(p, 'model_type', @(x) ismember(lower(x), {'heston-nandi', 'hn', 'gjr', 'duan'}));
    addParameter(p, 'contracts_per_day', 10, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'num_days', 10, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'batch_size', 100, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'json_folder', 'compare', @ischar);
    addParameter(p, 'true_json', 'true.json', @ischar);
    addParameter(p, 'initial_json', 'init.json', @ischar);
    addParameter(p, 'calibrated_json', 'results_best1bin.json', @ischar);
    
    parse(p, model_type, varargin{:});
    
    % Normalize model type
    model = lower(p.Results.model_type);
    if strcmp(model, 'hn')
        model = 'heston-nandi';
    end
    
    % Extract parameters
    CONTRACTS_PER_DAY = p.Results.contracts_per_day;
    NUM_DAYS = p.Results.num_days;
    BATCH_SIZE = p.Results.batch_size;
    json_folder = p.Results.json_folder;
    
    fprintf('=================================================================\n');
    fprintf('      UNIFIED GARCH OPTION PRICING COMPARISON\n');
    fprintf('      Model: %s\n', upper(model));
    fprintf('=================================================================\n\n');
    
    %% Load JSON parameter files
    fprintf('Loading parameter sets from JSON files...\n');
    
    % Try to load true parameters
    true_path = fullfile(json_folder, p.Results.true_json);
    if exist(true_path, 'file')
        true_data = jsondecode(fileread(true_path));
        fprintf('  ✓ Loaded true parameters from: %s\n', true_path);
    else
        fprintf('  ⚠ True parameters file not found: %s\n', true_path);
        fprintf('    Using default true parameters for %s\n', model);
        true_data = get_default_params(model, 'true');
    end
    
    % Load initial parameters
    init_path = fullfile(json_folder, p.Results.initial_json);
    if exist(init_path, 'file')
        init_data = jsondecode(fileread(init_path));
        fprintf('  ✓ Loaded initial parameters from: %s\n', init_path);
    else
        fprintf('  ⚠ Initial parameters file not found: %s\n', init_path);
        fprintf('    Using default initial parameters for %s\n', model);
        init_data = get_default_params(model, 'initial');
    end
    
    % Load calibrated parameters
    cal_path = fullfile(json_folder, p.Results.calibrated_json);
    if exist(cal_path, 'file')
        cal_data = jsondecode(fileread(cal_path));
        fprintf('  ✓ Loaded calibrated parameters from: %s\n', cal_path);
    else
        fprintf('  ⚠ Calibrated parameters file not found: %s\n', cal_path);
        fprintf('    Using default calibrated parameters for %s\n', model);
        cal_data = get_default_params(model, 'calibrated');
    end
    
    %% Display scaling configuration
    fprintf('\nSCALING CONFIGURATION:\n');
    fprintf('  Contracts per day: %d\n', CONTRACTS_PER_DAY);
    fprintf('  Number of days: %d\n', NUM_DAYS);
    fprintf('  Total contracts: %d\n', CONTRACTS_PER_DAY * NUM_DAYS * 2);
    fprintf('  Batch size: %d\n', BATCH_SIZE);
    fprintf('  Estimated memory usage: %.1f MB\n', ...
        (CONTRACTS_PER_DAY * NUM_DAYS * 2 * 12 * 8) / 1e6);
    
    %% Market parameters
    N = 1024;
    r = 0.05/252;
    S0 = 100;
    moneyness_range = [0.8, 1.2];
    maturity_range = [0.1, 1.0];
    
    %% Willow tree parameters
    m_h = 6;
    m_ht = 6;
    m_x = 30;
    gamma_h = 0.6;
    gamma_x = 0.8;
    
    %% Solver parameters
    itmax = 100;
    tol = 1e-6;
    
    %% Create parameter sets
    true_params = struct('name', 'true_params');
    initial_params = struct('name', 'initial_params');
    calibrated_params = struct('name', 'calibrated_params');
    
    % Set model-specific parameters
    if strcmp(model, 'heston-nandi')
        param_fields = {'omega', 'alpha', 'beta', 'gamma', 'lambda'};
    elseif strcmp(model, 'gjr')
        param_fields = {'omega', 'alpha', 'beta', 'lambda'};
    elseif strcmp(model, 'duan')
        param_fields = {'omega', 'alpha', 'beta', 'theta', 'lambda'};
    end
    
    for i = 1:length(param_fields)
        field = param_fields{i};
        if isfield(true_data, field)
            true_params.(field) = true_data.(field);
        end
        if isfield(init_data, field)
            initial_params.(field) = init_data.(field);
        end
        if isfield(cal_data, field)
            calibrated_params.(field) = cal_data.(field);
        end
    end
    
    param_sets = [true_params, initial_params, calibrated_params];
    
    %% Pre-generate randomness
    fprintf('\nPre-generating random numbers for GARCH simulation...\n');
    M = 1;
    numPoint = N + 1;
    rng(12345);
    Z_master = randn(numPoint + 1, M);
    fprintf('Random matrix Z generated: size = %d x %d\n', size(Z_master,1), size(Z_master,2));
    
    %% Process each parameter set
    total_contracts = CONTRACTS_PER_DAY * NUM_DAYS * 2;
    num_params = length(param_fields);
    headers = [{'S0', 'm', 'r', 'T', 'corp'}'; param_fields'; {'sigma', 'V'}'];
    
    for ps = 1:numel(param_sets)
        params = param_sets(ps);
        fprintf('\n>>> NOW PROCESSING ps=%d, name=%s\n', ps, params.name);
        setname = params.name;
        
        fprintf('\n============================================================\n');
        fprintf('  PROCESSING PARAMETER SET: %s\n', setname);
        fprintf('============================================================\n');
        
        %% Generate GARCH paths
        fprintf('\nGenerating scaled GARCH simulation for %s...\n', setname);
        Z = Z_master;
        
        switch model
            case 'heston-nandi'
                [S_paths, h0_paths] = mcHN(M, N, S0, Z, r, ...
                    params.omega, params.alpha, params.beta, params.gamma, params.lambda);
                c = params.gamma + params.lambda + 0.5;
            case 'gjr'
                [S_paths, h0_paths] = mcGJR(M, N, S0, Z, r, ...
                    params.omega, params.alpha, params.beta, params.lambda);
                c = 0;  % GJR doesn't use c parameter
            case 'duan'
                [S_paths, h0_paths] = mcDuan(M, N, S0, Z, r, ...
                    params.omega, params.alpha, params.beta, params.theta, params.lambda);
                c = 0;  % Duan doesn't use c parameter
        end
        
        fprintf('Generated %d GARCH paths for %s\n', M, setname);
        
        %% Initialize dataset
        dataset = zeros(5 + num_params + 2, total_contracts);
        dataset_idx = 1;
        
        fprintf('\nInitializing dataset structure for %s...\n', setname);
        fprintf('Dataset dimensions: %d metrics × %d contracts\n', ...
            size(dataset, 1), size(dataset, 2));
        
        %% Build tree structures
        fprintf('\nBuilding willow trees for %s (one-time setup)...\n', setname);
        
        try
            switch model
                case 'heston-nandi'
                    [hd, qhd] = genhDelta(h0_paths, params.beta, params.alpha, c, params.omega, m_h, gamma_h);
                    [nodes_ht] = TreeNodes_ht_HN(m_h, hd, qhd, gamma_h, params.alpha, params.beta, c, params.omega, N+1);
                case 'gjr'
                    [hd, qhd] = genhDelta(h0_paths, params.beta, params.alpha, 0, params.omega, m_h, gamma_h);
                    [nodes_ht] = TreeNodes_ht_GJR(m_h, hd, qhd, gamma_h, params.alpha, params.beta, params.lambda, params.omega, N+1);
                case 'duan'
                    [hd, qhd] = genhDelta(h0_paths, params.beta, params.alpha, 0, params.omega, m_h, gamma_h);
                    [nodes_ht] = TreeNodes_ht_Duan(m_h, hd, qhd, gamma_h, params.alpha, params.beta, params.theta, params.lambda, params.omega, N+1);
            end
            
            % Pre-compute tree nodes for different days
            tree_cache = cell(NUM_DAYS, 1);
            for day = 1:NUM_DAYS
                S_current = S_paths(end - CONTRACTS_PER_DAY + day, min(day, M));
                h_current = h0_paths(min(day, M));
                
                switch model
                    case 'heston-nandi'
                        [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_HN(m_x, gamma_x, r, hd, qhd, S_current, ...
                            params.alpha, params.beta, c, params.omega, N);
                        [q_Xt, P_Xt, ~] = Prob_Xt(nodes_ht, qhd, nodes_Xt, S_current, r, ...
                            params.alpha, params.beta, c, params.omega);
                    case 'gjr'
                        [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_GJR(m_x, gamma_x, r, hd, qhd, S_current, ...
                            params.alpha, params.beta, params.lambda, params.omega, N);
                        [q_Xt, P_Xt, ~] = Prob_Xt_GJR(nodes_ht, qhd, nodes_Xt, S_current, r, ...
                            params.alpha, params.beta, params.lambda, params.omega);
                    case 'duan'
                        [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_Duan(m_x, gamma_x, r, hd, qhd, S_current, ...
                            params.alpha, params.beta, params.theta, params.lambda, params.omega, N);
                        [q_Xt, P_Xt, ~] = Prob_Xt_Duan(nodes_ht, qhd, nodes_Xt, S_current, r, ...
                            params.alpha, params.beta, params.theta, params.lambda, params.omega);
                end
                
                nodes_S = exp(nodes_Xt);
                tree_cache{day} = struct('nodes_S', nodes_S, 'P_Xt', P_Xt, 'q_Xt', q_Xt, ...
                                        'S_current', S_current, 'h_current', h_current);
            end
            
            fprintf('Trees cached for %d days (%s)\n', NUM_DAYS, setname);
            
        catch ME
            fprintf('Error building trees for %s: %s\n', setname, ME.message);
            continue;
        end
        
        %% Batch processing
        num_batches = ceil(total_contracts / (BATCH_SIZE * 2));
        fprintf('\nProcessing %s in %d batches...\n', setname, num_batches);
        
        total_processed = 0;
        pricing_errors = 0;
        processing_times = zeros(num_batches, 1);
        
        for batch = 1:num_batches
            batch_start_time = tic;
            fprintf('\n[%s] Processing batch %d/%d...\n', setname, batch, num_batches);
            
            contracts_in_batch = min(BATCH_SIZE, ceil((total_contracts/2 - (batch-1)*BATCH_SIZE)));
            batch_start_idx = (batch - 1) * BATCH_SIZE + 1;
            
            batch_errors = 0;
            batch_processed = 0;
            
            for contract_in_batch = 1:contracts_in_batch
                contract_idx = batch_start_idx + contract_in_batch - 1;
                day = ceil(contract_idx / CONTRACTS_PER_DAY);
                if day > NUM_DAYS
                    break;
                end
                
                tree_data = tree_cache{day};
                S_current = tree_data.S_current;
                nodes_S = tree_data.nodes_S;
                P_Xt = tree_data.P_Xt;
                q_Xt = tree_data.q_Xt;
                
                rng(contract_idx);
                moneyness = moneyness_range(1) + (moneyness_range(2) - moneyness_range(1)) * rand();
                maturity_years = maturity_range(1) + (maturity_range(2) - maturity_range(1)) * rand();
                
                K_strike = moneyness * S_current;
                T_maturity = maturity_years;
                
                try
                    [V_C, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, 1);
                    [V_P, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, -1);
                    
                    if V_C < 0 || V_P < 0 || isnan(V_C) || isnan(V_P)
                        batch_errors = batch_errors + 1;
                        pricing_errors = pricing_errors + 1;
                        continue;
                    end
                    
                    % Implied volatility solver
                    switch model
                        case 'heston-nandi'
                            [impl_c, V0_c, it_c, conv_c] = impvol_improved(S_current, K_strike, T_maturity, r, V_C, 1, N, m_x, gamma_x, tol, itmax);
                            [impl_p, V0_p, it_p, conv_p] = impvol_improved(S_current, K_strike, T_maturity, r, V_P, -1, N, m_x, gamma_x, tol, itmax);
                        case 'gjr'
                            [impl_c, V0_c, it_c, conv_c] = impVol_GJR(S_current, K_strike, T_maturity, r, V_C, 1, N, m_x, gamma_x, tol, itmax);
                            [impl_p, V0_p, it_p, conv_p] = impVol_GJR(S_current, K_strike, T_maturity, r, V_P, -1, N, m_x, gamma_x, tol, itmax);
                        case 'duan'
                            [impl_c, V0_c, it_c, conv_c] = impVol_Duan(S_current, K_strike, T_maturity, r, V_C, 1, N, m_x, gamma_x, tol, itmax);
                            [impl_p, V0_p, it_p, conv_p] = impVol_Duan(S_current, K_strike, T_maturity, r, V_P, -1, N, m_x, gamma_x, tol, itmax);
                    end
                    
                    % Build parameter vector
                    param_values = zeros(num_params, 1);
                    for i = 1:length(param_fields)
                        param_values(i) = params.(param_fields{i});
                    end
                    
                    % Store call option
                    dataset(:, dataset_idx) = [S_current; K_strike/S_current; r; T_maturity; 1; param_values; impl_c; V_C];
                    dataset_idx = dataset_idx + 1;
                    
                    % Store put option
                    dataset(:, dataset_idx) = [S_current; K_strike/S_current; r; T_maturity; -1; param_values; impl_p; V_P];
                    dataset_idx = dataset_idx + 1;
                    
                    batch_processed = batch_processed + 2;
                    
                catch ME
                    batch_errors = batch_errors + 1;
                    pricing_errors = pricing_errors + 1;
                    
                    % Store error placeholders
                    param_values = zeros(num_params, 1);
                    for i = 1:length(param_fields)
                        param_values(i) = params.(param_fields{i});
                    end
                    
                    for k = 1:2
                        dataset(:, dataset_idx) = [S_current; K_strike/S_current; r; T_maturity; (-1)^k; param_values; 0.2; 0];
                        dataset_idx = dataset_idx + 1;
                    end
                end
            end
            
            processing_times(batch) = toc(batch_start_time);
            total_processed = total_processed + batch_processed;
            
            fprintf('  [%s] Batch %d complete: %d contracts processed, %d errors, %.2f seconds\n', ...
                setname, batch, batch_processed, batch_errors, processing_times(batch));
            
            if mod(batch, 5) == 0
                fprintf('  [%s] Memory cleanup...\n', setname);
                clear V_C V_P impl_c impl_p V0_c V0_p;
            end
        end
        
        %% Finalize and save dataset
        dataset = dataset(:, 1:dataset_idx-1);
        
        fprintf('\nSaving results for %s...\n', setname);
        filename = sprintf('%s/%s.csv', json_folder, setname);
        dataset_enhanced = [headers'; num2cell(dataset')];
        writecell(dataset_enhanced, filename);
        
        asset_filename = sprintf('assetprices_%s.csv', setname);
        writecell(['S', num2cell(S_paths')], asset_filename);
        
        %% Report statistics
        fprintf('\n=================================================================\n');
        fprintf('             PROCESSING COMPLETE FOR %s\n', upper(setname));
        fprintf('=================================================================\n');
        
        actual_contracts = size(dataset, 2);
        call_data = dataset(:, dataset(5,:) == 1);
        put_data = dataset(:, dataset(5,:) == -1);
        
        fprintf('\nSCALING RESULTS (%s):\n', setname);
        fprintf('  Planned contracts: %d\n', total_contracts);
        fprintf('  Actual contracts: %d\n', actual_contracts);
        fprintf('  Calls: %d, Puts: %d\n', size(call_data, 2), size(put_data, 2));
        fprintf('  Dataset saved as: %s\n', filename);
        
        fprintf('\nPERFORMANCE METRICS (%s):\n', setname);
        fprintf('  Total processing time: %.2f minutes\n', sum(processing_times) / 60);
        fprintf('  Average time per batch: %.2f seconds\n', mean(processing_times));
        fprintf('  Contracts per second: %.1f\n', actual_contracts / sum(processing_times));
        fprintf('  Pricing errors: %d (%.1f%%)\n', pricing_errors, 100*pricing_errors/max(actual_contracts,1));
        
        all_ivs = dataset(end-1,:);
        valid_ivs = all_ivs(all_ivs > 0.001 & all_ivs < 2.99);
        
        fprintf('\nDATA QUALITY (%s):\n', setname);
        fprintf('  Valid IVs: %d/%d (%.1f%%)\n', length(valid_ivs), length(all_ivs), ...
            100*length(valid_ivs)/max(length(all_ivs),1));
        if ~isempty(valid_ivs)
            fprintf('  IV range: [%.4f, %.4f] (%.1f%% - %.1f%% annualized)\n', ...
                min(valid_ivs), max(valid_ivs), ...
                min(valid_ivs)*sqrt(252)*100, max(valid_ivs)*sqrt(252)*100);
            fprintf('  Mean IV: %.4f (%.1f%% annualized)\n', ...
                mean(valid_ivs), mean(valid_ivs)*sqrt(252)*100);
        end
        
        fprintf('\n=================================================================\n');
    end
    
    %% Compute relative errors
    fprintf('\n\n=================================================================\n');
    fprintf('             RELATIVE ERROR ANALYSIS\n');
    fprintf('=================================================================\n\n');
    
    compute_relative_errors(json_folder, model);
    
    fprintf('\nAll parameter sets processed.\n');
    fprintf('To scale further, modify CONTRACTS_PER_DAY and NUM_DAYS parameters.\n');
end

function compute_relative_errors(folder, model)
    % Load datasets
    try
        init = readtable(fullfile(folder, 'initial_params.csv'));
        calib = readtable(fullfile(folder, 'calibrated_params.csv'));
        true_ = readtable(fullfile(folder, 'true_params.csv'));
    catch
        fprintf('Error: Could not load CSV files for relative error computation.\n');
        fprintf('Expected files:\n');
        fprintf('  %s\n', fullfile(folder, 'initial_params.csv'));
        fprintf('  %s\n', fullfile(folder, 'calibrated_params.csv'));
        fprintf('  %s\n', fullfile(folder, 'true_params.csv'));
        return;
    end
    
    % Apply filters
    valid_filter = (true_.sigma > 0.01) & (true_.V >= 0.5);
    
    if sum(valid_filter) == 0
        fprintf('Warning: No valid data points after filtering.\n');
        return;
    end
    
    % Calculate relative errors
    rel_err_init = abs(init.sigma(valid_filter) - true_.sigma(valid_filter)) ./ true_.sigma(valid_filter);
    rel_err_calib = abs(calib.sigma(valid_filter) - true_.sigma(valid_filter)) ./ true_.sigma(valid_filter);
    
    T_days = calib.T(valid_filter) .* 252;
    m = calib.m(valid_filter);
    
    % Assign categories for Maturity
    maturity_idx = zeros(size(T_days));
    maturity_idx(T_days < 30) = 1;
    maturity_idx(T_days >= 30 & T_days < 180) = 2;
    maturity_idx(T_days >= 180) = 3;
    
    % Assign categories for Moneyness
    moneyness_idx = zeros(size(m));
    moneyness_idx(m < 0.8) = 1;
    moneyness_idx(m >= 0.8 & m < 1.0) = 2;
    moneyness_idx(m >= 1.0 & m < 1.2) = 3;
    moneyness_idx(m >= 1.2) = 4;
    
    % Compute mean relative error
    valid_idx = (maturity_idx > 0) & (moneyness_idx > 0);
    
    mean_rel_err_matrix_init = accumarray([maturity_idx(valid_idx), moneyness_idx(valid_idx)], ...
        rel_err_init(valid_idx), [3, 4], @mean, NaN);
    
    mean_rel_err_matrix_calib = accumarray([maturity_idx(valid_idx), moneyness_idx(valid_idx)], ...
        rel_err_calib(valid_idx), [3, 4], @mean, NaN);
    
    % Create tables
    colNames = {'m_lt_0_8', 'm_0_8_to_1_0', 'm_1_0_to_1_2', 'm_gt_1_2'};
    rowNames = {'Days_lt_30', 'Days_30_to_180', 'Days_gt_180'};
    
    table_init = array2table(mean_rel_err_matrix_init, ...
        'VariableNames', colNames, 'RowNames', rowNames);
    
    table_calib = array2table(mean_rel_err_matrix_calib, ...
        'VariableNames', colNames, 'RowNames', rowNames);
    
    % Display results
    fprintf('Model: %s\n\n', upper(model));
    disp('======================================================');
    disp('   Mean Relative Error Table (Initial vs True)        ');
    disp('======================================================');
    disp(table_init);
    fprintf('\n');
    
    disp('======================================================');
    disp('   Mean Relative Error Table (Calibrated vs True)     ');
    disp('======================================================');
    disp(table_calib);
    fprintf('\n');
    
    % Summary statistics
    fprintf('SUMMARY STATISTICS:\n');
    fprintf('  Initial Parameters:\n');
    fprintf('    Mean relative error: %.4f (%.2f%%)\n', mean(rel_err_init), mean(rel_err_init)*100);
    fprintf('    Median relative error: %.4f (%.2f%%)\n', median(rel_err_init), median(rel_err_init)*100);
    fprintf('    Max relative error: %.4f (%.2f%%)\n', max(rel_err_init), max(rel_err_init)*100);
    fprintf('\n');
    fprintf('  Calibrated Parameters:\n');
    fprintf('    Mean relative error: %.4f (%.2f%%)\n', mean(rel_err_calib), mean(rel_err_calib)*100);
    fprintf('    Median relative error: %.4f (%.2f%%)\n', median(rel_err_calib), median(rel_err_calib)*100);
    fprintf('    Max relative error: %.4f (%.2f%%)\n', max(rel_err_calib), max(rel_err_calib)*100);
    fprintf('\n');
    fprintf('  Improvement (Initial → Calibrated):\n');
    fprintf('    Mean error reduction: %.2f%%\n', (1 - mean(rel_err_calib)/mean(rel_err_init))*100);
    fprintf('    Median error reduction: %.2f%%\n', (1 - median(rel_err_calib)/median(rel_err_init))*100);
    fprintf('\n');
end

function params = get_default_params(model, param_type)
    % Default parameter sets for different models
    switch model
        case 'heston-nandi'
            if strcmp(param_type, 'true')
                params.omega = 1e-6;
                params.alpha = 1.33e-6;
                params.beta = 0.8;
                params.gamma = 5;
                params.lambda = 0.2;
            elseif strcmp(param_type, 'initial')
                params.omega = 1.0e-6;
                params.alpha = 1.5e-6;
                params.beta = 0.5;
                params.gamma = 1.0;
                params.lambda = 0.1;
            else % calibrated
                params.omega = 1e-6;
                params.alpha = 1.33e-6;
                params.beta = 0.8;
                params.gamma = 5;
                params.lambda = 0.2;
            end
        case 'gjr'
            if strcmp(param_type, 'true')
                params.omega = 1e-6;
                params.alpha = 1.5e-6;
                params.beta = 0.85;
                params.lambda = 2.0;
            elseif strcmp(param_type, 'initial')
                params.omega = 1.0e-6;
                params.alpha = 2.0e-6;
                params.beta = 0.7;
                params.lambda = 1.0;
            else % calibrated
                params.omega = 1e-6;
                params.alpha = 1.5e-6;
                params.beta = 0.85;
                params.lambda = 2.0;
            end
        case 'duan'
            if strcmp(param_type, 'true')
                params.omega = 1e-6;
                params.alpha = 2e-6;
                params.beta = 0.9;
                params.theta = 0.5;
                params.lambda = 0.3;
            elseif strcmp(param_type, 'initial')
                params.omega = 1.0e-6;
                params.alpha = 3.0e-6;
                params.beta = 0.75;
                params.theta = 0.3;
                params.lambda = 0.2;
            else % calibrated
                params.omega = 1e-6;
                params.alpha = 2e-6;
                params.beta = 0.9;
                params.theta = 0.5;
                params.lambda = 0.3;
            end
    end
end
