clear all
% demo_scalable_varying_garch_with_validation.m
% Enhanced scalable Heston-Nandi GARCH option pricing with separate validation dataset
% Creates training set with normal parameters and validation set with extreme/stressed conditions

fprintf('=================================================================\n');
fprintf('   SCALABLE HESTON-NANDI GARCH WITH VALIDATION DATASET\n');
fprintf('=================================================================\n\n');

%% SCALING PARAMETERS - MODIFY THESE TO SCALE THE DATASET
CONTRACTS_PER_DAY = 50;     % Number of option contracts to generate per day
NUM_DAYS_TRAIN = 30;        % Number of trading days for training set
NUM_DAYS_VAL = 10;          % Number of trading days for validation set
BATCH_SIZE = 100;           % Process options in batches to manage memory

% GARCH Parameter Variation Settings
NUM_PARAM_SETS_TRAIN = 5;   % Number of parameter sets for training (normal conditions)
NUM_PARAM_SETS_VAL = 8;     % Number of parameter sets for validation (stressed conditions)

% Display configuration
total_train_contracts = CONTRACTS_PER_DAY * NUM_DAYS_TRAIN * 2;
total_val_contracts = CONTRACTS_PER_DAY * NUM_DAYS_VAL * 2;

fprintf('DATASET CONFIGURATION:\n');
fprintf('  TRAINING SET:\n');
fprintf('    Contracts per day: %d\n', CONTRACTS_PER_DAY);
fprintf('    Number of days: %d\n', NUM_DAYS_TRAIN);
fprintf('    Total contracts: %d\n', total_train_contracts);
fprintf('    Parameter sets: %d (normal conditions)\n', NUM_PARAM_SETS_TRAIN);
fprintf('  VALIDATION SET:\n');
fprintf('    Contracts per day: %d\n', CONTRACTS_PER_DAY);
fprintf('    Number of days: %d\n', NUM_DAYS_VAL);
fprintf('    Total contracts: %d\n', total_val_contracts);
fprintf('    Parameter sets: %d (stressed conditions)\n', NUM_PARAM_SETS_VAL);
fprintf('  Batch size: %d\n', BATCH_SIZE);

%% MARKET PARAMETERS
N = 504;                          % Number of time steps
r_base = 0.05/252;               % Base risk-free rate (daily)
S0 = 100;                        % Initial stock price

% TRAINING SET: Normal market conditions
train_moneyness_range = [0.85, 1.15];    % Normal moneyness range
train_maturity_range = [0.1, 1.0];       % Normal maturity range (years)
train_rate_multiplier = [0.8, 1.2];      % Normal rate variation

% VALIDATION SET: Stressed market conditions  
val_moneyness_range = [0.5, 2.0];        % Extreme moneyness range
val_maturity_range = [0.02, 2.0];        % Extreme maturity range (1 week to 2 years)
val_rate_multiplier = [0.1, 3.0];        % Extreme rate variation (near-zero to high rates)

%% GENERATE TRAINING GARCH PARAMETER SETS (NORMAL CONDITIONS)
fprintf('\nGenerating TRAINING GARCH parameter sets (normal market conditions)...\n');

% Normal parameter ranges based on empirical literature
train_alpha_range = [1.15e-6, 1.36e-6];
train_beta_range = [0.75, 0.85];
train_omega_range = [1e-7, 1e-6];
train_gamma_range = [50, 150];
train_lambda_range = [0.4, 0.6];

train_param_sets = generate_garch_params(NUM_PARAM_SETS_TRAIN, ...
    train_alpha_range, train_beta_range, train_omega_range, ...
    train_gamma_range, train_lambda_range, 'TRAINING');

%% GENERATE VALIDATION GARCH PARAMETER SETS (STRESSED CONDITIONS)
fprintf('\nGenerating VALIDATION GARCH parameter sets (stressed conditions)...\n');

% Stressed parameter ranges - wider and more extreme
val_alpha_range = [5e-7, 3e-6];     % Wider volatility innovation range
val_beta_range = [0.60, 0.95];      % Wider persistence range
val_omega_range = [1e-8, 5e-6];     % Wider long-run variance range
val_gamma_range = [10, 300];        % More extreme asymmetry
val_lambda_range = [0.1, 1.0];      % Wider risk premium range

val_param_sets = generate_garch_params(NUM_PARAM_SETS_VAL, ...
    val_alpha_range, val_beta_range, val_omega_range, ...
    val_gamma_range, val_lambda_range, 'VALIDATION');

% Add specific stress scenarios to validation set
fprintf('Adding specific stress scenarios to validation set...\n');
stress_scenarios = generate_stress_scenarios();
val_param_sets = [val_param_sets; stress_scenarios];
NUM_PARAM_SETS_VAL = size(val_param_sets, 1);

fprintf('Total validation parameter sets (including stress): %d\n', NUM_PARAM_SETS_VAL);

%% WILLOW TREE PARAMETERS
m_h = 6;
m_ht = 6;
m_x = 30;
gamma_h = 0.6;
gamma_x = 0.8;

%% SOLVER PARAMETERS
itmax = 100;
tol = 1e-6;

%% GENERATE DATASETS
fprintf('\n=== GENERATING TRAINING DATASET ===\n');
[train_dataset, train_stats] = generate_option_dataset(...
    train_param_sets, NUM_DAYS_TRAIN, CONTRACTS_PER_DAY, BATCH_SIZE, ...
    train_moneyness_range, train_maturity_range, train_rate_multiplier, ...
    r_base, S0, N, m_h, m_ht, m_x, gamma_h, gamma_x, tol, itmax, 'TRAIN');

fprintf('\n=== GENERATING VALIDATION DATASET ===\n');
[val_dataset, val_stats] = generate_option_dataset(...
    val_param_sets, NUM_DAYS_VAL, CONTRACTS_PER_DAY, BATCH_SIZE, ...
    val_moneyness_range, val_maturity_range, val_rate_multiplier, ...
    r_base, S0, N, m_h, m_ht, m_x, gamma_h, gamma_x, tol, itmax, 'VALIDATION');

%% SAVE RESULTS
fprintf('\n=== SAVING DATASETS ===\n');
headers = {'S0', 'moneyness', 'r', 'T', 'call_put', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'impl_vol', 'option_value'}';

% Save training dataset
train_filename = sprintf('train_garch_dataset_%dx%d_%dparams.csv', CONTRACTS_PER_DAY, NUM_DAYS_TRAIN, NUM_PARAM_SETS_TRAIN);
train_dataset_enhanced = [headers'; num2cell(train_dataset')];
writecell(train_dataset_enhanced, train_filename);

% Save validation dataset
val_filename = sprintf('val_garch_dataset_%dx%d_%dparams.csv', CONTRACTS_PER_DAY, NUM_DAYS_VAL, NUM_PARAM_SETS_VAL);
val_dataset_enhanced = [headers'; num2cell(val_dataset')];
writecell(val_dataset_enhanced, val_filename);

%% DATASET COMPARISON AND ANALYSIS
fprintf('\n=================================================================\n');
fprintf('              DATASET GENERATION COMPLETE\n');
fprintf('=================================================================\n');

compare_datasets(train_dataset, val_dataset, train_stats, val_stats, ...
                train_filename, val_filename);

%% HELPER FUNCTIONS

function param_sets = generate_garch_params(num_sets, alpha_range, beta_range, ...
    omega_range, gamma_range, lambda_range, dataset_type)
    
    fprintf('Generating %d %s parameter sets...\n', num_sets, dataset_type);
    
    param_sets = zeros(num_sets, 5);
    valid_sets = 0;
    max_attempts = num_sets * 100;
    attempts = 0;
    
    while valid_sets < num_sets && attempts < max_attempts
        attempts = attempts + 1;
        
        % Generate candidate parameters
        alpha = alpha_range(1) + (alpha_range(2) - alpha_range(1)) * rand();
        beta = beta_range(1) + (beta_range(2) - beta_range(1)) * rand();
        omega = omega_range(1) + (omega_range(2) - omega_range(1)) * rand();
        gamma = gamma_range(1) + (gamma_range(2) - gamma_range(1)) * rand();
        lambda = lambda_range(1) + (lambda_range(2) - lambda_range(1)) * rand();
        
        % Check stationarity condition
        c = gamma + lambda + 0.5;
        stationarity_condition = beta + alpha * c^2;
        
        % Relaxed stationarity for validation set to allow more extreme parameters
        if strcmp(dataset_type, 'VALIDATION')
            max_stationarity = 0.999;
        else
            max_stationarity = 0.995;
        end
        
        if stationarity_condition < max_stationarity
            valid_sets = valid_sets + 1;
            param_sets(valid_sets, :) = [alpha, beta, omega, gamma, lambda];
        end
    end
    
    if valid_sets < num_sets
        warning('Could only generate %d valid parameter sets out of %d requested', valid_sets, num_sets);
        param_sets = param_sets(1:valid_sets, :);
    end
    
    % Display parameter statistics
    fprintf('  %s parameter ranges:\n', dataset_type);
    fprintf('    Alpha:  [%.2e, %.2e] (mean: %.2e)\n', min(param_sets(:,1)), max(param_sets(:,1)), mean(param_sets(:,1)));
    fprintf('    Beta:   [%.3f, %.3f] (mean: %.3f)\n', min(param_sets(:,2)), max(param_sets(:,2)), mean(param_sets(:,2)));
    fprintf('    Omega:  [%.2e, %.2e] (mean: %.2e)\n', min(param_sets(:,3)), max(param_sets(:,3)), mean(param_sets(:,3)));
    fprintf('    Gamma:  [%.2f, %.2f] (mean: %.2f)\n', min(param_sets(:,4)), max(param_sets(:,4)), mean(param_sets(:,4)));
    fprintf('    Lambda: [%.3f, %.3f] (mean: %.3f)\n', min(param_sets(:,5)), max(param_sets(:,5)), mean(param_sets(:,5)));
    
    % Check stationarity
    c_values = param_sets(:,4) + param_sets(:,5) + 0.5;
    stationarity_values = param_sets(:,2) + param_sets(:,1) .* c_values.^2;
    fprintf('    Stationarity: [%.4f, %.4f]\n', min(stationarity_values), max(stationarity_values));
end

function stress_scenarios = generate_stress_scenarios()
    % Generate specific stress test scenarios for validation
    
    fprintf('Generating stress test scenarios...\n');
    
    stress_scenarios = [
        % Scenario 1: High volatility crisis (2008-style)
        2.5e-6, 0.85, 3e-6, 250, 0.8;
        
        % Scenario 2: Low volatility, high persistence
        8e-7, 0.94, 1e-7, 30, 0.2;
        
        % Scenario 3: High asymmetry (leverage effect)
        1.8e-6, 0.75, 2e-6, 300, 0.9;
        
        % Scenario 4: Near unit root (high persistence)
        1.2e-6, 0.92, 5e-7, 100, 0.5;
        
        % Scenario 5: Low gamma (reverse leverage)
        1.5e-6, 0.80, 1.5e-6, 10, 0.4;
    ];
    
    fprintf('Generated %d stress scenarios\n', size(stress_scenarios, 1));
end

function [dataset, stats] = generate_option_dataset(param_sets, num_days, contracts_per_day, ...
    batch_size, moneyness_range, maturity_range, rate_multiplier, r_base, S0, N, ...
    m_h, m_ht, m_x, gamma_h, gamma_x, tol, itmax, dataset_type)
    
    num_param_sets = size(param_sets, 1);
    total_contracts = contracts_per_day * num_days * 2;
    dataset = zeros(12, total_contracts);
    
    % Statistics tracking
    stats = struct();
    stats.total_processed = 0;
    stats.pricing_errors = 0;
    stats.solver_failures = 0;
    stats.processing_times = [];
    
    fprintf('Processing %s dataset: %d contracts\n', dataset_type, total_contracts);
    
    % Pre-generate GARCH paths
    fprintf('Generating GARCH paths for %d parameter sets...\n', num_param_sets);
    M_paths = min(num_param_sets, 50);
    numPoint = N + 1;
    Z_seeds = randn(numPoint + 1, M_paths);
    
    S_paths_all = cell(num_param_sets, 1);
    h0_paths_all = zeros(num_param_sets, 1);
    
    for p = 1:num_param_sets
        alpha_p = param_sets(p, 1);
        beta_p = param_sets(p, 2);
        omega_p = param_sets(p, 3);
        gamma_p = param_sets(p, 4);
        lambda_p = param_sets(p, 5);
        
        seed_idx = mod(p-1, M_paths) + 1;
        
        try
            [S_path, h0_path] = mcHN(1, N, S0, Z_seeds(:, seed_idx), r_base, ...
                omega_p, alpha_p, beta_p, gamma_p, lambda_p);
            S_paths_all{p} = S_path;
            h0_paths_all(p) = h0_path;
        catch ME
            fprintf('Warning: Path generation error for param set %d: %s\n', p, ME.message);
            % Use fallback parameters
            [S_path, h0_path] = mcHN(1, N, S0, Z_seeds(:, seed_idx), r_base, ...
                1e-6, 1.33e-6, 0.8, 100, 0.5);
            S_paths_all{p} = S_path;
            h0_paths_all(p) = h0_path;
        end
    end
    
    % Main processing loop
    num_batches = ceil(total_contracts / (batch_size * 2));
    dataset_idx = 1;
    
    for batch = 1:num_batches
        batch_start_time = tic;
        fprintf('Processing %s batch %d/%d...\n', dataset_type, batch, num_batches);
        
        contracts_in_batch = min(batch_size, ceil((total_contracts/2 - (batch-1)*batch_size)));
        batch_start_idx = (batch - 1) * batch_size + 1;
        
        tree_cache = containers.Map('KeyType', 'int32', 'ValueType', 'any');
        
        for contract_in_batch = 1:contracts_in_batch
            contract_idx = batch_start_idx + contract_in_batch - 1;
            day = ceil(contract_idx / contracts_per_day);
            
            if day > num_days, break; end
            
            % Assign parameter set
            param_idx = mod(contract_idx - 1, num_param_sets) + 1;
            
            % Get GARCH parameters
            alpha = param_sets(param_idx, 1);
            beta = param_sets(param_idx, 2);
            omega = param_sets(param_idx, 3);
            gamma = param_sets(param_idx, 4);
            lambda = param_sets(param_idx, 5);
            
            % Get GARCH path
            S_path = S_paths_all{param_idx};
            h0_path = h0_paths_all(param_idx);
            S_current = S_path(end);
            
            % Generate contract parameters (more extreme for validation)
            rng(contract_idx + 1000 * strcmp(dataset_type, 'VALIDATION'));
            moneyness = moneyness_range(1) + (moneyness_range(2) - moneyness_range(1)) * rand();
            maturity_years = maturity_range(1) + (maturity_range(2) - maturity_range(1)) * rand();
            rate_mult = rate_multiplier(1) + (rate_multiplier(2) - rate_multiplier(1)) * rand();
            
            K_strike = moneyness * S_current;
            T_maturity = maturity_years;
            r_current = r_base * rate_mult;
            
            % Build/get trees
            if ~isKey(tree_cache, param_idx)
                try
                    c = gamma + lambda + 0.5;
                    [hd, qhd] = genhDelta(h0_path, beta, alpha, c, omega, m_h, gamma_h);
                    [nodes_ht] = TreeNodes_ht_HN(m_ht, hd, qhd, gamma_h, alpha, beta, c, omega, N+1);
                    [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_HN(m_x, gamma_x, r_current, hd, qhd, S_current, alpha, beta, c, omega, N);
                    [q_Xt, P_Xt, ~] = Prob_Xt(nodes_ht, qhd, nodes_Xt, S_current, r_current, alpha, beta, c, omega);
                    nodes_S = exp(nodes_Xt);
                    
                    tree_data = struct('nodes_S', nodes_S, 'P_Xt', P_Xt, 'q_Xt', q_Xt);
                    tree_cache(param_idx) = tree_data;
                catch ME
                    stats.pricing_errors = stats.pricing_errors + 1;
                    continue;
                end
            end
            
            tree_data = tree_cache(param_idx);
            
            try
                % Price options
                [V_C, ~] = American(tree_data.nodes_S, tree_data.P_Xt, tree_data.q_Xt, ...
                    r_current, T_maturity, S_current, K_strike, 1);
                [V_P, ~] = American(tree_data.nodes_S, tree_data.P_Xt, tree_data.q_Xt, ...
                    r_current, T_maturity, S_current, K_strike, -1);
                
                % Calculate implied volatilities with error handling
                try
                    [impl_c, ~, ~] = impvol(S_current, K_strike, T_maturity, r_current, V_C, 1, N, m_x, gamma_x, tol, itmax);
                    [impl_p, ~, ~] = impvol(S_current, K_strike, T_maturity, r_current, V_P, -1, N, m_x, gamma_x, tol, itmax);
                catch
                    impl_c = 0.2;  % Default fallback
                    impl_p = 0.2;
                    stats.solver_failures = stats.solver_failures + 1;
                end
                
                % Store data
                common_data = [S_current; K_strike/S_current; r_current; T_maturity; 0; ...
                              alpha; beta; omega; gamma; lambda; 0; 0];
                
                % Call option
                common_data(5) = 1;
                common_data(11) = impl_c;
                common_data(12) = V_C;
                dataset(:, dataset_idx) = common_data;
                dataset_idx = dataset_idx + 1;
                
                % Put option
                common_data(5) = -1;
                common_data(11) = impl_p;
                common_data(12) = V_P;
                dataset(:, dataset_idx) = common_data;
                dataset_idx = dataset_idx + 1;
                
                stats.total_processed = stats.total_processed + 2;
                
            catch ME
                stats.pricing_errors = stats.pricing_errors + 1;
                % Store error placeholders
                for k = 1:2
                    dataset(:, dataset_idx) = [S_current; K_strike/S_current; r_current; T_maturity; (-1)^k; ...
                                              alpha; beta; omega; gamma; lambda; 0.2; 0];
                    dataset_idx = dataset_idx + 1;
                end
            end
        end
        
        stats.processing_times(end+1) = toc(batch_start_time);
        fprintf('  %s batch %d: %.2f seconds\n', dataset_type, batch, stats.processing_times(end));
        
        % Clear cache
        remove(tree_cache, keys(tree_cache));
    end
    
    % Trim dataset
    dataset = dataset(:, 1:dataset_idx-1);
    
    fprintf('%s dataset complete: %d contracts generated\n', dataset_type, size(dataset, 2));
end

function compare_datasets(train_dataset, val_dataset, train_stats, val_stats, train_filename, val_filename)
    
    fprintf('\nDATASET COMPARISON:\n');
    fprintf('==================\n');
    
    % Basic statistics
    fprintf('TRAINING SET:\n');
    fprintf('  File: %s\n', train_filename);
    fprintf('  Contracts: %d\n', size(train_dataset, 2));
    fprintf('  Processing errors: %d (%.1f%%)\n', train_stats.pricing_errors, ...
            100*train_stats.pricing_errors/size(train_dataset, 2));
    fprintf('  Processing time: %.2f minutes\n', sum(train_stats.processing_times)/60);
    
    fprintf('\nVALIDATION SET:\n');
    fprintf('  File: %s\n', val_filename);
    fprintf('  Contracts: %d\n', size(val_dataset, 2));
    fprintf('  Processing errors: %d (%.1f%%)\n', val_stats.pricing_errors, ...
            100*val_stats.pricing_errors/size(val_dataset, 2));
    fprintf('  Processing time: %.2f minutes\n', sum(val_stats.processing_times)/60);
    
    % Feature comparison
    feature_names = {'Moneyness', 'Interest_Rate', 'Time_to_Maturity', 'Alpha', 'Beta', 'Omega', 'Gamma', 'Lambda', 'Impl_Vol'};
    feature_indices = [2, 3, 4, 6, 7, 8, 9, 10, 11];
    
    fprintf('\nFEATURE RANGE COMPARISON:\n');
    fprintf('%-15s %15s %15s %10s\n', 'Feature', 'Train_Range', 'Val_Range', 'Coverage');
    fprintf('%-15s %15s %15s %10s\n', '-------', '----------', '--------', '--------');
    
    for i = 1:length(feature_indices)
        idx = feature_indices(i);
        train_vals = train_dataset(idx, :);
        val_vals = val_dataset(idx, :);
        
        train_min = min(train_vals);
        train_max = max(train_vals);
        val_min = min(val_vals);
        val_max = max(val_vals);
        
        % Coverage: how much of validation range is covered by training
        overlap_min = max(train_min, val_min);
        overlap_max = min(train_max, val_max);
        val_range = val_max - val_min;
        overlap_range = max(0, overlap_max - overlap_min);
        coverage = overlap_range / val_range * 100;
        
        fprintf('%-15s [%.4f,%.4f] [%.4f,%.4f] %8.1f%%\n', ...
                feature_names{i}, train_min, train_max, val_min, val_max, coverage);
    end
    
    % Data quality metrics
    fprintf('\nDATA QUALITY METRICS:\n');
    train_valid_iv = train_dataset(11, train_dataset(11,:) > 0.001 & train_dataset(11,:) < 2.99);
    val_valid_iv = val_dataset(11, val_dataset(11,:) > 0.001 & val_dataset(11,:) < 2.99);
    
    fprintf('TRAINING SET:\n');
    fprintf('  Valid IVs: %d/%d (%.1f%%)\n', length(train_valid_iv), size(train_dataset,2), ...
            100*length(train_valid_iv)/size(train_dataset,2));
    fprintf('  IV range: [%.4f, %.4f]\n', min(train_valid_iv), max(train_valid_iv));
    
    fprintf('VALIDATION SET:\n');
    fprintf('  Valid IVs: %d/%d (%.1f%%)\n', length(val_valid_iv), size(val_dataset,2), ...
            100*length(val_valid_iv)/size(val_dataset,2));
    fprintf('  IV range: [%.4f, %.4f]\n', min(val_valid_iv), max(val_valid_iv));
    
    % Distribution analysis
    fprintf('\nDISTRIBUTION STRESS TEST:\n');
    fprintf('  Moneyness extreme cases (Val): %.1f%% (<0.7 or >1.5)\n', ...
            100*sum(val_dataset(2,:) < 0.7 | val_dataset(2,:) > 1.5)/size(val_dataset,2));
    fprintf('  Short maturity cases (Val): %.1f%% (<0.08 years)\n', ...
            100*sum(val_dataset(4,:) < 0.08)/size(val_dataset,2));
    fprintf('  Long maturity cases (Val): %.1f%% (>1.5 years)\n', ...
            100*sum(val_dataset(4,:) > 1.5)/size(val_dataset,2));
    
    fprintf('\n=================================================================\n');
    fprintf('VALIDATION DATASET SUCCESSFULLY CREATED WITH ENHANCED STRESS CONDITIONS\n');
    fprintf('Training set covers normal market conditions\n');
    fprintf('Validation set includes extreme scenarios for robust testing\n');
    fprintf('=================================================================\n');
end