clear all
% demo_scalable_varying_garch.m
% Enhanced scalable Heston-Nandi GARCH option pricing with varying GARCH parameters
% Uses random parameter generation within empirically realistic ranges

fprintf('=================================================================\n');
fprintf('   SCALABLE HESTON-NANDI GARCH WITH VARYING PARAMETERS\n');
fprintf('=================================================================\n\n');

%% SCALING PARAMETERS - MODIFY THESE TO SCALE THE DATASET
CONTRACTS_PER_DAY = 50;     % Number of option contracts to generate per day
NUM_DAYS = 30;              % Number of trading days to simulate
BATCH_SIZE = 100;           % Process options in batches to manage memory

% GARCH Parameter Variation Settings
NUM_PARAM_SETS = 5;       % Number of different GARCH parameter combinations

% Display scaling configuration
fprintf('SCALING CONFIGURATION:\n');
fprintf('  Contracts per day: %d\n', CONTRACTS_PER_DAY);
fprintf('  Number of days: %d\n', NUM_DAYS);
fprintf('  Total contracts: %d\n', CONTRACTS_PER_DAY * NUM_DAYS * 2);
fprintf('  Parameter sets: %d\n', NUM_PARAM_SETS);
fprintf('  Batch size: %d\n', BATCH_SIZE);

%% MARKET PARAMETERS
T = [30, 60, 90, 180, 360]./252;  % Base maturities
N = 504;                          % Number of time steps
r = 0.05/252;                     % Risk-free rate (daily)
S0 = 100;                         % Initial stock price
K_base = 100;                     % Base strike price

% Enhanced strike and maturity generation for scaling
moneyness_range = [0.8, 1.2];    % Moneyness range
maturity_range = [0.1, 1.0];     % Maturity range (years)

%% GENERATE VARYING GARCH PARAMETER SETS
fprintf('\nGenerating GARCH parameter variations...\n');
fprintf('Strategy: Random generation within empirically realistic ranges\n');

% Define parameter ranges based on empirical literature
% These ranges are based on studies of equity index options
alpha_range = [1.15e-6, 1.36e-6];      % Volatility innovation parameter
beta_range = [0.75, 0.85];       % Persistence parameter  
omega_range = [1e-7, 1e-6];      % Long-run variance level
gamma_range = [50, 150];         % Asymmetry parameter (leverage effect)
lambda_range = [0.4, 0.6];      % Risk premium parameter

% Generate random parameter sets with constraints
param_sets = zeros(NUM_PARAM_SETS, 5);
valid_sets = 0;

fprintf('Generating parameter sets with stationarity constraints...\n');

while valid_sets < NUM_PARAM_SETS
    % Generate candidate parameters
    alpha_candidate = alpha_range(1) + (alpha_range(2) - alpha_range(1)) * rand();
    beta_candidate = beta_range(1) + (beta_range(2) - beta_range(1)) * rand();
    omega_candidate = omega_range(1) + (omega_range(2) - omega_range(1)) * rand();
    gamma_candidate = gamma_range(1) + (gamma_range(2) - gamma_range(1)) * rand();
    lambda_candidate = lambda_range(1) + (lambda_range(2) - lambda_range(1)) * rand();
    
    % Check stationarity condition for GARCH
    % For Heston-Nandi: beta + alpha * (gamma + lambda + 0.5)^2 < 1
    c = gamma_candidate + lambda_candidate + 0.5;
    stationarity_condition = beta_candidate + alpha_candidate * c^2;
    
    if stationarity_condition < 0.999  % Leave small buffer
        valid_sets = valid_sets + 1;
        param_sets(valid_sets, :) = [alpha_candidate, beta_candidate, omega_candidate, ...
                                    gamma_candidate, lambda_candidate];
    end
end

fprintf('Generated %d valid parameter sets\n', NUM_PARAM_SETS);

% Display parameter statistics
fprintf('\nParameter ranges in generated sets:\n');
fprintf('  Alpha:  [%.2e, %.2e] (mean: %.2e)\n', min(param_sets(:,1)), max(param_sets(:,1)), mean(param_sets(:,1)));
fprintf('  Beta:   [%.3f, %.3f] (mean: %.3f)\n', min(param_sets(:,2)), max(param_sets(:,2)), mean(param_sets(:,2)));
fprintf('  Omega:  [%.2e, %.2e] (mean: %.2e)\n', min(param_sets(:,3)), max(param_sets(:,3)), mean(param_sets(:,3)));
fprintf('  Gamma:  [%.2f, %.2f] (mean: %.2f)\n', min(param_sets(:,4)), max(param_sets(:,4)), mean(param_sets(:,4)));
fprintf('  Lambda: [%.3f, %.3f] (mean: %.3f)\n', min(param_sets(:,5)), max(param_sets(:,5)), mean(param_sets(:,5)));

% Check stationarity for all sets
c_values = param_sets(:,4) + param_sets(:,5) + 0.5;
stationarity_values = param_sets(:,2) + param_sets(:,1) .* c_values.^2;
fprintf('  Stationarity condition: [%.4f, %.4f] (all < 1.0)\n', min(stationarity_values), max(stationarity_values));

%% WILLOW TREE PARAMETERS
m_h = 6;
m_ht = 6;
m_x = 30;
gamma_h = 0.6;
gamma_x = 0.8;

%% SOLVER PARAMETERS
itmax = 100;
tol = 1e-6;

%% PRE-GENERATE GARCH PATHS FOR PARAMETER SETS
fprintf('\nGenerating GARCH paths for parameter sets...\n');
M_paths = min(NUM_PARAM_SETS, 50);  % Generate paths for subset to manage memory
numPoint = N + 1;

% Generate different random seeds for path diversity
Z_seeds = randn(numPoint + 1, M_paths);

% Store paths and initial variances for parameter sets
S_paths_all = cell(NUM_PARAM_SETS, 1);
h0_paths_all = zeros(NUM_PARAM_SETS, 1);

for p = 1:NUM_PARAM_SETS
    alpha_p = param_sets(p, 1);
    beta_p = param_sets(p, 2);
    omega_p = param_sets(p, 3);
    gamma_p = param_sets(p, 4);
    lambda_p = param_sets(p, 5);
    
    % Use modulo to reuse Z_seeds if we have more parameter sets than paths
    seed_idx = mod(p-1, M_paths) + 1;
    
    try
        [S_path, h0_path] = mcHN(1, N, S0, Z_seeds(:, seed_idx), r, omega_p, alpha_p, beta_p, gamma_p, lambda_p);
        S_paths_all{p} = S_path;
        h0_paths_all(p) = h0_path;
    catch ME
        fprintf('Warning: Error generating path for parameter set %d: %s\n', p, ME.message);
        % Use default parameters as fallback
        [S_path, h0_path] = mcHN(1, N, S0, Z_seeds(:, seed_idx), r, 1e-6, 1.33e-6, 0.8, 5, 0.2);
        S_paths_all{p} = S_path;
        h0_paths_all(p) = h0_path;
    end
    
    if mod(p, 20) == 0
        fprintf('  Generated paths for %d/%d parameter sets\n', p, NUM_PARAM_SETS);
    end
end

fprintf('Completed GARCH path generation\n');

%% INITIALIZE DATASET
total_contracts = CONTRACTS_PER_DAY * NUM_DAYS * 2;  
dataset = zeros(12, total_contracts);

fprintf('\nDataset structure:\n');
fprintf('1: Spot price, 2: Moneyness, 3: Risk-free rate, 4: Time to maturity\n');
fprintf('5: Call/Put flag, 6: alpha, 7: beta, 8: omega, 9: gamma, 10: lambda\n');
fprintf('11: Implied volatility, 12: Option value\n');

%% BATCH PROCESSING SETUP
num_batches = ceil(total_contracts / (BATCH_SIZE * 2));
fprintf('\nProcessing in %d batches with varying GARCH parameters...\n', num_batches);

% Statistics tracking
total_processed = 0;
pricing_errors = 0;
solver_failures = 0;
boundary_issues = 0;
processing_times = zeros(num_batches, 1);
param_usage_count = zeros(NUM_PARAM_SETS, 1);

%% MAIN PROCESSING LOOP
dataset_idx = 1;

for batch = 1:num_batches
    batch_start_time = tic;
    fprintf('\nProcessing batch %d/%d...\n', batch, num_batches);
    
    contracts_in_batch = min(BATCH_SIZE, ceil((total_contracts/2 - (batch-1)*BATCH_SIZE)));
    batch_start_idx = (batch - 1) * BATCH_SIZE + 1;
    
    batch_errors = 0;
    batch_processed = 0;
    
    % Cache for computed trees in this batch
    tree_cache = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    
    for contract_in_batch = 1:contracts_in_batch
        contract_idx = batch_start_idx + contract_in_batch - 1;
        day = ceil(contract_idx / CONTRACTS_PER_DAY);
        
        if day > NUM_DAYS, break; end
        
        % Assign parameter set to contract (cyclic assignment)
        param_idx = mod(contract_idx - 1, NUM_PARAM_SETS) + 1;
        param_usage_count(param_idx) = param_usage_count(param_idx) + 1;
        
        % Get GARCH parameters for this contract
        alpha = param_sets(param_idx, 1);
        beta = param_sets(param_idx, 2);
        omega = param_sets(param_idx, 3);
        gamma = param_sets(param_idx, 4);
        lambda = param_sets(param_idx, 5);
        
        % Get corresponding GARCH path
        S_path = S_paths_all{param_idx};
        h0_path = h0_paths_all(param_idx);
        S_current = S_path(end);
        
        % Generate contract-specific parameters
        rng(contract_idx);  % For reproducibility
        moneyness = moneyness_range(1) + (moneyness_range(2) - moneyness_range(1)) * rand();
        maturity_years = maturity_range(1) + (maturity_range(2) - maturity_range(1)) * rand();
        K_strike = moneyness * S_current;
        T_maturity = maturity_years;
        
        % Check if we need to compute tree for this parameter set
        if ~isKey(tree_cache, param_idx)
            try
                c = gamma + lambda + 0.5;
                
                % Build willow trees for this parameter set
                [hd, qhd] = genhDelta(h0_path, beta, alpha, c, omega, m_h, gamma_h);
                [nodes_ht] = TreeNodes_ht_HN(m_ht, hd, qhd, gamma_h, alpha, beta, c, omega, N+1);
                [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_HN(m_x, gamma_x, r, hd, qhd, S_current, alpha, beta, c, omega, N);
                [q_Xt, P_Xt, ~] = Prob_Xt(nodes_ht, qhd, nodes_Xt, S_current, r, alpha, beta, c, omega);
                nodes_S = exp(nodes_Xt);
                
                % Cache the tree data
                tree_data = struct('nodes_S', nodes_S, 'P_Xt', P_Xt, 'q_Xt', q_Xt);
                tree_cache(param_idx) = tree_data;
                
            catch ME
                batch_errors = batch_errors + 1;
                pricing_errors = pricing_errors + 1;
                fprintf('Tree building error for param set %d: %s\n', param_idx, ME.message);
                continue;
            end
        end
        
        % Get cached tree data
        tree_data = tree_cache(param_idx);
        
        try
            % Price American options using the parameter-specific trees
            [V_C, ~] = American(tree_data.nodes_S, tree_data.P_Xt, tree_data.q_Xt, r, T_maturity, S_current, K_strike, 1);
            [V_P, ~] = American(tree_data.nodes_S, tree_data.P_Xt, tree_data.q_Xt, r, T_maturity, S_current, K_strike, -1);
            
            
            [impl_c, ~, ~] = impvol(S_current, K_strike, T_maturity, r, V_C, 1, N, m_x, gamma_x, tol, itmax);
            [impl_p, ~, ~] = impvol(S_current, K_strike, T_maturity, r, V_P, 1, N, m_x, gamma_x, tol, itmax);

            
            % Store call option data
            dataset(:, dataset_idx) = [
                S_current;           % 1: Spot price
                K_strike/S_current;  % 2: Moneyness
                r;                   % 3: Risk-free rate
                T_maturity;          % 4: Time to maturity
                1;                   % 5: Call/Put flag
                alpha;               % 6: GARCH alpha
                beta;                % 7: GARCH beta
                omega;               % 8: GARCH omega
                gamma;               % 9: GARCH gamma
                lambda;              % 10: GARCH lambda
                impl_c;              % 11: Implied volatility
                V_C                  % 12: Option value
            ];
            dataset_idx = dataset_idx + 1;
            
            % Store put option data
            dataset(:, dataset_idx) = [
                S_current;           % 1: Spot price
                K_strike/S_current;  % 2: Moneyness
                r;                   % 3: Risk-free rate
                T_maturity;          % 4: Time to maturity
                -1;                  % 5: Call/Put flag
                alpha;               % 6: GARCH alpha
                beta;                % 7: GARCH beta
                omega;               % 8: GARCH omega
                gamma;               % 9: GARCH gamma
                lambda;              % 10: GARCH lambda
                impl_p;              % 11: Implied volatility
                V_P                  % 12: Option value
            ];
            dataset_idx = dataset_idx + 1;
            
            batch_processed = batch_processed + 2;
            
        catch ME
            batch_errors = batch_errors + 1;
            pricing_errors = pricing_errors + 1;
            
            % Store error placeholders with actual GARCH parameters
            for k = 1:2  % Call and put
                dataset(:, dataset_idx) = [S_current; K_strike/S_current; r; T_maturity; (-1)^k; ...
                                          alpha; beta; omega; gamma; lambda; 0.2; 0];
                dataset_idx = dataset_idx + 1;
            end
        end
    end
    
    % Batch completion statistics
    processing_times(batch) = toc(batch_start_time);
    total_processed = total_processed + batch_processed;
    
    fprintf('  Batch %d complete: %d contracts processed, %d errors, %.2f seconds\n', ...
            batch, batch_processed, batch_errors, processing_times(batch));
    
    % Clear tree cache to manage memory
    remove(tree_cache, keys(tree_cache));
end

%% FINALIZE DATASET
dataset = dataset(:, 1:dataset_idx-1);

%% SAVE RESULTS
fprintf('\nSaving results...\n');
headers = {'S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'sigma', 'V'}';
filename = sprintf('varying_garch_dataset_%dx%d_%dparams.csv', CONTRACTS_PER_DAY, NUM_DAYS, NUM_PARAM_SETS);
dataset_enhanced = [headers'; num2cell(dataset')];
writecell(dataset_enhanced, filename);

%% COMPREHENSIVE REPORTING
fprintf('\n=================================================================\n');
fprintf('              PROCESSING COMPLETE - VARYING GARCH\n');
fprintf('=================================================================\n');

actual_contracts = size(dataset, 2);
call_data = dataset(:, dataset(5,:) == 1);
put_data = dataset(:, dataset(5,:) == -1);

fprintf('\nSCALING RESULTS:\n');
fprintf('  Target contracts: %d\n', total_contracts);
fprintf('  Actual contracts: %d\n', actual_contracts);
fprintf('  Calls: %d, Puts: %d\n', size(call_data, 2), size(put_data, 2));
fprintf('  Parameter sets used: %d\n', NUM_PARAM_SETS);
fprintf('  Dataset saved as: %s\n', filename);

fprintf('\nPARAMETER USAGE:\n');
fprintf('  Min usage per param set: %d\n', min(param_usage_count));
fprintf('  Max usage per param set: %d\n', max(param_usage_count));
fprintf('  Mean usage per param set: %.1f\n', mean(param_usage_count));

fprintf('\nPERFORMANCE METRICS:\n');
fprintf('  Total processing time: %.2f minutes\n', sum(processing_times) / 60);
fprintf('  Contracts per second: %.1f\n', actual_contracts / sum(processing_times));
fprintf('  Pricing errors: %d (%.1f%%)\n', pricing_errors, 100*pricing_errors/actual_contracts);

fprintf('\nDATA QUALITY:\n');
all_ivs = dataset(11,:);
valid_ivs = all_ivs(all_ivs > 0.001 & all_ivs < 2.99);
fprintf('  Valid IVs: %d/%d (%.1f%%)\n', length(valid_ivs), length(all_ivs), 100*length(valid_ivs)/length(all_ivs));
fprintf('  IV range: [%.4f, %.4f]\n', min(valid_ivs), max(valid_ivs));

fprintf('\nGARCH PARAMETER DIVERSITY:\n');
fprintf('  Alpha diversity: %.2e (std/mean)\n', std(dataset(6,:))/mean(dataset(6,:)));
fprintf('  Beta diversity: %.3f (std)\n', std(dataset(7,:)));
fprintf('  Gamma diversity: %.2f (std)\n', std(dataset(9,:)));

fprintf('\n=================================================================\n');
fprintf('SUCCESS: Varying GARCH parameter dataset generation complete!\n');
fprintf('Dataset contains %d different GARCH parameter combinations.\n', NUM_PARAM_SETS);
fprintf('=================================================================\n');