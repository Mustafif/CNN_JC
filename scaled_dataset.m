function scaled_dataset(model_type, varargin)
% SCALED_DATASET Centralized option dataset generator for multiple GARCH models
%
% SYNTAX:
%   scaled_dataset('HN', ...)
%   scaled_dataset('Duan', ...)
%   scaled_dataset('GJR', ...)
%
% MODELS:
%   'HN'   - Heston-Nandi GARCH
%   'Duan' - Duan GARCH
%   'GJR'  - GJR-GARCH
%
% OPTIONAL NAME-VALUE PARAMETERS:
%   'ContractsPerDay'  - Number of contracts per day (default: 50)
%   'NumDays'          - Number of trading days (default: 20)
%   'BatchSize'        - Batch processing size (default: 100)
%   'NumParamSets'     - Number of GARCH parameter sets (default: 5)
%   'OutputFile'       - Output CSV filename (default: 'dataset_<model>.csv')
%   'MoneynessRange'   - [min max] moneyness (default: [0.8, 1.2])
%   'MaturityDays'     - Array of maturity days (default: [30, 60, 90, 180, 360])
%
% EXAMPLES:
%   % Generate HN-GARCH dataset with defaults
%   scaled_dataset('HN')
%
%   % Generate Duan GARCH dataset with custom parameters
%   scaled_dataset('Duan', 'ContractsPerDay', 100, 'NumDays', 30)
%
%   % Generate GJR-GARCH dataset with custom maturities
%   scaled_dataset('GJR', 'MaturityDays', [7, 14, 30, 60, 90])

fprintf('=================================================================\n');
fprintf('    CENTRALIZED GARCH OPTION PRICING DATASET GENERATOR\n');
fprintf('=================================================================\n\n');

%% PARSE INPUT ARGUMENTS
p = inputParser;
addRequired(p, 'model_type', @(x) ischar(x) && ismember(upper(x), {'HN', 'DUAN', 'GJR'}));
addParameter(p, 'ContractsPerDay', 50, @isnumeric);
addParameter(p, 'NumDays', 20, @isnumeric);
addParameter(p, 'BatchSize', 100, @isnumeric);
addParameter(p, 'NumParamSets', 5, @isnumeric);
addParameter(p, 'OutputFile', '', @ischar);
addParameter(p, 'MoneynessRange', [0.8, 1.2], @(x) isnumeric(x) && length(x)==2);
addParameter(p, 'MaturityDays', [30, 60, 90, 180, 360], @(x) isnumeric(x));

parse(p, model_type, varargin{:});

% Extract parameters
model_type = upper(p.Results.model_type);
CONTRACTS_PER_DAY = p.Results.ContractsPerDay;
NUM_DAYS = p.Results.NumDays;
BATCH_SIZE = p.Results.BatchSize;
NUM_PARAM_SETS = p.Results.NumParamSets;
moneyness_range = p.Results.MoneynessRange;
maturity_days = p.Results.MaturityDays;

% Create output folder with timestamp
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_folder = sprintf('generation_%s_%s', lower(model_type), timestamp);

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Set default output filename if not provided
if isempty(p.Results.OutputFile)
    output_file = sprintf('%s/dataset_%s.csv', output_folder, lower(model_type));
else
    output_file = sprintf('%s/%s', output_folder, p.Results.OutputFile);
end

%% DISPLAY CONFIGURATION
fprintf('MODEL SELECTED: %s GARCH\n', model_type);
fprintf('\nSCALING CONFIGURATION:\n');
fprintf('  Contracts per day: %d\n', CONTRACTS_PER_DAY);
fprintf('  Number of days: %d\n', NUM_DAYS);
fprintf('  Total contracts: %d (x2 for calls/puts)\n', CONTRACTS_PER_DAY * NUM_DAYS);
fprintf('  Batch size: %d\n', BATCH_SIZE);
fprintf('  Parameter sets: %d\n', NUM_PARAM_SETS);
fprintf('  Moneyness range: [%.2f, %.2f]\n', moneyness_range(1), moneyness_range(2));
fprintf('  Maturity days: [%s]\n', num2str(maturity_days));
fprintf('  Output folder: %s\n', output_folder);

%% MARKET PARAMETERS
N = 1260;                   % Number of time steps
r = 0.05/252;              % Risk-free rate (daily)
S0 = 100;                  % Initial stock price

%% WILLOW TREE PARAMETERS
m_h = 6;
m_x = 30;
gamma_h = 0.6;
gamma_x = 0.8;

%% SOLVER PARAMETERS
tol = 1e-4;
itmax = 60;

%% GENERATE GARCH PARAMETER SETS
fprintf('\nGenerating %d sets of %s GARCH parameters...\n', NUM_PARAM_SETS, model_type);

switch model_type
    case 'HN'
        % Add path to root data_gen if needed
        addpath(genpath('./data_gen'));
        param_sets = genHNGarchParams(NUM_PARAM_SETS);
        param_names = {'alpha', 'beta', 'omega', 'gamma', 'lambda'};
        
    case 'DUAN'
        % Add path to Duan_Garch folder
        addpath(genpath('./data_gen/Duan_Garch'));
        param_sets = genDuanGarchParams(NUM_PARAM_SETS);
        param_names = {'alpha', 'beta', 'omega', 'theta', 'lambda'};
        
    case 'GJR'
        % Add path to GJR_Garch folder
        addpath(genpath('./data_gen/GJR_Garch'));
        param_sets = genGJRGarchParams(NUM_PARAM_SETS);
        param_names = {'alpha', 'beta', 'omega', 'gamma', 'lambda'};
end

fprintf('Parameter sets generated successfully.\n');

%% GENERATE ASSET PRICE PATHS FOR EACH PARAMETER SET
fprintf('\nGenerating asset price paths...\n');
M = NUM_DAYS;  % One path per day
numPoint = N + 1;

% Store paths and initial variance for each parameter set
S_paths_all = cell(NUM_PARAM_SETS, 1);
h0_all = zeros(NUM_PARAM_SETS, 1);

for param_idx = 1:NUM_PARAM_SETS
    alpha = param_sets(param_idx, 1);
    beta = param_sets(param_idx, 2);
    omega = param_sets(param_idx, 3);
    param4 = param_sets(param_idx, 4);  % gamma, theta, or 0
    lambda = param_sets(param_idx, 5);
    
    % Generate random numbers for simulation
    rng(param_idx * 1000);  % Seed for reproducibility
    Z = randn(numPoint + 1, M);
    
    % Call appropriate Monte Carlo function
    switch model_type
        case 'HN'
            gamma = param4;
            [S_paths, h0_paths] = mcHN(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);
            
        case 'DUAN'
            theta = param4;
            [S_paths, h0_paths] = mcDuan(M, N, S0, Z, r, omega, alpha, beta, theta, lambda);
            
        case 'GJR'
            [S_paths, h0_paths] = mcGJR(M, N, S0, Z, r, omega, alpha, beta, lambda);
    end
    
    S_paths_all{param_idx} = S_paths;
    h0_all(param_idx) = h0_paths(1);
    
    fprintf('  Parameter set %d/%d: paths generated\n', param_idx, NUM_PARAM_SETS);
end

%% INITIALIZE DATASET
total_contracts = CONTRACTS_PER_DAY * NUM_DAYS * 2;  % x2 for calls and puts
dataset = zeros(12, total_contracts);

% Dataset structure:
% 1: Spot price, 2: Moneyness, 3: Risk-free rate, 4: Time to maturity
% 5: Call/Put flag, 6: alpha, 7: beta, 8: omega, 9: param4, 10: lambda
% 11: Implied volatility, 12: Option value

fprintf('\nInitializing dataset structure...\n');
fprintf('Dataset dimensions: %d metrics × %d contracts\n', size(dataset, 1), size(dataset, 2));

%% PRE-BUILD WILLOW TREES FOR EACH PARAMETER SET (OPTIMIZATION)
fprintf('\n=================================================================\n');
fprintf('                 PRE-BUILDING WILLOW TREES\n');
fprintf('=================================================================\n');
fprintf('Building trees once per parameter set (optimization)...\n\n');

tree_cache = cell(NUM_PARAM_SETS, 1);

for param_idx = 1:NUM_PARAM_SETS
    fprintf('  Building tree %d/%d...', param_idx, NUM_PARAM_SETS);
    
    alpha = param_sets(param_idx, 1);
    beta = param_sets(param_idx, 2);
    omega = param_sets(param_idx, 3);
    param4 = param_sets(param_idx, 4);
    lambda = param_sets(param_idx, 5);
    h0 = h0_all(param_idx);
    
    % Get a representative spot price for tree building
    S_paths = S_paths_all{param_idx};
    S_rep = mean(S_paths(end, :));  % Use mean final price as representative
    
    try
        switch model_type
            case 'HN'
                gamma = param4;
                c = gamma + lambda + 0.5;
                
                [hd, qhd] = genhDelta(h0, beta, alpha, c, omega, m_h, gamma_h);
                [nodes_ht] = TreeNodes_ht_HN(m_h, hd, qhd, gamma_h, alpha, beta, c, omega, N+1);
                [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_HN(m_x, gamma_x, r, hd, qhd, S_rep, alpha, beta, c, omega, N);
                [q_Xt, P_Xt, ~] = Prob_Xt(nodes_ht, qhd, nodes_Xt, S_rep, r, alpha, beta, c, omega);
                
                tree_cache{param_idx} = struct('hd', hd, 'qhd', qhd, 'nodes_ht', nodes_ht, ...
                                               'alpha', alpha, 'beta', beta, 'omega', omega, ...
                                               'c', c, 'r', r, 'h0', h0);
                
            case 'DUAN'
                theta = param4;
                
                [nodes_ht, qht, hmom1, hmom2, hmom3, hmom4_app] = TreeNodes_ht_D(m_h, h0, gamma_h, omega, beta, alpha, theta, lambda, N+1);
                
                tree_cache{param_idx} = struct('nodes_ht', nodes_ht, 'qht', qht, ...
                                               'hmom1', hmom1, 'hmom2', hmom2, ...
                                               'alpha', alpha, 'beta', beta, 'omega', omega, ...
                                               'theta', theta, 'lambda', lambda, 'r', r, 'h0', h0);
                
            case 'GJR'
                [nodes_ht, qht, hmom1, hmom2, hmom3, hmom4_app] = TreeNodes_ht_GJR(m_h, h0, gamma_h, omega, beta, alpha, lambda, N);
                
                tree_cache{param_idx} = struct('nodes_ht', nodes_ht, 'qht', qht, ...
                                               'hmom1', hmom1, 'hmom2', hmom2, ...
                                               'alpha', alpha, 'beta', beta, 'omega', omega, ...
                                               'lambda', lambda, 'r', r, 'h0', h0);
        end
        
        fprintf(' ✓\n');
    catch ME
        fprintf(' ✗ Error: %s\n', ME.message);
        tree_cache{param_idx} = [];
    end
end

fprintf('\nAll trees pre-built successfully!\n');

%% PREPARE TREE STRUCTURES AND PROCESS CONTRACTS
fprintf('\n=================================================================\n');
fprintf('                    PRICING OPTIONS\n');
fprintf('=================================================================\n');

num_batches = ceil(total_contracts / (BATCH_SIZE * 2));
fprintf('Processing in %d batches...\n', num_batches);

dataset_idx = 1;
total_processed = 0;
pricing_errors = 0;

for batch = 1:num_batches
    batch_start_time = tic;
    fprintf('\nBatch %d/%d...\n', batch, num_batches);
    
    contracts_in_batch = min(BATCH_SIZE, ceil((total_contracts/2 - (batch-1)*BATCH_SIZE)));
    batch_start_idx = (batch - 1) * BATCH_SIZE + 1;
    
    batch_errors = 0;
    batch_processed = 0;
    
    for contract_in_batch = 1:contracts_in_batch
        contract_idx = batch_start_idx + contract_in_batch - 1;
        
        % Determine which day and parameter set
        day = ceil(contract_idx / CONTRACTS_PER_DAY);
        if day > NUM_DAYS
            break;
        end
        
        % Cycle through parameter sets
        param_idx = mod(contract_idx - 1, NUM_PARAM_SETS) + 1;
        
        % Get cached tree for this parameter set
        tree_struct = tree_cache{param_idx};
        if isempty(tree_struct)
            batch_errors = batch_errors + 2;
            pricing_errors = pricing_errors + 2;
            continue;
        end
        
        % Get parameters
        alpha = param_sets(param_idx, 1);
        beta = param_sets(param_idx, 2);
        omega = param_sets(param_idx, 3);
        param4 = param_sets(param_idx, 4);
        lambda = param_sets(param_idx, 5);
        h0 = h0_all(param_idx);
        
        % Get current spot price
        S_paths = S_paths_all{param_idx};
        day_in_set = mod(day - 1, M) + 1;
        S_current = S_paths(end, day_in_set);
        
        % Generate random contract parameters
        rng(contract_idx + param_idx * 10000);
        moneyness = moneyness_range(1) + (moneyness_range(2) - moneyness_range(1)) * rand();
        
        % Randomly select from discrete maturity days
        maturity_idx = randi(length(maturity_days));
        T_days = maturity_days(maturity_idx);
        T_maturity = T_days / 252;  % Convert days to years (252 trading days)
        
        K_strike = moneyness * S_current;
        
        try
            % Use cached trees and only rebuild S-dependent parts
            switch model_type
                case 'HN'
                    gamma = param4;
                    % Retrieve cached tree components
                    hd = tree_struct.hd;
                    qhd = tree_struct.qhd;
                    nodes_ht = tree_struct.nodes_ht;
                    c = tree_struct.c;
                    
                    % Rebuild only S-dependent components
                    [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_HN(m_x, gamma_x, r, hd, qhd, S_current, alpha, beta, c, omega, N);
                    [q_Xt, P_Xt, ~] = Prob_Xt(nodes_ht, qhd, nodes_Xt, S_current, r, alpha, beta, c, omega);
                    nodes_S = exp(nodes_Xt);
                    
                    % Price options
                    [V_C, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, 1);
                    [V_P, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, -1);
                    
                    % Compute implied volatility
                    [impl_c, ~, ~] = impvol(S_current, K_strike, T_maturity, r, V_C, 1, N, m_x, gamma_x, tol, itmax);
                    [impl_p, ~, ~] = impvol(S_current, K_strike, T_maturity, r, V_P, -1, N, m_x, gamma_x, tol, itmax);
                    
                case 'DUAN'
                    theta = param4;
                    % Retrieve cached tree components
                    nodes_ht = tree_struct.nodes_ht;
                    qht = tree_struct.qht;
                    hmom1 = tree_struct.hmom1;
                    hmom2 = tree_struct.hmom2;
                    
                    % Rebuild S-dependent components
                    [nodes_Xt, mu, var, k3, k4] = TreeNodes_logSt_D(m_x, gamma_x, r, h0, omega, beta, alpha, theta, lambda, N, hmom1, hmom2);
                    nodes_Xt = nodes_Xt + log(S_current);
                    [q_Xt, P_Xt, tmpHt] = Probility_Xt2(nodes_ht, qht, nodes_Xt, S_current, r, omega, beta, alpha, theta, lambda);
                    nodes_S = exp(nodes_Xt);
                    
                    % Price options
                    [V_C, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, 1);
                    [V_P, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, -1);
                    
                    % Compute implied volatility
                    [impl_c, ~, ~] = impvol(S_current, K_strike, T_maturity, r, V_C, 1, N, m_x, gamma_x, tol, itmax);
                    [impl_p, ~, ~] = impvol(S_current, K_strike, T_maturity, r, V_P, -1, N, m_x, gamma_x, tol, itmax);
                    
                case 'GJR'
                    % Retrieve cached tree components
                    nodes_ht = tree_struct.nodes_ht;
                    qht = tree_struct.qht;
                    hmom1 = tree_struct.hmom1;
                    hmom2 = tree_struct.hmom2;
                    mu = r;
                    
                    % Rebuild S-dependent components
                    [nodes_Xt, mu1, var1, k31, k41] = TreeNodes_logSt_GJR(m_x, m_h, gamma_x, mu, omega, alpha, lambda, beta, h0, N, hmom1, hmom2);
                    nodes_Xt = nodes_Xt + log(S_current);
                    nodes_S = exp(nodes_Xt);
                    [q_Xt, P_Xt, tmpHt] = Probility_Xt2(nodes_ht, qht, nodes_Xt, S_current, mu, omega, beta, alpha, lambda);
                    
                    % Price options
                    [V_C, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, 1);
                    [V_P, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, -1);
                    
                    % Compute implied volatility
                    [impl_c, ~, ~] = impvol(S_current, K_strike, T_maturity, r, V_C, 1, N, m_x, gamma_x, tol, itmax);
                    [impl_p, ~, ~] = impvol(S_current, K_strike, T_maturity, r, V_P, -1, N, m_x, gamma_x, tol, itmax);
            end
            
            % Validate option prices
            if V_C < 0 || V_P < 0 || isnan(V_C) || isnan(V_P) || ...
               isnan(impl_c) || isnan(impl_p)
                batch_errors = batch_errors + 1;
                pricing_errors = pricing_errors + 1;
                continue;
            end
            
            % Store call option data
            dataset(:, dataset_idx) = [
                S_current;           % 1: Spot price
                K_strike/S_current;  % 2: Moneyness
                r;                   % 3: Risk-free rate
                T_maturity;          % 4: Time to maturity
                1;                   % 5: Call flag
                alpha;               % 6: GARCH alpha
                beta;                % 7: GARCH beta
                omega;               % 8: GARCH omega
                param4;              % 9: gamma/theta/0
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
                -1;                  % 5: Put flag
                alpha;               % 6: GARCH alpha
                beta;                % 7: GARCH beta
                omega;               % 8: GARCH omega
                param4;              % 9: gamma/theta/0
                lambda;              % 10: GARCH lambda
                impl_p;              % 11: Implied volatility
                V_P                  % 12: Option value
            ];
            dataset_idx = dataset_idx + 1;
            
            batch_processed = batch_processed + 2;
            
        catch ME
            batch_errors = batch_errors + 1;
            pricing_errors = pricing_errors + 1;
            
            % Store error placeholders
            for k = 1:2  % Call and put
                dataset(:, dataset_idx) = [S_current; K_strike/S_current; r; T_maturity; ...
                                          (-1)^k; alpha; beta; omega; param4; lambda; 0.2; 0];
                dataset_idx = dataset_idx + 1;
            end
        end
    end
    
    total_processed = total_processed + batch_processed;
    batch_time = toc(batch_start_time);
    
    fprintf('  Batch complete: %d contracts, %d errors, %.2f sec\n', ...
            batch_processed, batch_errors, batch_time);
end

%% FINALIZE DATASET
dataset = dataset(:, 1:dataset_idx-1);

%% CLEAN DATASET
fprintf('\n=================================================================\n');
fprintf('                    CLEANING DATASET\n');
fprintf('=================================================================\n');

original_size = size(dataset, 2);
fprintf('Original dataset size: %d contracts\n', original_size);

% Filter criteria:
% 1. Remove contracts with V < 0.5
% 2. Remove contracts with T == 1/252 (avoid daily options)
% 3. Remove contracts with invalid values (NaN, Inf)

valid_mask = (dataset(12, :) >= 0.5) & ...                    % V >= 0.5
             (abs(dataset(4, :) - 1/252) > 1e-10) & ...       % T != 1/252
             all(~isnan(dataset), 1) & ...                     % No NaN
             all(~isinf(dataset), 1);                          % No Inf

removed_low_value = sum(dataset(12, :) < 0.5);
removed_daily = sum(abs(dataset(4, :) - 1/252) < 1e-10);
removed_invalid = sum(any(isnan(dataset), 1) | any(isinf(dataset), 1));

dataset_clean = dataset(:, valid_mask);

fprintf('\nCleaning statistics:\n');
fprintf('  Removed (V < 0.5):          %d contracts\n', removed_low_value);
fprintf('  Removed (T = 1/252):        %d contracts\n', removed_daily);
fprintf('  Removed (NaN/Inf):          %d contracts\n', removed_invalid);
fprintf('  Total removed:              %d contracts (%.2f%%)\n', ...
        original_size - size(dataset_clean, 2), ...
        100*(original_size - size(dataset_clean, 2))/original_size);
fprintf('  Clean dataset size:         %d contracts\n', size(dataset_clean, 2));

%% DISPLAY SUMMARY STATISTICS
fprintf('\n=================================================================\n');
fprintf('                    DATASET STATISTICS (CLEAN)\n');
fprintf('=================================================================\n');
fprintf('Model: %s GARCH\n', model_type);
fprintf('Total contracts processed: %d\n', total_processed);
fprintf('Pricing errors: %d (%.2f%%)\n', pricing_errors, 100*pricing_errors/total_contracts);
fprintf('Final clean dataset size: %d contracts\n', size(dataset_clean, 2));

% Compute statistics on clean dataset
if size(dataset_clean, 2) > 0
    fprintf('\n--- Implied Volatility Statistics ---\n');
    calls_mask = dataset_clean(5, :) == 1;
    puts_mask = dataset_clean(5, :) == -1;
    
    if sum(calls_mask) > 0
        call_ivs = dataset_clean(11, calls_mask);
        fprintf('Calls - Mean IV: %.4f, Std: %.4f, Min: %.4f, Max: %.4f\n', ...
                mean(call_ivs), std(call_ivs), min(call_ivs), max(call_ivs));
    end
    
    if sum(puts_mask) > 0
        put_ivs = dataset_clean(11, puts_mask);
        fprintf('Puts  - Mean IV: %.4f, Std: %.4f, Min: %.4f, Max: %.4f\n', ...
                mean(put_ivs), std(put_ivs), min(put_ivs), max(put_ivs));
    end
    
    fprintf('\n--- Option Price Statistics ---\n');
    if sum(calls_mask) > 0
        call_prices = dataset_clean(12, calls_mask);
        fprintf('Calls - Mean: %.4f, Std: %.4f, Min: %.4f, Max: %.4f\n', ...
                mean(call_prices), std(call_prices), min(call_prices), max(call_prices));
    end
    
    if sum(puts_mask) > 0
        put_prices = dataset_clean(12, puts_mask);
        fprintf('Puts  - Mean: %.4f, Std: %.4f, Min: %.4f, Max: %.4f\n', ...
                mean(put_prices), std(put_prices), min(put_prices), max(put_prices));
    end
    
    fprintf('\n--- Maturity Statistics ---\n');
    maturities = dataset_clean(4, :);
    fprintf('T (years) - Mean: %.4f, Std: %.4f, Min: %.4f, Max: %.4f\n', ...
            mean(maturities), std(maturities), min(maturities), max(maturities));
    
    fprintf('\n--- Moneyness Statistics ---\n');
    moneyness = dataset_clean(2, :);
    fprintf('Moneyness - Mean: %.4f, Std: %.4f, Min: %.4f, Max: %.4f\n', ...
            mean(moneyness), std(moneyness), min(moneyness), max(moneyness));
end

%% SAVE RESULTS
fprintf('\n=================================================================\n');
fprintf('                    SAVING RESULTS\n');
fprintf('=================================================================\n');

% Create headers based on model type
switch model_type
    case 'HN'
        param4_name = 'gamma';
    case 'DUAN'
        param4_name = 'theta';
    case 'GJR'
        param4_name = 'gamma';
end

% Save cleaned dataset
fprintf('\n1. Saving cleaned dataset...\n');
headers = {'S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', param4_name, 'lambda', 'sigma', 'V'}';
dataset_with_headers = [headers'; num2cell(dataset_clean')];
write_csv_compat(dataset_with_headers, output_file);
fprintf('   ✓ Dataset saved to: %s\n', output_file);

% Save GARCH parameters
fprintf('\n2. Saving GARCH parameters...\n');
param_file = sprintf('%s/garch_params.csv', output_folder);
param_headers = {'Set', 'alpha', 'beta', 'omega', param4_name, 'lambda'}';
param_data = [(1:NUM_PARAM_SETS)', param_sets];
param_table = [param_headers'; num2cell(param_data)];
write_csv_compat(param_table, param_file);
fprintf('   ✓ GARCH parameters saved to: %s\n', param_file);

% Save asset price paths
fprintf('\n3. Saving asset price paths...\n');
assets_file = sprintf('%s/asset_prices.csv', output_folder);

% Combine all paths from different parameter sets
% Create a matrix with columns: Day, ParamSet, Price
assets_data = [];
for param_idx = 1:NUM_PARAM_SETS
    S_paths = S_paths_all{param_idx};
    num_paths = size(S_paths, 2);
    for path_idx = 1:num_paths
        day = path_idx;
        price = S_paths(end, path_idx);
        assets_data = [assets_data; day, param_idx, price];
    end
end

assets_headers = {'Day', 'ParamSet', 'FinalPrice'}';
assets_table = [assets_headers'; num2cell(assets_data)];
write_csv_compat(assets_table, assets_file);
fprintf('   ✓ Asset prices saved to: %s\n', assets_file);

% Save generation metadata
fprintf('\n4. Saving metadata...\n');
meta_file = sprintf('%s/metadata.txt', output_folder);
fid = fopen(meta_file, 'w');
fprintf(fid, '=================================================================\n');
fprintf(fid, 'DATASET GENERATION METADATA\n');
fprintf(fid, '=================================================================\n\n');
fprintf(fid, 'Generation timestamp: %s\n', timestamp);
fprintf(fid, 'Model: %s GARCH\n\n', model_type);

fprintf(fid, 'CONFIGURATION:\n');
fprintf(fid, '  Contracts per day: %d\n', CONTRACTS_PER_DAY);
fprintf(fid, '  Number of days: %d\n', NUM_DAYS);
fprintf(fid, '  Batch size: %d\n', BATCH_SIZE);
fprintf(fid, '  Parameter sets: %d\n', NUM_PARAM_SETS);
fprintf(fid, '  Moneyness range: [%.2f, %.2f]\n', moneyness_range(1), moneyness_range(2));
fprintf(fid, '  Maturity days: [%s]\n', num2str(maturity_days));

fprintf(fid, '\nMARKET PARAMETERS:\n');
fprintf(fid, '  Initial spot price (S0): %.2f\n', S0);
fprintf(fid, '  Risk-free rate (r): %.6f (daily), %.4f%% (annual)\n', r, r*252*100);
fprintf(fid, '  Time steps (N): %d\n', N);

fprintf(fid, '\nWILLOW TREE PARAMETERS:\n');
fprintf(fid, '  m_h: %d\n', m_h);
fprintf(fid, '  m_x: %d\n', m_x);
fprintf(fid, '  gamma_h: %.2f\n', gamma_h);
fprintf(fid, '  gamma_x: %.2f\n', gamma_x);

fprintf(fid, '\nRESULTS:\n');
fprintf(fid, '  Total contracts generated: %d\n', original_size);
fprintf(fid, '  Contracts removed (V < 0.5): %d\n', removed_low_value);
fprintf(fid, '  Contracts removed (T = 1/252): %d\n', removed_daily);
fprintf(fid, '  Contracts removed (invalid): %d\n', removed_invalid);
fprintf(fid, '  Final clean dataset: %d contracts\n', size(dataset_clean, 2));
fprintf(fid, '  Pricing errors: %d (%.2f%%)\n', pricing_errors, 100*pricing_errors/total_contracts);

if size(dataset_clean, 2) > 0
    fprintf(fid, '\nSTATISTICS:\n');
    fprintf(fid, '  Calls: %d, Puts: %d\n', sum(calls_mask), sum(puts_mask));
    if sum(calls_mask) > 0
        fprintf(fid, '  Call IV - Mean: %.4f, Std: %.4f\n', mean(dataset_clean(11, calls_mask)), std(dataset_clean(11, calls_mask)));
    end
    if sum(puts_mask) > 0
        fprintf(fid, '  Put IV  - Mean: %.4f, Std: %.4f\n', mean(dataset_clean(11, puts_mask)), std(dataset_clean(11, puts_mask)));
    end
end

fclose(fid);
fprintf('   ✓ Metadata saved to: %s\n', meta_file);

fprintf('\n=================================================================\n');
fprintf('                    GENERATION COMPLETE!\n');
fprintf('=================================================================\n');
fprintf('All files saved to folder: %s\n', output_folder);
fprintf('\nGenerated files:\n');
fprintf('  • %s (main dataset)\n', output_file);
fprintf('  • %s (GARCH parameters)\n', param_file);
fprintf('  • %s (asset prices)\n', assets_file);
fprintf('  • %s (metadata)\n', meta_file);
fprintf('=================================================================\n\n');

end
