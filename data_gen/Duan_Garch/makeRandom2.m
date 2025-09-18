clear all
% scaled_dataset_duan.m
% Enhanced scalable Duan GARCH option pricing with varying GARCH parameters
% Uses random parameter generation within empirically realistic ranges

fprintf('=================================================================\n');
fprintf('   SCALABLE DUAN GARCH WITH VARYING PARAMETERS\n');
fprintf('=================================================================\n\n');

%% SCALING PARAMETERS - MODIFY THESE TO SCALE THE DATASET
CONTRACTS_PER_DAY = 256;     % Number of option contracts to generate per day
NUM_DAYS = 60;              % Number of trading days to simulate
BATCH_SIZE = 100;           % Process options in batches to manage memory

% GARCH Parameter Variation Settings
NUM_PARAM_SETS = 12;       % Number of different GARCH parameter combinations

% Display scaling configuration
fprintf('SCALING CONFIGURATION:\n');
fprintf('  Contracts per day: %d\n', CONTRACTS_PER_DAY);
fprintf('  Number of days: %d\n', NUM_DAYS);
fprintf('  Total contracts: %d\n', CONTRACTS_PER_DAY * NUM_DAYS * 2);
fprintf('  Parameter sets: %d\n', NUM_PARAM_SETS);
fprintf('  Batch size: %d\n', BATCH_SIZE);

%% MARKET PARAMETERS
T = [50, 70, 80]./252;  % Base maturities
N = 504;                          % Number of time steps
r = 0.05/252;                     % Risk-free rate (daily)
S0 = 100;                         % Initial stock price

% Enhanced strike and maturity generation for scaling
moneyness_range = [0.8, 1.2];    % Moneyness range
maturity_range = [0.1, 2.0];     % Maturity range (years)

%% GENERATE VARYING DUAN GARCH PARAMETER SETS
fprintf('\nGenerating Duan GARCH parameter variations...\n');
fprintf('Strategy: Random generation within empirically realistic ranges\n');

% Define parameter ranges based on empirical literature
% These ranges are based on studies of equity index options
alpha_range = [1.15e-6, 1.26e-6];      % ARCH parameter
beta_range = [0.8, 0.85];             % GARCH parameter
omega_range = [1e-7, 1e-6];            % Long-run variance level
theta_range = [0.3, 0.45];             % Asymmetry parameter
lambda_range = [0.4, 0.5];             % Risk premium parameter

% Generate random parameter sets with constraints
param_sets = zeros(NUM_PARAM_SETS, 5);
valid_sets = 0;

fprintf('Generating parameter sets with stationarity constraints...\n');

while valid_sets < NUM_PARAM_SETS
    % Generate candidate parameters
    alpha_candidate = alpha_range(1) + (alpha_range(2) - alpha_range(1)) * rand();
    beta_candidate = beta_range(1) + (beta_range(2) - beta_range(1)) * rand();
    omega_candidate = omega_range(1) + (omega_range(2) - omega_range(1)) * rand();
    theta_candidate = theta_range(1) + (theta_range(2) - theta_range(1)) * rand();
    lambda_candidate = lambda_range(1) + (lambda_range(2) - lambda_range(1)) * rand();

    % Check stationarity condition for Duan GARCH
    % Condition 1: beta + alpha * theta^2 < 1
    % Condition 2: alpha + beta < 1
    if (beta_candidate + alpha_candidate * theta_candidate^2 < 0.999) && ...
       (alpha_candidate + beta_candidate < 0.999)
        valid_sets = valid_sets + 1;
        param_sets(valid_sets, :) = [alpha_candidate, beta_candidate, omega_candidate, ...
                                    theta_candidate, lambda_candidate];
    end
end

fprintf('Generated %d valid parameter sets\n', NUM_PARAM_SETS);

% Display parameter statistics
fprintf('\nParameter ranges in generated sets:\n');
fprintf('  Alpha:  [%.2e, %.2e] (mean: %.2e)\n', min(param_sets(:,1)), max(param_sets(:,1)), mean(param_sets(:,1)));
fprintf('  Beta:   [%.3f, %.3f] (mean: %.3f)\n', min(param_sets(:,2)), max(param_sets(:,2)), mean(param_sets(:,2)));
fprintf('  Omega:  [%.2e, %.2e] (mean: %.2e)\n', min(param_sets(:,3)), max(param_sets(:,3)), mean(param_sets(:,3)));
fprintf('  Theta:  [%.3f, %.3f] (mean: %.3f)\n', min(param_sets(:,4)), max(param_sets(:,4)), mean(param_sets(:,4)));
fprintf('  Lambda: [%.3f, %.3f] (mean: %.3f)\n', min(param_sets(:,5)), max(param_sets(:,5)), mean(param_sets(:,5)));

% Check stationarity for all sets
stationarity_values = param_sets(:,2) + param_sets(:,1) .* param_sets(:,4).^2;
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
fprintf('\nGenerating Duan GARCH paths for parameter sets...\n');
M_paths = min(NUM_PARAM_SETS, 50);  % Generate paths for subset to manage memory
numPoint = N + 1;

% Generate different random seeds for path diversity
Z_seeds = randn(numPoint, M_paths);

% Store paths and initial variances for parameter sets
S_paths_all = cell(NUM_PARAM_SETS, 1);
h0_paths_all = zeros(NUM_PARAM_SETS, 1);

for p = 1:NUM_PARAM_SETS
    alpha_p = param_sets(p, 1);
    beta_p = param_sets(p, 2);
    omega_p = param_sets(p, 3);
    theta_p = param_sets(p, 4);
    lambda_p = param_sets(p, 5);

    % Use modulo to reuse Z_seeds if we have more parameter sets than paths
    seed_idx = mod(p-1, M_paths) + 1;

    
    [S_path, h0_path] = mcDuan(M_paths, N, S0, Z_seeds(:, seed_idx), r, omega_p, alpha_p, beta_p, theta_p, lambda_p);
    S_paths_all{p} = S_path;
    h0_paths_all(p) = h0_path;

    if mod(p, 20) == 0
        fprintf('  Generated paths for %d/%d parameter sets\n', p, NUM_PARAM_SETS);
    end
end

fprintf('Completed Duan GARCH path generation\n');

%% INITIALIZE DATASET
total_contracts = CONTRACTS_PER_DAY * NUM_DAYS * 2;
dataset = zeros(12, total_contracts);

fprintf('\nDataset structure:\n');
fprintf('1: Spot price, 2: Moneyness, 3: Risk-free rate, 4: Time to maturity\n');
fprintf('5: Call/Put flag, 6: alpha, 7: beta, 8: omega, 9: theta, 10: lambda\n');
fprintf('11: Implied volatility, 12: Option value\n');

%% MAIN GENERATION LOOP WITH BATCHING
contract_idx = 1;
total_time = 0;
start_time_total = tic;
pricing_errors = 0;

% Pre-build trees for all parameter sets
fprintf('\nPre-building willow trees for all parameter sets...\n');
tree_cache = containers.Map('KeyType', 'int32', 'ValueType', 'any');

% Process in batches to manage memory
for batch_day = 1:BATCH_SIZE:NUM_DAYS
    batch_start = tic;
    batch_end = min(batch_day + BATCH_SIZE - 1, NUM_DAYS);
    batch_size_actual = batch_end - batch_day + 1;

    fprintf('\nProcessing days %d-%d (batch size: %d)\n', batch_day, batch_end, batch_size_actual);

    batch_errors = 0;

    % Process each day in the batch
    for day = batch_day:batch_end
        fprintf('  Day %d/%d: ', day, NUM_DAYS);
        day_errors = 0;
        day_contracts = 0;

        % Get stock price for this day
        % Use a parameter set that cycles through available sets
        param_idx_for_path = mod(day-1, NUM_PARAM_SETS) + 1;
        S_path = S_paths_all{param_idx_for_path};
        S_current = S_path(min(day*10, N+1));  % Sample from path

        % Process contracts for this day
        for contract = 1:CONTRACTS_PER_DAY
            % Randomly select parameter set for this contract
            param_idx = randi(NUM_PARAM_SETS);

            % Extract GARCH parameters
            alpha = param_sets(param_idx, 1);
            beta = param_sets(param_idx, 2);
            omega = param_sets(param_idx, 3);
            theta = param_sets(param_idx, 4);
            lambda = param_sets(param_idx, 5);
            h0_path = h0_paths_all(param_idx);

            % Generate contract-specific parameters
            rng(contract_idx);  % For reproducibility
            moneyness = moneyness_range(1) + (moneyness_range(2) - moneyness_range(1)) * rand();
            maturity_years = maturity_range(1) + (maturity_range(2) - maturity_range(1)) * rand();
            K_strike = moneyness * S_current;
            T_maturity = maturity_years;

            % Check if we need to compute tree for this parameter set
            if ~isKey(tree_cache, param_idx)
                try
                    % Build willow trees for this parameter set - Duan version
                    % First get the h tree
                    [nodes_ht, qht, hmom1, hmom2, hmom3, hmom4_app] = TreeNodes_ht_D(m_h, h0_path, gamma_h, omega, beta, alpha, theta, lambda, N+1);
                    % Then get the X tree
                    [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_D(m_x, gamma_x, r, h0_path, omega, beta, alpha, theta, lambda, N, hmom1, hmom2);
                    nodes_Xt = nodes_Xt + log(S_current);
                    % Get transition probabilities
                    [q_Xt, P_Xt, ~] = Probility_Xt2(nodes_ht, qht, nodes_Xt, S_current, r, omega, beta, alpha, theta, lambda);
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

                % Validate option prices
                if V_C < 0 || V_P < 0 || isnan(V_C) || isnan(V_P) || isinf(V_C) || isinf(V_P)
                    batch_errors = batch_errors + 1;
                    pricing_errors = pricing_errors + 1;
                    continue;
                end

                [sigma_c, ~, ~] = impvol(S_current, K_strike, T_maturity, r, V_C, 1, N, m_x, gamma_x, tol, itmax);
                [sigma_p, ~, ~] = impvol(S_current, K_strike, T_maturity, r, V_P, 1, N, m_x, gamma_x, tol, itmax);
               

                % Store results
                dataset(:, contract_idx) = [S_current, moneyness, r, T_maturity, 1, ...
                                           alpha, beta, omega, theta, lambda, ...
                                           sigma_c, V_C];
                contract_idx = contract_idx + 1;

                dataset(:, contract_idx) = [S_current, moneyness, r, T_maturity, -1, ...
                                           alpha, beta, omega, theta, lambda, ...
                                           sigma_p, V_P];
                contract_idx = contract_idx + 1;

                day_contracts = day_contracts + 2;

            catch ME
                batch_errors = batch_errors + 1;
                pricing_errors = pricing_errors + 1;
                fprintf('Pricing error: %s\n', ME.message);
                continue;
            end
        end

        fprintf('%d contracts generated, %d errors\n', day_contracts, day_errors);
    end

    batch_time = toc(batch_start);
    fprintf('Batch completed in %.2f seconds\n', batch_time);
    fprintf('Total contracts so far: %d\n', contract_idx - 1);
    fprintf('Batch errors: %d\n', batch_errors);
end

% Trim dataset to actual size
dataset = dataset(:, 1:contract_idx-1);

total_time = toc(start_time_total);
fprintf('\n=================================================================\n');
fprintf('GENERATION COMPLETE\n');
fprintf('  Total contracts generated: %d\n', size(dataset, 2));
fprintf('  Total pricing errors: %d\n', pricing_errors);
fprintf('  Total time: %.2f seconds\n', total_time);
fprintf('  Average time per contract: %.4f seconds\n', total_time / size(dataset, 2));
fprintf('=================================================================\n');

% %% SAVE DATASET
% filename = sprintf('duan_garch_dataset_%ddays_%dcontracts.mat', NUM_DAYS, size(dataset, 2));
% save(filename, 'dataset', 'param_sets', 'NUM_DAYS', 'CONTRACTS_PER_DAY');
% fprintf('\nDataset saved to: %s\n', filename);
% headers = {'S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'sigma', 'V'}';
% dataset = [headers'; num2cell(dataset')];
% % Also save as CSV for compatibility
% csv_filename = sprintf('duan_garch_dataset_%ddays_%dcontracts.csv', NUM_DAYS, size(dataset, 2));
% writecell(dataset, csv_filename);
% fprintf('CSV saved to: %s\n', csv_filename);
fprintf('\nSaving results...\n');
headers = {'S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'sigma', 'V'}';
filename = sprintf('duan_garch_dataset_%dx%d_%dparams.csv', CONTRACTS_PER_DAY, NUM_DAYS, NUM_PARAM_SETS);
dataset_enhanced = [headers'; num2cell(dataset')];
writecell(dataset_enhanced, filename);

% %% DISPLAY DATASET STATISTICS
% fprintf('\nDataset Statistics:\n');
% fprintf('  Spot prices: [%.2f, %.2f]\n', min(dataset(1,:)), max(dataset(1,:)));
% fprintf('  Moneyness: [%.3f, %.3f]\n', min(dataset(2,:)), max(dataset(2,:)));
% fprintf('  Maturities: [%.3f, %.3f] years\n', min(dataset(4,:)), max(dataset(4,:)));
% fprintf('  Call options: %d\n', sum(dataset(5,:) == 1));
% fprintf('  Put options: %d\n', sum(dataset(5,:) == -1));
% fprintf('  Implied volatilities: [%.3f, %.3f]\n', min(dataset(11,:)), max(dataset(11,:)));
% fprintf('  Option values: [%.2f, %.2f]\n', min(dataset(12,:)), max(dataset(12,:)));
% 
% % Parameter distribution in dataset
% fprintf('\nParameter Distribution in Dataset:\n');
% unique_alphas = unique(dataset(6,:));
% fprintf('  Unique alpha values: %d\n', length(unique_alphas));
% unique_betas = unique(dataset(7,:));
% fprintf('  Unique beta values: %d\n', length(unique_betas));
% unique_omegas = unique(dataset(8,:));
% fprintf('  Unique omega values: %d\n', length(unique_omegas));
% unique_thetas = unique(dataset(9,:));
% fprintf('  Unique theta values: %d\n', length(unique_thetas));
% unique_lambdas = unique(dataset(10,:));
% fprintf('  Unique lambda values: %d\n', length(unique_lambdas));
