clear all
% demo_scalable.m
% Scalable version of Heston-Nandi GARCH option pricing with flexible dataset generation
% Allows easy scaling of contracts per day and number of days

fprintf('=================================================================\n');
fprintf('      SCALABLE HESTON-NANDI GARCH OPTION PRICING DEMO\n');
fprintf('=================================================================\n\n');

%% SCALING PARAMETERS - MODIFY THESE TO SCALE THE DATASET
CONTRACTS_PER_DAY = 250;     % Number of option contracts to generate per day
NUM_DAYS = 60;              % Number of trading days to simulate
BATCH_SIZE = 100;           % Process options in batches to manage memory

% Display scaling configuration
fprintf('SCALING CONFIGURATION:\n');
fprintf('  Contracts per day: %d\n', CONTRACTS_PER_DAY);
fprintf('  Number of days: %d\n', NUM_DAYS);
fprintf('  Total contracts: %d\n', CONTRACTS_PER_DAY * NUM_DAYS * 2); % x2 for calls and puts
fprintf('  Batch size: %d\n', BATCH_SIZE);
fprintf('  Estimated memory usage: %.1f MB\n', (CONTRACTS_PER_DAY * NUM_DAYS * 2 * 12 * 8) / 1e6);

%% MARKET PARAMETERS
T = [30, 60, 90, 180, 360]./252;  % Base maturities
N = 504;                          % Number of time steps
r = 0.05/252;                     % Risk-free rate (daily)
S0 = 100;                         % Initial stock price
K_base = 100;                     % Base strike price

% Enhanced strike and maturity generation for scaling
moneyness_range = [0.8, 1.2];    % Moneyness range
maturity_range = [0.1, 1.0];     % Maturity range (years)

%% GARCH PARAMETERS
alpha = 1.33e-6;
beta = 0.8;
omega = 1e-6;
gamma = 5;
lambda = 0.2;

%% WILLOW TREE PARAMETERS
m_h = 6;
m_ht = 6;
m_x = 30;
gamma_h = 0.6;
gamma_x = 0.8;

%% SOLVER PARAMETERS
itmax = 100;
tol = 1e-6;

%% GENERATE SCALED GARCH PATHS
fprintf('\nGenerating scaled GARCH simulation...\n');
M = 1;  % Generate multiple paths for different days
numPoint = N + 1;
Z = randn(numPoint + 1, M);

[S_paths, h0_paths] = mcHN(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);

fprintf('Generated %d GARCH paths\n', M);

%% INITIALIZE DATASET
total_contracts = CONTRACTS_PER_DAY * NUM_DAYS * 2;  % x2 for calls and puts
dataset = zeros(12, total_contracts);  % Core dataset with essential metrics

% Dataset structure:
% 1: Spot price, 2: Moneyness, 3: Risk-free rate, 4: Time to maturity
% 5: Call/Put flag, 6: alpha, 7: beta, 8: omega, 9: gamma, 10: lambda
% 11: Implied volatility, 12: Option value

fprintf('\nInitializing dataset structure...\n');
fprintf('Dataset dimensions: %d metrics × %d contracts\n', size(dataset, 1), size(dataset, 2));

%% PREPARE TREE STRUCTURES (ONCE FOR EFFICIENCY)
fprintf('\nBuilding willow trees (one-time setup)...\n');
c = gamma + lambda + 0.5;  % Risk-neutral parameter

try
    [hd, qhd] = genhDelta(h0_paths(1), beta, alpha, c, omega, m_h, gamma_h);
    [nodes_ht] = TreeNodes_ht_HN(m_ht, hd, qhd, gamma_h, alpha, beta, c, omega, N+1);
    
    % Pre-compute tree nodes for different scenarios
    tree_cache = cell(NUM_DAYS, 1);
    for day = 1:NUM_DAYS
        S_current = S_paths(end - CONTRACTS_PER_DAY + day, min(day, M));
        h_current = h0_paths(min(day, M));
        
        [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_HN(m_x, gamma_x, r, hd, qhd, S_current, alpha, beta, c, omega, N);
        [q_Xt, P_Xt, ~] = Prob_Xt(nodes_ht, qhd, nodes_Xt, S_current, r, alpha, beta, c, omega);
        nodes_S = exp(nodes_Xt);
        
        tree_cache{day} = struct('nodes_S', nodes_S, 'P_Xt', P_Xt, 'q_Xt', q_Xt, ...
                                'S_current', S_current, 'h_current', h_current);
    end
    
    fprintf('Trees cached for %d days\n', NUM_DAYS);
    
catch ME
    fprintf('Error building trees: %s\n', ME.message);
    return;
end

%% BATCH PROCESSING SETUP
num_batches = ceil(total_contracts / (BATCH_SIZE * 2));  % Divide by 2 since we process calls and puts together
fprintf('\nProcessing in %d batches...\n', num_batches);

% Statistics tracking
total_processed = 0;
pricing_errors = 0;
solver_failures = 0;
boundary_issues = 0;
processing_times = zeros(num_batches, 1);

%% MAIN PROCESSING LOOP
dataset_idx = 1;

for batch = 1:num_batches
    batch_start_time = tic;
    fprintf('\nProcessing batch %d/%d...\n', batch, num_batches);
    
    % Calculate batch boundaries
    contracts_in_batch = min(BATCH_SIZE, ceil((total_contracts/2 - (batch-1)*BATCH_SIZE)));
    batch_start_idx = (batch - 1) * BATCH_SIZE + 1;
    
    batch_errors = 0;
    batch_processed = 0;
    
    for contract_in_batch = 1:contracts_in_batch
        contract_idx = batch_start_idx + contract_in_batch - 1;
        
        % Determine which day this contract belongs to
        day = ceil(contract_idx / CONTRACTS_PER_DAY);
        contract_in_day = mod(contract_idx - 1, CONTRACTS_PER_DAY) + 1;
        
        if day > NUM_DAYS
            break;
        end
        
        % Get cached tree data for this day
        tree_data = tree_cache{day};
        S_current = tree_data.S_current;
        nodes_S = tree_data.nodes_S;
        P_Xt = tree_data.P_Xt;
        q_Xt = tree_data.q_Xt;
        
        % Generate random contract parameters for this specific contract
        rng(contract_idx);  % Seed for reproducibility
        
        % Random moneyness and maturity
        moneyness = moneyness_range(1) + (moneyness_range(2) - moneyness_range(1)) * rand();
        maturity_years = maturity_range(1) + (maturity_range(2) - maturity_range(1)) * rand();
        
        K_strike = moneyness * S_current;
        T_maturity = maturity_years;
        
        try
            % Price American options
            [V_C, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, 1);
            [V_P, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, -1);
            
            % Validate option prices
            if V_C < 0 || V_P < 0 || isnan(V_C) || isnan(V_P)
                batch_errors = batch_errors + 1;
                pricing_errors = pricing_errors + 1;
                continue;
            end
            
            % Compute implied volatilities
            if exist('impvol_improved', 'file') == 2
                % Use improved solver if available
                [impl_c, V0_c, it_c, conv_c] = impvol_improved(S_current, K_strike, T_maturity, r, V_C, 1, N, m_x, gamma_x, tol, itmax);
                [impl_p, V0_p, it_p, conv_p] = impvol_improved(S_current, K_strike, T_maturity, r, V_P, -1, N, m_x, gamma_x, tol, itmax);
                
                if ~conv_c || ~conv_p
                    solver_failures = solver_failures + 1;
                end
                if impl_c <= 0.002 || impl_c >= 2.99 || impl_p <= 0.002 || impl_p >= 2.99
                    boundary_issues = boundary_issues + 1;
                end
                
            else
                % Fallback solver with improved initial guesses
                z = zq(m_x, gamma_x);
                [P, q] = gen_PoWiner(T_maturity, N, z);
                
                % Call option implied volatility
                sigma_min = 0.001; sigma_max = 3.0;
                init_guess_c = 0.2 + 0.1 * abs(moneyness - 1);
                init_guess_c = max(sigma_min, min(sigma_max, init_guess_c));
                
                [impl_c, V0_c, it_c, conv_c] = solve_iv_bisection(S_current, K_strike, T_maturity, r, V_C, 1, ...
                                                                  sigma_min, sigma_max, init_guess_c, tol, itmax, N, z);
                
                % Put option implied volatility
                init_guess_p = 0.2 + 0.1 * abs(moneyness - 1);
                init_guess_p = max(sigma_min, min(sigma_max, init_guess_p));
                
                [impl_p, V0_p, it_p, conv_p] = solve_iv_bisection(S_current, K_strike, T_maturity, r, V_P, -1, ...
                                                                  sigma_min, sigma_max, init_guess_p, tol, itmax, N, z);
            end
            
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
            
            batch_processed = batch_processed + 2;  % Call and put
            
        catch ME
            batch_errors = batch_errors + 1;
            pricing_errors = pricing_errors + 1;
            
            % Store error placeholders
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
    
    % Memory management
    if mod(batch, 5) == 0
        fprintf('  Memory cleanup...\n');
        clear V_C V_P impl_c impl_p V0_c V0_p;
    end
end

%% FINALIZE DATASET
% Trim unused columns if any
dataset = dataset(:, 1:dataset_idx-1);

%% SAVE RESULTS
fprintf('\nSaving results...\n');
headers = {'S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'sigma', 'V'}';
filename = sprintf('scalable_hn_dataset_%dx%d.csv', CONTRACTS_PER_DAY, NUM_DAYS);
dataset_enhanced = [headers'; num2cell(dataset')];
writecell(dataset_enhanced, filename);

%% COMPREHENSIVE REPORTING
fprintf('\n=================================================================\n');
fprintf('                    PROCESSING COMPLETE\n');
fprintf('=================================================================\n');

% Final statistics
actual_contracts = size(dataset, 2);
call_data = dataset(:, dataset(5,:) == 1);
put_data = dataset(:, dataset(5,:) == -1);

fprintf('\nSCALING RESULTS:\n');
fprintf('  Planned contracts: %d\n', total_contracts);
fprintf('  Actual contracts: %d\n', actual_contracts);
fprintf('  Calls: %d, Puts: %d\n', size(call_data, 2), size(put_data, 2));
fprintf('  Dataset saved as: %s\n', filename);
fprintf('  File size: %.2f MB\n', dir(filename).bytes / 1e6);

fprintf('\nPERFORMANCE METRICS:\n');
fprintf('  Total processing time: %.2f minutes\n', sum(processing_times) / 60);
fprintf('  Average time per batch: %.2f seconds\n', mean(processing_times));
fprintf('  Contracts per second: %.1f\n', actual_contracts / sum(processing_times));
fprintf('  Pricing errors: %d (%.1f%%)\n', pricing_errors, 100*pricing_errors/actual_contracts);
fprintf('  Solver failures: %d (%.1f%%)\n', solver_failures, 100*solver_failures/actual_contracts);
fprintf('  Boundary issues: %d (%.1f%%)\n', boundary_issues, 100*boundary_issues/actual_contracts);

% Data quality analysis
all_ivs = dataset(11,:);
valid_ivs = all_ivs(all_ivs > 0.001 & all_ivs < 2.99);
call_ivs = call_data(11,:);
put_ivs = put_data(11,:);

fprintf('\nDATA QUALITY:\n');
fprintf('  Valid IVs: %d/%d (%.1f%%)\n', length(valid_ivs), length(all_ivs), 100*length(valid_ivs)/length(all_ivs));
fprintf('  IV range: [%.4f, %.4f] (%.1f%% - %.1f%% annualized)\n', ...
        min(valid_ivs), max(valid_ivs), min(valid_ivs)*sqrt(252)*100, max(valid_ivs)*sqrt(252)*100);
fprintf('  Mean IV: %.4f (%.1f%% annualized)\n', mean(valid_ivs), mean(valid_ivs)*sqrt(252)*100);

% Moneyness and maturity distribution
moneyness_data = dataset(2,:);
maturity_data = dataset(4,:);

fprintf('\nDATA DISTRIBUTION:\n');
fprintf('  Moneyness range: [%.3f, %.3f]\n', min(moneyness_data), max(moneyness_data));
fprintf('  Maturity range: [%.3f, %.3f] years\n', min(maturity_data), max(maturity_data));
fprintf('  Spot price range: [%.2f, %.2f]\n', min(dataset(1,:)), max(dataset(1,:)));

fprintf('\n=================================================================\n');
if length(valid_ivs)/length(all_ivs) > 0.95 && pricing_errors < actual_contracts*0.05
    fprintf('SUCCESS: Scalable dataset generation completed successfully!\n');
else
    fprintf('PARTIAL SUCCESS: Some data quality issues detected.\n');
end
fprintf('=================================================================\n');

%% HELPER FUNCTION FOR BISECTION METHOD
function [impl_vol, V0, iterations, converged] = solve_iv_bisection(S, K, T, r, target_price, cp, ...
                                                                   sigma_min, sigma_max, init_guess, tol, itmax, N, z)
    % Bisection method for implied volatility
    a = sigma_min;
    b = sigma_max;
    sigma = init_guess;
    iterations = 0;
    V0 = 0;
    
    while abs(V0 - target_price) > tol && iterations < itmax
        Xnodes = nodes_Winer(T, N, z, r, sigma);
        nodes = S .* exp(Xnodes);
        V0 = American(nodes, gen_PoWiner(T, N, z), zq(length(z), 0.8), r, T, S, K, cp);
        
        if V0 > target_price
            b = sigma;
        else
            a = sigma;
        end
        sigma = (a + b) / 2;
        iterations = iterations + 1;
        
        if (b - a) < 1e-10
            break;
        end
    end
    
    impl_vol = sigma;
    converged = (abs(V0 - target_price) <= tol);
end

fprintf('\nScalable dataset generation complete!\n');
fprintf('To scale further, modify CONTRACTS_PER_DAY and NUM_DAYS at the top of the script.\n');