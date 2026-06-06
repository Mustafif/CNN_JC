function compare_dataset(config_file)
% COMPARE_DATASET Centralized comparison of true, initial, and calibrated GARCH parameters
%
% SYNTAX:
%   compare_dataset('compare_config.toml')
%
% DESCRIPTION:
%   Generates option pricing datasets using three parameter sets (true, initial, calibrated)
%   with the SAME randomness to enable fair comparison. Supports multiple GARCH models.
%
% INPUT:
%   config_file - Path to TOML configuration file
%
% CONFIGURATION FILE FORMAT (TOML):
%   See compare_config.toml for a complete example with all options
%
% OUTPUT:
%   Generates CSV files in the output folder:
%   - true_params.csv       : Dataset with true parameters
%   - initial_params.csv    : Dataset with initial parameters
%   - calibrated_params.csv : Dataset with calibrated parameters
%   - assetprices_*.csv     : Asset price paths (optional)
%
% EXAMPLE:
%   compare_dataset('compare_config.toml')

if nargin < 1
    config_file = 'compare_config.toml';
end

fprintf('=================================================================\n');
fprintf('    CENTRALIZED GARCH PARAMETER COMPARISON FRAMEWORK\n');
fprintf('=================================================================\n\n');

%% LOAD CONFIGURATION FROM TOML
fprintf('Loading configuration from: %s\n', config_file);

% Use Python helper to read TOML and convert to JSON
[status, json_str] = system(sprintf('python3 read_toml_config.py "%s"', config_file));

if status ~= 0
    error('Failed to read TOML config file. Error: %s', json_str);
end

config = jsondecode(json_str);

% Check for errors
if isfield(config, 'error')
    error('Configuration error: %s', config.error);
end

%% EXTRACT CONFIGURATION
model_type = upper(config.model.type);
fprintf('Model type: %s GARCH\n\n', model_type);

% Scaling parameters
CONTRACTS_PER_DAY = config.scaling.contracts_per_day;
NUM_DAYS = config.scaling.num_days;
BATCH_SIZE = config.scaling.batch_size;

% Market parameters
N = config.market.N;
r = config.market.r;
S0 = config.market.S0;
moneyness_range = config.market.moneyness_range;
maturity_range = config.market.maturity_range;

% Willow tree parameters
m_h = config.willow_tree.m_h;
m_x = config.willow_tree.m_x;
gamma_h = config.willow_tree.gamma_h;
gamma_x = config.willow_tree.gamma_x;

% Solver parameters
itmax = config.solver.itmax;
tol = config.solver.tol;

% Noise parameters
noise_level = config.noise.level;
noise_seed = config.noise.seed;

% Output parameters
output_folder = config.output.folder;
save_asset_prices = config.output.save_asset_prices;

% Create output folder
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% DISPLAY CONFIGURATION
fprintf('SCALING CONFIGURATION:\n');
fprintf('  Contracts per day: %d\n', CONTRACTS_PER_DAY);
fprintf('  Number of days: %d\n', NUM_DAYS);
fprintf('  Total contracts: %d\n', CONTRACTS_PER_DAY * NUM_DAYS * 2);
fprintf('  Batch size: %d\n', BATCH_SIZE);
fprintf('  Estimated memory usage: %.1f MB\n\n', ...
    (CONTRACTS_PER_DAY * NUM_DAYS * 2 * 12 * 8) / 1e6);

fprintf('MARKET PARAMETERS:\n');
fprintf('  Time steps: %d\n', N);
fprintf('  Risk-free rate: %.6f (daily)\n', r);
fprintf('  Initial stock price: %.2f\n', S0);
fprintf('  Moneyness range: [%.2f, %.2f]\n', moneyness_range(1), moneyness_range(2));
fprintf('  Maturity range: [%.2f, %.2f] years\n\n', maturity_range(1), maturity_range(2));

fprintf('NOISE CONFIGURATION:\n');
fprintf('  Noise level: %.4f (%.1f%%)\n', noise_level, noise_level * 100);
fprintf('  Noise seed: %d\n\n', noise_seed);

%% BUILD PARAMETER SETS
true_params.name = 'true_params';
true_params.omega = config.parameters.true.omega;
true_params.alpha = config.parameters.true.alpha;
true_params.beta = config.parameters.true.beta;
true_params.gamma = config.parameters.true.gamma;
true_params.lambda = config.parameters.true.lambda;

initial_params.name = 'initial_params';
initial_params.omega = config.parameters.initial.omega;
initial_params.alpha = config.parameters.initial.alpha;
initial_params.beta = config.parameters.initial.beta;
initial_params.gamma = config.parameters.initial.gamma;
initial_params.lambda = config.parameters.initial.lambda;

calibrated_params.name = 'calibrated_params';
calibrated_params.omega = config.parameters.calibrated.omega;
calibrated_params.alpha = config.parameters.calibrated.alpha;
calibrated_params.beta = config.parameters.calibrated.beta;
calibrated_params.gamma = config.parameters.calibrated.gamma;
calibrated_params.lambda = config.parameters.calibrated.lambda;

param_sets = [true_params, initial_params, calibrated_params];

fprintf('PARAMETER SETS:\n');
for i = 1:length(param_sets)
    ps = param_sets(i);
    fprintf('  %s:\n', ps.name);
    fprintf('    omega=%.6e, alpha=%.6e, beta=%.4f, gamma=%.4f, lambda=%.4f\n', ...
        ps.omega, ps.alpha, ps.beta, ps.gamma, ps.lambda);
end
fprintf('\n');

%% PRE-GENERATE RANDOMNESS FOR GARCH SIMULATION (SHARED ACROSS PARAM SETS)
fprintf('Pre-generating random numbers for GARCH simulation...\n');
M = 1;
numPoint = N + 1;

% Fix seed for reproducibility
rng(12345);
Z_master = randn(numPoint + 1, M);

fprintf('Random matrix Z generated: size = %d x %d\n', size(Z_master,1), size(Z_master,2));

% Initialize noise generator if needed
if noise_level > 0
    if noise_seed >= 0
        noise_rng = RandStream('mt19937ar', 'Seed', noise_seed);
    else
        noise_rng = RandStream('mt19937ar', 'Seed', randi([0, 2^31-1]));
    end
    fprintf('Noise will be added to option prices\n');
end

%% DATASET HEADERS
headers = {'S0', 'm', 'r', 'T', 'corp', ...
           'alpha', 'beta', 'omega', 'gamma', 'lambda', ...
           'sigma', 'V'}';

%% LOOP OVER PARAMETER SETS
total_contracts = CONTRACTS_PER_DAY * NUM_DAYS * 2;

for ps = 1:numel(param_sets)
    params = param_sets(ps);
    fprintf('\n>>> NOW PROCESSING: %s\n', params.name);
    
    alpha = params.alpha;
    beta = params.beta;
    omega = params.omega;
    gamma = params.gamma;
    lambda = params.lambda;
    setname = params.name;
    
    fprintf('\n============================================================\n');
    fprintf('  PROCESSING PARAMETER SET: %s\n', setname);
    fprintf('============================================================\n');
    
    %% GENERATE GARCH PATHS FOR THIS PARAMETER SET
    fprintf('\nGenerating GARCH simulation for %s...\n', setname);
    
    Z = Z_master;  % Use same randomness for all parameter sets
    
    % Calculate risk-neutral parameter first (needed for mcDuan)
    c = gamma + lambda + 0.5;  % Risk-neutral parameter
    
    % Add model-specific paths before calling model functions
    if strcmp(model_type, 'DUAN')
        addpath('Duan_Garch');
    elseif strcmp(model_type, 'GJR')
        addpath('GJR_Garch');
    end

    % Select appropriate MC function based on model type
    switch model_type
        case 'HN'
            [S_paths, h0_paths] = mcHN(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);
        case 'DUAN'
            [S_paths, h0_paths] = mcDuan(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);
        case 'GJR'
            [S_paths, h0_paths] = mcGJR(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);
        otherwise
            error('Unsupported model type: %s', model_type);
    end
    
    fprintf('Generated %d GARCH paths for %s\n', M, setname);
    
    %% INITIALIZE DATASET
    dataset = zeros(12, total_contracts);
    dataset_idx = 1;
    
    fprintf('Initializing dataset structure for %s...\n', setname);
    
    %% PREPARE TREE STRUCTURES
    fprintf('Building willow trees for %s...\n', setname);
    
    try
        [hd, qhd] = genhDelta(h0_paths(1), beta, alpha, c, omega, m_h, gamma_h);
        
        % Build tree nodes based on model type
        switch model_type
            case 'HN'
                [nodes_ht] = TreeNodes_ht_HN(m_h, hd, qhd, gamma_h, alpha, beta, c, omega, N+1);
            case 'DUAN'
                [nodes_ht, qht, hmom1, hmom2, ~, ~] = TreeNodes_ht_D(m_h, h0_paths(1), gamma_h, omega, beta, alpha, gamma, lambda, N+1);
            otherwise
                error('Tree construction not implemented for model: %s', model_type);
        end
        
        % Build basic tree structures for pricing
        S_current = S_paths(1, 1);  % Use first stock price
        switch model_type
            case 'HN'
                [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_HN(m_x, gamma_x, r, hd, qhd, S_current, alpha, beta, c, omega, N);
                [q_Xt, P_Xt, ~] = Prob_Xt(nodes_ht, qhd, nodes_Xt, S_current, r, alpha, beta, c, omega);
            case 'DUAN'
                [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_D(m_x, gamma_x, r, h0_paths(1), omega, beta, alpha, gamma, lambda, N, hmom1, hmom2);
                nodes_Xt = nodes_Xt + log(S_current);
                [q_Xt, P_Xt, ~] = Probility_Xt2(nodes_ht, qht, nodes_Xt, S_current, r, omega, beta, alpha, gamma, lambda);
            otherwise
                error('Tree construction not implemented for model: %s', model_type);
        end
        
        nodes_S = exp(nodes_Xt);
        
        % Create simple tree cache for all days (same trees for simplicity)
        tree_cache = cell(NUM_DAYS, 1);
        for day = 1:NUM_DAYS
            tree_cache{day} = struct('nodes_S', nodes_S, 'P_Xt', P_Xt, ...
                                     'q_Xt', q_Xt, 'S_current', S_current);
        end
        
        fprintf('Trees built successfully for %s\n', setname);
        
    catch ME
        fprintf('Error building trees for %s: %s\n', setname, ME.message);
        fprintf('Stack trace:\n');
        for k = 1:length(ME.stack)
            fprintf('  File: %s, Line: %d, Function: %s\n', ...
                ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
        end
        continue;
    end
    
    %% BATCH PROCESSING SETUP
    num_batches = ceil(total_contracts / (BATCH_SIZE * 2));
    fprintf('\nProcessing %s in %d batches...\n', setname, num_batches);
    
    total_processed = 0;
    pricing_errors = 0;
    processing_times = zeros(num_batches, 1);
    
    %% MAIN PROCESSING LOOP
    for batch = 1:num_batches
        batch_start_time = tic;
        fprintf('\n[%s] Processing batch %d/%d...\n', setname, batch, num_batches);
        
        contracts_in_batch = min(BATCH_SIZE, ...
            ceil((total_contracts/2 - (batch-1)*BATCH_SIZE)));
        batch_start_idx = (batch - 1) * BATCH_SIZE + 1;
        
        batch_errors = 0;
        batch_processed = 0;
        
        for contract_in_batch = 1:contracts_in_batch
            contract_idx = batch_start_idx + contract_in_batch - 1;
            
            day = ceil(contract_idx / CONTRACTS_PER_DAY);
            if day > NUM_DAYS
                break;
            end
            
            % Get cached tree data
            tree_data = tree_cache{day};
            S_current = tree_data.S_current;
            nodes_S = tree_data.nodes_S;
            P_Xt = tree_data.P_Xt;
            q_Xt = tree_data.q_Xt;
            
            % Generate contract parameters with reproducible randomness
            rng(contract_idx);
            
            moneyness = moneyness_range(1) + (moneyness_range(2) - moneyness_range(1)) * rand();
            maturity_years = maturity_range(1) + (maturity_range(2) - maturity_range(1)) * rand();
            
            K_strike = moneyness * S_current;
            T_maturity = maturity_years;
            
            try
                % Price American options
                [V_C, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, 1);
                [V_P, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, -1);
                
                % Add noise if configured (multiplicative, matching scaled_dataset)
                if noise_level > 0
                    noise_c = 1 + noise_level * randn(noise_rng);
                    noise_p = 1 + noise_level * randn(noise_rng);
                    V_C = V_C * noise_c;
                    V_P = V_P * noise_p;
                end
                
                % Validate
                if V_C < 0 || V_P < 0 || isnan(V_C) || isnan(V_P)
                    batch_errors = batch_errors + 1;
                    pricing_errors = pricing_errors + 1;
                    continue;
                end
                
                % Compute implied volatility using impvol_fzero (consistent with scaled_dataset)
                [impl_c, ~, ~] = impvol_fzero(S_current, K_strike, T_maturity, r, V_C, 1, N, m_x, gamma_x);
                [impl_p, ~, ~] = impvol_fzero(S_current, K_strike, T_maturity, r, V_P, -1, N, m_x, gamma_x);
                
                % Store call option data
                dataset(:, dataset_idx) = [
                    S_current;
                    K_strike/S_current;
                    r;
                    T_maturity;
                    1;
                    alpha;
                    beta;
                    omega;
                    gamma;
                    lambda;
                    impl_c;
                    V_C
                ];
                dataset_idx = dataset_idx + 1;
                
                % Store put option data
                dataset(:, dataset_idx) = [
                    S_current;
                    K_strike/S_current;
                    r;
                    T_maturity;
                    -1;
                    alpha;
                    beta;
                    omega;
                    gamma;
                    lambda;
                    impl_p;
                    V_P
                ];
                dataset_idx = dataset_idx + 1;
                
                batch_processed = batch_processed + 2;
                
            catch ME
                batch_errors = batch_errors + 1;
                pricing_errors = pricing_errors + 1;
                % Skip failed contracts (no placeholder rows), consistent with scaled_dataset
                continue;
            end
        end
        
        processing_times(batch) = toc(batch_start_time);
        total_processed = total_processed + batch_processed;
        
        fprintf('  [%s] Batch %d complete: %d contracts, %d errors, %.2f sec\n', ...
            setname, batch, batch_processed, batch_errors, processing_times(batch));
    end
    
    %% FINALIZE DATASET
    dataset = dataset(:, 1:dataset_idx-1);
    
    %% SAVE RESULTS
    fprintf('\nSaving results for %s...\n', setname);
    
    filename = sprintf('%s/%s.csv', output_folder, setname);
    dataset_enhanced = [headers'; num2cell(dataset')];
    writecell(dataset_enhanced, filename);
    
    % Save asset prices if configured
    if save_asset_prices
        asset_filename = sprintf('%s/assetprices_%s.csv', output_folder, setname);
        writecell(['S', num2cell(S_paths')], asset_filename);
    end
    
    %% REPORT
    fprintf('\n=================================================================\n');
    fprintf('             PROCESSING COMPLETE FOR %s\n', upper(setname));
    fprintf('=================================================================\n');
    
    actual_contracts = size(dataset, 2);
    fprintf('  Actual contracts: %d\n', actual_contracts);
    fprintf('  Dataset saved: %s\n', filename);
    fprintf('  Total time: %.2f minutes\n', sum(processing_times) / 60);
    fprintf('  Pricing errors: %d (%.1f%%)\n', pricing_errors, 100*pricing_errors/max(actual_contracts,1));
    fprintf('=================================================================\n');

    % Clean up model-specific paths
    if strcmp(model_type, 'DUAN')
        rmpath('Duan_Garch');
    elseif strcmp(model_type, 'GJR')
        rmpath('GJR_Garch');
    end
end

fprintf('\n=================================================================\n');
fprintf('ALL PARAMETER SETS PROCESSED SUCCESSFULLY!\n');
fprintf('=================================================================\n');
fprintf('Output folder: %s\n', output_folder);
fprintf('Files generated:\n');
for i = 1:length(param_sets)
    fprintf('  - %s.csv\n', param_sets(i).name);
end
if save_asset_prices
    fprintf('  - assetprices_*.csv (asset price paths)\n');
end
fprintf('\nTo analyze results, run: rel_err_results\n');
fprintf('=================================================================\n');

end