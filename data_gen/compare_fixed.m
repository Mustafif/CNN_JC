clear all; clc;
% demo_scalable_3sets.m
% Scalable version of Heston-Nandi GARCH option pricing with flexible dataset generation
% Now handles 3 different GARCH parameter sets with the SAME randomness.

cal_path = 'compare/results_01.json';
jsonStr = fileread(cal_path);
cal_data = jsondecode(jsonStr);

init_path = 'compare/init.json';
% Load initial conditions from JSON file
initStr = fileread(init_path);
init_data = jsondecode(initStr);

fprintf('=================================================================\n');
fprintf('      SCALABLE HESTON-NANDI GARCH OPTION PRICING DEMO\n');
fprintf('=================================================================\n\n');

%% SCALING PARAMETERS - MODIFY THESE TO SCALE THE DATASET
CONTRACTS_PER_DAY = 10;     % Number of option contracts to generate per day
NUM_DAYS         = 20;      % Number of trading days to simulate
BATCH_SIZE       = 100;     % Process options in batches to manage memory

% Display scaling configuration
fprintf('SCALING CONFIGURATION:\n');
fprintf('  Contracts per day: %d\n', CONTRACTS_PER_DAY);
fprintf('  Number of days: %d\n', NUM_DAYS);
fprintf('  Total contracts: %d\n', CONTRACTS_PER_DAY * NUM_DAYS * 2); % x2 for calls and puts
fprintf('  Batch size: %d\n', BATCH_SIZE);
fprintf('  Estimated memory usage: %.1f MB\n', ...
    (CONTRACTS_PER_DAY * NUM_DAYS * 2 * 12 * 8) / 1e6);

%% MARKET PARAMETERS
N       = 1024;                         % Number of time steps
T       = [30, 60, 90, 180];  % Base maturities
r       = 0.05/252;                     % Risk-free rate (daily)
S0      = 100;                          % Initial stock price
K_base  = 100;                          % Base strike price

% Strike generation for scaling
moneyness_range = [0.8, 1.2];           % Moneyness range

%% GARCH PARAMETER SETS

% TRUE (market) parameters
true_params.name   = 'true_params';
true_params.omega  = 1e-6;
true_params.alpha  = 1.33e-6;
true_params.beta   = 0.8;
true_params.gamma  = 5;
true_params.lambda = 0.2;

% INITIAL guess parameters (PUT YOUR VALUES HERE)
initial_params.name   = 'initial_params';
initial_params.omega  = init_data.omega;
initial_params.alpha  = init_data.alpha;
initial_params.beta   = init_data.beta;
initial_params.gamma  = init_data.gamma;
initial_params.lambda = init_data.lambda;

% CALIBRATED parameters (PUT YOUR VALUES HERE)
calibrated_params.name   = 'calibrated_params';
calibrated_params.omega  = cal_data.omega;
calibrated_params.alpha  = cal_data.alpha;
calibrated_params.beta   = cal_data.beta;
calibrated_params.gamma  = cal_data.gamma;
calibrated_params.lambda = cal_data.lambda;

param_sets = [true_params, initial_params, calibrated_params];

%% WILLOW TREE PARAMETERS
m_h     = 6;
m_ht    = 6;
m_x     = 30;
gamma_h = 0.6;
gamma_x = 0.8;

%% SOLVER PARAMETERS
itmax = 100;
tol   = 1e-6;

%% PRE-GENERATE RANDOMNESS FOR GARCH SIMULATION (SHARED ACROSS PARAM SETS)
fprintf('\nPre-generating random numbers for GARCH simulation...\n');
M        = 1;               % Number of paths
numPoint = N + 1;

% Fix seed for reproducibility across runs (optional but nice)
rng(12345);
Z_master = randn(numPoint + 1, M);

fprintf('Random matrix Z generated: size = %d x %d\n', size(Z_master,1), size(Z_master,2));

%% LOOP OVER PARAMETER SETS
total_contracts = CONTRACTS_PER_DAY * NUM_DAYS * 2;  % x2 for calls and puts
headers = {'S0', 'm', 'r', 'T', 'corp', ...
           'alpha', 'beta', 'omega', 'gamma', 'lambda', ...
           'sigma', 'V'}';

for ps = 1:numel(param_sets)
    params  = param_sets(ps);
    fprintf('\n>>> NOW PROCESSING ps=%d, name=%s\n', ps, params.name);
    alpha   = params.alpha;
    beta    = params.beta;
    omega   = params.omega;
    gamma   = params.gamma;
    lambda  = params.lambda;
    setname = params.name;

    fprintf('\n============================================================\n');
    fprintf('  PROCESSING PARAMETER SET: %s\n', setname);
    fprintf('============================================================\n');

    %% GENERATE SCALED GARCH PATHS FOR THIS PARAMETER SET
    fprintf('\nGenerating scaled GARCH simulation for %s...\n', setname);

    Z = Z_master;  % same randomness for *all* parameter sets

    [S_paths, h0_paths] = mcHN(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);
    fprintf('Generated %d GARCH paths for %s\n', M, setname);

    %% INITIALIZE DATASET FOR THIS PARAMETER SET

    dataset     = zeros(12, total_contracts);  % 12 metrics × contracts
    dataset_idx = 1;

    fprintf('\nInitializing dataset structure for %s...\n', setname);
    fprintf('Dataset dimensions: %d metrics × %d contracts\n', ...
        size(dataset, 1), size(dataset, 2));

    %% PREPARE TREE STRUCTURES (ONCE PER PARAMETER SET)

    fprintf('\nBuilding willow trees for %s (one-time setup)...\n', setname);
    c = gamma + lambda + 0.5;  % Risk-neutral parameter

    try
        [hd, qhd] = genhDelta(h0_paths, beta, alpha, c, omega, m_h, gamma_h);
% DIAGNOSTIC - paste output
fprintf('Calling TreeNodes_ht_HN with:\n');
fprintf('  m_h=%d, gamma_h=%.4f, alpha=%.6e, beta=%.6f\n', m_h, gamma_h, alpha, beta);
fprintf('  gamma(c)=%.4f, omega=%.6e, N=%d\n', c, omega, N);
fprintf('  hd size: %dx%d\n', size(hd,1), size(hd,2));
fprintf('  qhd size: %dx%d\n', size(qhd,1), size(qhd,2));

[nodes_ht] = TreeNodes_ht_HN(m_h, hd, qhd, gamma_h, alpha, beta, c, omega, N+1);
fprintf('TreeNodes_ht_HN succeeded\n');

% Check nodes_ht output
fprintf('nodes_ht: size=%dx%d, complex=%d, anynan=%d, anyinf=%d\n', ...
    size(nodes_ht,1), size(nodes_ht,2), ...
    any(~isreal(nodes_ht(:))), any(isnan(nodes_ht(:))), any(isinf(nodes_ht(:))));

        % Pre-compute tree nodes for different days
        tree_cache = cell(NUM_DAYS, 1);
        for day = 1:NUM_DAYS
            % You may want a different mapping day -> S, but keep your logic
            S_current = S_paths(end - CONTRACTS_PER_DAY + day, min(day, M));
            h_current = h0_paths(min(day, M));

            % [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_HN( ...
            %     m_x, gamma_x, r, hd, qhd, S_current, ...
            %     alpha, beta, c, omega, N);
            % 
            % [q_Xt, P_Xt, ~] = Prob_Xt(nodes_ht, qhd, nodes_Xt, ...
            %                           S_current, r, alpha, beta, c, omega);
            
            try
    [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_HN(m_x, gamma_x, r, hd, qhd, S_current, alpha, beta, c, omega, N);
    fprintf('TreeNodes_logSt_HN succeeded\n');
catch ME3
    fprintf('TreeNodes_logSt_HN failed: %s\n', ME3.message);
    for ii = 1:length(ME3.stack)
        fprintf('  %s line %d\n', ME3.stack(ii).name, ME3.stack(ii).line);
    end
    rethrow(ME3);
end

try
    [q_Xt, P_Xt, ~] = Prob_Xt(nodes_ht, qhd, nodes_Xt, S_current, r, alpha, beta, c, omega);
    fprintf('Prob_Xt succeeded\n');
catch ME4
    fprintf('Prob_Xt failed: %s\n', ME4.message);
    for ii = 1:length(ME4.stack)
        fprintf('  %s line %d\n', ME4.stack(ii).name, ME4.stack(ii).line);
    end
end


            nodes_S = exp(nodes_Xt);

            tree_cache{day} = struct('nodes_S', nodes_S, 'P_Xt', P_Xt, ...
                                     'q_Xt', q_Xt, 'S_current', S_current, ...
                                     'h_current', h_current);
        end

        fprintf('Trees cached for %d days (%s)\n', NUM_DAYS, setname);

    catch ME
        fprintf('Error building trees for %s: %s\n', setname, ME.message);
        continue;  % Skip this parameter set if tree fails
    end

    %% PRE-GENERATE ALL MONEYNESS VALUES (avoids rng reseeding inside loop)
    rng(42);  % Fixed seed for reproducibility
    num_unique_contracts = CONTRACTS_PER_DAY * NUM_DAYS;
    all_moneyness = moneyness_range(1) + ...
                    (moneyness_range(2) - moneyness_range(1)) * rand(1, num_unique_contracts);

    %% BATCH PROCESSING SETUP FOR THIS PARAMETER SET

    num_batches      = ceil(total_contracts / (BATCH_SIZE * 2));  % calls+puts together
    fprintf('\nProcessing %s in %d batches...\n', setname, num_batches);

    % Statistics tracking
    total_processed  = 0;
    pricing_errors   = 0;
    solver_failures  = 0;
    boundary_issues  = 0;
    processing_times = zeros(num_batches, 1);

    %% MAIN PROCESSING LOOP

    for batch = 1:num_batches
        batch_start_time = tic;
        fprintf('\n[%s] Processing batch %d/%d...\n', setname, batch, num_batches);

        % Calculate batch boundaries
        contracts_in_batch = min(BATCH_SIZE, ...
            ceil((total_contracts/2 - (batch-1)*BATCH_SIZE)));
        batch_start_idx = (batch - 1) * BATCH_SIZE + 1;

        batch_errors    = 0;
        batch_processed = 0;

        % Pre-extract batch data for parfor (broadcast variables)
        contracts_range = 1:contracts_in_batch;
        batch_results   = cell(contracts_in_batch, 1);

        parfor contract_in_batch = contracts_range
            contract_idx    = batch_start_idx + contract_in_batch - 1;
            day             = ceil(contract_idx / CONTRACTS_PER_DAY);
            if day > NUM_DAYS
                batch_results{contract_in_batch} = [];
                continue;
            end
            contract_in_day = mod(contract_idx - 1, CONTRACTS_PER_DAY) + 1;

            % Get cached tree data for this day
            tree_data = tree_cache{day};
            S_current = tree_data.S_current;
            nodes_S   = tree_data.nodes_S;
            P_Xt      = tree_data.P_Xt;
            q_Xt      = tree_data.q_Xt;

            % Use pregenerated moneyness and fixed maturity
            moneyness  = all_moneyness(contract_idx);
            K_strike   = moneyness * S_current;
            T_maturity = T(mod(contract_in_day - 1, length(T)) + 1);

            try
                [V_C, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike,  1);
                [V_P, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, -1);

                if V_C < 0 || V_P < 0 || isnan(V_C) || isnan(V_P)
                    batch_results{contract_in_batch} = 'error';
                    continue;
                end

                [impl_c, ~, ~, ~] = impvol_improved(S_current, K_strike, T_maturity, r, V_C,  1, N, m_x, gamma_x, tol, itmax);
                [impl_p, ~, ~, ~] = impvol_improved(S_current, K_strike, T_maturity, r, V_P, -1, N, m_x, gamma_x, tol, itmax);

                batch_results{contract_in_batch} = [
                    S_current; moneyness; r; T_maturity;  1; alpha; beta; omega; gamma; lambda; impl_c; V_C;
                    S_current; moneyness; r; T_maturity; -1; alpha; beta; omega; gamma; lambda; impl_p; V_P
                ];

            catch
                batch_results{contract_in_batch} = 'error';
            end
        end

        % Collect parfor results sequentially
        for contract_in_batch = 1:contracts_in_batch
            res = batch_results{contract_in_batch};
            if isempty(res) || (ischar(res) && strcmp(res, 'error'))
                batch_errors   = batch_errors + 1;
                pricing_errors = pricing_errors + 1;
                continue;
            end
            % res is 24x1: first 12 = call, next 12 = put
            dataset(:, dataset_idx)   = res(1:12);
            dataset(:, dataset_idx+1) = res(13:24);
            dataset_idx     = dataset_idx + 2;
            batch_processed = batch_processed + 2;
        end

        % Batch completion statistics
        processing_times(batch) = toc(batch_start_time);
        total_processed         = total_processed + batch_processed;

        fprintf('  [%s] Batch %d complete: %d contracts processed, %d errors, %.2f seconds\n', ...
                setname, batch, batch_processed, batch_errors, processing_times(batch));

        % Memory management
        if mod(batch, 5) == 0
            fprintf('  [%s] Memory cleanup...\n', setname);
            clear V_C V_P impl_c impl_p V0_c V0_p;
        end
    end

    %% FINALIZE DATASET FOR THIS PARAMETER SET

    dataset = dataset(:, 1:dataset_idx-1);

    %% SAVE RESULTS FOR THIS PARAMETER SET

    fprintf('\nSaving results for %s...\n', setname);

    filename = sprintf('compare/%s.csv', setname);
    dataset_enhanced = [headers'; num2cell(dataset')];
    writecell(dataset_enhanced, filename);

    % Save asset price path for this parameter set
    asset_filename = sprintf('assetprices_%s.csv', setname);
    writecell(['S', num2cell(S_paths')], asset_filename);

    %% COMPREHENSIVE REPORTING FOR THIS PARAMETER SET

    fprintf('\n=================================================================\n');
    fprintf('             PROCESSING COMPLETE FOR %s\n', upper(setname));
    fprintf('=================================================================\n');

    actual_contracts = size(dataset, 2);
    call_data        = dataset(:, dataset(5,:) == 1);
    put_data         = dataset(:, dataset(5,:) == -1);

    fprintf('\nSCALING RESULTS (%s):\n', setname);
    fprintf('  Planned contracts: %d\n', total_contracts);
    fprintf('  Actual contracts: %d\n', actual_contracts);
    fprintf('  Calls: %d, Puts: %d\n', size(call_data, 2), size(put_data, 2));
    fprintf('  Dataset saved as: %s\n', filename);
    fprintf('  File size: %.2f MB\n', dir(filename).bytes / 1e6);

    fprintf('\nPERFORMANCE METRICS (%s):\n', setname);
    fprintf('  Total processing time: %.2f minutes\n', sum(processing_times) / 60);
    fprintf('  Average time per batch: %.2f seconds\n', mean(processing_times));
    fprintf('  Contracts per second: %.1f\n', actual_contracts / sum(processing_times));
    fprintf('  Pricing errors: %d (%.1f%%)\n', pricing_errors, 100*pricing_errors/max(actual_contracts,1));
    fprintf('  Solver failures: %d (%.1f%%)\n', solver_failures, 100*solver_failures/max(actual_contracts,1));
    fprintf('  Boundary issues: %d (%.1f%%)\n', boundary_issues, 100*boundary_issues/max(actual_contracts,1));

    % Data quality analysis
    all_ivs  = dataset(11,:);
    valid_ivs = all_ivs(all_ivs > 0.001 & all_ivs < 2.99);
    call_ivs = call_data(11,:);
    put_ivs  = put_data(11,:);

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

    % Moneyness and maturity distribution
    moneyness_data = dataset(2,:);
    maturity_data  = dataset(4,:);

    fprintf('\nDATA DISTRIBUTION (%s):\n', setname);
    fprintf('  Moneyness range: [%.3f, %.3f]\n', min(moneyness_data), max(moneyness_data));
    fprintf('  Maturity range: [%.3f, %.3f] years\n', min(maturity_data), max(maturity_data));
    fprintf('  Spot price range: [%.2f, %.2f]\n', min(dataset(1,:)), max(dataset(1,:)));

    fprintf('\n=================================================================\n');
    if length(valid_ivs)/max(length(all_ivs),1) > 0.95 && ...
       pricing_errors < actual_contracts*0.05
        fprintf('SUCCESS: Dataset generation for %s completed successfully!\n', setname);
    else
        fprintf('PARTIAL SUCCESS: Some data quality issues detected for %s.\n', setname);
    end
    fprintf('=================================================================\n');

end

fprintf('\nAll parameter sets processed.\n');
fprintf('To scale further, modify CONTRACTS_PER_DAY and NUM_DAYS at the top of the script.\n');