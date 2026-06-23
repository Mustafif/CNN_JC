%% ------------------ GJR GARCH - PARALLEL + BATCHED VERSION ------------------
clear all; clc;

fprintf('=================================================================\n');
fprintf('   GJR GARCH - PARALLEL + BATCHED VERSION\n');
fprintf('=================================================================\n\n');

%% ------------------ PARAMETERS ------------------
OBS_DAYS          = 20;
CONTRACTS_PER_TTM = 4;
T_days            = [30 60 90 180 360];
NUM_PARAM_SETS    = 25;
BATCH_SIZE        = 5;   % <-- tune this: lower = less RAM, more I/O
N                 = 504;
r                 = 0.05/252;
S0                = 100;

moneyness_range   = [0.8, 1.2];
obs_days          = round(linspace(1, N, OBS_DAYS));
num_maturities    = numel(T_days);

contracts_per_param_set = OBS_DAYS * num_maturities * CONTRACTS_PER_TTM;
total_contracts         = NUM_PARAM_SETS * contracts_per_param_set;
num_batches             = ceil(NUM_PARAM_SETS / BATCH_SIZE);

fprintf('CONFIGURATION:\n');
fprintf('  Param sets:         %d\n', NUM_PARAM_SETS);
fprintf('  Batch size:         %d param sets\n', BATCH_SIZE);
fprintf('  Num batches:        %d\n', num_batches);
fprintf('  Total contracts:    %d (x2 = %d rows)\n', total_contracts, total_contracts*2);

%% ------------------ WILLOW TREE & SOLVER PARAMETERS ------------------
m_h     = 6;
m_ht    = 6;
m_x     = 30;
gamma_h = 0.6;
gamma_x = 0.8;
itmax   = 100;
tol     = 1e-6;

%% ------------------ GARCH PARAMETER GENERATION ------------------
alpha_range  = [0.01, 0.15];
beta_range   = [0.75, 0.85];
omega_range  = [1e-7, 1e-6];
mu_range  = [-5e-4,    5e-4  ];
lambda_range = [0.01, 0.15];

param_sets = zeros(NUM_PARAM_SETS, 5);
valid_sets = 0;
while valid_sets < NUM_PARAM_SETS
    alpha  = alpha_range(1)  + diff(alpha_range)  * rand();
    beta   = beta_range(1)   + diff(beta_range)   * rand();
    omega  = omega_range(1)  + diff(omega_range)  * rand();
    mu  = mu_range(1)  + diff(mu_range)  * rand();
    lambda = lambda_range(1) + diff(lambda_range) * rand();
    if beta + alpha + lambda/2 < 0.999
        valid_sets = valid_sets + 1;
        param_sets(valid_sets, :) = [alpha, beta, omega, mu, lambda];
    end
end

%% ------------------ GARCH PATH SIMULATION (all sets upfront) ------------------
M_paths  = min(NUM_PARAM_SETS, 50);
numPoint = N + 1;
Z_seeds  = randn(numPoint + 1, M_paths);

S_paths_all  = cell(NUM_PARAM_SETS, 1);
h0_paths_all = zeros(NUM_PARAM_SETS, 1);

for p = 1:NUM_PARAM_SETS
    seed_idx = mod(p-1, M_paths) + 1;
    [S_path, h0] = mcGJR(1, N, S0, Z_seeds(:, seed_idx), param_sets(p, 4), ...
                         param_sets(p,3), param_sets(p,1), ...
                         param_sets(p,2), param_sets(p,5));
    S_paths_all{p}  = S_path;
    h0_paths_all(p) = h0;
end

%% ------------------ ADD PATHS ------------------
currentFolder = pwd;
addpath(currentFolder);
addpath(genpath(currentFolder));
pctRunOnAll addpath(currentFolder);
pctRunOnAll addpath(genpath(currentFolder));

%% ------------------ BATCHED PROCESSING ------------------
headers      = {'S0','m','r','T','corp','alpha','beta','omega','gamma','lambda','sigma','V'};
partial_files = cell(num_batches, 1);
total_elapsed = 0;

for batch = 1:num_batches

    p_start = (batch-1)*BATCH_SIZE + 1;
    p_end   = min(batch*BATCH_SIZE, NUM_PARAM_SETS);
    batch_p = p_start:p_end;
    B       = numel(batch_p);   % actual size (last batch may be smaller)

    fprintf('\n--- Batch %d/%d  (param sets %d:%d) ---\n', ...
            batch, num_batches, p_start, p_end);

    %% -- Build contract map for this batch only --
    batch_contracts = B * contracts_per_param_set;
    batch_map       = zeros(batch_contracts, 4);  % [local_p, d, t, k]
    row = 1;
    for lp = 1:B
        for d = 1:OBS_DAYS
            for t = 1:num_maturities
                for k = 1:CONTRACTS_PER_TTM
                    batch_map(row,:) = [lp, d, t, k];
                    row = row + 1;
                end
            end
        end
    end

    %% -- Pre-cache willow trees for this batch --
    fprintf('  Caching trees...\n');
    tic;

    % Slice only what this batch needs — avoids broadcasting entire cell arrays
    batch_params   = param_sets(batch_p, :);
    batch_h0       = h0_paths_all(batch_p);
    batch_S_paths  = S_paths_all(batch_p);

    tree_cache = cell(B, OBS_DAYS);

    parfor lp = 1:B
        alpha  = batch_params(lp,1);
        beta   = batch_params(lp,2);
        omega  = batch_params(lp,3);
        mu  = batch_params(lp,4);
        lambda = batch_params(lp,5);
        h0     = batch_h0(lp);

        local_cache = cell(OBS_DAYS, 1);

        [nodes_ht, qht, hmom1, hmom2, ~, ~] = ...
            TreeNodes_ht_GJR(m_h, h0, gamma_h, omega, beta, alpha, lambda, N+1);

        for d = 1:OBS_DAYS
            day_idx = obs_days(d);
            S_cur   = batch_S_paths{lp}(day_idx);

            [nodes_Xt, ~, ~, ~, ~] = ...
                TreeNodes_logSt_GJR(m_x, m_h, gamma_x, mu, ...
                                    omega, alpha, lambda, beta, ...
                                    h0, N, hmom1, hmom2);
            nodes_Xt = nodes_Xt + log(S_cur);

            [q_Xt, P_Xt, ~] = ...
                Probility_Xt2(nodes_ht, qht, nodes_Xt, ...
                              S_cur, mu, omega, beta, alpha, lambda);

            local_cache{d} = struct( ...
                'nodes_S', exp(nodes_Xt), ...
                'P_Xt',    P_Xt, ...
                'q_Xt',    q_Xt, ...
                'S_cur',   S_cur);
        end
        local_cache_out{lp} = local_cache;   %#ok<AGROW>
    end
    % Unpack parfor output into tree_cache
    for lp = 1:B
        tree_cache(lp,:) = local_cache_out{lp};
    end
    clear local_cache_out

    tree_time = toc;
    fprintf('  Trees done in %.2fs\n', tree_time);

    %% -- Price contracts for this batch --
    fprintf('  Pricing %d contracts...\n', batch_contracts);
    tic;

    results = cell(batch_contracts, 1);

    parfor idx = 1:batch_contracts
        lp = batch_map(idx,1);
        d  = batch_map(idx,2);
        t  = batch_map(idx,3);

        tree = tree_cache{lp, d};

        rng(idx + (batch-1)*1e6);   % unique seeds across batches
        moneyness     = moneyness_range(1) + diff(moneyness_range)*rand();
        maturity_days = T_days(t);
        maturity_years = maturity_days / 252;
        K = moneyness * tree.S_cur;

        [V_C, ~] = American(tree.nodes_S, tree.P_Xt, tree.q_Xt, r, ...
                             maturity_years, tree.S_cur, K,  1);
        [V_P, ~] = American(tree.nodes_S, tree.P_Xt, tree.q_Xt, r, ...
                             maturity_years, tree.S_cur, K, -1);

        [impl_c, ~, ~] = impvol(tree.S_cur, K, maturity_years, r, V_C,  1, ...
                                 N, m_x, gamma_x, tol, itmax);
        [impl_p, ~, ~] = impvol(tree.S_cur, K, maturity_years, r, V_P,  1, ...
                                 N, m_x, gamma_x, tol, itmax);

        % Use global param index for the output
        global_p = batch_p(lp);     %#ok<PFBNS>
        params = param_sets(global_p, :)';

        results{idx} = [ ...
            [tree.S_cur; K/tree.S_cur; r; maturity_years;  1; params; impl_c; V_C], ...
            [tree.S_cur; K/tree.S_cur; r; maturity_years; -1; params; impl_p; V_P]  ...
        ];
    end

    price_time = toc;
    fprintf('  Pricing done in %.2fs\n', price_time);
    total_elapsed = total_elapsed + tree_time + price_time;

    %% -- Assemble and write partial CSV --
    dataset_batch = zeros(12, batch_contracts * 2);
    for i = 1:batch_contracts
        col = (i-1)*2 + 1;
        dataset_batch(:, col:col+1) = results{i};
    end

    partial_file = sprintf('GJR_partial_batch%02d.csv', batch);
    if batch == 1
        % Write with header on first batch
        writecell([headers; num2cell(dataset_batch')], partial_file);
    else
        % No header on subsequent batches (we'll concatenate below)
        writematrix(dataset_batch', partial_file);
    end
    partial_files{batch} = partial_file;
    fprintf('  Written: %s\n', partial_file);

    %% -- Free batch memory explicitly --
    clear tree_cache results dataset_batch batch_map
    clear batch_params batch_h0 batch_S_paths
end

%% ------------------ CONCATENATE PARTIAL FILES ------------------
fprintf('\nConcatenating %d partial files...\n', num_batches);

filename = sprintf('GJR_%dK.csv', total_contracts * 2 / 1000);

% Read batch 1 (has header) and write to final file
final_data = readcell(partial_files{1});

for batch = 2:num_batches
    chunk = readmatrix(partial_files{batch});
    final_data = [final_data; num2cell(chunk)];  %#ok<AGROW>
end

writecell(final_data, filename);

% Clean up partials
for batch = 1:num_batches
    delete(partial_files{batch});
end

%% ------------------ SUMMARY ------------------
fprintf('=================================================================\n');
fprintf('COMPLETE\n');
fprintf('  Total elapsed:  %.2f seconds\n', total_elapsed);
fprintf('  Rows saved:     %d\n', total_contracts * 2);
fprintf('  File:           %s\n', filename);
fprintf('=================================================================\n');