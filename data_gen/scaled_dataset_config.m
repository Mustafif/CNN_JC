function scaled_dataset_config(varargin)
% Two-phase GARCH option pricing dataset generator.
%
% Phase 1: Pre-cache all willow trees across all parameter sets.
% Phase 2: Price all American options using cached trees.
%
% USAGE:
%   scaled_dataset_optimized_configurable()
%   scaled_dataset_optimized_configurable('config', 'scaled_dataset_config.toml')

    fprintf('=================================================================\n');
    fprintf('  TWO-PHASE GARCH OPTION PRICING DATASET GENERATOR\n');
    fprintf('  Phase 1: Tree Caching | Phase 2: Option Pricing\n');
    fprintf('=================================================================\n\n');

    %% PARSE INPUTS
    p = inputParser;
    addParameter(p, 'config',            'scaled_dataset_config.toml', @ischar);
    addParameter(p, 'model_type',        '',  @ischar);
    addParameter(p, 'noise_level',       [],  @(x) isempty(x) || (isnumeric(x) && x >= 0));
    addParameter(p, 'contracts_per_day', [],  @(x) isempty(x) || (isnumeric(x) && x > 0));
    addParameter(p, 'num_days',          [],  @(x) isempty(x) || (isnumeric(x) && x > 0));
    addParameter(p, 'output_folder',     '',  @ischar);
    parse(p, varargin{:});

    %% LOAD CONFIG
    config_file = p.Results.config;
    fprintf('Loading configuration from: %s\n', config_file);
    if ~exist(config_file, 'file')
        error('Config file not found: %s', config_file);
    end
    config = read_toml_config_matlab(config_file);

    %% EXTRACT PARAMETERS (command-line overrides config)
    model_type = upper(p.Results.model_type);
    if isempty(model_type), model_type = upper(config.model.type); end
    if ~ismember(model_type, {'HN', 'DUAN', 'GJR'})
        error('Invalid model type: %s. Must be HN, DUAN, or GJR.', model_type);
    end

    CONTRACTS_PER_DAY = p.Results.contracts_per_day;
    if isempty(CONTRACTS_PER_DAY), CONTRACTS_PER_DAY = config.scaling.contracts_per_day; end

    NUM_DAYS = p.Results.num_days;
    if isempty(NUM_DAYS), NUM_DAYS = config.scaling.num_days; end

    BATCH_SIZE     = config.scaling.batch_size;
    NUM_PARAM_SETS = config.scaling.num_param_sets;

    N                     = config.market.N;
    r                     = config.market.r;
    S0                    = config.market.S0;
    moneyness_range       = config.market.moneyness_range;
    original_maturity_days = config.market.maturity_days;

    m_h     = config.willow_tree.m_h;
    m_x     = config.willow_tree.m_x;
    gamma_h = config.willow_tree.gamma_h;
    gamma_x = config.willow_tree.gamma_x;

    NUM_WORKERS = config.performance.num_workers;

    NOISE_LEVEL = p.Results.noise_level;
    if isempty(NOISE_LEVEL), NOISE_LEVEL = config.noise.level; end
    NOISE_SEED = config.noise.seed;

    output_folder = p.Results.output_folder;
    if isempty(output_folder)
        if isempty(config.output.folder) || strcmp(config.output.folder, '')
            timestamp     = datestr(now, 'yyyymmdd_HHMMSS');
            output_folder = sprintf('generation_%s_%s', lower(model_type), timestamp);
        else
            output_folder = config.output.folder;
        end
    end

    output_filename = config.output.filename;
    if isempty(output_filename)
        output_filename = sprintf('dataset_%s', lower(model_type));
    end

    %% PARALLEL POOL
    pool_obj = gcp('nocreate');
    if isempty(pool_obj)
        if NUM_WORKERS == 0, NUM_WORKERS = feature('numcores'); end
        fprintf('Starting parallel pool with %d workers...\n', NUM_WORKERS);
        try
            parpool(NUM_WORKERS);
        catch
            fprintf('Could not start parallel pool, using serial processing.\n');
        end
    else
        fprintf('Using existing parallel pool (%d workers).\n', pool_obj.NumWorkers);
    end

    if ~exist(output_folder, 'dir'), mkdir(output_folder); end

    %% PRE-GENERATE SHARED RANDOM DATA
    fprintf('\nPre-generating shared random data...\n');
    rng(12345);
    M        = 1;
    numPoint = N + 1;
    Z        = randn(numPoint + 1, M);

    num_unique_contracts = CONTRACTS_PER_DAY * NUM_DAYS;
    all_moneyness = moneyness_range(1) + ...
                    (moneyness_range(2) - moneyness_range(1)) * rand(1, num_unique_contracts);
    fprintf('Random data ready.\n\n');

    %% GENERATE PARAMETER SETS
    fprintf('Generating %d parameter sets for %s-GARCH...\n', NUM_PARAM_SETS, model_type);
    param_sets = generate_param_sets(model_type, NUM_PARAM_SETS);
    fprintf('Parameter sets ready.\n\n');

    % =====================================================================
    %  PHASE 1 — PRE-CACHE ALL WILLOW TREES
    % =====================================================================
    fprintf('=================================================================\n');
    fprintf('  PHASE 1: PRE-CACHING WILLOW TREES\n');
    fprintf('  %d param sets x %d days = %d trees to build\n', ...
            NUM_PARAM_SETS, NUM_DAYS, NUM_PARAM_SETS * NUM_DAYS);
    fprintf('=================================================================\n\n');

    % all_tree_caches{ps} = cell(NUM_DAYS, 1) of tree structs
    % all_S_paths{ps}     = price path for that param set
    all_tree_caches = cell(NUM_PARAM_SETS, 1);
    all_S_paths     = cell(NUM_PARAM_SETS, 1);

    phase1_start = tic;

    % Add model path once before parfor (parfor workers inherit the path)
    switch model_type
        case 'DUAN', addpath('Duan_Garch');
        case 'GJR',  addpath('GJR_Garch');
    end

    parfor ps = 1:NUM_PARAM_SETS
        fprintf('  [Phase 1] Building trees for param set %d/%d...\n', ps, NUM_PARAM_SETS);

        params = param_sets(ps);
        [tree_cache, S_paths] = build_tree_cache(model_type, params, ...
                                                  M, N, S0, Z, r, NUM_DAYS, ...
                                                  m_h, gamma_h, m_x, gamma_x);

        all_tree_caches{ps} = tree_cache;
        all_S_paths{ps}     = S_paths;
    end

    fprintf('\nPhase 1 complete in %.2f seconds.\n\n', toc(phase1_start));

    % =====================================================================
    %  PHASE 2 — PRICE ALL AMERICAN OPTIONS
    % =====================================================================
    fprintf('=================================================================\n');
    fprintf('  PHASE 2: PRICING AMERICAN OPTIONS\n');
    fprintf('  %d param sets x %d contracts/day x %d days = %d contracts (x2 call/put)\n', ...
            NUM_PARAM_SETS, CONTRACTS_PER_DAY, NUM_DAYS, ...
            NUM_PARAM_SETS * CONTRACTS_PER_DAY * NUM_DAYS);
    fprintf('=================================================================\n\n');

    phase2_start  = tic;
    all_datasets  = cell(NUM_PARAM_SETS, 1);
    param_times   = zeros(NUM_PARAM_SETS, 1);

    parfor ps = 1:NUM_PARAM_SETS
        fprintf('  [Phase 2] Pricing contracts for param set %d/%d...\n', ps, NUM_PARAM_SETS);

        params     = param_sets(ps);
        tree_cache = all_tree_caches{ps};

        alpha  = params.alpha;
        beta   = params.beta;
        omega  = params.omega;
        lambda = params.lambda;
        % GJR has no gamma parameter; use 0 so the dataset column is consistent
        gamma  = 0;
        if isfield(params, 'gamma'), gamma = params.gamma; end

        ps_start = tic;
        all_datasets{ps} = price_contracts(tree_cache, all_moneyness, ...
                                           original_maturity_days, r, N, ...
                                           m_x, gamma_x, alpha, beta, omega, gamma, lambda, ...
                                           CONTRACTS_PER_DAY, NUM_DAYS, BATCH_SIZE, ...
                                           NOISE_LEVEL, NOISE_SEED);
        param_times(ps) = toc(ps_start);

        fprintf('  [Phase 2] Param set %d done in %.2f s.\n', ps, param_times(ps));
    end

    fprintf('\nPhase 2 complete in %.2f seconds.\n', toc(phase2_start));
    fprintf('  Average per param set: %.2f s | Total serial time: %.2f s\n\n', ...
            mean(param_times), sum(param_times));

    %% COMBINE, CLEAN, AND SAVE
    fprintf('Combining and saving results...\n');
    combined = horzcat(all_datasets{:});
    original_size = size(combined, 2);

    valid_mask = (combined(12,:) >= 0.5) & ...
                 (combined(4,:)  >= 1)   & ...
                 all(~isnan(combined), 1) & ...
                 all(~isinf(combined), 1);

    dataset_clean = combined(:, valid_mask);
    cleaned_size  = size(dataset_clean, 2);
    fprintf('  Kept %d / %d contracts (removed %.1f%%)\n', ...
            cleaned_size, original_size, 100*(original_size-cleaned_size)/original_size);

    output_file = sprintf('%s/%s.csv', output_folder, output_filename);
    headers     = {'S0','m','r','T','corp','alpha','beta','omega','gamma','lambda','sigma','V'}';
    writecell([headers'; num2cell(dataset_clean')], output_file);

    save_param_sets(output_folder, param_sets, model_type);
    if config.output.save_asset_prices
        save_asset_files(output_folder, all_S_paths, output_filename);
    end

    fprintf('\n=================================================================\n');
    fprintf('  COMPLETE\n');
    fprintf('  Valid contracts : %d\n', cleaned_size);
    fprintf('  Output          : %s (%.2f MB)\n', output_file, dir(output_file).bytes/1e6);
    fprintf('  Total wall time : %.2f min\n', (toc(phase1_start))/60);
    fprintf('=================================================================\n');
end

% =========================================================================
%  PHASE 1 HELPER — build all day-trees for one parameter set
% =========================================================================

function [tree_cache, S_paths] = build_tree_cache(model_type, params, ...
                                                   M, N, S0, Z, r, NUM_DAYS, ...
                                                   m_h, gamma_h, m_x, gamma_x)
% Returns tree_cache{day} structs and the price path for this param set.

    alpha  = params.alpha;
    beta   = params.beta;
    omega  = params.omega;
    lambda = params.lambda;
    % step   = [];   % computed after we have S_paths

    switch upper(model_type)
        % -----------------------------------------------------------------
        case 'HN'
            gamma = params.gamma;
            [S_paths, h0_paths] = mcHN(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);

            c               = gamma + lambda + 0.5;
            [hd, qhd]       = genhDelta(h0_paths(1), beta, alpha, c, omega, m_h, gamma_h);
            [nodes_ht]      = TreeNodes_ht_HN(m_h, hd, qhd, gamma_h, alpha, beta, c, omega, N+1);
            step            = floor(size(S_paths,1) / NUM_DAYS);

            tree_cache = cell(NUM_DAYS, 1);
            for day = 1:NUM_DAYS
                S_current = S_paths(min(day*step, size(S_paths,1)), 1);
                [nodes_Xt, ~,~,~,~] = TreeNodes_logSt_HN(m_x, gamma_x, r, hd, qhd, ...
                                                           S_current, alpha, beta, c, omega, N);
                [q_Xt, P_Xt, ~]     = Prob_Xt(nodes_ht, qhd, nodes_Xt, S_current, r, ...
                                               alpha, beta, c, omega);
                tree_cache{day} = struct('nodes_S',   exp(nodes_Xt), ...
                                         'P_Xt',      P_Xt, ...
                                         'q_Xt',      q_Xt, ...
                                         'S_current', S_current, ...
                                         'N_steps',   N);
            end

        % -----------------------------------------------------------------
        case 'DUAN'
            gamma = params.gamma;
            [S_paths, h0_paths] = mcDuan(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);

            [nodes_ht, qht, hmom1, hmom2, ~, ~] = TreeNodes_ht_D(m_h, h0_paths(1), gamma_h, ...
                                                                    omega, beta, alpha, gamma, lambda, N+1);
            [nodes_Xt_base, ~,~,~,~] = TreeNodes_logSt_D(m_x, gamma_x, r, h0_paths(1), ...
                                                           omega, beta, alpha, gamma, lambda, N, hmom1, hmom2);
            step = floor(size(S_paths,1) / NUM_DAYS);

            tree_cache = cell(NUM_DAYS, 1);
            for day = 1:NUM_DAYS
                S_current = S_paths(min(day*step, size(S_paths,1)), 1);
                nodes_Xt  = nodes_Xt_base + log(S_current);
                [q_Xt, P_Xt, ~] = Probility_Xt2(nodes_ht, qht, nodes_Xt, S_current, r, ...
                                                  omega, beta, alpha, gamma, lambda);
                tree_cache{day} = struct('nodes_S',   exp(nodes_Xt), ...
                                         'P_Xt',      P_Xt, ...
                                         'q_Xt',      q_Xt, ...
                                         'S_current', S_current, ...
                                         'N_steps',   N);
            end

        % -----------------------------------------------------------------
        case 'GJR'
            [S_paths, h0_paths] = mcGJR(M, N, S0, Z, r, omega, alpha, beta, lambda);

            h0 = h0_paths(1);
            [nodes_ht, qht, hmom1, hmom2, ~, ~] = TreeNodes_ht_GJR(m_h, h0, gamma_h, ...
                                                                      omega, beta, alpha, lambda, N+1);
            [nodes_Xt_base, ~,~,~,~] = TreeNodes_logSt_GJR(m_x, m_h, gamma_x, r, ...
                                                              omega, alpha, lambda, beta, h0, N, hmom1, hmom2);
            step = floor(size(S_paths,1) / NUM_DAYS);

            tree_cache = cell(NUM_DAYS, 1);
            for day = 1:NUM_DAYS
                S_current = S_paths(min(day*step, size(S_paths,1)), 1);
                nodes_Xt  = nodes_Xt_base + log(S_current);
                [q_Xt, P_Xt, ~] = Probility_Xt2(nodes_ht, qht, nodes_Xt, S_current, r, ...
                                                  omega, beta, alpha, 0, lambda);
                tree_cache{day} = struct('nodes_S',   exp(nodes_Xt), ...
                                         'P_Xt',      P_Xt, ...
                                         'q_Xt',      q_Xt, ...
                                         'S_current', S_current, ...
                                         'N_steps',   N);
            end

        otherwise
            error('Unknown model type: %s', model_type);
    end
end

% =========================================================================
%  PHASE 2 HELPER — price all contracts for one parameter set
% =========================================================================

function dataset = price_contracts(tree_cache, all_moneyness, original_maturity_days, ...
                                   r, ~, m_x, gamma_x, ...
                                   alpha, beta, omega, gamma, lambda, ...
                                   CONTRACTS_PER_DAY, NUM_DAYS, BATCH_SIZE, ...
                                   NOISE_LEVEL, NOISE_SEED)
% Prices calls and puts for every (day, contract) combination.
% T_maturity is in trading days throughout; matches daily r.

    num_contracts = CONTRACTS_PER_DAY * NUM_DAYS;
    dataset       = zeros(12, num_contracts * 2);
    write_idx     = 1;

    % Noise stream (serial path only; parfor path seeds per-contract)
    % noise_rng = [];
    if NOISE_LEVEL > 0
        seed = NOISE_SEED;
        if seed < 0, seed = randi([0, 2^31-1]); end
        noise_rng = RandStream('mt19937ar', 'Seed', seed); %#ok<NASGU>
    end

    num_batches = ceil(num_contracts / BATCH_SIZE);

    for batch = 1:num_batches
        b_start      = (batch-1)*BATCH_SIZE + 1;
        b_end        = min(batch*BATCH_SIZE, num_contracts);
        b_size       = b_end - b_start + 1;

        batch_calls  = zeros(12, b_size);
        batch_puts   = zeros(12, b_size);

        for i = 1:b_size
            contract_idx = b_start + i - 1;
            day          = ceil(contract_idx / CONTRACTS_PER_DAY);
            if day > NUM_DAYS, continue; end

            contract_in_day = mod(contract_idx-1, CONTRACTS_PER_DAY) + 1;
            mat_idx         = mod(contract_in_day-1, length(original_maturity_days)) + 1;
            base_mat        = original_maturity_days(mat_idx);
            remaining_days  = base_mat - (day - 1);
            if remaining_days < 1, continue; end

            T_maturity = remaining_days;   % trading days; consistent with daily r

            td        = tree_cache{day};
            S_current = td.S_current;
            nodes_S   = td.nodes_S;
            P_Xt      = td.P_Xt;
            q_Xt      = td.q_Xt;
            N_iv      = td.N_steps;

            moneyness = all_moneyness(contract_idx);
            K_strike  = moneyness * S_current;

            try
                [V_C, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike,  1);
                [V_P, ~] = American(nodes_S, P_Xt, q_Xt, r, T_maturity, S_current, K_strike, -1);

                if V_C < 0 || V_P < 0 || isnan(V_C) || isnan(V_P), continue; end

                if NOISE_LEVEL > 0
                    rng_c = RandStream('mt19937ar', 'Seed', NOISE_SEED + contract_idx*2);
                    rng_p = RandStream('mt19937ar', 'Seed', NOISE_SEED + contract_idx*2 + 1);
                    V_C   = V_C * (1 + NOISE_LEVEL * randn(rng_c));
                    V_P   = V_P * (1 + NOISE_LEVEL * randn(rng_p));
                end

                [impl_c, ~, ~] = impvol_fzero(S_current, K_strike, T_maturity, r, V_C,  1, N_iv, m_x, gamma_x);
                [impl_p, ~, ~] = impvol_fzero(S_current, K_strike, T_maturity, r, V_P, -1, N_iv, m_x, gamma_x);

                batch_calls(:,i) = [S_current; moneyness; r; T_maturity;  1; alpha; beta; omega; gamma; lambda; impl_c; V_C];
                batch_puts(:,i)  = [S_current; moneyness; r; T_maturity; -1; alpha; beta; omega; gamma; lambda; impl_p; V_P];
            catch
                continue;
            end
        end

        combined  = [batch_calls, batch_puts];
        valid     = combined(:, any(combined, 1));
        n_valid   = size(valid, 2);
        dataset(:, write_idx:write_idx+n_valid-1) = valid;
        write_idx = write_idx + n_valid;
    end

    dataset = dataset(:, 1:write_idx-1);
end

% =========================================================================
%  PARAMETER GENERATION  (unchanged logic, condensed)
% =========================================================================

function param_sets = generate_param_sets(model_type, num_sets)
    param_sets = struct('alpha',{}, 'beta',{}, 'omega',{}, 'gamma',{}, 'lambda',{});

    switch upper(model_type)
        case 'HN'
            ranges = struct('omega',[1e-7,1e-6], 'alpha',[1.15e-6,1.36e-6], ...
                            'beta',[0.75,0.85],  'gamma',[50,150], 'lambda',[0.4,0.6]);
            valid_fn = @(o,a,b,g,l) (b + a*g^2 < 1) && (a+b < 1);

        case 'DUAN'
            ranges = struct('omega',[1e-7,1e-6], 'alpha',[1.15e-6,1.36e-6], ...
                            'beta',[0.75,0.85],  'gamma',[0.25,0.5], 'lambda',[0.4,0.6]);
            valid_fn = @(o,a,b,g,l) (b + a*g^2 < 1) && (a+b < 1);

        case 'GJR'
            ranges = struct('omega',[1e-7,1e-6], 'alpha',[0.01,0.15], ...
                            'beta',[0.75,0.85],  'gamma',[0,0], 'lambda',[0.01,0.15]);
            valid_fn = @(o,a,b,g,l) (b + a + l/2 < 0.999) && (o>0) && (a>=0) && (b>=0) && (a+l>=0);

        otherwise
            error('Unknown model type: %s', model_type);
    end

    for i = 1:num_sets
        valid = false; attempts = 0;
        while ~valid && attempts < 1000
            o = sample(ranges.omega);
            a = sample(ranges.alpha);
            b = sample(ranges.beta);
            g = sample(ranges.gamma);
            l = sample(ranges.lambda);
            if valid_fn(o,a,b,g,l)
                param_sets(i) = struct('alpha',a,'beta',b,'omega',o,'gamma',g,'lambda',l);
                valid = true;
            end
            attempts = attempts + 1;
        end
        if ~valid
            warning('Failed to generate valid %s parameters for set %d.', model_type, i);
        end
    end
end

function v = sample(range)
    v = range(1) + diff(range) * rand();
end

% =========================================================================
%  SAVE HELPERS  (unchanged)
% =========================================================================

function save_param_sets(output_folder, param_sets, model_type)
    num_sets      = length(param_sets);
    params_matrix = zeros(num_sets, 5);
    for i = 1:num_sets
        params_matrix(i,:) = [param_sets(i).omega, param_sets(i).alpha, ...
                               param_sets(i).beta,  param_sets(i).gamma, param_sets(i).lambda];
    end
    headers  = {'set_id','omega','alpha','beta','gamma','lambda'}';
    set_ids  = (1:num_sets)';
    csv_data = [headers'; num2cell([set_ids, params_matrix])];
    csv_file = sprintf('%s/garch_parameters_%s.csv', output_folder, lower(model_type));
    writecell(csv_data, csv_file);
    fprintf('  Saved parameters: %s\n', csv_file);
end

function save_asset_files(output_folder, all_S_paths, output_filename)
    nonempty = all_S_paths(~cellfun('isempty', all_S_paths));
    if isempty(nonempty), warning('No asset price paths to save.'); return; end

    num_rows   = size(nonempty{1}, 1);
    total_cols = sum(cellfun(@(x) size(x,2), nonempty));
    combined   = zeros(num_rows, total_cols);
    col = 1;
    for ps = 1:length(all_S_paths)
        S = all_S_paths{ps};
        if isempty(S), continue; end
        nc = size(S,2);
        combined(:, col:col+nc-1) = double(S);
        col = col + nc;
    end
    csv_file = sprintf('%s/%s_asset_prices.csv', output_folder, output_filename);
    writematrix(combined, csv_file);
    fprintf('  Saved asset prices: %s\n', csv_file);
end

% =========================================================================
%  CONFIG READER  (unchanged)
% =========================================================================

function config = read_toml_config_matlab(filename)
    fid = fopen(filename, 'r');
    if fid < 0, error('Cannot open file: %s', filename); end
    config          = struct();
    current_section = '';
    while ~feof(fid)
        line = fgetl(fid);
        if ~ischar(line), continue; end
        line = strtrim(line);
        if isempty(line) || line(1) == '#', continue; end

        if line(1) == '[' && line(end) == ']'
            section_parts   = strsplit(line(2:end-1), '.');
            current_section = strjoin(section_parts, '.');
            sp = section_parts;
            if ~isfield(config, sp{1}), config.(sp{1}) = struct(); end
            if length(sp) == 2
                if ~isfield(config.(sp{1}), sp{2})
                    config.(sp{1}).(sp{2}) = struct();
                end
            end
            continue;
        end

        if contains(line, '=')
            parts     = strsplit(line, '=');
            key       = strtrim(parts{1});
            value_str = strtrim(strjoin(parts(2:end), '='));
            ci        = find(value_str == '#', 1);
            if ~isempty(ci), value_str = strtrim(value_str(1:ci-1)); end
            value     = parse_toml_value(value_str);

            sp = strsplit(current_section, '.');
            if length(sp) == 2
                config.(sp{1}).(sp{2}).(key) = value;
            elseif isscalar(sp) && ~isempty(sp{1})
                config.(sp{1}).(key) = value;
            else
                config.(key) = value;
            end
        end
    end
    fclose(fid);
end

function value = parse_toml_value(str)
    str = strtrim(str);
    if (str(1)=='"'  && str(end)=='"' ), value = str(2:end-1); return; end
    if (str(1)==''''&& str(end)==''''), value = str(2:end-1); return; end
    if strcmpi(str,'true'),  value = true;  return; end
    if strcmpi(str,'false'), value = false; return; end
    if str(1)=='[' && str(end)==']'
        elems = strsplit(str(2:end-1), ',');
        value = cellfun(@(e) str2double(strtrim(e)), elems);
        value = value(~isnan(value));
        return;
    end
    num_val = str2double(str);
    if ~isnan(num_val), value = num_val; return; end
    value = str;
end