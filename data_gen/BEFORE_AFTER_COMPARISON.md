# Before & After Comparison

## CRITICAL BUG: Tree Building Functions

### Before (BROKEN ❌)
```
Model Type Selection:  model_type = 'DUAN' or 'GJR'
                                    ↓
                       Validation passes ✓
                                    ↓
              Parfor Loop (all parameter sets)
                                    ↓
         [S, h0] = mcHN(...)  ← WRONG! Should be mcDuan or mcGJR
                                    ↓
         [nodes_ht] = TreeNodes_ht_HN(...)  ← WRONG!
                                    ↓
         [nodes_Xt] = TreeNodes_logSt_HN(...)  ← WRONG!
                                    ↓
              Dataset generated with HN model
                      (not DUAN or GJR!)
                                    ↓
                    RESULT: INVALID DATASET ❌
```

### After (FIXED ✓)
```
Model Type Selection:  model_type = 'DUAN' or 'GJR'
                                    ↓
                       Validation passes ✓
                                    ↓
              Parfor Loop (all parameter sets)
                                    ↓
            switch upper(model_type)
         /              |              \
      'HN'          'DUAN'           'GJR'
        |              |               |
    mcHN          mcDuan          mcGJR
        |              |               |
TreeNodes_ht_HN   TreeNodes_ht_D  TreeNodes_ht_GJR
        |              |               |
TreeNodes_        TreeNodes_      TreeNodes_
logSt_HN          logSt_D         logSt_GJR
        |              |               |
    Prob_Xt       Probility_Xt    Probility_Xt
        |              |               |
  HN Dataset      DUAN Dataset    GJR Dataset
        |              |               |
   CORRECT ✓      CORRECT ✓      CORRECT ✓
```

---

## Code Comparison

### PARFOR LOOP

#### Before:
```matlab
parfor ps = 1:NUM_PARAM_SETS
    fprintf('  [Worker] Processing parameter set %d/%d...\n', ps, NUM_PARAM_SETS);
    
    params = param_sets(ps);
    alpha = params.alpha;
    beta = params.beta;
    omega = params.omega;
    gamma = params.gamma;      % Extracted but passed to all models
    lambda = params.lambda;
    
    ps_start = tic;
    
    % Generate GARCH paths - ALWAYS HN
    [S_paths, h0_paths] = mcHN(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);
    
    % Build trees - ALWAYS HN
    c = gamma + lambda + 0.5;
    [hd, qhd] = genhDelta(h0_paths(1), beta, alpha, c, omega, m_h, gamma_h);
    [nodes_ht] = TreeNodes_ht_HN(m_h, hd, qhd, gamma_h, alpha, beta, c, omega, N+1);
    
    % Cache trees for all days
    tree_cache = cell(NUM_DAYS, 1);
    for day = 1:NUM_DAYS
        step = floor(size(S_paths, 1) / NUM_DAYS);
        S_current = S_paths(min(day * step, size(S_paths, 1)), 1);
        
        [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_HN(m_x, gamma_x, r, hd, qhd, S_current, alpha, beta, c, omega, N);
        [q_Xt, P_Xt, ~] = Prob_Xt(nodes_ht, qhd, nodes_Xt, S_current, r, alpha, beta, c, omega);
        nodes_S = exp(nodes_Xt);
        
        tree_cache{day} = struct(...);
    end
    
    % Process contracts
    dataset = process_contracts_with_noise(..., gamma, lambda, ...);
    
    all_datasets{ps} = dataset;
    all_processing_times(ps) = toc(ps_start);
end
```

#### After:
```matlab
parfor ps = 1:NUM_PARAM_SETS
    fprintf('  [Worker] Processing parameter set %d/%d...\n', ps, NUM_PARAM_SETS);
    
    params = param_sets(ps);
    ps_start = tic;
    
    % Route to correct model processing function
    switch upper(model_type)
        case 'HN'
            dataset = process_parameter_set_HN(params, M, N, S0, Z, r, NUM_DAYS, ...
                                               m_h, gamma_h, m_x, gamma_x, ...
                                               all_moneyness, T_years, BATCH_SIZE, ...
                                               NOISE_LEVEL, NOISE_SEED, CONTRACTS_PER_DAY, all_S_paths, ps);
        case 'DUAN'
            dataset = process_parameter_set_DUAN(params, M, N, S0, Z, r, NUM_DAYS, ...
                                                 m_h, gamma_h, m_x, gamma_x, ...
                                                 all_moneyness, T_years, BATCH_SIZE, ...
                                                 NOISE_LEVEL, NOISE_SEED, CONTRACTS_PER_DAY, all_S_paths, ps);
        case 'GJR'
            dataset = process_parameter_set_GJR(params, M, N, S0, Z, r, NUM_DAYS, ...
                                                m_h, gamma_h, m_x, gamma_x, ...
                                                all_moneyness, T_years, BATCH_SIZE, ...
                                                NOISE_LEVEL, NOISE_SEED, CONTRACTS_PER_DAY, all_S_paths, ps);
        otherwise
            error('Unknown model type: %s', model_type);
    end
    
    all_datasets{ps} = dataset;
    all_processing_times(ps) = toc(ps_start);
    
    fprintf('  [Worker] Parameter set %d complete: %.2f seconds\n', ps, all_processing_times(ps));
end
```

---

### PARAMETER GENERATION

#### Before:
```matlab
function param_sets = generate_param_sets(model_type, num_sets)
    param_sets = struct('alpha', {}, 'beta', {}, 'omega', {}, 'gamma', {}, 'lambda', {});
    % model_type parameter is IGNORED!
    
    for i = 1:num_sets
        baseline_omega = 1e-6;
        baseline_alpha = 1.33e-6;
        baseline_beta = 0.8;
        baseline_gamma = 5.0;      % SAME for all models (WRONG!)
        baseline_lambda = 0.2;
        
        scale = 0.8 + 0.4 * rand();
        
        param_sets(i).omega = baseline_omega * scale;
        param_sets(i).alpha = baseline_alpha * scale;
        param_sets(i).beta = baseline_beta * (0.95 + 0.1 * rand());
        param_sets(i).gamma = baseline_gamma * (0.8 + 0.4 * rand());  % Used for all!
        param_sets(i).lambda = baseline_lambda * (0.8 + 0.4 * rand());
    end
end
```

#### After:
```matlab
function param_sets = generate_param_sets(model_type, num_sets)
    % Generate parameter sets for specified model type
    
    if strcmpi(model_type, 'HN')
        param_sets = struct('alpha', {}, 'beta', {}, 'omega', {}, 'gamma', {}, 'theta', {}, 'lambda', {});
        baseline_omega = 1e-6;
        baseline_alpha = 1.33e-6;
        baseline_beta = 0.8;
        baseline_gamma = 5.0;      % HN uses gamma
        baseline_theta = 0;        % HN doesn't use theta
        baseline_lambda = 0.2;
        
    elseif strcmpi(model_type, 'DUAN')
        param_sets = struct('alpha', {}, 'beta', {}, 'omega', {}, 'gamma', {}, 'theta', {}, 'lambda', {});
        baseline_omega = 1e-6;
        baseline_alpha = 1.33e-6;
        baseline_beta = 0.8;
        baseline_gamma = 0;        % DUAN doesn't use gamma
        baseline_theta = 0.5;      % DUAN uses theta
        baseline_lambda = 0.2;
        
    elseif strcmpi(model_type, 'GJR')
        param_sets = struct('alpha', {}, 'beta', {}, 'omega', {}, 'gamma', {}, 'theta', {}, 'lambda', {});
        baseline_omega = 1e-6;
        baseline_alpha = 1.33e-6;
        baseline_beta = 0.8;
        baseline_gamma = 0;        % GJR doesn't use gamma
        baseline_theta = 0;        % GJR doesn't use theta
        baseline_lambda = 0.2;     % GJR leverage parameter
    else
        error('Unknown model type: %s', model_type);
    end
    
    for i = 1:num_sets
        scale = 0.8 + 0.4 * rand();
        
        param_sets(i).omega = baseline_omega * scale;
        param_sets(i).alpha = baseline_alpha * scale;
        param_sets(i).beta = baseline_beta * (0.95 + 0.1 * rand());
        param_sets(i).gamma = baseline_gamma * (0.8 + 0.4 * rand());   % Per model
        param_sets(i).theta = baseline_theta * (0.8 + 0.4 * rand());   % Per model
        param_sets(i).lambda = baseline_lambda * (0.8 + 0.4 * rand());
    end
end
```

---

## Results

### HN Model (model_type='HN')
| Before | After |
|--------|-------|
| ✓ Uses mcHN | ✓ Uses mcHN |
| ✓ Uses TreeNodes_ht_HN | ✓ Uses TreeNodes_ht_HN |
| ✓ Uses TreeNodes_logSt_HN | ✓ Uses TreeNodes_logSt_HN |
| ✓ CORRECT | ✓ CORRECT |

### DUAN Model (model_type='DUAN')
| Before | After |
|--------|-------|
| ❌ Uses mcHN (WRONG) | ✓ Uses mcDuan |
| ❌ Uses TreeNodes_ht_HN (WRONG) | ✓ Uses TreeNodes_ht_D |
| ❌ Uses TreeNodes_logSt_HN (WRONG) | ✓ Uses TreeNodes_logSt_D |
| ❌ INVALID DATASET | ✓ CORRECT DATASET |

### GJR Model (model_type='GJR')
| Before | After |
|--------|-------|
| ❌ Uses mcHN (WRONG) | ✓ Uses mcGJR |
| ❌ Uses TreeNodes_ht_HN (WRONG) | ✓ Uses TreeNodes_ht_GJR |
| ❌ Uses TreeNodes_logSt_HN (WRONG) | ✓ Uses TreeNodes_logSt_GJR |
| ❌ INVALID DATASET | ✓ CORRECT DATASET |

---

## Impact

### Before Fix
- Only HN model works correctly
- DUAN results are using HN model (invalid)
- GJR results are using HN model (invalid)
- Any research using DUAN or GJR datasets is compromised

### After Fix
- HN model works correctly ✓
- DUAN model works correctly ✓
- GJR model works correctly ✓
- All three models can be used independently
- Parameter sets are model-appropriate
- Datasets are now valid for all three model types

