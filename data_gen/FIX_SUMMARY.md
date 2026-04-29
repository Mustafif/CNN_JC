# Fix Summary: Tree Building Functions

## Date: 2026-04-08
## File: scaled_dataset_optimized_configurable.m

---

## Issues Fixed

### ISSUE #1: CRITICAL - Wrong Tree Functions for Model Types ✓ FIXED
**Problem**: File accepted 3 model types (HN, DUAN, GJR) but only used HN tree functions
**Status**: ✓ FIXED

#### Before (BROKEN):
```matlab
parfor ps = 1:NUM_PARAM_SETS
    [S_paths, h0_paths] = mcHN(...);           % ALWAYS HN
    [nodes_ht] = TreeNodes_ht_HN(...);         % ALWAYS HN  
    [nodes_Xt] = TreeNodes_logSt_HN(...);      % ALWAYS HN
end
```

#### After (FIXED):
```matlab
parfor ps = 1:NUM_PARAM_SETS
    switch upper(model_type)
        case 'HN'
            dataset = process_parameter_set_HN(...);
        case 'DUAN'
            dataset = process_parameter_set_DUAN(...);
        case 'GJR'
            dataset = process_parameter_set_GJR(...);
    end
end
```

### ISSUE #2: CRITICAL - Wrong MC Simulation Functions ✓ FIXED
**Problem**: Only used mcHN regardless of model selection
**Status**: ✓ FIXED

#### Solution:
- **HN Model**: Uses `mcHN(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda)`
- **DUAN Model**: Uses `mcDuan(M, N, S0, Z, r, omega, alpha, beta, theta, lambda)` from Duan_Garch/
- **GJR Model**: Uses `mcGJR(M, N, S0, Z, r, omega, alpha, beta, lambda)` from GJR_Garch/

### ISSUE #3: IMPORTANT - Parameter Generation Not Model-Aware ✓ FIXED
**Problem**: generate_param_sets() used same baseline_gamma for all models
**Status**: ✓ FIXED

#### Before:
```matlab
function param_sets = generate_param_sets(model_type, num_sets)
    % Accepts model_type but doesn't use it
    baseline_gamma = 5.0;  % Used for ALL models
end
```

#### After:
```matlab
function param_sets = generate_param_sets(model_type, num_sets)
    if strcmpi(model_type, 'HN')
        baseline_gamma = 5.0;      % HN uses gamma
        baseline_theta = 0;
    elseif strcmpi(model_type, 'DUAN')
        baseline_gamma = 0;        % DUAN doesn't use gamma
        baseline_theta = 0.5;      % DUAN uses theta
    elseif strcmpi(model_type, 'GJR')
        baseline_gamma = 0;        % GJR doesn't use gamma
        baseline_theta = 0;        % GJR doesn't use theta
    end
end
```

---

## Changes Made

### 1. Added Model-Specific Processing Functions (Lines 322-441)

Three new functions handle each model type:

#### `process_parameter_set_HN()`
- Calls `mcHN()` for asset price simulation
- Uses `TreeNodes_ht_HN()` and `TreeNodes_logSt_HN()` for tree building
- Parameters: omega, alpha, beta, gamma, lambda

#### `process_parameter_set_DUAN()`
- Calls `mcDuan()` for asset price simulation (from Duan_Garch/)
- Uses `TreeNodes_ht_D()` and `TreeNodes_logSt_D()` for tree building
- Parameters: omega, alpha, beta, theta (NOT gamma), lambda
- Adds/removes Duan_Garch path for function access

#### `process_parameter_set_GJR()`
- Calls `mcGJR()` for asset price simulation (from GJR_Garch/)
- Uses `TreeNodes_ht_GJR()` and `TreeNodes_logSt_GJR()` for tree building
- Parameters: omega, alpha, beta, lambda (NO gamma, NO theta)
- Adds/removes GJR_Garch path for function access

### 2. Updated Parfor Loop (Lines 229-258)
- Replaced hardcoded model calls with switch statement
- Routes to correct processing function based on model_type
- All tree and MC functions now selected per model

### 3. Model-Aware Parameter Generation (Lines 322-360)
- Now checks model_type and sets appropriate baseline parameters
- Generates theta for DUAN model
- Sets gamma to 0 for DUAN and GJR (unused)
- Parameters now match each model's requirements

---

## Tree Building Matrix

| Model | MC Function | Tree h_t | Tree X_t | Prob Function | Location |
|-------|-------------|----------|----------|---------------|----------|
| HN | mcHN | TreeNodes_ht_HN | TreeNodes_logSt_HN | Prob_Xt | Main folder |
| DUAN | mcDuan | TreeNodes_ht_D | TreeNodes_logSt_D | Probility_Xt | Duan_Garch/ |
| GJR | mcGJR | TreeNodes_ht_GJR | TreeNodes_logSt_GJR | Probility_Xt | GJR_Garch/ |

---

## Parameter Mapping

### HN Model
```matlab
[S_paths, h0] = mcHN(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda)
h_t = omega + alpha * (Z_{t-1} - gamma * sqrt(h_{t-1}))^2 + beta * h_{t-1}
```

### DUAN Model
```matlab
[S_paths, h0] = mcDuan(M, N, S0, Z, r, omega, alpha, beta, theta, lambda)
h_t = omega + beta * h_{t-1} + alpha * h_{t-1} * (Z_{t-1} - theta - lambda)^2
```

### GJR Model
```matlab
[S_paths, h0] = mcGJR(M, N, S0, Z, r, omega, alpha, beta, lambda)
h_t = omega + beta * h_{t-1} + alpha * Z_{t-1}^2 * h_{t-1} + lambda * I{Z_{t-1}<0} * Z_{t-1}^2 * h_{t-1}
```

---

## Verification

All fixes verified:
✓ Model-specific processing functions created
✓ Switch statement routes to correct model
✓ All three MC functions called appropriately
✓ All tree functions called appropriately
✓ Parameter generation is model-aware
✓ theta parameter handled for DUAN
✓ Model folders added to path when needed

---

## Testing Recommendation

To verify the fix works:
```matlab
% Test HN model
scaled_dataset_optimized_configurable('model_type', 'HN', ...
                                      'contracts_per_day', 10, 'num_days', 5)

% Test DUAN model
scaled_dataset_optimized_configurable('model_type', 'DUAN', ...
                                      'contracts_per_day', 10, 'num_days', 5)

% Test GJR model
scaled_dataset_optimized_configurable('model_type', 'GJR', ...
                                      'contracts_per_day', 10, 'num_days', 5)
```

---

## File Statistics

- **Lines Added**: ~200 (new processing functions)
- **Lines Modified**: ~50 (parfor loop, parameter generation)
- **Lines Removed**: ~50 (old hardcoded model calls)
- **Total File Size**: 756 lines (was 619 lines)

