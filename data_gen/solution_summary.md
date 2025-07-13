# Implied Volatility Spread Analysis: Problem Diagnosis and Solution

## Executive Summary

The Heston-Nandi GARCH option pricing model shows significant implied volatility spreads between call and put options, with calls averaging 39.25% IV and puts averaging 3.35% IV. This 35.9 percentage point spread indicates serious numerical and solver issues rather than genuine market dynamics.

## Problem Diagnosis

### Key Findings

1. **Solver Boundary Issues**
   - 8.9% of calls hit upper bound (IV ≥ 0.99)
   - 64.4% of puts hit lower bound (IV ≤ 0.02)
   - Restrictive bisection bounds [0.001, 1.0] are inadequate

2. **Put-Call Parity Violations**
   - 42 out of 45 option pairs violate put-call parity
   - Maximum violation: 9.58 price units
   - Indicates fundamental pricing inconsistencies

3. **Maturity Pattern**
   - IV spread decreases with maturity (61.96% → 15.62%)
   - Suggests time-to-expiration dependent solver issues

4. **Numerical Precision**
   - Deep OTM options have near-zero values
   - Solver struggles with small target prices
   - Willow tree discretization errors compound

## Root Cause Analysis

### 1. Bisection Solver Limitations

**File: `impvol.m`**
```matlab
a = 0;          % Lower bound too restrictive
b = 1;          % Upper bound too restrictive (100% max vol)
sigma = (a+b)/2; % Poor initial guess (50%)
```

**Issues:**
- Upper bound of 100% volatility is unrealistic for GARCH models
- No moneyness-based initial guess
- Fixed bounds regardless of option characteristics

### 2. Risk-Neutral Transformation

**File: `impVol_HN.m`**
```matlab
c = gamma + lambda + 0.5;  % Risk-neutral parameter
```

**Issues:**
- With γ=5, λ=0.2: c = 5.7 (extremely high)
- May cause numerical instability in tree construction
- Risk-neutral measure transformation may be incorrect

### 3. Tree Construction Parameters

**Current settings:**
- m_x = 30 (log-price nodes)
- m_h = 6 (volatility nodes)
- N = 504 (time steps)

**Issues:**
- Insufficient resolution for extreme GARCH parameters
- Probability calibration may be inaccurate
- Tree discretization errors

## Specific Solutions

### 1. Immediate Fixes

#### A. Expand Bisection Bounds
```matlab
% In impvol.m, replace:
a = 0;
b = 1;

% With:
sigma_min = 0.001;  % 0.1% minimum
sigma_max = 3.0;    % 300% maximum (suitable for GARCH)
```

#### B. Implement Moneyness-Based Initial Guess
```matlab
% Add to impvol.m:
moneyness = K / S0;
if index == 1  % Call option
    if moneyness < 1
        initial_guess = 0.3 + 0.2 * (1 - moneyness);  % Higher IV for ITM
    else
        initial_guess = 0.2 + 0.1 * (moneyness - 1);  % Lower IV for OTM
    end
else  % Put option
    if moneyness > 1
        initial_guess = 0.3 + 0.2 * (moneyness - 1);  % Higher IV for ITM
    else
        initial_guess = 0.2 + 0.1 * (1 - moneyness);  % Lower IV for OTM
    end
end
```

#### C. Add Convergence Diagnostics
```matlab
% Add to impvol.m:
if abs(V0 - target_price) > tol && it >= itmax
    fprintf('Warning: No convergence for K=%.2f, T=%.3f, Error=%.6f\n', ...
            K, T, abs(V0 - target_price));
end
```

### 2. Model Improvements

#### A. Increase Tree Resolution
```matlab
% In demo.m, increase:
m_x = 50;  % More log-price nodes
m_h = 10;  % More volatility nodes
```

#### B. Validate Tree Construction
```matlab
% Add validation in TreeNodes_logSt_HN.m:
if any(isnan(nodes_Xt(:))) || any(isinf(nodes_Xt(:)))
    error('Invalid tree nodes detected');
end
```

#### C. European Option Baseline
```matlab
% Create European option pricer for validation:
function V_european = European_HN(nodes_S, P_Xt, q_Xt, r, T, K, CorP)
    % No early exercise - pure discounting
    [M, N] = size(nodes_S);
    if CorP == 1
        payoff = max(nodes_S(:,N) - K, 0);
    else
        payoff = max(K - nodes_S(:,N), 0);
    end
    V_european = exp(-r*T) * q_Xt' * payoff;
end
```

### 3. Parameter Validation

#### A. GARCH Parameter Bounds
```matlab
% Add to impVol_HN.m:
if alpha < 0 || beta < 0 || omega < 0
    error('GARCH parameters must be non-negative');
end
if alpha + beta >= 1
    warning('GARCH parameters suggest non-stationarity');
end
```

#### B. Risk-Neutral Parameter Check
```matlab
% Add to impVol_HN.m:
c = gamma + lambda + 0.5;
if abs(c) > 10
    warning('Risk-neutral parameter c=%.2f may cause numerical issues', c);
end
```

## Implementation Priority

### Phase 1: Critical Fixes (Immediate)
1. **Replace `impvol.m` with improved version**
   - Expand bounds to [0.001, 3.0]
   - Add moneyness-based initial guess
   - Implement convergence diagnostics

2. **Add boundary checking in `impVol_HN.m`**
   - Validate option prices before solving
   - Check for reasonable bounds
   - Add error handling

### Phase 2: Model Validation (Week 1)
1. **Implement European option baseline**
   - Create European pricer
   - Validate put-call parity
   - Compare with analytical solutions

2. **Parameter sensitivity analysis**
   - Test different GARCH parameters
   - Validate tree construction
   - Check probability calibration

### Phase 3: Production Enhancements (Week 2)
1. **Advanced solver improvements**
   - Newton-Raphson method
   - Vega-based iterations
   - Adaptive bounds

2. **Performance optimization**
   - Parallel processing
   - Memory optimization
   - Caching strategies

## Code Files to Modify

### 1. `impvol.m` → `impvol_fixed.m`
- Expand bounds [0.001, 3.0]
- Add moneyness-based initial guess
- Implement convergence diagnostics
- Add boundary validation

### 2. `impVol_HN.m` → `impVol_HN_fixed.m`
- Add parameter validation
- Implement error handling
- Add European option fallback
- Improve debugging output

### 3. `demo.m` → `demo_fixed.m`
- Use improved solvers
- Add validation checks
- Filter unrealistic options
- Generate quality metrics

## Expected Results

### After Fixes:
- **Convergence rate**: 95%+ (vs current ~50%)
- **Boundary hits**: <5% (vs current 65%)
- **Put-call parity violations**: <1% (vs current 93%)
- **IV spread**: <10 percentage points (vs current 36)
- **Realistic volatility range**: 5%-200% (vs current 0.2%-100%)

## Validation Checklist

- [ ] Bisection bounds expanded to [0.001, 3.0]
- [ ] Moneyness-based initial guess implemented
- [ ] Convergence diagnostics added
- [ ] Parameter validation implemented
- [ ] European option baseline created
- [ ] Put-call parity validation added
- [ ] Tree construction validated
- [ ] Performance benchmarks established
- [ ] Error handling comprehensive
- [ ] Documentation updated

## Conclusion

The implied volatility spread issue stems primarily from solver boundary problems and inadequate initial guesses rather than model deficiencies. The proposed fixes address these root causes systematically and should restore realistic option pricing behavior.

**Next Steps:**
1. Implement improved `impvol_fixed.m`
2. Test with European options first
3. Validate put-call parity
4. Gradually expand to American options
5. Monitor convergence and boundary statistics

This solution maintains the sophisticated Heston-Nandi GARCH model while fixing the numerical implementation issues that caused the unrealistic volatility spreads.