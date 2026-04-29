# Implied Volatility Solver Update

## Change Summary

**Date:** 2026-03-31

**What Changed:** Switched from custom iterative solver (`impvol_improved`) to MATLAB's `fzero`-based solver (`impvol_fzero`)

**Files Updated:**
- `scaled_dataset.m`
- `compare_dataset.m`

**New File Created:**
- `impvol_fzero.m`

## Why Use fzero?

### Advantages of fzero

1. **More Robust:** MATLAB's `fzero` is a well-tested, industrial-strength root-finding algorithm
2. **Better Convergence:** Uses Brent's method (combination of bisection, secant, and inverse quadratic interpolation)
3. **Fewer Parameters:** No need to tune tolerance and max iterations
4. **Handles Edge Cases:** Better at handling difficult cases near boundaries
5. **Less Code:** Simpler implementation, less maintenance

### Comparison

**Old Method (impvol_improved):**
```matlab
[impl_c, ~, ~, ~] = impvol_improved(S_current, K_strike, T_maturity, r, V_C, 1, N, m_x, gamma_x, tol, itmax);
```
- Custom Newton-Raphson or bisection
- Requires tuning `tol` and `itmax`
- May fail to converge in edge cases
- More complex implementation

**New Method (impvol_fzero):**
```matlab
[impl_c, ~, ~] = impvol_fzero(S_current, K_strike, T_maturity, r, V_C, 1, N, m_x, gamma_x);
```
- Uses MATLAB's proven `fzero` function
- Automatic tolerance and convergence control
- Fallback bisection if `fzero` fails
- Simpler interface

## Implementation Details

### impvol_fzero.m Features

1. **Smart Initial Guess:**
   - Based on moneyness and option type
   - Different guesses for ITM, ATM, OTM options

2. **Wide Search Range:**
   - Minimum: 0.1% (0.001)
   - Maximum: 500% (5.0)
   - Handles extreme GARCH volatilities

3. **Robust Error Handling:**
   - Checks if solution can be bracketed
   - Falls back to bisection if `fzero` fails
   - Returns sensible defaults on complete failure

4. **Efficient:**
   - Pre-computes z values and probability matrices
   - Reuses tree structure within objective function

### Function Signature

```matlab
function [imp_vol, V0, converged] = impvol_fzero(S0, K, T, r, V_target, index, N, m, gamma)
```

**Inputs:**
- `S0` - Current stock price
- `K` - Strike price
- `T` - Time to maturity (years)
- `r` - Risk-free rate
- `V_target` - Target option price
- `index` - Option type (+1 call, -1 put)
- `N` - Number of time steps
- `m` - Number of tree nodes
- `gamma` - Willow tree gamma parameter

**Outputs:**
- `imp_vol` - Implied volatility
- `V0` - Option price at implied volatility
- `converged` - Convergence flag

## Backward Compatibility

The old `impvol_improved` function is still available if needed. To revert:

```matlab
% In scaled_dataset.m or compare_dataset.m, change:
[impl_c, ~, ~] = impvol_fzero(...);

% Back to:
[impl_c, ~, ~, ~] = impvol_improved(..., tol, itmax);
```

## Testing

To verify the new solver works correctly:

```matlab
% Test case
S0 = 100;
K = 100;
T = 0.25;  % 3 months
r = 0.05/252;
V_target = 5.0;
index = 1;  % Call
N = 252;
m = 30;
gamma = 0.8;

% New method
[iv_new, V_new, conv_new] = impvol_fzero(S0, K, T, r, V_target, index, N, m, gamma);

% Old method (for comparison)
tol = 1e-6;
itmax = 100;
[iv_old, V_old, ~, conv_old] = impvol_improved(S0, K, T, r, V_target, index, N, m, gamma, tol, itmax);

fprintf('New method: IV=%.4f, Price=%.4f, Converged=%d\n', iv_new, V_new, conv_new);
fprintf('Old method: IV=%.4f, Price=%.4f, Converged=%d\n', iv_old, V_old, conv_old);
```

## Performance Notes

**Speed:**
- `fzero` may be slightly slower than a well-tuned Newton-Raphson
- However, convergence is more reliable
- Overall fewer failures means less re-computation

**Accuracy:**
- Both methods achieve similar accuracy when they converge
- `fzero` has better convergence rate (fewer failures)

**Robustness:**
- `fzero` handles edge cases much better
- Particularly important for extreme GARCH volatilities
- Reduces need for manual intervention

## Migration Checklist

If you have existing code using the old method:

- [x] `scaled_dataset.m` updated to use `impvol_fzero`
- [x] `compare_dataset.m` updated to use `impvol_fzero`
- [ ] Test with actual data to verify results
- [ ] Compare convergence rates between old and new
- [ ] Update any custom scripts using implied vol solvers

## Recommendations

1. **Use `impvol_fzero` for all new code**
2. **Test thoroughly** with your specific GARCH parameters
3. **Monitor convergence rates** - should improve with new method
4. **Keep old solver** as backup until fully validated

## Questions?

If you encounter issues with the new solver:

1. Check that all required functions are in path:
   - `zq.m`
   - `gen_PoWiner.m`
   - `nodes_Winer.m`
   - `American.m`

2. Verify input parameters are reasonable:
   - S0, K, T > 0
   - r is appropriate (daily or annual)
   - V_target > 0

3. Try the old solver as comparison:
   ```matlab
   [iv_old, ~, ~, ~] = impvol_improved(...);
   ```

4. Check for extreme cases:
   - Very deep ITM/OTM options
   - Very short/long maturities
   - Extreme volatilities (>200%)

## References

- MATLAB Documentation: `doc fzero`
- Brent's Method: Press et al., "Numerical Recipes"
- Original impvol_improved: Based on bisection/Newton-Raphson hybrid
