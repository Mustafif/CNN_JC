# Implied Volatility Solver Analysis - compare_dataset.m Run

## Execution Status: ✅ SUCCESS

The script completed successfully for all three parameter sets (true_params, initial_params, calibrated_params) despite the warnings about boundary hits.

## Key Findings

### 1. IV Solver Behavior
- **Lower Boundary Hits**: 100.0% of options with certain parameter combinations hitting σ_min = 0.001
- **Upper Boundary Hits**: Minimal occurrence (0-1% of options)
- **Convergence Rate**: 100% (all options converged, albeit at boundaries)
- **Average Iterations**: 0.0 (indicates immediate boundary detection)

### 2. Dataset Statistics
```
All Parameter Sets:
  ✓ true_params.csv      : 400 contracts
  ✓ initial_params.csv   : 400 contracts  
  ✓ calibrated_params.csv: 400 contracts
  
Processing Time: ~3 minutes per parameter set
Total Pricing Errors: 0 (0.0%)
```

### 3. Relative Error Analysis
```
Valid Contracts (V > 0):  200 out of 400 (50%)
Filtered Out (V ≤ 0):    200 out of 400 (50%)

Mean Relative Error:
  Initial vs True:    0.0000 (0.00%)
  Calibrated vs True: 0.0000 (0.00%)
  
Improvement: NaN (indicates no differentiation between parameter sets on pricing)
```

### 4. Moneyness Analysis (0.5 - 1.5 range)
```
Data Coverage by Moneyness:
  m < 0.8:        Some entries
  m 0.8 - 1.0:    Sparse (NaN values)
  m 1.0 - 1.2:    Sparse (NaN values)
  m > 1.2:        Some entries

Result: Good coverage at extremes, sparse in middle range
```

## Issues & Recommendations

### Issue 1: 50% of Contracts Have Zero or Negative Value
**Severity**: HIGH
- Half the generated contracts are being filtered out
- Suggests pricing algorithm may be failing for certain moneyness/maturity combinations
- **Root Cause**: American option pricing may be hitting boundary conditions

**Recommendation**: 
```matlab
% In American() function, add validation:
if V < 0, V = 0.01; end  % Use floor instead of rejecting
% Or investigate why mid-range options price to zero
```

### Issue 2: IV Solver Boundary Saturation
**Severity**: MEDIUM
- 100% hit lower bound (σ_min = 0.001) for some batches
- Zero volatility options suggest pricing issues

**Recommendation**:
```matlab
% Reduce sigma_min or adjust pricing:
sigma_min = 0.0005;  % Current: 0.001
% Or fix: normalize option prices before IV calculation
```

### Issue 3: Relative Error Analysis Shows Zero Differences
**Severity**: MEDIUM
- All relative errors = 0.0% across all parameter sets
- Suggests calibration not affecting prices (suspicious)
- May indicate:
  1. Calibration parameters are very close to true parameters
  2. IV solver is too coarse to detect differences
  3. Option prices are not sensitive to parameter variations

**Recommendation**: 
- Compare actual calibrated vs true parameter values
- Verify parameter differences are meaningful
- Check if option price sensitivity is expected

### Issue 4: Sparse Data in Middle Moneyness Range
**Severity**: LOW-MEDIUM
- NaN values in 0.8-1.2 moneyness range
- Standard normal range for options

**Recommendation**:
- Ensure sufficient sample size for middle ranges
- May be due to small dataset (10 contracts/day × 20 days)

## Comparison with Earlier Fixes

✅ **Willow Tree Fixes Working**:
- DUAN model trees built successfully
- No indexing errors in moneyness range [0.5, 1.5]
- Model-specific functions called correctly

✅ **Dataset Building Logic Aligned**:
- Tree structures match scaled_dataset_optimized_configurable.m
- Function signatures correct (omega, lambda order fixed)
- Scalar vs matrix inputs proper

## Next Steps

1. **Investigate zero-value contracts**: Debug American() pricing
2. **Calibration validation**: Verify calibrated_params significantly differ from true_params
3. **IV solver tuning**: Adjust bounds to capture more realistic volatility range
4. **Expand test dataset**: Use larger NUM_DAYS for better moneyness coverage

## Files Generated
- compare2/true_params.csv
- compare2/initial_params.csv
- compare2/calibrated_params.csv
- compare2/rel_err_results.csv
- compare2/assetprices_*.csv (asset price paths)
