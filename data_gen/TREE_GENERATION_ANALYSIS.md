# Tree Generation Performance Analysis

## Current Bottleneck Breakdown

For **60K contracts × 15 param sets** with N=50 time steps, m_x=50 nodes:

| Operation | Time | Frequency |
|-----------|------|-----------|
| TreeNodes_logSt_HN | **0.1-0.3 sec** | 900K times |
| Prob_Xt | **0.05-0.1 sec** | 900K times |
| American pricing | 0.05 sec | 900K times |
| IV calculation* | 0.023 sec | 900K times (OPTIMIZED) |
| **Total** | **~14-18 hours** | |

*IV calculation is now optimized with caching. Tree generation is the main bottleneck.

## Root Cause: What Makes Tree Generation Slow

### 1. TreeNodes_logSt_HN: 4th-Order Moment Computation
**Current Algorithm:**
- Computes 5 sequences: diffB0, diffB1, diffB2, diffB3, diffB4 (N steps each)
- Each step involves complex arithmetic with divisions and multiplications
- Then computes 5 sequences: diffA0, diffA1, diffA2, diffA3, diffA4 (N steps each)
- Finally computes 4-dimensional MGF expansions
- Total: **~1000+ scalar operations per call**

**Why it's slow:**
- Sequential loop with no vectorization (can't skip dependencies)
- Heavy use of scalar operations instead of vector operations
- Complex arithmetic (divisions, roots, exponentials)
- Numerical stability checks add overhead

**Estimated breakdown:**
- DiffB/DiffA computation: ~70% of time (deeply sequential)
- MGF calculation: ~20% of time
- Tree node generation: ~10% of time

### 2. Prob_Xt: Transition Probability Matrix
**Current Algorithm:**
- For each time step t and node combination:
  - Computes normcdf bounds for X transitions
  - Computes normcdf bounds for h transitions
  - Nested double loop: m_h × m_x operations
- Total: **~2500 normcdf() calls per contract** (50 nodes × 50 nodes)

**Why it's slow:**
- normcdf() is expensive (involves Phi error function)
- Called 2500 times per contract
- Not vectorized
- 900K contracts × 2500 calls = 2.25 billion normcdf() calls total

## Optimization Strategies (Ranked by Impact)

### CRITICAL OPTIMIZATIONS (Can Implement)

#### 1. **Vectorize TreeNodes_logSt_HN** (Est. 3-5× speedup)
Current: Each step computed sequentially in loops
Better: Use array operations for independent calculations

**Strategy:**
```matlab
% Current (slow):
for row = 2:numPoint
    diffB0(row) = beta*diffB0(row-1) + ... % Serial dependency
end

% Better (vectorized where possible):
% Pre-compute powers, divisions, exponentials as matrices
% Use matrix operations instead of scalar loops
```

**Challenge:** diffB sequences have strict dependencies (row i depends on row i-1)
**Solution:** Can't fully vectorize, but can:
- Pre-compute common subexpressions outside loops
- Vectorize the MGF calculations (currently scalar)
- Cache repeated division operations

**Estimated speedup: 2-3×** (from reducing loop overhead)

#### 2. **Vectorize Prob_Xt normcdf Calls** (Est. 4-6× speedup)
Current: 2500 separate normcdf() calls per contract
Better: Batch calls into vector operations

**Strategy:**
```matlab
% Current (slow - 2500 calls):
for j = 1:m_x
    p = normcdf(z_bounds); % Scalar call repeated
end

% Better (vectorized - 1 call):
% Stack all bounds into arrays
% Call vectorized normcdf once
p_all = normcdf(all_bounds); % Single vectorized call
```

**Implementation:**
- Instead of loop over m_x, create arrays of all bounds
- Use vectorized normcdf to compute all at once
- Reshape results back

**Estimated speedup: 4-6×** (vectorized normcdf is much faster)

#### 3. **Cache Probability Matrix for Repeated Maturities** (Est. 2-3× speedup)
Current: Recalculate Prob_Xt for same (S0, params, N) on each day
Better: Cache by (S0, day, param_set)

**Challenge:** S0 changes daily, so can't cache indefinitely
**Solution:** Cache within parameter set (reuse same probability matrix for all contracts on same day)

**Estimated speedup: 1.5-2×** (high variance by days)

#### 4. **Pre-compute Common Subexpressions** (Est. 1.5-2× speedup)
Current: Redundant calculations within TreeNodes_logSt_HN
Better: Calculate once, reuse

**Examples:**
- `1 - 2*alpha*diffB0` is computed 4 times per row
- `sqrt(var)` is computed multiple times
- Exponentials with same base

**Estimated speedup: 1.5×** (modest but easy)

### MODERATE OPTIMIZATIONS (Requires More Work)

#### 5. **Approximation for Tree Generation** (Est. 5-10× speedup)
Use approximate moment-matching instead of exact 4th-order

**Trade-off:** Slightly less accurate but much faster
- Use 2nd-order moments instead of 4th
- Skip Prob_Xt refinement on low-probability paths

**Estimated speedup: 5-10×** but with accuracy loss

#### 6. **Parallel Computation of Multiple Trees** (Est. 2-3× speedup)
Current: Serial computation within parfor
Better: Vectorize tree computation

**Challenge:** parfor already parallelizes over parameter sets
**Solution:** Within each param set, compute trees for multiple maturities in parallel

**Estimated speedup: 1.5-2×** (limited by dependencies)

### DIFFICULT OPTIMIZATIONS (Would Require Major Refactoring)

#### 7. **GPU Acceleration** (Est. 10-50× speedup)
Port TreeNodes_logSt_HN and Prob_Xt to GPU

**Effort:** Very high (requires CUDA or gpuArray)
**Speedup:** 10-50× for matrix operations

#### 8. **C++ MEX Implementation** (Est. 5-15× speedup)
Implement core loops in C++

**Effort:** High (requires C++ programming)
**Speedup:** 5-15× from compiled code + vectorization

## Quick Wins (Can Implement Now)

### Quick Win #1: Vectorize Prob_Xt normcdf
```matlab
% Stack all normcdf bounds into single array
% Single vectorized call instead of 2500 scalar calls
% Expected: 4-6× speedup
% Effort: 20-30 minutes
```

### Quick Win #2: Pre-compute Tree Subexpressions  
```matlab
% Cache divisions: inv_denom = 1.0 ./ (1 - 2*alpha*diffB0)
% Cache powers: diffB0_sq = diffB0.^2
% Expected: 1.5-2× speedup
% Effort: 30-40 minutes
```

### Quick Win #3: Memoize Within Day
```matlab
% Within each day, S0 is constant
% Reuse Prob_Xt computation for all contracts same day
% Expected: 1.5-2× speedup  
% Effort: 40-50 minutes
```

## Recommended Approach

1. **Immediate**: Vectorize Prob_Xt (4-6× speedup, ~30 min work)
2. **Next**: Pre-compute subexpressions in TreeNodes (1.5-2× speedup, ~40 min work)
3. **Later**: Approximation schemes for additional speedup

**Combined Potential: 6-12× speedup**, bringing 60K contracts down to **2-3 hours** instead of ~6-8 hours.

## Summary Table

| Optimization | Speedup | Effort | Impact |
|---|---|---|---|
| Vectorize Prob_Xt | 4-6× | 30 min | Very High |
| Pre-compute subexpr | 1.5-2× | 40 min | High |
| Cache by day | 1.5-2× | 50 min | Medium |
| Approximate moments | 5-10× | 2 hours | High but accuracy loss |
| GPU acceleration | 10-50× | 1-2 days | Very High but complex |
| **Combined (1-3)** | **~6-12×** | **2 hours** | **Very High** |

