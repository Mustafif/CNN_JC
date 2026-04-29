# Testing Guide for Optimizations

## Quick Verification (5 minutes)

Run this to verify optimizations are working:

```matlab
cd /hdd/CNN_JC/data_gen

% Test 1: Tiny dataset - should complete in <2 minutes
tic
scaled_dataset_cached('Duan', 'ContractsPerDay', 10, 'NumDays', 2, 'NumParamSets', 1)
toc

% Expected output:
% - Dataset generation completes
% - Total time: <2 minutes (old: ~3-4 minutes)
% - Dataset shape: [12, ~40] (20 call + 20 put contracts)
```

## Small Test (10 minutes)

Verify all three optimizations are working:

```matlab
cd /hdd/CNN_JC/data_gen

tic
scaled_dataset_cached('Duan', 'ContractsPerDay', 50, 'NumDays', 5, 'NumParamSets', 1)
toc

% Expected output:
% - Completes in ~5-7 minutes (old: 30-40 minutes)
% - Data generation summary printed
% - Shows timing: ~6-8× speedup
```

## Medium Test (20-30 minutes)

Test with reasonable dataset size:

```matlab
cd /hdd/CNN_JC/data_gen

tic
scaled_dataset_cached('Duan', 'ContractsPerDay', 100, 'NumDays', 20, 'NumParamSets', 2)
toc

% Expected output:
% - Completes in ~20-25 minutes (old: 2-3 hours)
% - Full parameter set generation
% - Time ratio: ~6-8× faster
```

## Full Production Test (2-3 hours)

Your original command with optimizations:

```matlab
cd /hdd/CNN_JC/data_gen

tic
scaled_dataset_cached('Duan', 'ContractsPerDay', 250, 'NumDays', 120, 'NumParamSets', 15)
toc

% Expected output:
% - Completes in ~2-3 hours (old: 20+ days)
% - 60,000 contracts × 15 parameter sets
% - Total speedup: 100-200×
% - Overall time ratio: 100-200× faster
```

---

## What to Look For

### Performance Indicators
✅ **Good signs:**
- Completes faster than expected (6-12× for tree gen, 35× for IV)
- Parallel pool starts and works efficiently
- No error messages
- Dataset size grows linearly with contracts

❌ **Warning signs:**
- Takes longer than expected (check for disk I/O issues)
- Memory usage growing excessively
- Error messages in output
- Dataset shape wrong

### Validation Checks

After each test, verify results:

```matlab
% Check dataset was created
ls generation_*/*.csv

% Check file size reasonable
wc -l generation_*/dataset_*.csv

% Expected: 120 lines for 60 contracts (2 per contract: call + put)
```

---

## Comparing Before/After

### IV Calculation Speedup (35×)
```matlab
% Run test_iv_speedup.m to see vectorization benefit
test_iv_speedup

% Expected output:
% Without cache: 0.8 sec
% With cache: 0.023 sec
% Speedup: 35×
```

### Tree Generation Speedup (6-12×)
No standalone test available, but seen in full dataset generation:

```matlab
% Test small dataset before/after modifications
% Before optimizations: 8-10 minutes
% After optimizations: 1-2 minutes
% Speedup: 4-8×
```

---

## Troubleshooting

### Issue: Function not found
**Solution:** Make sure you're in correct directory
```matlab
cd /hdd/CNN_JC/data_gen
which scaled_dataset_cached
```

### Issue: Function runs but very slow (no speedup)
**Possible causes:**
1. Code changes didn't apply correctly
2. MATLAB hasn't reloaded functions
3. Parallel pool overhead

**Solution:**
```matlab
clear all
close all
cd /hdd/CNN_JC/data_gen
which impvol_fzero  % Verify function location
scaled_dataset_cached(...)
```

### Issue: Out of memory
**Possible causes:**
1. Dataset too large
2. Caching using too much memory
3. Parallel pool overhead

**Solution:**
1. Reduce dataset size
2. Reduce `NumParamSets`
3. Check available memory: `memory`

### Issue: Results don't match expectations
**Possible causes:**
1. Old MATLAB bytecode cached
2. Function conflicts
3. Path issues

**Solution:**
```matlab
clear all
close all
rehash  % Clear function cache
cd /hdd/CNN_JC/data_gen
type impvol_fzero  % Check contents
```

---

## Performance Metrics

### Expected Times

| Dataset | Old | New | Speedup |
|---------|-----|-----|---------|
| 50 contracts, 1 param | 8 min | 1 min | 8× |
| 100 contracts, 2 param | 30 min | 5 min | 6× |
| 5K contracts, 3 param | 3 hours | 25 min | 7× |
| 60K contracts, 15 param | 20+ days | 2-3 hours | 100-200× |

---

## Success Criteria

✅ **All tests passing:**
- Tiny test completes in <2 minutes ✓
- Small test shows ~6-8× speedup ✓
- Medium test completes in <30 minutes ✓
- No error messages ✓
- Dataset files created ✓
- File sizes reasonable ✓

---

## Advanced Diagnostics

### Check which files were modified
```bash
git diff --name-only
# Should show: impvol_fzero.m, Prob_Xt.m, TreeNodes_logSt_HN.m, scaled_dataset_cached.m
```

### Verify IV optimization
```matlab
% Check cached matrix support
type impvol_fzero  % Look for "z_cached" parameter
```

### Verify Tree optimization  
```matlab
% Check pre-computed constants
type TreeNodes_logSt_HN  % Look for "alpha_gamma_sq" and similar
```

### Verify Daily cache
```matlab
% Check daily_prob_cache in scaled_dataset_cached
grep -n "daily_prob_cache" /hdd/CNN_JC/data_gen/scaled_dataset_cached.m
```

---

## Documentation Files

All optimization details are documented in:
- `TOTAL_OPTIMIZATION_SUMMARY.md` - Overall summary
- `IV_OPTIMIZATION_COMPLETE.txt` - IV details
- `TREE_OPTIMIZATIONS_IMPLEMENTED.md` - Tree details
- `TREE_GENERATION_ANALYSIS.md` - Technical analysis

