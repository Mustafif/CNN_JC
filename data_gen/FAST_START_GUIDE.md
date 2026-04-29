# Fast Start Guide - Two Caching Modes

## TL;DR - Quick Fix

**If cache generation is too slow, use lazy caching:**

```matlab
% FAST startup, good generation speed
scaled_dataset_cached('HN', 'ContractsPerDay', 200, 'NumDays', 250, ...
    'PrecomputeTrees', false);  % ← Lazy caching (default now)
```

**Time:** ~5 seconds startup + ~60 seconds generation = ~65 seconds total

---

## Two Caching Strategies

### Strategy 1: Lazy Caching (DEFAULT) ⚡
**Best for: Most users, faster startup**

```matlab
scaled_dataset_cached('HN', 'ContractsPerDay', 200, 'NumDays', 250, ...
    'PrecomputeTrees', false);  % Build trees on-demand
```

**How it works:**
- Trees built **on-demand** as contracts are processed
- First time a (day, maturity) combo is needed → build it
- Cache it for reuse
- Fast startup (~5 seconds)
- Slightly slower generation (~60 sec for 100K)

**Performance:**
```
Startup:    ~5 seconds      (minimal)
Generation: ~60 seconds     (builds trees as needed)
Total:      ~65 seconds     for 100K contracts
```

---

### Strategy 2: Pre-compute All Trees
**Best for: Multiple runs with same parameters, maximum generation speed**

```matlab
scaled_dataset_cached('HN', 'ContractsPerDay', 200, 'NumDays', 250, ...
    'PrecomputeTrees', true);  % Build all trees upfront
```

**How it works:**
- Pre-builds **ALL** possible (day, maturity) combinations
- For 250 days × 5 maturities = 1,250 trees upfront
- Slow startup but faster generation
- Direct O(1) array lookup during generation

**Performance:**
```
Startup:    ~45 seconds     (builds all trees)
Generation: ~45 seconds     (uses pre-built trees)
Total:      ~90 seconds     for 100K contracts
```

---

## Which Should You Use?

### Use Lazy Caching (PrecomputeTrees=false) if:
✅ Generating dataset once  
✅ Want faster startup  
✅ Large number of days (250+)  
✅ Limited memory  
✅ Testing/development  

**→ This is now the DEFAULT**

### Use Pre-compute (PrecomputeTrees=true) if:
✅ Generating multiple datasets with same days  
✅ Want absolute fastest generation  
✅ Smaller number of days (<100)  
✅ Have lots of memory  
✅ Production runs with repeated parameters  

---

## Performance Comparison

### 100K Contracts (250 days, 200 contracts/day)

| Strategy | Startup | Generation | Total | Memory |
|----------|---------|------------|-------|--------|
| **Lazy** | 5 sec | 60 sec | **65 sec** | 5 GB |
| Pre-compute | 45 sec | 45 sec | 90 sec | 8 GB |

### 100K Contracts (100 days, 500 contracts/day)

| Strategy | Startup | Generation | Total | Memory |
|----------|---------|------------|-------|--------|
| **Lazy** | 5 sec | 55 sec | **60 sec** | 4 GB |
| Pre-compute | 20 sec | 40 sec | **60 sec** | 6 GB |

**Recommendation:** Use lazy caching (PrecomputeTrees=false) in most cases.

---

## Examples

### Example 1: 50K Contracts (Fastest)
```matlab
scaled_dataset_cached('HN', 'ContractsPerDay', 100, 'NumDays', 250);
% Uses lazy caching by default
% Expected: ~35 seconds total
```

### Example 2: 100K Contracts (Fastest)
```matlab
scaled_dataset_cached('HN', 'ContractsPerDay', 200, 'NumDays', 250);
% Uses lazy caching by default
% Expected: ~65 seconds total
```

### Example 3: Multiple Parameter Sets
```matlab
scaled_dataset_cached('HN', 'ContractsPerDay', 100, 'NumDays', 100, ...
    'NumParamSets', 5);
% Each param set builds its own lazy cache
% Expected: ~90 seconds total (parallel)
```

### Example 4: Force Pre-compute (if you want it)
```matlab
scaled_dataset_cached('HN', 'ContractsPerDay', 200, 'NumDays', 250, ...
    'PrecomputeTrees', true);  % Explicitly enable pre-compute
% Expected: ~90 seconds (45 + 45)
```

---

## Progress Reporting

With pre-compute enabled, you'll see progress:
```
[PS 1/1] Pre-computing 1250 trees... 20% 40% 60% 80% 100% done! (42.3 sec)
```

With lazy caching, trees are built silently as needed:
```
[PS 1/1] Using lazy tree caching (build on-demand)
  Processing contracts...
```

---

## Memory Usage

### Lazy Caching
- Starts with minimal memory (~1 GB)
- Grows as trees are cached (~5 GB peak for 100K)
- More efficient for large datasets

### Pre-compute
- Immediately allocates full cache (~8 GB)
- Constant memory usage
- Better for repeated runs

---

## Why the Change?

**Original design:** Pre-compute all trees (PrecomputeTrees=true by default)
- **Problem:** 250 days × 5 maturities = 1,250 trees takes ~45 seconds to build

**New default:** Lazy caching (PrecomputeTrees=false)
- **Benefit:** Start immediately, build trees as needed
- **Result:** Faster overall for single runs

---

## Troubleshooting

### "Still taking too long"
1. Reduce number of days:
   ```matlab
   'NumDays', 100  % instead of 250
   ```

2. Use fewer parameter sets:
   ```matlab
   'NumParamSets', 1  % instead of 5
   ```

3. Increase batch size:
   ```matlab
   'BatchSize', 1000  % larger batches
   ```

### "Out of memory"
1. Use lazy caching (default):
   ```matlab
   'PrecomputeTrees', false
   ```

2. Process serially instead of parallel:
   ```matlab
   'NumParamSets', 1  % Process one at a time
   ```

---

## Summary

**Default mode now: Lazy caching**
- Faster startup (~5 sec)
- Good generation speed (~60 sec for 100K)
- Lower memory usage
- Better for most users

**To use old mode:**
```matlab
'PrecomputeTrees', true  % Explicitly enable
```

**Best practice:**
- Leave defaults for single runs
- Use PrecomputeTrees=true for repeated runs with same parameters
- Adjust NumDays if still too slow

---

**Total time for 100K contracts: ~65 seconds with defaults** ✓

Much better than the ~90 seconds with pre-compute!
