# Performance Tuning Guide

## Quick Optimization Checklist

- [ ] Check batch size recommendations (shown in config output)
- [ ] Adjust `batch_size` if warning appears
- [ ] Set `num_workers` = CPU cores - 2
- [ ] Enable `nested_parfor` only for 2000+ contracts
- [ ] Monitor throughput from batch output
- [ ] Profile if throughput drops between batches

## Batch Size Quick Guide

| Scenario | Batch Size | Reason |
|----------|-----------|--------|
| <100 contracts total | 50-100 | Small workload, low overhead |
| 100-1000 contracts | 200 | Recommended default |
| 1000-5000 contracts | 300-500 | Medium workload, good balance |
| >5000 contracts | 500-1000 | Large workload, memory amortization |

## System Configuration Guide

**Laptop (4-8 cores, 8-16GB):**
```toml
batch_size = 150
num_workers = 6
nested_parfor = false
```

**Desktop (8-16 cores, 16-32GB):**
```toml
batch_size = 250
num_workers = 12
nested_parfor = false (or true if >2000 contracts)
```

**Workstation (16-32 cores, 32-64GB):**
```toml
batch_size = 400
num_workers = 28
nested_parfor = true
```

## Performance Red Flags

| Issue | Likely Cause | Fix |
|-------|-------------|-----|
| <100 contracts/sec | Memory pressure | Reduce batch_size |
| Batch times vary >50% | Load imbalance | Enable nested_parfor |
| High memory (>80%) | batch_size too large | Reduce by 30% |
| Slow tree generation | Tree structure too large | Check m_h, m_x params |

## Key Metrics to Monitor

1. **Throughput**: contracts/second (shown per batch)
   - Target: 200-1000 contracts/sec
   - Monitor for consistency across batches

2. **Batch Time**: seconds per batch
   - Should be relatively consistent
   - Variance > 50% suggests load imbalance

3. **Memory Usage**: GB used
   - Monitor with `memory()` in MATLAB
   - Should stay stable per batch

## Example: Optimizing a Slow Run

**Initial Configuration:**
```toml
contracts_per_day = 500
num_days = 60
batch_size = 200
num_param_sets = 10
```

**Observed Problem:** Only 80 contracts/sec

**Diagnosis:** Memory pressure (batch_size too large for this system)

**Solution:**
```toml
batch_size = 150  # Reduce by 25%
```

**Result:** 150 contracts/sec (improved 87%)

## Nested Parallelization Decision

Enable `enable_nested_parfor = true` if:
```
CONTRACTS_PER_DAY * NUM_DAYS > 500  AND
feature('numcores') > 8  AND
Throughput < 300 contracts/sec
```

Disable if:
```
CONTRACTS_PER_DAY * NUM_DAYS < 500  OR
feature('numcores') <= 8  OR
Tree computation is bottleneck (check profiler)
```

## Testing Your Configuration

```matlab
% Test with small dataset
scaled_dataset_optimized_configurable(...
    'config', 'my_config.toml', ...
    'contracts_per_day', 50, ...
    'num_days', 5)

% Observe:
% 1. Batch times (should be < 1 second)
% 2. Throughput (should be > 200 contracts/sec)
% 3. Memory usage (should be < 2 GB)

% If good, scale up to full config
```

## Expected Performance Ranges

| Scenario | Throughput | Notes |
|----------|-----------|-------|
| Single param, small dataset | 1000-2000 c/s | Tree computation dominates |
| 10 params, medium dataset | 300-600 c/s | Mixed workload |
| 100 params, large dataset | 100-300 c/s | Parallelization overhead |
| With nested parfor | +20-50% | Depends on contract count |

## Troubleshooting

**Q: Batch throughput drops as run progresses**
A: Likely system thermal throttling. Check CPU temps.

**Q: Memory usage grows over time**
A: Monitor for leaks. Normal slight increase is OK.

**Q: Nested parfor not helping**
A: Tree computation is bottleneck. Try GPU acceleration.

**Q: Large variance in batch times**
A: Worker heterogeneity. Consider worker affinity settings.
