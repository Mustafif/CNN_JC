# Discrete Maturity Days Update

## Date: 2026-03-31

## Change Summary

Updated the maturity specification from a continuous range to **discrete maturity days** to match real-world option markets.

---

## What Changed

### Before ❌
```matlab
'MaturityRange', [0.1, 1.0]  % Continuous range in years
% Generated random values: 0.15, 0.37, 0.82, etc. years
```

### After ✅
```matlab
'MaturityDays', [30, 60, 90, 180, 360]  % Discrete days
% Randomly selects from: 30, 60, 90, 180, or 360 days
% Converts to years: T = days / 252 (trading days)
```

---

## Why This Matters

### Real Option Markets
Standard option contracts have **discrete maturities**:
- **Weekly**: 7 days
- **Bi-weekly**: 14 days  
- **Monthly**: 30 days
- **Quarterly**: 90 days
- **Semi-annual**: 180 days
- **Annual**: 360 days

### Previous Issue
Random continuous maturities like 0.237 years (≈ 59.7 days) are:
- **Unrealistic** - no such contracts trade
- **Inconsistent** - hard to compare across datasets
- **Non-standard** - doesn't match market conventions

### New Behavior
Each contract now has a maturity from the discrete set:
- **Realistic** - matches actual option products
- **Consistent** - same maturities across runs
- **Standard** - aligns with market practices

---

## Usage

### Default Maturities
```matlab
scaled_dataset('HN')
% Uses default: [30, 60, 90, 180, 360] days
```

### Custom Maturities
```matlab
% Short-term only
scaled_dataset('Duan', 'MaturityDays', [7, 14, 30])

% Include weeklies and monthlies
scaled_dataset('GJR', 'MaturityDays', [7, 14, 30, 60, 90])

% Long-term options
scaled_dataset('HN', 'MaturityDays', [180, 360, 540, 720])
```

---

## Implementation Details

### Code Changes

**Contract Generation:**
```matlab
% OLD: Continuous random maturity
maturity_years = maturity_range(1) + (maturity_range(2) - maturity_range(1)) * rand();
T_maturity = maturity_years;

% NEW: Discrete maturity selection
maturity_idx = randi(length(maturity_days));
T_days = maturity_days(maturity_idx);
T_maturity = T_days / 252;  % Convert to years (252 trading days)
```

### Data Cleaning
The cleaning filter `T != 1/252` now makes more sense:
- `1/252 ≈ 0.00397` years = **1 day**
- Ensures no single-day options (which are edge cases)
- With discrete maturities `[30, 60, 90, ...]`, this is automatically satisfied

---

## Output Changes

### Console Output
```
SCALING CONFIGURATION:
  Contracts per day: 50
  Number of days: 20
  ...
  Maturity days: [30 60 90 180 360]  ← NEW!
```

### Metadata File
```
CONFIGURATION:
  ...
  Maturity days: [30 60 90 180 360]  ← NEW!
```

### Dataset
- **T column**: Still in years (e.g., 0.119, 0.238, 0.357, ...)
- **Values**: Now correspond to exact day counts
  - `30/252 = 0.119` years
  - `60/252 = 0.238` years
  - `90/252 = 0.357` years
  - `180/252 = 0.714` years
  - `360/252 = 1.429` years

---

## Examples

### Example 1: Standard Options
```matlab
scaled_dataset('HN', ...
    'ContractsPerDay', 100, ...
    'NumDays', 30, ...
    'MaturityDays', [30, 60, 90, 180, 360])

% Generates 6,000 contracts (100 × 30 × 2)
% Each contract has maturity: 30, 60, 90, 180, or 360 days
% Distribution: ~1,200 contracts per maturity
```

### Example 2: Weekly Options
```matlab
scaled_dataset('Duan', ...
    'ContractsPerDay', 50, ...
    'NumDays', 20, ...
    'MaturityDays', [7, 14])

% Generates 2,000 contracts
% All contracts are either 7-day or 14-day weeklies
```

### Example 3: Research Dataset
```matlab
scaled_dataset('GJR', ...
    'ContractsPerDay', 200, ...
    'NumDays', 50, ...
    'MaturityDays', [30, 60, 90])

% Generates 20,000 contracts
% Only near-term options (1-3 months)
% Good for studying short-term volatility
```

---

## Backward Compatibility

### Breaking Change ⚠️
The parameter name changed:
- **OLD**: `'MaturityRange', [0.1, 1.0]`
- **NEW**: `'MaturityDays', [30, 60, 90, 180, 360]`

### Migration
```matlab
% OLD CODE (will error)
scaled_dataset('HN', 'MaturityRange', [0.1, 1.0])

% NEW CODE
scaled_dataset('HN', 'MaturityDays', [30, 60, 90, 180, 360])

% Or just use defaults
scaled_dataset('HN')  % Uses [30, 60, 90, 180, 360] automatically
```

---

## Benefits

✅ **Realistic**: Matches real option markets  
✅ **Consistent**: Same maturities every time  
✅ **Flexible**: Easy to customize for research needs  
✅ **Clear**: Days are more intuitive than decimal years  
✅ **Standard**: Aligns with market conventions  

---

## Files Modified

1. **`scaled_dataset.m`**
   - Changed parameter from `MaturityRange` to `MaturityDays`
   - Updated contract generation to select discrete maturities
   - Updated display and metadata output

2. **`DATASET_GENERATOR_README.md`**
   - Updated parameter table
   - Updated examples

3. **`QUICK_REFERENCE.txt`**
   - Updated parameter descriptions
   - Updated examples

4. **`MATURITY_DAYS_UPDATE.md`** (NEW)
   - Complete documentation of this change

---

## Testing

```matlab
% Test with defaults
scaled_dataset('HN', 'ContractsPerDay', 10, 'NumDays', 5)

% Test with custom maturities
scaled_dataset('Duan', ...
    'ContractsPerDay', 10, ...
    'NumDays', 5, ...
    'MaturityDays', [7, 30, 90])

% Check metadata.txt to verify discrete maturities
% Check dataset CSV - T column should have exact values from days/252
```

---

## Summary

Maturity specification is now **discrete and realistic**, matching how options actually trade in markets. This makes generated datasets more suitable for training models that will be deployed in real-world scenarios.

