# Centralized Parameter Comparison System - File Index

## 📚 Documentation Files (Read These First!)

### Getting Started
- **`GETTING_STARTED.md`** ⭐ **START HERE!** Complete guide with examples
- **`COMPARE_QUICK_REFERENCE.txt`** Quick command reference card
- **`SYSTEM_OVERVIEW.txt`** Visual diagrams and workflow

### Detailed Documentation  
- **`COMPARE_README.md`** Full documentation and reference
- **`COMPARE_SYSTEM_SUMMARY.md`** Implementation details and features

---

## 🔧 Core System Files (Required)

### Main Functions
- **`compare_dataset.m`** Main function to generate comparison datasets
- **`rel_err_results.m`** Analysis function for relative errors

### Configuration
- **`compare_config.toml`** ⚙️ Main configuration file (edit this!)
- **`read_toml_config.py`** Python helper to parse TOML

---

## 📖 Examples & Templates

### Example Scripts
- **`example_compare_usage.m`** 5 usage examples with code
- **`test_compare_system.m`** System verification test

### Example Configs
- **`compare_config.toml`** Main config (HN model, no noise)
- **`compare_config_duan_example.toml`** Example (Duan model, 3% noise)

---

## 📁 Output Structure

When you run `compare_dataset('compare_config.toml')`, it creates:

```
compare/  (or your specified folder)
├── true_params.csv              # Dataset with true parameters
├── initial_params.csv           # Dataset with initial parameters
├── calibrated_params.csv        # Dataset with calibrated parameters
├── assetprices_true_params.csv  # Asset price paths
├── assetprices_initial_params.csv
├── assetprices_calibrated_params.csv
├── rel_err_results.csv          # Analysis results (after running rel_err_results)
└── rel_err_heatmaps.png         # Visualizations (if plot=true)
```

---

## 🚀 Quick Start Guide

### 1️⃣ First Time Setup

```bash
# Install Python TOML package
pip install toml

# Test the system
cd data_gen
matlab -r "test_compare_system"
```

### 2️⃣ Generate Datasets

```matlab
% In MATLAB, navigate to data_gen folder
cd data_gen

% Generate comparison datasets
compare_dataset('compare_config.toml')
```

### 3️⃣ Analyze Results

```matlab
% Basic analysis
rel_err_results()

% With plots
rel_err_results('plot', true)
```

---

## 📝 Common Tasks

### Task 1: Change Noise Level
```toml
# Edit compare_config.toml
[noise]
level = 0.02  # 2% noise
seed = 42
```
Then run: `compare_dataset('compare_config.toml')`

### Task 2: Switch Model
```toml
# Edit compare_config.toml
[model]
type = "Duan"  # or "HN" or "GJR"
```

### Task 3: Modify Parameters
```toml
# Edit compare_config.toml
[parameters.calibrated]
omega = 1.0e-6
alpha = 1.2e-6
beta = 0.8
gamma = 5.0
lambda = 0.2
```

### Task 4: Generate Larger Dataset
```toml
# Edit compare_config.toml
[scaling]
contracts_per_day = 100
num_days = 50
```

---

## 🔍 Which File Should I Read?

### If you want to...

**Get started quickly** → `GETTING_STARTED.md`

**See quick commands** → `COMPARE_QUICK_REFERENCE.txt`

**Understand the workflow** → `SYSTEM_OVERVIEW.txt`

**Learn all features** → `COMPARE_README.md`

**Understand implementation** → `COMPARE_SYSTEM_SUMMARY.md`

**See code examples** → `example_compare_usage.m`

**Test the system** → Run `test_compare_system.m`

**Configure parameters** → Edit `compare_config.toml`

---

## 🎯 Key Features

✅ **Centralized config** - All parameters in one TOML file  
✅ **Fair comparison** - Same randomness across all parameter sets  
✅ **Noise simulation** - Add realistic noise to test robustness  
✅ **Multi-model support** - HN, Duan, and GJR GARCH models  
✅ **Enhanced analysis** - Automatic filtering and categorization  
✅ **Memory efficient** - Batch processing for large datasets  
✅ **Well documented** - Multiple guides and examples  

---

## 🆘 Troubleshooting

### Problem: "Failed to read TOML"
**Solution:**
```bash
pip install toml
python3 read_toml_config.py compare_config.toml
```

### Problem: "Function not found"
**Solution:** Run `test_compare_system` to check dependencies

### Problem: Memory errors
**Solution:** Reduce `contracts_per_day` or increase `batch_size` in config

### Problem: Wrong results
**Solution:** Check GARCH parameters are valid (e.g., beta < 1)

---

## 📞 Need Help?

1. Read `GETTING_STARTED.md` for complete guide
2. Check `COMPARE_QUICK_REFERENCE.txt` for quick commands
3. Run `test_compare_system` to verify setup
4. Review `example_compare_usage.m` for code examples

---

## 🔄 Relationship to Other Files

This system is similar to `scaled_dataset.m` but specifically designed for **parameter comparison**.

- `scaled_dataset.m` - General dataset generation with varying parameters
- `compare_dataset.m` - Fair comparison of specific parameter sets (this system)

Both use the same underlying willow tree and pricing functions.

---

## 🎓 Learning Path

**Beginner:**
1. Read `GETTING_STARTED.md`
2. Run `test_compare_system`
3. Run example: `compare_dataset('compare_config.toml')`
4. Analyze: `rel_err_results()`

**Intermediate:**
5. Read `COMPARE_README.md`
6. Modify `compare_config.toml` (change noise level)
7. Run again and compare results
8. Try different models

**Advanced:**
9. Read `COMPARE_SYSTEM_SUMMARY.md`
10. Create multiple configs for noise study
11. Run systematic comparison studies
12. Integrate with your ML pipeline

---

## 📊 Example Workflow

```matlab
% Step 1: Test system
test_compare_system

% Step 2: Run default config (no noise)
compare_dataset('compare_config.toml')
rel_err_results('plot', true)

% Step 3: Edit config to add 2% noise
% [Edit compare_config.toml: set noise.level = 0.02]

% Step 4: Run with noise
compare_dataset('compare_config.toml')
rel_err_results('plot', true)

% Step 5: Compare results!
```

---

**Ready to start? Open `GETTING_STARTED.md`!** 🚀
