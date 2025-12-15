# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a research project implementing a **Calibration Neural Network for Joint Calibration** of GARCH models using both historical returns data and option prices. The goal is to calibrate GARCH(1,1) and Heston-Nandi GARCH parameters by combining physical measure (P) and risk-neutral measure (Q) data through a joint calibration objective function.

## Core Architecture

### Three-Tier System

1. **Data Generation Layer** (`data_gen/`, MATLAB)
   - Willow tree-based option pricing for Heston-Nandi GARCH models
   - Generates synthetic training/validation datasets with market data and GARCH parameters
   - Primary script: `data_gen/newgen.m` - scalable dataset generator with normal and stressed market conditions
   - Produces CSV files with columns: `S0`, `moneyness`, `r`, `T`, `call_put`, `alpha`, `beta`, `omega`, `gamma`, `lambda`, `impl_vol`, `option_value`

2. **Willow Tree Construction** (`willow_tree_constr/`, Python)
   - Implements willow tree numerical method for GARCH option pricing
   - Key modules:
     - `Treenodes_logSt_HN.py` - constructs log stock price tree nodes
     - `Treenodes_ht_HN.py` - constructs volatility tree nodes
     - `prob_Xt.py`, `prob_ht.py` - probability distributions for tree nodes
     - `zq.py` - generates discrete density distribution function `{z, q}`
   - Pricing algorithm converts GARCH stochastic processes into discrete tree structures

3. **Neural Network Calibration** (`ForwardANN/`, Python/PyTorch)
   - Two primary models:
     - **VolatilityNet**: Predicts implied volatility from option characteristics
     - **OptionPriceNet**: Directly predicts option prices
   - Main files:
     - `option_neural_calibration.py` - neural calibration system
     - `garch_calibration.py` - traditional GARCH calibration
     - `advanced_training.py`, `simple_train.py` - training scripts
   - Uses sophisticated loss functions: Combined Loss, Adaptive Loss, Focal Loss

### Data Flow

```
MATLAB (data_gen/) → CSV datasets → Python (ForwardANN/) → Trained NN → Calibrated GARCH params
                                  ↘ Python (willow_tree_constr/) for validation
```

### Key Parameters

**GARCH Parameters** (target outputs):
- `omega` (ω): Long-run variance
- `alpha` (α): ARCH coefficient (volatility innovation)
- `beta` (β): GARCH coefficient (persistence)
- `gamma` (γ): Asymmetry parameter (Heston-Nandi)
- `lambda` (λ): Risk premium parameter

**Model Constraints**:
- Stationarity: `beta + alpha * (gamma + lambda + 0.5)^2 < 1`
- All parameters must be non-negative

## Common Development Commands

### Python Environment Setup

```bash
# Install dependencies using Poetry
poetry install

# Activate virtual environment
poetry shell
```

### Data Generation (MATLAB)

```bash
# Generate training/validation datasets
# Open MATLAB and run:
cd data_gen
newgen  # Main scalable dataset generator

# Alternative generators for specific GARCH variants:
# - demo.m: Standard HN-GARCH demo
# - GJR_Garch/demo.m: GJR-GARCH variant
# - Duan_Garch/demo.m: Duan GARCH variant
```

### Neural Network Training

```bash
# Train volatility prediction model
cd ForwardANN
python run_neural_calibration.py --target volatility --epochs 1000

# Train price prediction model
python run_neural_calibration.py --target price --epochs 1000

# Train both models
python run_neural_calibration.py --target both --epochs 1000 --gpu

# Run hyperparameter study
python run_neural_calibration.py --target volatility --hyperparameter-study

# Simple training script (more straightforward)
python simple_train.py
```

### Traditional GARCH Calibration

```bash
cd ForwardANN
python run_garch_calibration.py
```

### Testing

```bash
# Test neural calibration
cd ForwardANN
python test_neural_calibration.py

# Test specific modules
cd willow_tree_constr
python -m pytest tests/test_zq.py

# Basic data generation test
cd tests
python test_gen_data.py
```

Note: This project does not use a standard test runner. Testing is done through individual test scripts located in `tests/` and `ForwardANN/test_*.py`.

## Important Implementation Details

### MATLAB Data Generation

**Scaling Parameters** (in `data_gen/newgen.m`):
- `CONTRACTS_PER_DAY`: Number of option contracts per trading day (default: 50)
- `NUM_DAYS_TRAIN`: Trading days for training set (default: 30)
- `NUM_DAYS_VAL`: Trading days for validation set (default: 10)
- `NUM_PARAM_SETS_TRAIN`: Parameter sets for training (default: 5)
- `NUM_PARAM_SETS_VAL`: Parameter sets for validation with stress scenarios (default: 8+)

**Parameter Ranges**:
- Training: Normal market conditions
  - `alpha_range`: [1.15e-6, 1.36e-6]
  - `beta_range`: [0.75, 0.85]
  - `moneyness_range`: [0.85, 1.15]
  - `maturity_range`: [0.1, 1.0] years
- Validation: Stressed conditions (wider ranges, extreme scenarios)

**Willow Tree Parameters**:
- `m_h = 6`: Number of volatility nodes
- `m_ht = 6`: Number of volatility history nodes
- `m_x = 30`: Number of log-price nodes
- `gamma_h = 0.6`, `gamma_x = 0.8`: Distribution shape parameters

### Neural Network Architecture

**Input Features** (10 total):
- Market data: `S0`, `moneyness (m=K/S0)`, `r`, `T`, `corp` (call/put indicator)
- GARCH parameters: `alpha`, `beta`, `omega`, `gamma`, `lambda`

**VolatilityNet Architecture**:
```
Input(10) → Linear(128) → ReLU → BatchNorm → Dropout(0.15)
          → Linear(256) → ReLU → BatchNorm → Dropout(0.15)
          → Linear(128) → ReLU → BatchNorm → Dropout(0.15)
          → Linear(64)  → ReLU → BatchNorm → Dropout(0.15)
          → Linear(1)   → Softplus
```

**OptionPriceNet Architecture**:
```
Input(10) → Linear(256) → ReLU → BatchNorm → Dropout(0.2)
          → Linear(512) → ReLU → BatchNorm → Dropout(0.2)
          → Linear(256) → ReLU → BatchNorm → Dropout(0.2)
          → Linear(128) → ReLU → BatchNorm → Dropout(0.2)
          → Linear(1)   → Softplus
```

### Loss Function Components

**Combined Loss** (default for volatility):
```
L_total = 0.35*L_huber + 0.40*L_relative + 0.15*L_quantile + 0.10*L_constraint
```

Where:
- `L_huber`: Robust to outliers (δ=0.05)
- `L_relative`: Scale-invariant relative error
- `L_quantile`: Better tail behavior (α=0.15)
- `L_constraint`: Penalty for unrealistic values

### Known Issues and Solutions

**Implied Volatility Solver** (documented in `data_gen/solution_summary.md`):
- Issue: Significant IV spreads between calls/puts due to restrictive bisection bounds
- Solution: Expand bounds to [0.001, 3.0] instead of [0, 1.0]
- Use moneyness-based initial guesses instead of fixed 0.5
- See `solution_summary.md` for detailed fixes

**Put-Call Parity Violations**:
- Caused by numerical precision issues in willow tree discretization
- Increase tree resolution: `m_x = 50`, `m_h = 10` for better accuracy
- Validate with European options before American options

## File Organization

```
.
├── cnn_jc/                    # Core Python module (mostly prototype code)
│   ├── cnn.py                 # MLPRegressor-based neural network
│   └── garch.py               # GARCH data generation utilities
├── data_gen/                  # MATLAB data generation (155+ .m files)
│   ├── newgen.m              # Main scalable dataset generator ⭐
│   ├── demo.m                # Standard demo
│   ├── impvol.m              # Implied volatility solver
│   ├── GJR_Garch/            # GJR-GARCH variant
│   └── Duan_Garch/           # Duan GARCH variant
├── ForwardANN/               # Primary neural network implementation ⭐
│   ├── option_neural_calibration.py    # Main NN calibration system
│   ├── garch_calibration.py            # Traditional GARCH methods
│   ├── run_neural_calibration.py       # CLI interface
│   ├── test_neural_calibration.py      # Testing utilities
│   ├── advanced_training.py            # Advanced training loops
│   └── simple_train.py                 # Simplified training
├── willow_tree_constr/       # Willow tree Python implementation
│   ├── main.py               # Demo script
│   ├── Treenodes_logSt_HN.py # Log stock price tree
│   ├── prob_Xt.py            # Stock price probabilities
│   └── zq.py                 # Discrete density distribution
├── python/                   # Experimental Python scripts
├── thesis/                   # Research documentation
├── tests/                    # Basic tests
└── pyproject.toml           # Poetry dependencies
```

## Research Context

**Primary Goal**: Joint calibration of GARCH parameters using both:
- **Physical measure (P)**: Historical asset returns
- **Risk-neutral measure (Q)**: Option market prices

**Methodology**:
1. Generate synthetic option data with known GARCH parameters (MATLAB)
2. Train neural networks to learn inverse mapping: market observables → GARCH params
3. Validate against traditional calibration methods
4. Apply to real market data

**Risk Neutralization**:
- Converts physical GARCH to risk-neutral using Radon-Nikodym derivative
- For HN-GARCH: `R_t = r + λ*h_t + ε_t` under P
- Under Q: `R_t = r - 0.5*h_t + ε_t*`
- Risk-neutral innovations: `ε_t* = ε_t + λ*√h_t`

## Dependencies

**Python** (via Poetry):
- Core: `numpy`, `pandas`, `scipy`, `scikit-learn`
- Neural Networks: `torch` (PyTorch) - not in pyproject.toml, add separately
- GARCH: `arch`
- Visualization: `matplotlib`
- Utilities: `joblib`, `dill`

**MATLAB**:
- Financial Toolbox (for option pricing validation)
- Optimization Toolbox (for parameter calibration)
- Statistics and Machine Learning Toolbox

**System Requirements**:
- Python 3.10+
- MATLAB R2020a or later recommended
- GPU recommended for neural network training (optional)

## Working with This Codebase

### When Modifying Data Generation

1. Adjust scaling parameters in `data_gen/newgen.m`
2. Ensure parameter ranges satisfy stationarity constraints
3. Validate generated CSV format matches expected schema
4. Check for put-call parity violations in output
5. Monitor convergence rates of IV solver

### When Training Neural Networks

1. Verify input CSV has all required columns
2. Choose appropriate loss function for target (volatility vs price)
3. Monitor for overfitting with validation metrics
4. Save models with full state: `calibrator.save_model(path)`
5. Check predictions are physically reasonable (positive, bounded)

### When Debugging Willow Tree

1. Validate tree construction parameters (`m_h`, `m_x`, `N`)
2. Check probability distributions sum to 1
3. Ensure no NaN/Inf in tree nodes
4. Compare European option prices with analytical solutions
5. Verify risk-neutral parameter `c = gamma + lambda + 0.5` is reasonable (<10)

### Documentation Files

- `ForwardANN/GARCH_CALIBRATION_README.md`: Traditional GARCH calibration
- `ForwardANN/NEURAL_CALIBRATION_README.md`: Neural network calibration
- `ForwardANN/MODEL_IMPROVEMENTS.md`: Model enhancement notes
- `data_gen/solution_summary.md`: IV solver issues and fixes
- `summary.typ`, `datagen.typ`, `pseudo.typ`: Research documentation (Typst format)

## Code Style and Patterns

- MATLAB: Standard MATLAB conventions, extensive comments
- Python: Mix of functional and OOP, PyTorch conventions in ForwardANN
- Neural networks use separate classes for models, calibrators, and trainers
- Data preprocessing includes standardization/normalization before training
- Model checkpointing and early stopping are standard practices
