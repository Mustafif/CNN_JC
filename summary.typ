#set heading(outlined: true)
= A Neural Network Based Framework for Financial Models Summary 

== 4.3 Backward Pass (Heston Model)

  $sigma_"imp"$
  | $K$, $tau$, $S_0$, $r$ -> Heston-CaNN -> $rho$, $k$, $v_0$, #math.overline("v"), $gamma$

=== How does this relate to our research?

- We hope to be able to use historical asset prices and option prices to be able to calibrate our parameters. This mean, our input depends on the different parameters that make these prices/returns, which would be strike prices, initial price, rate, time to maturity and the implicit volatility. 

- We then send these input to the CaNN where it would train the model

- After we train the model, we can use our Joint objective function to return the calibrated parameters, where in our case, since we are just caring about the GARCH(1, 1) model would be $omega, alpha, beta$. 


=== Sampling Training Data 

Found in Table 5 of the paper: 

#table(
  columns: (auto, auto, auto, auto),
  inset: 10pt,
  align: horizon,
  table.header(
    [], [*Parameters*], [*Range*], [*Samples*]
  ),
  "Market data", 
  [
    Moneyness, $m=S_0/K$ \
    Time to maturity, $tau$ \
    Risk free rate, $r$ \
    European call/put price, $V/K$
  ], 
  [
    $[0.85, 1.15]$\
    $[0.5, 2.0]$(year)\
    $0.03$\
    $(0.0, 0.6)$
  ], 
  [
    $5$\
    $7$\
    Fixed, 
    -
  ], 
  "Black-Scholes", 
  "Implied Volatility", 
  $(0.2, 0.5)$, 
  $35$
)


#rect[
  We can use this as a way to sample all our different input parameters, and be able to measure how 
  more accurate the model becomes and its calibration. 
]

During the calibration, they use the total squared error measure $J(Theta)$: 
$
  J(Theta) = sum omega(sigma_"imp"^"ANN"-sigma^*_"imp")^2  + #math.overline($lambda$) #math.abs($Theta$) $

=== Averaged performance of the Backward pass of the CaNN: 

- Need to list CPU and GPU spec
  - OS: Linux pop-os 6.6.10-76060610-generic
  - CPU: AMD Ryzen 5 5600G
  - GPU: Radeon 6600


Abosulute deviation from $theta^*$, Error measure and computational cost. 

Error Measure: $J(Theta)$, MJ and Data Points\
Computational Cost: CPU, GPU time and Function evaluations

#pagebreak()

= Summary of Joint Calibrarion Artificial Neural Networks 

== Goals & Designs

Our goal is to be able to use two forms of input data to be able to calibrate an optimum parameters of the GARCH(1, 1) model. This will be calibrated using joint calibration which will take into account the log likelihood of returns and option prices to be able to consider both physical and risk neutral measures. 

The design will make use of a backward pass artificial neural network, where we will do the following: 

Market Data ($S, K, S_0, r, tau$, sigma | $R_t$) → Joint Calibration Neural Network → GARCH Params ($omega, alpha, beta$)

Where we consider the risk neutral parameters:  
- $S$: Price 
- $K$: Strike Price
- $S_0$: Initial Price 
- $r$: Risk-free rate (fixed)
- $sigma$: Volatility
- $tau$: Time to maturity

And we consider the following physical measure: 
- $R_t$: Log return at time $t$


Our goal is to be able to use the Joint Calibration to have the most optimum calibrated GARCH parameters that takes into account both measures. This will come into the calibration phase, where the idea is to utilize the Joint Calibration formula as the objective function for minimization. 


=== Input Data
So firstly we need to discuss how we get our input parameters? 

Risk-Neutral Measure: 
- Option Prices (European, American, Asian, etc.)

Physical Measure: 
- Historical Asset Prices (change in difference to get log return)
  - Where: $R_t equiv ln(S_t/S_"t-1")$
  - Can be easily done using `numpy.diff(numpy.log(S))` 


=== Joint Calibration Neural Network

This neural network will be written in Python with a minimum required version of 3.10, and will require the 
following dependencies: 

```toml
[tool.poetry.dependencies]
python = "^3.10"
numpy = "^1.26.4"
pandas = "^2.2.2"
arch = "^7.0.0"
scikit-learn = "^1.5.0"
matplotlib = "^3.9.0"
scipy = "^1.13.1"
```

The architecture currently will follow simarly to the paper, as the following: 
#align(center)[
#table(
  columns: (auto, auto),
  inset: 10pt,
  table.header(
    [*Parameters*], [*Options*]
  ),
  "Hidden Layers", $4$, 
  [
    Neurons(each layer)\
    Activation\
    Dropout rate\
    Batch-normalization\
    Initialization\
    Optimizer\
    Batch size\
  ], 
  [
    $200$\
    ReLu\
    $0.0$\
    No\
    Glorot_uniform\
    Adam\
    $1024$
  ]
)
]
#linebreak()
#image("ann.png")

Where we then hope in the regression phase, to be able to use the Joint Calibration formula in a batching method to be able to calibrate our parameters for the GARCH model. 

== Essential Equations

These are from GARCH Option Valuation Theory and Evidence: 

=== Return Log Likelihood: 

$
ln L^R ∝ -1/2 sum^T_"t=1" {ln(h(t)) + (R(t) - mu_t - gamma_t)^2/h(t)}
$

where: 
- $h(t)$ is the conditional variance 
- $R(t)$ represents the return at time $t$
- $mu_t$ is the conditional mean of the returns at time $t$
- $gamma_t$ is an adjustment term 

=== Options Log Likelihood: 

$
ln L^O ∝ -1/2 sum^N_"i=1" {ln("IVRMSE"^2) + (e_"i,t"/"IVRMSE")^2}
$

Implied Volatility root mean squared error ($"IVRMSE"$) loss function follows: 
$
"IVRMSE" approx sqrt(1/N_T sum_"i, t"^N_T e_"i,t"^2)
$

where: 
- $N_T$: total number of option prices in the sample


The verga weighted option error follows as: 
$
e_"i,t" = (C_"i,t" - C_"i,t" (h_t (xi^*)))/"Vega"_" i,t"
$

where: 
- $"Vega"_"i,t"$ is the Black-Scholes sensitivity of the option prices with respect to volatility
- $xi^*$ is the vector of risk-neutral parameters to be estimated 
- $C_"i,t"$ is the market option price
- $C_"i,t" (h_t (xi^*))$ is the model price

=== Joint Log Likelihood

$
L_"joint" = (T+N_T)/2 L^R/T + (T+N_T)/2 L^O/N
$
where: 
- $T$ is the number of days in the return sample
- $N_T$ is the total number of option contracts

== Creating Synthetic Data For European Options and Returns

```python 
import numpy as np
from scipy.stats import norm

# Parameters for GBM
S0 = 100    # Initial stock price
r = 0.1    # risk free rate
sigma = 0.2 # Volatility (annual)
T = 1.0     # Time period in years
dt = 1/252  # Time step (daily)
n_steps = int(T / dt)

# Generate random Brownian motion
np.random.seed(42)
W = np.random.standard_normal(size=n_steps)
W = np.cumsum(W) * np.sqrt(dt)

# Simulate stock prices
t = np.linspace(0, T, n_steps)
S = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * W)

# Calculate returns
returns = (S[1:] - S[:-1]) / S[:-1]
log_returns = np.diff(np.log(S))
```

#rect[The goal is to be able to train off of American options using the Willow Tree method to generate synthetic data. ]
