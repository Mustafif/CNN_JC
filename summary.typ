#outline(
  title: "Outline", 
  depth: 4
)
#pagebreak()
#set page(footer: context [
  #h(1fr)
  #counter(page).display(
    "1/1",
    both: true,
  )
], paper: "us-letter")

#set par(leading: 0.55em, first-line-indent: 1.8em, justify: true)
#set text(font: "New Computer Modern")
#set heading(numbering: none)
#show par: set block(spacing: 0.55em)
#show heading: set block(above: 1.4em, below: 1em)


= A Neural Network Based Framework for Financial Models Summary 

== 4.3 Backward Pass (Heston Model)

  $sigma_"imp"$
  | $K$, $tau$, $S_0$, $r$ -> Heston-CaNN -> $rho$, $k$, $v_0$, #math.overline("v"), $gamma$

=== How does this relate to our research?

- We hope to be able to use historical asset prices and option prices to be able to calibrate our parameters. This mean, our input depends on the different parameters that make these prices/returns, which would be strike prices, initial price, rate, time to maturity and the implied volatility. 

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
#set par(first-line-indent: 0em)
#let arrow = text(size: 1.5em, "\u{2193}")

#align(center)[
  #stack(
    dir: ttb,
    spacing: 1em,
    align(center)[Market Data ($S, K, S_0, r, tau$, sigma | $R_t$)],
    arrow,
    align(center)[Joint Calibration Neural Network],
    arrow,
    align(center)[GARCH Params ($omega, alpha, beta, gamma, lambda$)]
  )
]
\

Where we consider the parameters under *P* measure:  
- $S$: Price (a time series of asset prices eg. (10-year daily prices))
- $K$: Strike Price
- $S_0$: Initial Price 
- $r$: Risk-free rate (fixed)
- $sigma$: Implied Volatility
- $tau$: Time to maturity

And we consider the following physical measure: 
- $R_t$: Log return at time $t$ (several GARCH models)


Our goal is to be able to use the Joint Calibration to have the most optimum calibrated GARCH parameters that takes into account both measures. This will come into the calibration phase, where the idea is to utilize the Joint Calibration formula as the objective function for minimization. 

=== Risk Neutralization for One-Component Gaussian Models

- $Q$: Risk-Neutral Measure 
- $P$: Physical Measure

Using the Radon-Nikodym derivative, we can convert the physical measure to the risk-neutral measure. 
Let $z_t$ i.i.d $N(0, 1)$, then $gamma_t = 1/2 h_t$ since $exp(gamma_t) = E_"t-1" [exp(epsilon_t)]$

The Radon-Nikodym derivative is defined as:

$
"dQ"/"dP" bar F_t = exp(-sum^t_"i=1" ((mu_i - r_i)/h_i epsilon_i + 1/2 ((mu_i -r_i)/h_i)^2 h_i))
$

==== NGARCH(1, 1)

For NGARCH(1, 1) using $epsilon_t^* = epsilon_t + mu_t - r_t$, the volatility process under $Q$ becomes: 

$
h_t = omega + beta h_"t-1" + alpha (epsilon^*_"t-1" - mu_"t-1" + r_"t-1" )^2  => epsilon^*_t bar F_t tilde N(0, h_t) \
R_t equiv ln (S_t/S_"t-1" ) = r_t - 1/2 h_t + epsilon^*_t => epsilon^*_t bar F_"t-1" tilde N(0, h_t)\
E^Q [S_t/S_"t-1" bar F_"t-1"] = exp(r_t)
$

==== Duan

The Physical GARCH Model Duan (1995) comes in the following form: 

$
R_t equiv ln (S_t/S_"t-1" ) = r_t + lambda sqrt(h_t) - 1/2 h_t + epsilon_t\
h_t = omega + beta h_"t-1" + alpha epsilon^2_"t-1"
$

Assume: 
- $lambda$: price of risk (const.)
- $mu_t = r_t + lambda sqrt(h_t)$ or $lambda = (mu_t - r_t)/sqrt(h_t)$


This corresponds to the following RN-Derivative: 

$
"dQ"/"dP" bar F_t = exp(-sum^t_"i=1" (epsilon_i/sqrt(h_i) lambda + 1/2 lambda^2))
$

With risk-neutral innovations: 
$
epsilon^*_t &= epsilon_t + mu_t - r_t \
&= epsilon_t+lambda sqrt(h_t)
$

The Risk-Neutral GARCH becomes: 

$
R_t equiv ln(S_t/S_"t-1") = r-1/2 h_t + epsilon_t^*\
h_t = omega + beta h_"t-1" + alpha (epsilon_"t-1"^* - lambda sqrt(h_"t-1"))^2
$

==== HN-GARCH(1, 1)

Starting with the following model of Heston and Nandi (2000): 

$
R_t equiv ln(S_t/S_"t-1") = r + lambda h_t + epsilon_t\
h_t = omega + beta h_"t-1" + alpha (z_"t-1" - c sqrt(h_"t-1"))^2
$

Assume $r_t = r, mu_t = r+lambda h_t + 0.5 h_t$

RN-Derivative: 

$
"dQ"/"dP" bar F_t = exp(-sum^t_"i=1" ((lambda + 1/2)epsilon_i + 1/2(lambda+1/2)^2 h_i))\
\
epsilon_t^* = epsilon_t + lambda h_t + 0.5h_t\
R_t equiv ln(S_t/S_"t-1") = r-1/2 h_t + epsilon_t^*\
h_t = omega + beta h_"t-1" + alpha (z^*_"t-1" - (c+lambda + 1/2)sqrt(h_"t-1"))^2
$


=== Input Data
So firstly we need to discuss how we get our input parameters? 

Risk-Neutral Measure: 
- American Option Prices

Physical Measure: 
- Historical Asset Prices (change in difference to get log return)
  - Where: $R_t equiv ln(S_t/S_"t-1") = mu_t - gamma_t + epsilon_t$
  - $mu_t$: Conditional mean of the returns at time $t$
  - Assume $gamma_t$ is defined from $exp(gamma_t) = E_"t-1" [exp(epsilon_t)]$
  - $epsilon_t$: the normal innovation at time $t$ where $epsilon_t|F_"t-1" tilde N(0, h_t)$
  - $h_t$: Conditional variance at time $t$
    - $h_t = omega + alpha epsilon_"t-1"^2 + beta h_"t-1"$


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

#pagebreak()

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
underbrace(e_"i,t" = (C_"i,t" - C_"i,t" (h_t (xi^*)))/"Vega"_" i,t", 
    "We may also choose to use relative error if Vega is hard or expensive to calculate"
)
$



where: 
- $"Vega"_"i,t"$ is the Black-Scholes sensitivity of the option prices with respect to volatility
- $xi^*$ is the vector of risk-neutral parameters to be estimated 
- $C_"i,t" - C_"i,t" (h_t (xi^*))$: The corresponding implied volatility from the option price. 

=== Joint Log Likelihood

$
L_"joint" = (T+N_T)/2 L^R/T + (T+N_T)/2 L^O/N
$
where: 
- $T$ is the number of days in the return sample
- $N_T$ is the total number of option contracts

#pagebreak()

#include "datagen.typ"