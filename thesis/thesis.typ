#import "@preview/ilm:1.2.1": *
#import "styles.typ": *

#let abstract = [
  *Abstract Coming Soon!*
]

#show: ilm.with(
  title: [Joint Calibration Thesis],
  author: "Mustafif Khan",
  date: datetime.today(),
  abstract: abstract,
  bibliography: bibliography("refs.bib"),
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true)
)

#pagebreak()

#include "intro.typ"


#pagebreak()

= Estimation with GARCH Models

In our paper we will be taking a look at two different GARCH models and how they are represented
under both $bb(P)$ and $bb(Q)$ measures. The two models we will take a closer look is the Heston-Nandi
GARCH(1, 1) model and the Duan (1995) model.

== GARCH Processes

Before we are able to look, we need the general picture of the GARCH process where under the physical measure $bb(P)$,
we assume the underlying stock price process follows a conditional distribution $D$ where we can express the
log returns $R_t$ at time period $t$ as:

$
R_t equiv ln(S_t/S_(t-1)) &= mu_t - gamma_t + epsilon_t #h(3.3cm) epsilon_t | F_(t-1) tilde D(0, h_t) \
      &= mu_t - gamma_t + sqrt(h_t)z_t #h(2.5cm) z_t | F_(t-1) tilde D(0,1)
$


$epsilon_t = sqrt(h_t)z_t$

- $S_t$: Stock price at time $t$
- $h_t$: Conditional variance of the log return in period $t$

== Parameter Definitions

- $omega$: Long term average variance constant
- $alpha$: Coefficient for lagged innovation
- $beta$: Coefficient for lagged variance
- $gamma$: The Asymmetry coefficient
- $lambda$: Price of risk or risk premium
- $r$: Risk-free rate or discount rate

Relationships:

- $alpha$ and $beta$ ensures model's stationarity ($alpha + beta < 1$)
- $gamma > 0$ indicates negative shocks have a larger impact on future volatility.

== Risk Neutralization

- $bb(P)$: Physical Measure
- $bb(Q)$: Risk-Neutral Measure

Using the Radon-Nikodym derivative, we can convert the physical measure to the risk-neutral measure.
Let $z_t$ i.i.d $N(0, 1)$, then $gamma_t = 1/2 h_t$ since $exp(gamma_t) = E_(t-1) [exp(epsilon_t)]$

The Radon-Nikodym derivative is defined as:

$
"dQ"/"dP" bar F_t = exp(-sum^t_(i=1) ((mu_i - r_i)/h_i epsilon_i + 1/2 ((mu_i -r_i)/h_i)^2 h_i))
$ <RN-GARCH>

We also get under the RN-Derivative:

Defined by $epsilon_t | F_(t-1) tilde N(-(mu_t - r_t), h_t)$

$
ln(S_t/S_(t-1)) = r_t - 1/2 h_t + epsilon^*_t
$

with $epsilon^*_t | F_(t-1) tilde N(0, h_t)$ and $E^Q [S_t/S_(t-1) | F_(t-1)] = exp(r_t)$


=== Heston-Nandi GARCH
$bb(P)$: \
$
R_t &equiv ln(S_t/S_(t-1)) = r + lambda h_t + epsilon_t \
h_t &= omega + beta h_(t-1) + alpha (epsilon_(t-1) - c sqrt(h_(t-1)))^2
$ <HNP>

Assume:
- $r_t = r$
- $mu_t = r + (lambda + 1/2) h_t$

To risk neutralize @HNP we subsitute it along with the assumptions stated into @RN-GARCH to get the
corresponding RN-Derivative for the Heston-Nandi Model:

$
"dQ"/"dP" bar F_t = exp(-sum^t_(i=1) ((lambda + 1/2)epsilon_i + 1/2(lambda+1/2)^2 h_i))
$

Risk-neutral innovations of the form:

$epsilon^*_t = epsilon_t + lambda h_t + 1/2 h_t$

$bb(Q)$:\
$
R_t equiv ln(S_t/S_(t-1)) = r - 1/2 h_t + epsilon_t^* \
h_t = omega + beta h_(t-1) + alpha (z^*_(t-1) - (gamma+lambda + 1/2)sqrt(h_(t-1)))^2
$ <HNQ>

Using $z^*_t tilde^Q N(0, 1)$, $epsilon_t^* = sqrt(h_t) z_t^*$ and $rho^* = gamma + lambda + 1/2$ into
@HNQ we get @HNQ_NEW:

$
R_t = (r- 1/2 h_t) + sqrt(h_t)z_t^* \
h_t = omega + beta h_(t-1) + alpha (z_(t-1)^* - rho^*sqrt(h_(t-1)))^2
$ <HNQ_NEW>

#pagebreak()
=== Duan (1995)

$bb(P)$:

$
R_t equiv ln (S_t/S_(t-1) ) = r_t + lambda sqrt(h_t) - 1/2 h_t + epsilon_t\
h_t = omega + beta h_(t-1) + alpha epsilon^2_(t-1)
$ <DuanP>

Assume:

- Price of risk $lambda$ is assumed to be constant
- $r_t = r$
- $mu_t = r + lambda sqrt(h_t)$ or $lambda = (mu_t - r) / sqrt(h_t)$

To risk neutralize @DuanP we subsitute it along with the assumptions stated into @RN-GARCH to get the
corresponding RN-Derivative for the Duan (1995) Model:

$
"dQ"/"dP" bar F_t = exp(-sum^t_(i=1) (epsilon_i/sqrt(h_i) lambda + 1/2 lambda^2))
$

Risk-neutral innovations of the form:

$epsilon^*_t &= epsilon_t + mu_t - r_t = epsilon_t+lambda sqrt(h_t)$

$bb(Q)$:

$
R_t equiv ln(S_t/S_(t-1)) = r - 1/2 h_t + epsilon_t^*\
h_t = omega + beta h_(t-1) + alpha (epsilon_(t-1)^* - lambda sqrt(h_(t-1)))^2
$

Let $epsilon_t^* = z_t^* sqrt(h_t)$ with $z_t^* tilde^Q N(0, 1)$:

$
R_t = r - 1/2 h_t + z^*_t sqrt(h_t)\
h_t = omega + beta h_(t-1) + alpha(sqrt(h_(t-1))(z^*_(t-1)-lambda))
$

#pagebreak()

== Log Likelihoods

=== Return Log Likelihood

The log likelihood of the return process is calculated under the $bb(P)$ measure.

Let $Y_1$ represent the returns log likelihood, $h_i$ be the daily returns, and $N$ be the number of days in the returns sample, we can then compute it as:

$
Y_1 = -1/2 sum^N_(i=1) { ln(h_i) + (R_i - mu_i + gamma)^2 / h_i }
$ <LLR>

From the GARCH Process, we can notice that $R_i - mu_i + gamma$ in the formula is equivalent to the following:

$
R_t &= mu_t - gamma + sqrt(h_t)z_t \
sqrt(h_t)z_t &= R_t - mu_t + gamma
$

Substituting the above into @LLR we get the following:

$
Y_1 = -1/2 sum^N_(i=1) { ln(h_i) + z_i^2 } #h(2cm) z_i tilde D(0, 1)
$<LLR_NEW>

#note(
[
This derivation with the formula with @LLR_NEW, allows us to calculate the
Log Likelihood agnostic towards whichever GARCH model we'd like to use, whether Heston-Nandi or
Duan.
]
)

=== Options Log Likelihood

The log likelihood of the options process is calculated under the $bb(Q)$ measure.

Assume
$
underbrace(sigma_(i,t), "Imp vol. from market price") = underbrace(sigma_(i, t) (C_(i, t) (h_t (xi^*))), "Imp vol. from GARCH model") + epsilon_(i, t)
$ where
$epsilon_(i, t) tilde N(0, sigma_epsilon^2)$


Let $Y_2$ represent the options log likelihood and $M$ be the number of options, we can then compute it as:

$
Y_2 = -1/2 sum^M_(i=1) { 2ln(sigma_epsilon) + (overbrace(sigma_(i), "Imp vol. on market price") - overbrace(sigma_("imp", i), "Impl vol. from models"))^2 / sigma_epsilon^2 }
$

=== Joint Log Likelihood

$
Y_"joint" = (N+M)/(2N) Y_1 + (N+M)/(2M) Y_2
$

In our Calibration Artificial Neural Network, we will use $-Y_"joint"$ as our objective function to to solve for the parameters $theta = (omega, alpha, beta, gamma, lambda, sigma_epsilon)$, with the non-negativity constraint on the parameters in  $theta$.


= Conclusion
