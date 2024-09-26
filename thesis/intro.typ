#import "styles.typ": *

= Introduction

The financial modelling world tends to look at American Options with a shy eye, to be able to be exercised at any day before the maturity presents
unique challeges compared to simpler styles like the European Options. When we take a look at timeseries way to forecast American options
pricing, we often either consider the _daily log returns_ of the underlying asset *or* _implicit volatility_ of the option prices.
So we have estimation taking into account only one measure, we are either looking at historical data of the underlying asset in the Physical measure
or looking forward with the implicit volatility which is in the Risk-Neutral measure.

// Example usage:
#definition(
  title: "Risk-Neutral Measure",
  [
  A probability measure such that the expectation of the returns for all assets is equivalent to the risk-free rate.
  Under this paper, we will denote this measure as $bb(Q)$.
  ]
)

#definition(
title: "Physical Measure",
[
  A probability measure that takes into account the real-world proabilities that reflect the actual likelihood
  of events which occur in the financial markets. Under this paper, we will denote this measure as $bb(P)$.
]
)

Our goal in this paper is to take into account both of these probability measures to better estimate American Options under
a GARCH model. To accomplish this, we will be using a technique called *Joint Calibration* which will be used as an objective
function for an Artificial Neural Network which will use both historical asset prices and implicit volatilities to calibrate GARCH model
parameters.

In particular, this paper will focus on two GARCH models, the _Heston-Nandi GARCH(1, 1)_ and the _Duan (1995)_ model, which will be further discussed later on.

== Motivation

// - Complexity that Joint MLE brings with GARCH model is very hard to do with traditional optimization methods
// - As your data increases, becomes more computationally expensive
// - The best way to handle this, is to use Artificial Neural Network to calibrate the GARCH parameters, and this can be built to scale


The primary motivation of this paper is to extend GARCH model estimation to encompass not only one, but both probability measures.
We seek to consider not only the risk-neutral perspective but also the real-world physical probabilities derived from historical asset data.
By incorporating both measures, we aim to enhance estimation using Joint Maximum Likelihood Estimation (MLE).

Where does a Calibration Artificial Neural Network (CaNN) fit into our goal? Why can't we stick with traditional optimization methods?
The issue arises from the complexity of the GARCH model and the Joint MLE, how do we use two measures, how do we handle the issue of non-convex optimization problems and local minima.
This calibration process becomes increasingly computationally expensive as the data size grows. How can we address these challenges?

Using Artificial Neural Networks to calibrate financial models has been done before and is still being explored to assess its effectiveness.
We also aim to evaluate its potential advantages. One advantage is its ability to use the Joint MLE as an objective function as we train the network on both
historical prices and implicit volatilies as input to get calibrated parameters as output. Additionally, Artificial
Neural Networks can scale to accommodate larger data samples as needed.

== Related Works
// - A list of related work that uses some ideas related to what we are doing
// - What we are contributing
// - Papers used for inspiration


#let related(body) = block(
  width: 100%,
  fill: color.rgb("d9b99b"),
  inset: 12pt,
  radius: 4pt,
  [
    #body
  ]
)

#related([
*"Estimating and using GARCH models with VIX data for option valuation"* \ by Juho Kanniainen, Binghuan Lin and Hanxue Yang.

- This paper utilizes the GARCH model and the Joint MLE to use information on VIX to improve the empirical performance of the models for pricing options on the S&P 500.
- VIX (Cboe Volatility Index) follows a European Exercise Style, our paper focuses on American Options which will be harder to price, and take into account the more complex exercise style.
])

#related([
*"FX Volatility Calibration Using Artificial Neural Networks"* \ by Alexander Winter, Kellogg College, University of Oxford

- This paper explores effectively learning model parameters using an ANN with the Heston stochastic volatility model, and how it can perform orders of magnitudes faster than traditional optimization-based methods.
- This paper is another great example of using ANN as an application to calibrate financial models, which we also hope to demonstrate in this paper, while this paper focuses on the Heston Model, and we focus on GARCH model, we hope to apply some of the ideas brought in this paper like data gathering to aid us.
])

#related(
[
*"A neural network-based framework for financial model calibration"* \ by Shuaiqiang Liu, Anastasia Borovykh, Lech A. Grzelak, and Cornelis W.Oosterlee

- This paper explores an approach to a CaNN (Calibration Neural Network) to calibrate financial asset pricing model parameters using an Artificial Neural Network.
- The paper addresses challenges in calibrating financial models, particularly the issue of non-convex optimization problems and local minima.
- In this paper, we were particularly interested in the backward pass approach to be the way we would go about calibrating our model to get the parameters.
]
)


== American Options

#definition(title: "American Option", [
  An American option is a financial derivative that gives the holder of the contract the right, but not the obligation, to buy or sell an underlying asset at a specified price at any time from the purchase to the expiration date.
])

#note(
  [
    After the contract expiration date, the options contract no longer exists and the options holder can neither sell the contract to another trader nor exercise it.
]
)

// what is an equity option
// different type of strike options (itm, atm,otm)
// The OCC
// Their specifications on monthly and weekly equity options

#definition(title: "Equity Option", [
An equity option (also known as stock option), gives an investor the right, but not an obligation to buy a call or sell a put
at a set strike price before the contract's expirations date. #cite(<nyse_options_equity>)
])

=== Types of Strike Options

1. *In-the-money (ITM)*: When a call option's strike price is below the stock price, or a put option's strike price is above the stock price, the investor gains from the difference between the strike and stock price.
2. *At-the-money (ATM)*: Both put and call options, the strike and stock prices are the same.
3. *Out-of-the-money (OTM)*: When a call option’s strike price is above the stock price or put option’s strike price is lower than the stock price. The investor is in a loss from the difference between the stock and strike price.

The strike prices for listed options are set by criteria established by the OCC.

#definition(title: "The Options Clearing Corporation (OCC)", [
A financial institution that facilitates the transactions between the exchanges of payments with financial derivatives. It currently operates under the jurisdiction
of both the SEC and CFTC, where specifically under the SEC jurisdiction clears transactions for put and call options products.#cite(<occ_defn>)
])

=== Equity Options Criteria Under the OCC #cite(<occ_eq_spec>)

Each equity option covers 100 shares of the underlying security, which for our purposes will be an individual shares of stock.

*Strike Price Intervals*:

Let $K$ represent the strike price.

The general criteria is as follows:
- \$2.50 increments for $K < \$25$
- \$5.00 increments for $\$25 <= K < \$200$
- \$10.00 increments for $K > \$200$

#note([
Some exchanges, such as Nasdaq also provide \$1 and \$0.50 intervals for lower priced stocks, those which tighter regulations, higher volatilites, or other reasons.
])

*Expiration Months*: Generally options contracts typically expire in the two nearest upcoming months and two additional months within the January, February, or March quarterly cycle.
However, some exchange programs provide the flexibility to list options with expiration months beyond this standard schedule.

*Expiration Dates*:
- Monthly options expire on the third Friday of the expiration month.
- When the expiration date falls on a holiday, the expiration date is on the Thursday before the third Friday.

=== Weekly Options Criteria Under the OCC #cite(<occ_weekly_spec>)

Weekly options are designed to be a short-term option contract which will follow the same standard specifications listed above.
The main difference comes to the expiration date being expired on the date listed on the contract, this could typically be the last trading
day of the week.

== GARCH Model

// we will do individual american stocks like AAPL to try and simulate
// we need to look at Nasdaq and figure out how the following work:
// - strike prices being set (moneyness)
// - expiration dates
// - amount of contracts








== Structure

#todo[

- This is where we will discuss why we will use two different GARCH models
- An initial look at a simple Artificial Neural Network
- Use some of the summary parts for this
- What American Option we will use, specifications
- Training idea, synthetic data could come here or a "Data Gather" section
]
