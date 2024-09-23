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

== Motivation

#todo[
  Why didnt traditional optimization work?

  - Willow Tree not being efficient, very slow with this idea
  - What exactly were the problems with the initial idea of this
  - What Artificial Neural Networks comes to try and solve, with the advantages it brings
]

#pagebreak()
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
*"Estimating and using GARCH models with VIX data for option valuation"* by Juho Kanniainen, Binghuan Lin and Hanxue Yang.

- This paper utilizes the GARCH model and the Joint MLE to use information on VIX to improve the empirical performance of the models for pricing options on the S&P 500.
- VIX (Cboe Volatility Index) follows a European Exercise Style, our paper focuses on American Options which will be harder to price, and take into account the more complex exercise style.
])

#related([
*"FX Volatility Calibration Using Artificial Neural Networks"* by Alexander Winter, Kellogg College, University of Oxford

- This paper explores effectively learning model parameters using an ANN with the Heston stochastic volatility model, and how it can perform orders of magnitudes faster than traditional optimization-based methods.
- This paper is another great example of using ANN as an application to calibrate financial models, which we also hope to demonstrate in this paper, while this paper focuses on the Heston Model, and we focus on GARCH model, we hope to apply some of the ideas brought in this paper like data gathering to aid us.

])


== Structure

#todo[

- This is where we will discuss why we will use two different GARCH models
- An initial look at a simple Artificial Neural Network
- Use some of the summary parts for this
- What American Option we will use, specifications
- Training idea, synthetic data could come here or a "Data Gather" section
]
