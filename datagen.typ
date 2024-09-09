== Training Data Generation

#align(center)[
// #rect(stroke: color.red)[
//   *TODO: Add American option pricing in Python*\
//   Requires:
//   - Heston-Nandi GARCH Model
//   - Monte-Carlo Simulation
//   - Understanding of American Option Pricing
// ]
#rect(fill: color.yellow)[
  *Current Idea*

1. Given a set of parameters for GARCH (in Physical measure)
2. Given the initial asset price $S_0$, use Monte Carlo method to simulate a path of asset prices, \
$S_1, S_2, ... S_N$, with say $N=500$ (Under *P* measure)
3. Select last 30-50 days on the path, for each day, use the selected asset price (under *Q*) as the initial price to generate American option prices with various strike prices (11-17) and maturities (7 days to 1 year). *Pay attention to the transformation from the physical measure to the risk-neutral measure*.
]]

#align(center)[
*Pseudo Code*
]

#let include_code_file(file_path, lang) = {
    raw(read(file_path), lang: lang)
}


=== The Steps that we are following: 

1. Initialize Option and Monte Carlo Parameters 
  - `r`: The risk-free rate 
  - `S0`: The initial asset price 
  - `h0`: Initial volatility 
  - `N`: Number of time steps for simulation 
  - `M`: Number of Monte Carlo paths 

2. Initialize HN-GARCH parameters under P measure 
  - $theta = (alpha, beta, omega, gamma, lambda)$

3. Simulate paths using Monte Carlo Simulation 

4. Risk Neutralize HN-GARCH parameters
  - $theta^* = (alpha_Q, rho, omega_Q, gamma_Q, lambda_Q)$

5. Initialize Willow Tree parameters

6. Generate data for the days up to the maturity

#line(length: 100%)

=== Project Structure: 

*MATLAB Files:*


#align(center)[
  #rect(fill: blue)[
#table(
  columns: 3,
  rows: 6,
  align: center, 
  stroke: none,
  
    [`American.m`],
    [`gen_PoWiner.m`],
    [`nodes_Winer.m`],
    [`Prob_Xt.m`],
    [`zq.m`],
    [`datagen.m`],
  
  
    [`impVol_HN.m`],
    [`impvol.m`],
    [`probcali.m`],
    [`main.m`],
    [`Prob_ht.m`],
    [`sign.m`],
  
  
    [`genhDelta.m`],
    [`Prob.m`],
    [`TreeNodes_ht_HN.m`],
    [`Treenodes_JC_h.m`],
    [`Treenodes_JC_X.m`],
    [`TreeNodes_logSt_HN.m`],
  
)
]]



*Dependencies:* `f_hhh.mexa64`

*Output Files:* `annual.csv`, `half.csv`, `quarter.csv`, `week.csv`

#pagebreak()

=== The `datagen()` Function 

#rect(stroke: orange)[
#include_code_file("data_gen/datagen.m", "matlab")
]

#pagebreak()

=== The `main` Code 

This will contain generating data for different maturities and configuring the 
parameters for both the Option and HN-GARCH. 

#rect(stroke: orange)[
#include_code_file("data_gen/main.m", "matlab")
]


*Downloading the Code*: Available under *#underline(link("https://github.com/Mustafif/CNN_JC/releases/download/alpha.1/data_gen.zip", "Github"))*


#pagebreak()

== Generated Data

#let csv_data(path) = align(center)[#table(columns: 3, fill: (rgb("EAF2F5"), none), ..csv(path).flatten())]

=== Week

#csv_data("data_gen/week.csv")

=== 3 Months 

#csv_data("data_gen/quarter.csv")

=== 6 Months 

#csv_data("data_gen/half.csv")

=== One Year

#csv_data("data_gen/annual.csv")



// *Link to Code*

// #rect(fill:color.yellow)[
//   Once Python code is able to produce similar results to Matlab, it will have an initial Github release,
//   with this link. It will be ZIP file under the name `JC_WT_DataGen`
// ]

// #show link: underline

// #link("https://github.com/Mustafif/CNN_JC/releases/tag/alpha.1")[
//   Alpha 1 Release
// ]
