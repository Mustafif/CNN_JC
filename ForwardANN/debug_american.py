import numpy as np
from zq import zq
from gen_PoWiner import gen_powiener as gen_PoWiner
from nodes_Winer import nodes_wiener as nodes_Winer
from American import American

# Test parameters
S0 = 100
K = np.array([90, 100, 110])  # Strike prices
T = 1.0                       # Maturity
r = 0.05                      # Risk-free rate
sigma = 0.25                  # Volatility
N = 50                        # Number of time steps
m = 20                        # Number of nodes per time step
gamma = 3.0                   # Sampling parameter for z

# Generate the tree
z, _, _, _ = zq(m, gamma)
P, q = gen_PoWiner(T, N, z)
Xnodes = nodes_Winer(T, N, z, r, sigma)
nodes = S0 * np.exp(Xnodes)

# Print some nodes for inspection
print("Sample stock price nodes at terminal time:")
print(nodes[:5, -1])  # First 5 nodes at maturity

# Price both call and put options for each strike
print("\n===== CALL OPTIONS =====")
call_prices = []
for strike in K:
    price, option_values = American(nodes, P, q, r, T, S0, strike, 1)  # 1 for call
    call_prices.append(price)
    print(f"Call option with K={strike}: price={price:.6f}")
    print(f"Immediate exercise at t=0: {max(S0-strike, 0):.6f}")
    print(f"Sample terminal payoffs: {np.maximum(nodes[:5, -1] - strike, 0)}")
    print(f"Sample option values at t=0: {option_values[:5, 0]}")
    print()

print("\n===== PUT OPTIONS =====")
put_prices = []
for strike in K:
    price, option_values = American(nodes, P, q, r, T, S0, strike, -1)  # -1 for put
    put_prices.append(price)
    print(f"Put option with K={strike}: price={price:.6f}")
    print(f"Immediate exercise at t=0: {max(strike-S0, 0):.6f}")
    print(f"Sample terminal payoffs: {np.maximum(strike - nodes[:5, -1], 0)}")
    print(f"Sample option values at t=0: {option_values[:5, 0]}")
    print()

print("\nSummary:")
print("Call prices:", call_prices)
print("Put prices:", put_prices)

# European Black-Scholes prices for comparison (if available, assuming BS formula is defined elsewhere)
try:
    from black_scholes import bs_price
    print("\nEuropean BS prices for comparison:")
    for strike in K:
        c = bs_price(S0, strike, T, r, sigma, 'c')
        p = bs_price(S0, strike, T, r, sigma, 'p')
        print(f"K={strike}: Call={c:.6f}, Put={p:.6f}")
except ImportError:
    print("\nBlack-Scholes module not available for comparison")