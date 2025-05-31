import numpy as np
from zq import zq
from gen_PoWiner import gen_powiener as gen_PoWiner
from nodes_Winer import nodes_wiener as nodes_Winer
from American import American
import matplotlib.pyplot as plt

def impvol(S0, K, T, r, V, index, N, m, gamma, tol=1e-5, itmax=100):
    """
    Compute implied volatilities of American options via Willow Tree and bisection.

    Parameters:
        S0 : float - Initial stock price
        K  : ndarray - Strike price matrix (k x n)
        T  : ndarray - Maturities (length n)
        r  : float - Risk-free interest rate
        V  : ndarray - Observed American option prices (k x n)
        index : ndarray - +1 for call, -1 for put (k x n)
        N  : int - Number of time steps in the tree
        m  : int - Number of nodes per time step
        gamma : float - Sampling parameter for z
        tol : float - Tolerance for bisection
        itmax : int - Maximum iterations

    Returns:
        imp_vol : ndarray - Implied volatilities (k x n)
    """
    n = len(T)
    k = K.shape[0]
    z, _, _, _ = zq(m, gamma)
    imp_vol = np.zeros((k, n))

    for i in range(n):  # for each maturity
        P, q = gen_PoWiner(T[i], N, z)

        for j in range(k):
            V0 = 1e10
            it = 0
            a, b = 0.0, 1.0
            sigma = 0.5 * (a + b)

            while abs(V0 - V[j, i]) > tol and it < itmax:
                Xnodes = nodes_Winer(T[i], N, z, r, sigma)
                nodes = S0 * np.exp(Xnodes)
                V0, _ = American(nodes, P, q, r, T[i], S0, K[j], index[j, i])

                if V0 > V[j, i]:
                    b = sigma
                else:
                    a = sigma

                sigma = 0.5 * (a + b)
                it += 1

            imp_vol[j, i] = sigma

    return imp_vol

# ==== TEST CASE ====
S0 = 100
K = np.array([[90], [100], [110]])  # Strike matrix (3x1)
T = np.array([1.0])                 # One maturity
r = 0.05
index = np.full((3, 1), -1)         # All Put options
true_sigma = 0.25
N = 50
m = 20
gamma = 3.0

# Generate true option prices
z, _, _, _ = zq(m, gamma)
print("z values:")
print(z)
P, q = gen_PoWiner(T[0], N, z)
print("First few rows and columns of P[0]:")
print(P[0, 0:3, 0:3])
print("q values:")
print(q)
Xnodes = nodes_Winer(T[0], N, z, r, true_sigma)
print("Sample of Xnodes (first few rows and columns):")
print(Xnodes[0:3, 0:3])
nodes = S0 * np.exp(Xnodes)
print("Sample of stock price nodes (first few rows and columns):")
print(nodes[0:3, 0:3])

V = np.zeros((3, 1))
for i in range(3):
    # Debug the option pricing calculation
    print(f"\nCalculating for strike K={K[i][0]}, option type={index[i][0]} (1=Call, -1=Put)")
    # Check immediate exercise value at t=0 for this option
    if index[i][0] == 1:  # Call
        immediate_exercise = max(S0 - K[i][0], 0)
    else:  # Put
        immediate_exercise = max(K[i][0] - S0, 0)
    print(f"Immediate exercise value at t=0: {immediate_exercise}")
    
    # Calculate and print terminal payoffs for a few nodes
    if index[i][0] == 1:  # Call
        terminal_payoffs = np.maximum(nodes[:, -1] - K[i][0], 0)
    else:  # Put
        terminal_payoffs = np.maximum(K[i][0] - nodes[:, -1], 0)
    print(f"Sample terminal payoffs (first 3 nodes): {terminal_payoffs[0:3]}")
    
    V[i, :], option_values = American(nodes, P, q, r, T[0], S0, K[i][0], index[i][0])
    print(f"Option price for K={K[i]}: {V[i, 0]}")
    print(f"Option values at t=0 nodes (first 3): {option_values[0:3, 0]}")

print("\nFinal option prices V:")
print(V)

# Run implied vol solver
implied_vols = impvol(S0, K, T, r, V, index, N, m, gamma)
print("Implied volatilities:")
print(implied_vols)

# Test case to verify the correct behavior of the code
print("\nAdjusting option values manually to make 2nd and 3rd options have value 0:")
V_manual = np.array([[3.98493162], [0.0], [0.0]])
print("Manual option values V_manual:")
print(V_manual)

# Run implied vol solver with manual values
print("\nRunning implied vol solver with manual option values:")
implied_vols_manual = impvol(S0, K, T, r, V_manual, index, N, m, gamma)
print("Implied volatilities with manual option values:")
print(implied_vols_manual)

# Plot the volatility smile
plt.figure(figsize=(10, 6))
plt.plot(K.flatten(), implied_vols.flatten(), 'bo-', label='Original')
plt.plot(K.flatten(), implied_vols_manual.flatten(), 'ro-', label='With V=[3.98, 0, 0]')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility Smile')
plt.legend()
plt.grid(True)
plt.savefig('implied_volatility_test.png')
print("\nVolatility smile plot saved to 'implied_volatility_test.png'")

# Check with extreme low volatility
print("\nExtreme low volatility test (should make non-ITM options close to zero):")
extreme_sigma = 0.001  # Extremely low volatility
X_extreme = nodes_Winer(T[0], N, z, r, extreme_sigma)
nodes_extreme = S0 * np.exp(X_extreme)

# Price with extremely low vol
V_extreme = np.zeros((3, 1))
for i in range(3):
    V_extreme[i, :], _ = American(nodes_extreme, P, q, r, T[0], S0, K[i][0], index[i][0])

print("Option prices with extreme low volatility:")
print(V_extreme)
