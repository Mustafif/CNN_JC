# 4. Generate Monte-Carlo paths for GARCH model
num_paths = ... # Number of paths for Monte Carlo simulation
num_point = N+1 # Number of points in each path

Z = np.random.randn(num_point+1, num_paths)
Z1 = np.random.randn(num_point, num_paths)
ht = np.full((num_point+1, num_paths), np.nan)
ht[0, :] = h0 * np.ones(num_paths)
Xt = np.full((num_point + 1, num_paths), np.nan)
Xt[0, :] = np.log(S_0) * np.ones(num_paths)

for i in range(1, num_point):
  ht[i,:] = omega + alpha * (Z[i-1,:] - gamma * np.sqrt(ht[i-1,:]))**2 + beta * ht[i-1,:]
  Xt[i,:] = Xt[i-1,:] + (r - 0.5 * ht[i,:]) + np.sqrt(ht[i,:]) * Z[i,:]

ht[num_point,:] = omega + alpha * (Z[num_point-1,:] - gamma * 
                  ... np.sqrt(ht[num_point-1,:]))**2 + beta * ht[num_point-1,:]
S = np.exp(Xt)