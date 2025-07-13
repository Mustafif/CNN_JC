import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("stage1b_impl_90days.csv")
# Compute strike price
df['strike'] = df['S0'] * df['m']
df['impl'] = df['V']  # Use V as implied volatility

# # Average over duplicate strikes (e.g., multiple corp values)
# df_mean = df.groupby('strike', as_index=False)['impl'].mean()

# # Sort by strike
# df_mean = df_mean.sort_values(by='strike')

options = df[df['corp'] == -1] # put options
call = df[df['corp'] == 1] # call options

# Plot volatility smile
plt.figure(figsize=(8, 5))
# plt.plot(df_mean['strike'], df_mean['impl'], marker='o')
plt.plot(options['strike'].sort_values(), options['impl'].sort_values(), marker='o')
# # plt.scatter(df['strike'], df['impl'], alpha=0.4)
# for corp_val in [-1, 1]:
#     subset = df[df['corp'] == corp_val]
#     plt.scatter(subset['strike'], subset['V'], label=f'corp={corp_val}', alpha=0.3)
# plt.legend()
plt.plot(call['strike'].sort_values(), call['impl'].sort_values(), marker='o')
plt.xlabel("Strike Price (K)")
plt.ylabel("Implied Volatility (V)")
plt.title("Volatility Smile")
plt.grid(True)
plt.tight_layout()
plt.show()
