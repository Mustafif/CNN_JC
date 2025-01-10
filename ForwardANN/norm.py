import pandas as pd
import numpy as np

def normalize_option_data(file_path):
    """
    Normalize option pricing data using finance-specific methods.

    Parameters:
    file_path (str): Path to the CSV file

    Returns:
    tuple: (original_df, normalized_df)
    """
    # Read CSV file with specified dtypes
    dtype_dict = {
        'S0': np.float64,
        'K': np.float64,
        'r': np.float64,
        'T': np.float64,
        'corp': np.int64,
        'alpha': np.float64,
        'beta': np.float64,
        'omega': np.float64,
        'gamma': np.float64,
        'lambda': np.float64,
        'V': np.float64
    }
    df = pd.read_csv(file_path, dtype=dtype_dict)

    # Print column data types
    print("DataFrame column dtypes:")
    print(df.dtypes)
    print("\n")
    normalized_df = df.copy()

    # 1. Normalize option prices (V) by the spot price (S0)
    normalized_df['V'] = df['V'] / df['S0']

    # 2. Express strike prices as moneyness (K/S0)
    normalized_df['K'] = df['K'] / df['S0']

    # 3. Keep time to maturity (T) as is - it's already in a reasonable scale

    # 4. Keep interest rate (r) as is - it's already in percentage form

    # 5. Keep corp as is - it's a binary indicator

    # 6. For the other parameters (alpha, beta, omega, gamma, lambda)
    # Only normalize if they vary significantly
    for param in ['alpha', 'beta', 'omega', 'gamma', 'lambda']:
        if df[param].std() > 0:
            # Use log transformation for very small values
            if df[param].mean() < 1e-4:
                normalized_df[param] = np.log10(df[param] + 1e-20)
            else:
                # Keep as is if already in reasonable range
                normalized_df[param] = df[param]

    # 7. Normalize S0 to 1 since we've used it as denominator
    normalized_df['S0'] = 1.0

    return df, normalized_df

def display_option_statistics(original_df, normalized_df):
    """
    Display statistics relevant for option pricing data
    """
    stats = pd.DataFrame()

    # Calculate moneyness categories
    original_df['moneyness'] = original_df['K'] / original_df['S0']

    print("Option Price Statistics by Moneyness:")
    moneyness_stats = original_df.groupby(
        pd.cut(original_df['moneyness'],
               bins=[0, 0.95, 1.05, float('inf')],
               labels=['ITM', 'ATM', 'OTM'])
    )['V'].describe()

    print("\nNormalized Price Statistics by Moneyness:")
    norm_moneyness_stats = normalized_df.groupby(
        pd.cut(normalized_df['K'],
               bins=[0, 0.95, 1.05, float('inf')],
               labels=['ITM', 'ATM', 'OTM'])
    )['V'].describe()

    return moneyness_stats, norm_moneyness_stats

if __name__ == "__main__":
    file_path = 'test_dataset.csv'
    original_df, normalized_df = normalize_option_data(file_path)
    moneyness_stats, norm_moneyness_stats = display_option_statistics(original_df, normalized_df)

    # Save normalized data
    normalized_df.to_csv('normalized_option_data.csv', index=False)
