import numpy as np
from arch import arch_model
from scipy.stats import norm


import numpy as np
from arch import arch_model

def generate_garch_data(n_samples, mean, std_dev, alpha, beta, omega):
    """
    Generate GARCH(1,1) data
    :param n_samples: The number of samples to generate
    :param mean: The mean of the data
    :param std_dev: The standard deviation of the data
    :param alpha: GARCH alpha parameter
    :param beta: GARCH beta parameter
    :param omega: GARCH omega parameter
    :return: returns, volatility
    """
    # Generate some random data
    np.random.seed(0)
    returns = np.random.normal(mean, std_dev, n_samples)

    # Fit a GARCH(1, 1) model to the data with specified parameters
    am = arch_model(returns, vol='GARCH', p=1, q=1, o=0, dist='normal', mean='Zero', lags=0, rescale=False)
    am.volatility.parch = [alpha, beta, omega]
    res = am.fit(disp='off')

    # Generate the volatility
    volatility = res.conditional_volatility

    return returns, volatility
