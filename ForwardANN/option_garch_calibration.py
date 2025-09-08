import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import RBFInterpolator, griddata
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import warnings
from datetime import datetime, timedelta

@dataclass
class OptionData:
    """Container for option market data"""
    S0: float  # Current stock price
    K: np.ndarray  # Strike prices
    T: np.ndarray  # Time to expiration
    r: float  # Risk-free rate
    option_prices: np.ndarray  # Market option prices
    option_types: np.ndarray  # 1 for calls, -1 for puts
    implied_vols: Optional[np.ndarray] = None  # Market implied volatilities

@dataclass
class VolatilitySurface:
    """Container for volatility surface"""
    strikes: np.ndarray
    maturities: np.ndarray
    volatilities: np.ndarray  # 2D array: volatilities[i,j] = vol(strike[i], maturity[j])
    moneyness: Optional[np.ndarray] = None

class BlackScholesModel:
    """Black-Scholes option pricing model"""

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes call option price"""
        if T <= 0:
            return max(S - K, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes put option price"""
        if T <= 0:
            return max(K - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price

    @staticmethod
    def option_price(S: float, K: float, T: float, r: float, sigma: float,
                    option_type: int) -> float:
        """Calculate option price (1 for call, -1 for put)"""
        if option_type == 1:
            return BlackScholesModel.call_price(S, K, T, r, sigma)
        else:
            return BlackScholesModel.put_price(S, K, T, r, sigma)

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega (sensitivity to volatility)"""
        if T <= 0:
            return 0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        return vega

    @staticmethod
    def implied_volatility(S: float, K: float, T: float, r: float,
                          market_price: float, option_type: int,
                          max_iter: int = 100, tol: float = 1e-6) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        if T <= 0:
            return 0

        # Initial guess
        sigma = 0.2

        for _ in range(max_iter):
            price = BlackScholesModel.option_price(S, K, T, r, sigma, option_type)
            vega = BlackScholesModel.vega(S, K, T, r, sigma)

            if abs(vega) < tol:
                break

            price_diff = price - market_price

            if abs(price_diff) < tol:
                break

            sigma = sigma - price_diff / vega
            sigma = max(0.001, min(5.0, sigma))  # Bound sigma

        return sigma

class GARCHVolatilitySurface:
    """GARCH-based volatility surface model"""

    def __init__(self, base_garch_params: Dict[str, float],
                 surface_params: Optional[Dict[str, float]] = None):
        """
        Initialize GARCH volatility surface

        Parameters:
        -----------
        base_garch_params : dict
            Base GARCH parameters {'omega', 'alpha', 'beta', 'gamma'}
        surface_params : dict, optional
            Surface parametrization {'kappa_K', 'kappa_T', 'rho'}
        """
        self.garch_params = base_garch_params
        self.surface_params = surface_params or {
            'kappa_K': 0.1,  # Strike sensitivity
            'kappa_T': 0.05, # Maturity sensitivity
            'rho': -0.5      # Leverage correlation
        }

    def _base_volatility(self, T: float) -> float:
        """Calculate base volatility using GARCH long-run variance"""
        omega = self.garch_params['omega']
        alpha = self.garch_params['alpha']
        beta = self.garch_params['beta']

        # Long-run variance
        long_run_var = omega / (1 - alpha - beta)
        base_vol = np.sqrt(long_run_var)

        # Term structure effect
        vol_T = base_vol * (1 + self.surface_params['kappa_T'] * np.sqrt(T))

        return vol_T

    def _strike_adjustment(self, moneyness: float, T: float) -> float:
        """Calculate strike-dependent volatility adjustment"""
        kappa_K = self.surface_params['kappa_K']

        # Smile effect: higher volatility for OTM options
        smile_effect = kappa_K * (moneyness - 1)**2

        # Leverage effect: higher volatility for low strikes (puts)
        leverage_effect = self.surface_params['rho'] * np.log(moneyness) * np.sqrt(T)

        return smile_effect + leverage_effect

    def volatility(self, S: float, K: Union[float, np.ndarray],
                  T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate volatility for given strike(s) and maturity(ies)"""
        K = np.asarray(K)
        T = np.asarray(T)

        # Calculate moneyness
        moneyness = K / S

        # Base volatility from GARCH
        base_vol = self._base_volatility(T)

        # Strike adjustment
        strike_adj = self._strike_adjustment(moneyness, T)

        # Total volatility
        total_vol = base_vol * np.exp(strike_adj)

        return total_vol

    def generate_surface(self, S: float, strikes: np.ndarray,
                        maturities: np.ndarray) -> VolatilitySurface:
        """Generate complete volatility surface"""
        K_grid, T_grid = np.meshgrid(strikes, maturities)

        vol_surface = np.zeros_like(K_grid)

        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                vol_surface[i, j] = self.volatility(S, K, T)

        moneyness_grid = K_grid / S

        return VolatilitySurface(
            strikes=strikes,
            maturities=maturities,
            volatilities=vol_surface,
            moneyness=moneyness_grid
        )

class OptionGARCHCalibrator:
    """Calibrate GARCH models to option market data"""

    def __init__(self, option_data: OptionData):
        self.option_data = option_data
        self.bs_model = BlackScholesModel()
        self.calibrated_params = None
        self.fitted_surface = None

        # Calculate implied volatilities if not provided
        if self.option_data.implied_vols is None:
            self._calculate_implied_volatilities()

    def _calculate_implied_volatilities(self):
        """Calculate implied volatilities from market prices"""
        n_options = len(self.option_data.option_prices)
        implied_vols = np.zeros(n_options)

        for i in range(n_options):
            try:
                iv = self.bs_model.implied_volatility(
                    self.option_data.S0,
                    self.option_data.K[i],
                    self.option_data.T[i],
                    self.option_data.r,
                    self.option_data.option_prices[i],
                    self.option_data.option_types[i]
                )
                implied_vols[i] = iv
            except:
                implied_vols[i] = 0.2  # Fallback volatility

        self.option_data.implied_vols = implied_vols

    def _objective_function(self, params: np.ndarray) -> float:
        """Objective function for calibration"""
        # Unpack parameters
        omega, alpha, beta = params[:3]

        # GARCH constraints
        if alpha < 0 or beta < 0 or omega < 0 or alpha + beta >= 1:
            return 1e10

        if len(params) > 3:
            kappa_K, kappa_T, rho = params[3:6]
        else:
            kappa_K, kappa_T, rho = 0.1, 0.05, -0.5

        # Create GARCH surface model
        garch_params = {'omega': omega, 'alpha': alpha, 'beta': beta}
        surface_params = {'kappa_K': kappa_K, 'kappa_T': kappa_T, 'rho': rho}

        surface_model = GARCHVolatilitySurface(garch_params, surface_params)

        # Calculate model prices
        model_prices = np.zeros(len(self.option_data.option_prices))

        for i in range(len(model_prices)):
            model_vol = surface_model.volatility(
                self.option_data.S0,
                self.option_data.K[i],
                self.option_data.T[i]
            )

            model_prices[i] = self.bs_model.option_price(
                self.option_data.S0,
                self.option_data.K[i],
                self.option_data.T[i],
                self.option_data.r,
                model_vol,
                self.option_data.option_types[i]
            )

        # Calculate weighted squared errors
        price_errors = (model_prices - self.option_data.option_prices)**2

        # Weight by vega to focus on liquid options
        weights = np.array([
            self.bs_model.vega(
                self.option_data.S0,
                self.option_data.K[i],
                self.option_data.T[i],
                self.option_data.r,
                self.option_data.implied_vols[i]
            ) for i in range(len(price_errors))
        ])

        weights = weights / np.sum(weights)  # Normalize weights

        weighted_error = np.sum(weights * price_errors)

        return weighted_error

    def calibrate(self, method: str = 'differential_evolution',
                 include_surface_params: bool = True,
                 maxiter: int = 1000) -> Dict[str, float]:
        """
        Calibrate GARCH model to option data

        Parameters:
        -----------
        method : str
            Optimization method ('differential_evolution' or 'L-BFGS-B')
        include_surface_params : bool
            Whether to calibrate surface parameters
        maxiter : int
            Maximum iterations
        """

        # Parameter bounds
        if include_surface_params:
            bounds = [
                (1e-6, 0.1),    # omega
                (1e-6, 0.3),    # alpha
                (1e-6, 0.99),   # beta
                (0.01, 0.5),    # kappa_K
                (0.01, 0.2),    # kappa_T
                (-0.9, 0.1)     # rho
            ]
            initial_guess = [0.01, 0.05, 0.9, 0.1, 0.05, -0.5]
        else:
            bounds = [
                (1e-6, 0.1),    # omega
                (1e-6, 0.3),    # alpha
                (1e-6, 0.99)    # beta
            ]
            initial_guess = [0.01, 0.05, 0.9]

        print(f"Calibrating GARCH model using {method}...")
        print(f"Number of options: {len(self.option_data.option_prices)}")

        if method == 'differential_evolution':
            result = differential_evolution(
                self._objective_function,
                bounds,
                maxiter=maxiter,
                seed=42,
                disp=True
            )
        else:
            result = minimize(
                self._objective_function,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter}
            )

        if result.success:
            print("✓ Calibration successful!")
            self.calibrated_params = result.x

            # Store calibrated parameters
            params_dict = {
                'omega': result.x[0],
                'alpha': result.x[1],
                'beta': result.x[2]
            }

            if include_surface_params and len(result.x) > 3:
                params_dict.update({
                    'kappa_K': result.x[3],
                    'kappa_T': result.x[4],
                    'rho': result.x[5]
                })

            print("Calibrated parameters:")
            for param, value in params_dict.items():
                print(f"  {param}: {value:.6f}")

            print(f"Final objective value: {result.fun:.6f}")

            return params_dict
        else:
            print("✗ Calibration failed!")
            print(f"Reason: {result.message}")
            return None

    def create_fitted_surface(self, n_strikes: int = 20, n_maturities: int = 10) -> VolatilitySurface:
        """Create fitted volatility surface"""
        if self.calibrated_params is None:
            raise ValueError("Model must be calibrated first")

        # Create GARCH surface model with calibrated parameters
        garch_params = {
            'omega': self.calibrated_params[0],
            'alpha': self.calibrated_params[1],
            'beta': self.calibrated_params[2]
        }

        if len(self.calibrated_params) > 3:
            surface_params = {
                'kappa_K': self.calibrated_params[3],
                'kappa_T': self.calibrated_params[4],
                'rho': self.calibrated_params[5]
            }
        else:
            surface_params = {'kappa_K': 0.1, 'kappa_T': 0.05, 'rho': -0.5}

        surface_model = GARCHVolatilitySurface(garch_params, surface_params)

        # Create grid
        min_strike = np.min(self.option_data.K) * 0.8
        max_strike = np.max(self.option_data.K) * 1.2
        strikes = np.linspace(min_strike, max_strike, n_strikes)

        min_maturity = max(np.min(self.option_data.T), 1/365)  # At least 1 day
        max_maturity = np.max(self.option_data.T) * 1.2
        maturities = np.linspace(min_maturity, max_maturity, n_maturities)

        # Generate surface
        self.fitted_surface = surface_model.generate_surface(
            self.option_data.S0, strikes, maturities
        )

        return self.fitted_surface

    def calculate_pricing_errors(self) -> Dict[str, float]:
        """Calculate pricing errors for calibrated model"""
        if self.calibrated_params is None:
            raise ValueError("Model must be calibrated first")

        # Recalculate model prices with calibrated parameters
        model_prices = np.zeros(len(self.option_data.option_prices))

        garch_params = {
            'omega': self.calibrated_params[0],
            'alpha': self.calibrated_params[1],
            'beta': self.calibrated_params[2]
        }

        if len(self.calibrated_params) > 3:
            surface_params = {
                'kappa_K': self.calibrated_params[3],
                'kappa_T': self.calibrated_params[4],
                'rho': self.calibrated_params[5]
            }
        else:
            surface_params = {'kappa_K': 0.1, 'kappa_T': 0.05, 'rho': -0.5}

        surface_model = GARCHVolatilitySurface(garch_params, surface_params)

        for i in range(len(model_prices)):
            model_vol = surface_model.volatility(
                self.option_data.S0,
                self.option_data.K[i],
                self.option_data.T[i]
            )

            model_prices[i] = self.bs_model.option_price(
                self.option_data.S0,
                self.option_data.K[i],
                self.option_data.T[i],
                self.option_data.r,
                model_vol,
                self.option_data.option_types[i]
            )

        # Calculate error metrics
        price_errors = model_prices - self.option_data.option_prices

        rmse = np.sqrt(np.mean(price_errors**2))
        mae = np.mean(np.abs(price_errors))
        mape = np.mean(np.abs(price_errors / self.option_data.option_prices)) * 100

        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Max_Error': np.max(np.abs(price_errors)),
            'R_squared': 1 - np.var(price_errors) / np.var(self.option_data.option_prices)
        }

    def plot_volatility_surface(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot the fitted volatility surface"""
        if self.fitted_surface is None:
            self.create_fitted_surface()

        fig = plt.figure(figsize=figsize)

        # 3D surface plot
        ax1 = fig.add_subplot(121, projection='3d')

        K_grid, T_grid = np.meshgrid(
            self.fitted_surface.strikes,
            self.fitted_surface.maturities
        )

        surf = ax1.plot_surface(
            K_grid, T_grid, self.fitted_surface.volatilities * 100,
            cmap='viridis', alpha=0.8
        )

        ax1.set_xlabel('Strike')
        ax1.set_ylabel('Maturity')
        ax1.set_zlabel('Volatility (%)')
        ax1.set_title('GARCH-based Volatility Surface')

        # 2D contour plot
        ax2 = fig.add_subplot(122)

        contour = ax2.contour(
            K_grid, T_grid, self.fitted_surface.volatilities * 100,
            levels=15, colors='black', alpha=0.6, linewidths=0.5
        )

        filled_contour = ax2.contourf(
            K_grid, T_grid, self.fitted_surface.volatilities * 100,
            levels=15, cmap='viridis', alpha=0.8
        )

        # Plot market data points
        ax2.scatter(
            self.option_data.K,
            self.option_data.T,
            c=self.option_data.implied_vols * 100,
            s=50, cmap='viridis', edgecolors='white', linewidth=1
        )

        ax2.set_xlabel('Strike')
        ax2.set_ylabel('Maturity')
        ax2.set_title('Volatility Surface (Top View)')

        plt.colorbar(filled_contour, ax=ax2, label='Volatility (%)')
        plt.tight_layout()

        return fig

    def plot_implied_vol_smile(self, target_maturity: float = None, figsize: Tuple[int, int] = (10, 6)):
        """Plot implied volatility smile for a specific maturity"""
        if target_maturity is None:
            target_maturity = np.median(self.option_data.T)

        # Find options close to target maturity
        maturity_tolerance = 0.05  # 5% tolerance
        mask = np.abs(self.option_data.T - target_maturity) <= maturity_tolerance

        if np.sum(mask) == 0:
            print(f"No options found near maturity {target_maturity:.2f}")
            return None

        # Extract data for this maturity
        strikes = self.option_data.K[mask]
        market_vols = self.option_data.implied_vols[mask]

        # Sort by strike
        sort_idx = np.argsort(strikes)
        strikes = strikes[sort_idx]
        market_vols = market_vols[sort_idx]

        # Calculate model volatilities
        if self.calibrated_params is not None:
            garch_params = {
                'omega': self.calibrated_params[0],
                'alpha': self.calibrated_params[1],
                'beta': self.calibrated_params[2]
            }

            if len(self.calibrated_params) > 3:
                surface_params = {
                    'kappa_K': self.calibrated_params[3],
                    'kappa_T': self.calibrated_params[4],
                    'rho': self.calibrated_params[5]
                }
            else:
                surface_params = {'kappa_K': 0.1, 'kappa_T': 0.05, 'rho': -0.5}

            surface_model = GARCHVolatilitySurface(garch_params, surface_params)

            # Create fine grid for smooth curve
            strike_grid = np.linspace(np.min(strikes), np.max(strikes), 100)
            model_vols = surface_model.volatility(
                self.option_data.S0, strike_grid, target_maturity
            )

        fig, ax = plt.subplots(figsize=figsize)

        # Plot market data
        moneyness_market = strikes / self.option_data.S0
        ax.scatter(moneyness_market, market_vols * 100,
                  c='red', s=50, label='Market Data', zorder=5)

        # Plot model curve if available
        if self.calibrated_params is not None:
            moneyness_grid = strike_grid / self.option_data.S0
            ax.plot(moneyness_grid, model_vols * 100,
                   'b-', linewidth=2, label='GARCH Model', alpha=0.8)

        ax.set_xlabel('Moneyness (K/S)')
        ax.set_ylabel('Implied Volatility (%)')
        ax.set_title(f'Volatility Smile (T = {target_maturity:.2f})')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add vertical line at ATM
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='ATM')

        return fig

def load_option_data_from_csv(filepath: str, S0: float, r: float) -> OptionData:
    """Load option data from CSV file"""
    df = pd.read_csv(filepath)

    # Assume CSV has columns: K (strike), T (maturity), option_price, option_type
    # option_type should be 1 for calls, -1 for puts

    required_columns = ['K', 'T', 'option_price', 'option_type']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")

    return OptionData(
        S0=S0,
        K=df['K'].values,
        T=df['T'].values,
        r=r,
        option_prices=df['option_price'].values,
        option_types=df['option_type'].values
    )

def example_calibration():
    """Example of GARCH option calibration"""
    print("GARCH Option Calibration Example")
    print("=" * 50)

    # Generate synthetic option data
    np.random.seed(42)
    S0 = 100
    r = 0.02

    # Create option grid
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    maturities = np.array([0.25, 0.5, 1.0])

    K_grid, T_grid = np.meshgrid(strikes, maturities)
    K_flat = K_grid.flatten()
    T_flat = T_grid.flatten()

    # Generate "market" prices using a known volatility surface
    true_surface = GARCHVolatilitySurface(
        {'omega': 0.01, 'alpha': 0.05, 'beta': 0.9},
        {'kappa_K': 0.15, 'kappa_T': 0.1, 'rho': -0.3}
    )

    bs_model = BlackScholesModel()
    option_types = np.ones(len(K_flat))  # All calls for simplicity

    market_prices = np.zeros(len(K_flat))
    for i in range(len(K_flat)):
        vol = true_surface.volatility(S0, K_flat[i], T_flat[i])
        market_prices[i] = bs_model.call_price(S0, K_flat[i], T_flat[i], r, vol)
        # Add some noise
        market_prices[i] += np.random.normal(0, 0.01 * market_prices[i])

    # Create option data
    option_data = OptionData(
        S0=S0,
        K=K_flat,
        T=T_flat,
        r=r,
        option_prices=market_prices,
        option_types=option_types
    )

    print(f"Generated {len(market_prices)} synthetic option prices")

    # Calibrate model
    calibrator = OptionGARCHCalibrator(option_data)

    # Calibrate with surface parameters
    calibrated_params = calibrator.calibrate(
        method='differential_evolution',
        include_surface_params=True,
        maxiter=50  # Reduced for example
    )

    if calibrated_params is not None:
        # Calculate errors
        errors = calibrator.calculate_pricing_errors()
        print("\nPricing Errors:")
        for metric, value in errors.items():
            print(f"  {metric}: {value:.6f}")

        # Create plots
        try:
            # Volatility surface
            fig1 = calibrator.plot_volatility_surface()
            fig1.savefig('garch_option_surface.png', dpi=300, bbox_inches='tight')
            print("Volatility surface saved as 'garch_option_surface.png'")

            # Volatility smile
            fig2 = calibrator.plot_implied_vol_smile(target_maturity=0.5)
            fig2.savefig('garch_vol_smile.png', dpi=300, bbox_inches='tight')
            print("Volatility smile saved as 'garch_vol_smile.png'")

        except Exception as e:
            print(f"Could not create plots: {e}")

    return calibrator

if __name__ == "__main__":
    # Run example
    calibrator = example_calibration()
