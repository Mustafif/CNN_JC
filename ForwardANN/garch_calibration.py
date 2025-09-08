import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, t as student_t
import warnings
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class GARCHResults:
    """Container for GARCH model estimation results"""
    params: np.ndarray
    loglikelihood: float
    aic: float
    bic: float
    hessian: Optional[np.ndarray] = None
    std_errors: Optional[np.ndarray] = None
    t_stats: Optional[np.ndarray] = None
    p_values: Optional[np.ndarray] = None
    converged: bool = False
    iterations: int = 0

class GARCHModelBase(ABC):
    """Abstract base class for GARCH models"""

    def __init__(self, distribution='normal'):
        self.distribution = distribution
        self.params = None
        self.results = None
        self.fitted = False

    @abstractmethod
    def _variance_recursion(self, params: np.ndarray, resid: np.ndarray) -> np.ndarray:
        """Calculate conditional variance recursion"""
        pass

    @abstractmethod
    def _param_names(self) -> List[str]:
        """Return parameter names"""
        pass

    @abstractmethod
    def _param_bounds(self) -> List[Tuple[float, float]]:
        """Return parameter bounds for optimization"""
        pass

    def loglikelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """Calculate log-likelihood for given parameters"""
        T = len(returns)

        # Calculate conditional variance
        sigma2 = self._variance_recursion(params, returns)

        # Avoid numerical issues
        sigma2 = np.maximum(sigma2, 1e-8)

        # Calculate log-likelihood based on distribution
        if self.distribution == 'normal':
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
        elif self.distribution == 't':
            # Student-t distribution with degrees of freedom as last parameter
            nu = params[-1]
            if nu <= 2:
                return -np.inf
            ll = np.sum(
                student_t.logpdf(returns / np.sqrt(sigma2), df=nu) - 0.5 * np.log(sigma2)
            )
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        return ll if np.isfinite(ll) else -np.inf

    def _negative_loglikelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """Negative log-likelihood for minimization"""
        return -self.loglikelihood(params, returns)

    def fit(self, returns: np.ndarray, method='L-BFGS-B', maxiter=1000,
            use_global=True) -> GARCHResults:
        """
        Fit GARCH model to return series

        Parameters:
        -----------
        returns : np.ndarray
            Return series (should be demeaned)
        method : str
            Optimization method
        maxiter : int
            Maximum iterations
        use_global : bool
            Whether to use global optimization as fallback
        """
        returns = np.asarray(returns).flatten()

        # Get parameter bounds
        bounds = self._param_bounds()

        # Initial parameter guess
        initial_params = self._get_initial_params(returns)

        # Local optimization
        try:
            result = minimize(
                self._negative_loglikelihood,
                initial_params,
                args=(returns,),
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter}
            )

            if not result.success and use_global:
                # Try global optimization as fallback
                print("Local optimization failed, trying global optimization...")
                result = differential_evolution(
                    self._negative_loglikelihood,
                    bounds,
                    args=(returns,),
                    maxiter=maxiter // 10,
                    seed=42
                )

        except Exception as e:
            raise RuntimeError(f"Optimization failed: {e}")

        # Store results
        self.params = result.x
        self.fitted = True

        # Calculate model statistics
        ll = self.loglikelihood(self.params, returns)
        k = len(self.params)
        n = len(returns)
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        # Calculate standard errors if possible
        std_errors = None
        t_stats = None
        p_values = None

        try:
            hessian = self._numerical_hessian(self.params, returns)
            if np.all(np.linalg.eigvals(-hessian) > 0):  # Check positive definite
                cov_matrix = np.linalg.inv(-hessian)
                std_errors = np.sqrt(np.diag(cov_matrix))
                t_stats = self.params / std_errors
                p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))
        except:
            hessian = None
            warnings.warn("Could not compute standard errors")

        self.results = GARCHResults(
            params=self.params,
            loglikelihood=ll,
            aic=aic,
            bic=bic,
            hessian=hessian,
            std_errors=std_errors,
            t_stats=t_stats,
            p_values=p_values,
            converged=result.success,
            iterations=result.nit if hasattr(result, 'nit') else 0
        )

        return self.results

    def _numerical_hessian(self, params: np.ndarray, returns: np.ndarray,
                          eps: float = 1e-5) -> np.ndarray:
        """Compute numerical Hessian matrix"""
        n = len(params)
        hessian = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                params_pp = params.copy()
                params_pm = params.copy()
                params_mp = params.copy()
                params_mm = params.copy()

                params_pp[i] += eps
                params_pp[j] += eps

                params_pm[i] += eps
                params_pm[j] -= eps

                params_mp[i] -= eps
                params_mp[j] += eps

                params_mm[i] -= eps
                params_mm[j] -= eps

                hessian[i, j] = (
                    self._negative_loglikelihood(params_pp, returns) -
                    self._negative_loglikelihood(params_pm, returns) -
                    self._negative_loglikelihood(params_mp, returns) +
                    self._negative_loglikelihood(params_mm, returns)
                ) / (4 * eps**2)

        return hessian

    def _get_initial_params(self, returns: np.ndarray) -> np.ndarray:
        """Get initial parameter estimates"""
        # Simple moment-based initial estimates
        unconditional_var = np.var(returns)

        if hasattr(self, '_get_model_initial_params'):
            return self._get_model_initial_params(returns, unconditional_var)
        else:
            # Default fallback
            return np.array([0.01, 0.05, 0.9])

    def conditional_variance(self, returns: np.ndarray,
                           params: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate conditional variance series"""
        if params is None:
            if not self.fitted:
                raise ValueError("Model must be fitted first")
            params = self.params

        return self._variance_recursion(params, returns)

    def forecast_variance(self, returns: np.ndarray, horizon: int = 1,
                         params: Optional[np.ndarray] = None) -> np.ndarray:
        """Forecast conditional variance"""
        if params is None:
            if not self.fitted:
                raise ValueError("Model must be fitted first")
            params = self.params

        # Calculate current conditional variance
        sigma2 = self._variance_recursion(params, returns)
        current_var = sigma2[-1]
        current_resid = returns[-1]

        # Multi-step forecast
        forecasts = np.zeros(horizon)

        for h in range(horizon):
            if h == 0:
                forecasts[h] = self._one_step_forecast(params, current_var, current_resid)
            else:
                # For h > 1, residual is assumed to be zero (unconditional expectation)
                forecasts[h] = self._one_step_forecast(params, forecasts[h-1], 0.0)

        return forecasts

    @abstractmethod
    def _one_step_forecast(self, params: np.ndarray, current_var: float,
                          current_resid: float) -> float:
        """Calculate one-step ahead variance forecast"""
        pass

    def summary(self) -> str:
        """Return model summary"""
        if not self.fitted:
            return "Model not fitted"

        param_names = self._param_names()

        summary = f"\n{self.__class__.__name__} Model Results\n"
        summary += "=" * 50 + "\n"
        summary += f"Distribution: {self.distribution}\n"
        summary += f"Log-likelihood: {self.results.loglikelihood:.6f}\n"
        summary += f"AIC: {self.results.aic:.6f}\n"
        summary += f"BIC: {self.results.bic:.6f}\n"
        summary += f"Converged: {self.results.converged}\n\n"

        summary += "Parameter Estimates:\n"
        summary += "-" * 50 + "\n"
        summary += f"{'Parameter':<12} {'Estimate':<12} {'Std Error':<12} {'t-stat':<12} {'p-value':<12}\n"
        summary += "-" * 60 + "\n"

        for i, name in enumerate(param_names):
            estimate = self.results.params[i]
            std_err = self.results.std_errors[i] if self.results.std_errors is not None else np.nan
            t_stat = self.results.t_stats[i] if self.results.t_stats is not None else np.nan
            p_val = self.results.p_values[i] if self.results.p_values is not None else np.nan

            summary += f"{name:<12} {estimate:<12.6f} {std_err:<12.6f} {t_stat:<12.6f} {p_val:<12.6f}\n"

        return summary

class GARCH11(GARCHModelBase):
    """Standard GARCH(1,1) model"""

    def _variance_recursion(self, params: np.ndarray, resid: np.ndarray) -> np.ndarray:
        """GARCH(1,1) variance recursion: sigma2_t = omega + alpha * resid2_{t-1} + beta * sigma2_{t-1}"""
        omega, alpha, beta = params[:3]

        T = len(resid)
        sigma2 = np.zeros(T)

        # Initial variance (unconditional variance estimate)
        sigma2[0] = np.var(resid) if np.var(resid) > 0 else 0.01

        for t in range(1, T):
            sigma2[t] = omega + alpha * resid[t-1]**2 + beta * sigma2[t-1]

        return sigma2

    def _param_names(self) -> List[str]:
        if self.distribution == 't':
            return ['omega', 'alpha', 'beta', 'nu']
        return ['omega', 'alpha', 'beta']

    def _param_bounds(self) -> List[Tuple[float, float]]:
        bounds = [
            (1e-8, 1.0),    # omega > 0
            (1e-8, 1.0),    # alpha > 0
            (1e-8, 0.999),  # beta > 0, alpha + beta < 1
        ]

        if self.distribution == 't':
            bounds.append((2.1, 50.0))  # degrees of freedom

        return bounds

    def _get_model_initial_params(self, returns: np.ndarray, unconditional_var: float) -> np.ndarray:
        """Get initial GARCH(1,1) parameters"""
        omega = 0.01 * unconditional_var
        alpha = 0.05
        beta = 0.9

        params = [omega, alpha, beta]

        if self.distribution == 't':
            params.append(10.0)  # Initial degrees of freedom

        return np.array(params)

    def _one_step_forecast(self, params: np.ndarray, current_var: float,
                          current_resid: float) -> float:
        """GARCH(1,1) one-step forecast"""
        omega, alpha, beta = params[:3]
        return omega + alpha * current_resid**2 + beta * current_var

class GJRGARCH(GARCHModelBase):
    """GJR-GARCH model with asymmetric effects"""

    def _variance_recursion(self, params: np.ndarray, resid: np.ndarray) -> np.ndarray:
        """GJR-GARCH variance recursion with leverage effect"""
        omega, alpha, gamma, beta = params[:4]

        T = len(resid)
        sigma2 = np.zeros(T)

        # Initial variance
        sigma2[0] = np.var(resid) if np.var(resid) > 0 else 0.01

        for t in range(1, T):
            leverage = 1.0 if resid[t-1] < 0 else 0.0
            sigma2[t] = (omega +
                        alpha * resid[t-1]**2 +
                        gamma * resid[t-1]**2 * leverage +
                        beta * sigma2[t-1])

        return sigma2

    def _param_names(self) -> List[str]:
        if self.distribution == 't':
            return ['omega', 'alpha', 'gamma', 'beta', 'nu']
        return ['omega', 'alpha', 'gamma', 'beta']

    def _param_bounds(self) -> List[Tuple[float, float]]:
        bounds = [
            (1e-8, 1.0),     # omega > 0
            (1e-8, 1.0),     # alpha > 0
            (-1.0, 1.0),     # gamma (can be negative)
            (1e-8, 0.999),   # beta > 0
        ]

        if self.distribution == 't':
            bounds.append((2.1, 50.0))

        return bounds

    def _get_model_initial_params(self, returns: np.ndarray, unconditional_var: float) -> np.ndarray:
        """Get initial GJR-GARCH parameters"""
        omega = 0.01 * unconditional_var
        alpha = 0.05
        gamma = 0.05  # Leverage effect
        beta = 0.85

        params = [omega, alpha, gamma, beta]

        if self.distribution == 't':
            params.append(10.0)

        return np.array(params)

    def _one_step_forecast(self, params: np.ndarray, current_var: float,
                          current_resid: float) -> float:
        """GJR-GARCH one-step forecast"""
        omega, alpha, gamma, beta = params[:4]
        leverage = 1.0 if current_resid < 0 else 0.0
        return (omega + alpha * current_resid**2 +
                gamma * current_resid**2 * leverage + beta * current_var)

class EGARCH(GARCHModelBase):
    """Exponential GARCH model"""

    def _variance_recursion(self, params: np.ndarray, resid: np.ndarray) -> np.ndarray:
        """EGARCH variance recursion in log space"""
        omega, alpha, gamma, beta = params[:4]

        T = len(resid)
        log_sigma2 = np.zeros(T)

        # Initial log variance
        initial_var = np.var(resid) if np.var(resid) > 0 else 0.01
        log_sigma2[0] = np.log(initial_var)

        for t in range(1, T):
            z = resid[t-1] / np.sqrt(np.exp(log_sigma2[t-1])) if log_sigma2[t-1] > -10 else resid[t-1]
            log_sigma2[t] = (omega +
                           alpha * (np.abs(z) - np.sqrt(2/np.pi)) +
                           gamma * z +
                           beta * log_sigma2[t-1])

        return np.exp(log_sigma2)

    def _param_names(self) -> List[str]:
        if self.distribution == 't':
            return ['omega', 'alpha', 'gamma', 'beta', 'nu']
        return ['omega', 'alpha', 'gamma', 'beta']

    def _param_bounds(self) -> List[Tuple[float, float]]:
        bounds = [
            (-10.0, 10.0),   # omega (can be negative in log space)
            (-1.0, 1.0),     # alpha
            (-1.0, 1.0),     # gamma (asymmetry)
            (-0.999, 0.999), # beta (persistence)
        ]

        if self.distribution == 't':
            bounds.append((2.1, 50.0))

        return bounds

    def _get_model_initial_params(self, returns: np.ndarray, unconditional_var: float) -> np.ndarray:
        """Get initial EGARCH parameters"""
        omega = np.log(unconditional_var) * 0.01
        alpha = 0.1
        gamma = -0.05  # Negative for leverage effect
        beta = 0.9

        params = [omega, alpha, gamma, beta]

        if self.distribution == 't':
            params.append(10.0)

        return np.array(params)

    def _one_step_forecast(self, params: np.ndarray, current_var: float,
                          current_resid: float) -> float:
        """EGARCH one-step forecast"""
        omega, alpha, gamma, beta = params[:4]

        log_var = np.log(current_var)
        z = current_resid / np.sqrt(current_var)

        next_log_var = (omega +
                       alpha * (np.abs(z) - np.sqrt(2/np.pi)) +
                       gamma * z +
                       beta * log_var)

        return np.exp(next_log_var)

class GARCHCalibrator:
    """Main class for GARCH model calibration and comparison"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def add_model(self, name: str, model: GARCHModelBase):
        """Add a GARCH model to the calibrator"""
        self.models[name] = model

    def calibrate_all(self, returns: np.ndarray, **fit_kwargs) -> Dict[str, GARCHResults]:
        """Calibrate all models and return results"""
        returns = self._prepare_returns(returns)

        print("Calibrating GARCH models...")
        print("=" * 50)

        for name, model in self.models.items():
            print(f"Fitting {name}...")
            try:
                result = model.fit(returns, **fit_kwargs)
                self.results[name] = result
                print(f"✓ {name}: LL = {result.loglikelihood:.4f}, AIC = {result.aic:.4f}")
            except Exception as e:
                print(f"✗ {name}: Failed - {e}")
                self.results[name] = None

        return self.results

    def _prepare_returns(self, returns: np.ndarray) -> np.ndarray:
        """Prepare return series (remove mean, handle missing values)"""
        returns = np.asarray(returns).flatten()

        # Remove NaN values
        returns = returns[~np.isnan(returns)]

        # Demean returns
        returns = returns - np.mean(returns)

        return returns

    def compare_models(self) -> pd.DataFrame:
        """Compare fitted models"""
        if not self.results:
            raise ValueError("No models have been fitted")

        comparison_data = []

        for name, result in self.results.items():
            if result is not None:
                comparison_data.append({
                    'Model': name,
                    'Log-Likelihood': result.loglikelihood,
                    'AIC': result.aic,
                    'BIC': result.bic,
                    'Parameters': len(result.params),
                    'Converged': result.converged
                })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('AIC')  # Sort by AIC (lower is better)

        return df

    def best_model(self, criterion='aic') -> Tuple[str, GARCHResults]:
        """Select best model based on information criterion"""
        if not self.results:
            raise ValueError("No models have been fitted")

        valid_results = {k: v for k, v in self.results.items() if v is not None}

        if not valid_results:
            raise ValueError("No models fitted successfully")

        if criterion.lower() == 'aic':
            best_name = min(valid_results.keys(), key=lambda k: valid_results[k].aic)
        elif criterion.lower() == 'bic':
            best_name = min(valid_results.keys(), key=lambda k: valid_results[k].bic)
        elif criterion.lower() == 'loglik':
            best_name = max(valid_results.keys(), key=lambda k: valid_results[k].loglikelihood)
        else:
            raise ValueError("Criterion must be 'aic', 'bic', or 'loglik'")

        return best_name, valid_results[best_name]

    def plot_volatility(self, returns: np.ndarray, model_names: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (12, 8)):
        """Plot conditional volatility for fitted models"""
        if not self.results:
            raise ValueError("No models have been fitted")

        returns = self._prepare_returns(returns)

        if model_names is None:
            model_names = [name for name, result in self.results.items() if result is not None]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot returns
        ax1.plot(returns, alpha=0.7, color='black', linewidth=0.5)
        ax1.set_ylabel('Returns')
        ax1.set_title('Returns and Conditional Volatility')
        ax1.grid(True, alpha=0.3)

        # Plot conditional volatilities
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))

        for name, color in zip(model_names, colors):
            if name in self.models and self.results[name] is not None:
                model = self.models[name]
                vol = np.sqrt(model.conditional_variance(returns)) * 100  # Convert to percentage
                ax2.plot(vol, label=name, color=color, linewidth=1.5)

        ax2.set_ylabel('Conditional Volatility (%)')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def diagnostic_tests(self, returns: np.ndarray, model_name: str) -> Dict[str, float]:
        """Perform diagnostic tests on model residuals"""
        if model_name not in self.models or self.results[model_name] is None:
            raise ValueError(f"Model {model_name} not fitted")

        returns = self._prepare_returns(returns)
        model = self.models[model_name]

        # Calculate standardized residuals
        sigma2 = model.conditional_variance(returns)
        std_resid = returns / np.sqrt(sigma2)

        # Ljung-Box test on standardized residuals (approximate)
        def ljung_box_stat(residuals, lags=10):
            n = len(residuals)
            acf = np.correlate(residuals, residuals, mode='full')
            acf = acf[n-1:][:lags+1] / acf[n-1]
            lb_stat = n * (n + 2) * np.sum([(acf[i]**2) / (n - i) for i in range(1, lags+1)])
            return lb_stat

        # Ljung-Box test on squared standardized residuals
        def ljung_box_squared(residuals, lags=10):
            return ljung_box_stat(residuals**2 - 1, lags)

        # ARCH-LM test (approximate)
        def arch_lm_test(residuals, lags=5):
            squared_resid = residuals**2
            n = len(squared_resid)

            # Create lagged matrix
            X = np.column_stack([squared_resid[i:n-lags+i] for i in range(lags)])
            y = squared_resid[lags:]

            # Simple regression R²
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)

            ss_res = np.sum((y - y_mean)**2)
            ss_tot = len(y) * np.var(y)

            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            lm_stat = len(y) * r_squared

            return lm_stat

        diagnostics = {
            'Ljung-Box (standardized residuals)': ljung_box_stat(std_resid),
            'Ljung-Box (squared residuals)': ljung_box_squared(std_resid),
            'ARCH-LM test': arch_lm_test(std_resid),
            'Jarque-Bera (approximate)': self._jarque_bera_test(std_resid)
        }

        return diagnostics

    def _jarque_bera_test(self, residuals: np.ndarray) -> float:
        """Approximate Jarque-Bera test for normality"""
        n = len(residuals)
        skewness = np.mean(((residuals - np.mean(residuals)) / np.std(residuals))**3)
        kurtosis = np.mean(((residuals - np.mean(residuals)) / np.std(residuals))**4)

        jb_stat = n * (skewness**2 / 6 + (kurtosis - 3)**2 / 24)
        return jb_stat

def example_calibration():
    """Example of how to use the GARCH calibration framework"""

    # Generate sample data (replace with real return data)
    np.random.seed(42)
    T = 1000

    # Simulate GARCH(1,1) process
    omega, alpha, beta = 0.01, 0.05, 0.9
    returns = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)  # Unconditional variance

    for t in range(T):
        epsilon = np.random.normal(0, 1)
        returns[t] = np.sqrt(sigma2[t]) * epsilon
        if t < T - 1:
            sigma2[t+1] = omega + alpha * returns[t]**2 + beta * sigma2[t]

    print("GARCH Calibration Example")
    print("=" * 50)
    print(f"Generated {T} synthetic returns")
    print(f"True parameters: ω={omega}, α={alpha}, β={beta}")
    print()

    # Create calibrator
    calibrator = GARCHCalibrator()

    # Add models
    calibrator.add_model('GARCH(1,1)', GARCH11())
    calibrator.add_model('GJR-GARCH', GJRGARCH())
    calibrator.add_model('EGARCH', EGARCH())
    calibrator.add_model('GARCH(1,1)-t', GARCH11(distribution='t'))

    # Calibrate all models
    results = calibrator.calibrate_all(returns)

    print("\nModel Comparison:")
    print(calibrator.compare_models())

    # Get best model
    best_name, best_result = calibrator.best_model('aic')
    print(f"\nBest model (AIC): {best_name}")

    # Print detailed results for best model
    print(calibrator.models[best_name].summary())

    # Diagnostic tests
    print(f"\nDiagnostic Tests for {best_name}:")
    diagnostics = calibrator.diagnostic_tests(returns, best_name)
    for test, stat in diagnostics.items():
        print(f"{test}: {stat:.4f}")

    # Plot volatility
    try:
        fig = calibrator.plot_volatility(returns)
        fig.savefig('garch_volatility.png', dpi=300, bbox_inches='tight')
        print(f"\nVolatility plot saved as 'garch_volatility.png'")
    except Exception as e:
        print(f"Could not create plot: {e}")

    return calibrator, returns

if __name__ == "__main__":
    # Run example
    calibrator, sample_returns = example_calibration()
