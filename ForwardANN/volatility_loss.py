import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VolatilityLoss(nn.Module):
    """
    Comprehensive loss function specifically designed for implied volatility prediction.
    Combines multiple loss components to capture different aspects of volatility modeling.
    """

    def __init__(self,
                 huber_weight=0.4,
                 relative_weight=0.3,
                 quantile_weight=0.2,
                 constraint_weight=0.1,
                 huber_delta=0.1,
                 quantile_alpha=0.1):
        super().__init__()

        self.huber_weight = huber_weight
        self.relative_weight = relative_weight
        self.quantile_weight = quantile_weight
        self.constraint_weight = constraint_weight
        self.huber_delta = huber_delta
        self.quantile_alpha = quantile_alpha

        self.huber_loss = nn.HuberLoss(delta=huber_delta)

    def relative_error_loss(self, pred, target):
        """Relative error loss - important for volatility as it's often compared in relative terms"""
        epsilon = 1e-6
        relative_error = torch.abs((pred - target) / (target + epsilon))
        return torch.mean(relative_error)

    def quantile_loss(self, pred, target, alpha=None):
        """Quantile loss for better tail behavior in volatility prediction"""
        if alpha is None:
            alpha = self.quantile_alpha

        error = target - pred
        loss = torch.where(error >= 0,
                          alpha * error,
                          (alpha - 1) * error)
        return torch.mean(loss)

    def volatility_constraint_loss(self, pred):
        """Penalty for unrealistic volatility values"""
        # Penalize extremely low volatilities (below 1%)
        low_vol_penalty = torch.relu(0.01 - pred)

        # Penalize extremely high volatilities (above 300%)
        high_vol_penalty = torch.relu(pred - 3.0)

        return torch.mean(low_vol_penalty + high_vol_penalty)

    def forward(self, pred, target):
        """
        Combined loss function for volatility prediction

        Args:
            pred: Predicted volatility values
            target: Target volatility values

        Returns:
            Combined loss value
        """
        # Huber loss for robustness to outliers
        huber = self.huber_loss(pred, target)

        # Relative error loss
        relative = self.relative_error_loss(pred, target)

        # Quantile loss for tail behavior
        quantile = self.quantile_loss(pred, target)

        # Volatility constraint loss
        constraint = self.volatility_constraint_loss(pred)

        # Combine all losses
        total_loss = (self.huber_weight * huber +
                     self.relative_weight * relative +
                     self.quantile_weight * quantile +
                     self.constraint_weight * constraint)

        return total_loss, {
            'huber': huber.item(),
            'relative': relative.item(),
            'quantile': quantile.item(),
            'constraint': constraint.item(),
            'total': total_loss.item()
        }


class AdaptiveVolatilityLoss(nn.Module):
    """
    Adaptive loss that adjusts weights based on volatility regime
    """

    def __init__(self, low_vol_threshold=0.15, high_vol_threshold=0.4):
        super().__init__()
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold
        self.base_loss = VolatilityLoss()

    def forward(self, pred, target):
        # Classify volatility regime
        avg_vol = torch.mean(target)

        if avg_vol < self.low_vol_threshold:
            # Low volatility regime - focus more on relative accuracy
            weights = {
                'huber_weight': 0.2,
                'relative_weight': 0.5,
                'quantile_weight': 0.2,
                'constraint_weight': 0.1
            }
        elif avg_vol > self.high_vol_threshold:
            # High volatility regime - focus more on absolute accuracy
            weights = {
                'huber_weight': 0.6,
                'relative_weight': 0.2,
                'quantile_weight': 0.1,
                'constraint_weight': 0.1
            }
        else:
            # Medium volatility regime - balanced approach
            weights = {
                'huber_weight': 0.4,
                'relative_weight': 0.3,
                'quantile_weight': 0.2,
                'constraint_weight': 0.1
            }

        # Update base loss weights
        for key, value in weights.items():
            setattr(self.base_loss, key, value)

        return self.base_loss(pred, target)


class FocalVolatilityLoss(nn.Module):
    """
    Focal loss variant for volatility prediction that focuses on hard examples
    """

    def __init__(self, alpha=0.25, gamma=2.0, base_loss='huber'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

        if base_loss == 'huber':
            self.base_criterion = nn.HuberLoss(reduction='none')
        elif base_loss == 'mse':
            self.base_criterion = nn.MSELoss(reduction='none')
        else:
            self.base_criterion = nn.L1Loss(reduction='none')

    def forward(self, pred, target):
        # Calculate base loss
        base_loss = self.base_criterion(pred, target)

        # Calculate relative error for focal weighting
        epsilon = 1e-6
        relative_error = torch.abs((pred - target) / (target + epsilon))

        # Focal weight calculation
        pt = torch.exp(-relative_error)  # Confidence score
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Apply focal weighting
        focal_loss = focal_weight * base_loss

        return torch.mean(focal_loss)


class WeightedVolatilityLoss(nn.Module):
    """
    Loss function that weights samples based on moneyness and time to expiration
    """

    def __init__(self):
        super().__init__()
        self.base_loss = nn.HuberLoss(reduction='none')

    def calculate_weights(self, features):
        """
        Calculate sample weights based on option characteristics

        Args:
            features: Input features tensor containing S0, m, T, etc.

        Returns:
            Weight tensor for each sample
        """
        # Extract relevant features (assuming specific order)
        # S0 (index 0), m (index 1), T (index 3)
        moneyness = features[:, 1]  # m = K/S0
        time_to_exp = features[:, 3]  # T

        # Weight ATM options more heavily
        atm_weight = torch.exp(-5 * (moneyness - 1.0) ** 2)

        # Weight shorter-term options more heavily
        time_weight = torch.exp(-2 * time_to_exp)

        # Combine weights
        total_weight = atm_weight * time_weight

        # Normalize weights
        total_weight = total_weight / torch.mean(total_weight)

        return total_weight

    def forward(self, pred, target, features):
        # Calculate sample weights
        weights = self.calculate_weights(features)

        # Calculate base loss
        losses = self.base_loss(pred, target)

        # Apply weights
        weighted_losses = losses * weights.unsqueeze(1)

        return torch.mean(weighted_losses)


def create_volatility_loss(loss_type='combined', **kwargs):
    """
    Factory function to create different types of volatility losses

    Args:
        loss_type: Type of loss ('combined', 'adaptive', 'focal', 'weighted')
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function instance
    """
    if loss_type == 'combined':
        return VolatilityLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveVolatilityLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalVolatilityLoss(**kwargs)
    elif loss_type == 'weighted':
        return WeightedVolatilityLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
