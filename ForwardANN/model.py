import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SpectralNorm

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels)
        self.linear1 = nn.Linear(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.linear2 = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x

        # First transformation
        out = self.bn1(x)
        out = F.relu(out)
        out = self.linear1(out)
        out = self.dropout(out)  # Add dropout after first linear

        # Second transformation
        out = self.bn2(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = self.dropout(out)  # Add dropout after second linear

        # Scaled residual connection
        return 0.1 * out + identity  # Scale the residual to improve stability

class FinancialResidualBlock(nn.Module):
    def __init__(self, channels, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Dropout(dropout_rate),
            nn.Linear(channels, channels),
            nn.Dropout(dropout_rate * 0.8),
            SpectralNorm(nn.Linear(channels, channels))  # Add spectral normalization
        )

    def forward(self, x):
        return x + 0.3 * self.block(x)  # Reduced residual scaling


class CaNNModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(CaNNModel, self).__init__()
        input_features = 19
        neurons = 200  # Reduced capacity

        # Input layer
        self.input_layer = nn.Linear(input_features, neurons)
        self.input_bn = nn.BatchNorm1d(neurons)

        # Financial residual blocks
        self.fin_res1 = FinancialResidualBlock(neurons, dropout_rate)
        self.fin_res2 = FinancialResidualBlock(neurons, dropout_rate)
        self.fin_res3 = FinancialResidualBlock(neurons, dropout_rate)  # Optional third block

        # Output layer
        self.output_bn = nn.BatchNorm1d(neurons)
        self.output_layer = nn.Linear(neurons, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input processing
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = F.relu(x)

        # Financial residual blocks
        x = self.fin_res1(x)
        x = self.fin_res2(x)
        x = self.fin_res3(x)  # Optional third block

        # Output processing
        x = self.output_bn(x)
        x = F.relu(x)
        x = F.softplus(self.output_layer(x))

        return x

# class CaNNModel(nn.Module):
#     def __init__(self, dropout_rate=0.3):
#         super(CaNNModel, self).__init__()
#         input_features = 19
#         neurons = 200  # Reduced capacity
#         # Input layer
#         self.input_layer = nn.Linear(input_features, neurons)
#         self.input_bn = nn.BatchNorm1d(neurons)

#         # Residual blocks
#         self.res1 = ResidualBlock(neurons, dropout_rate)
#         self.financial_res_block = FinancialResidualBlock(neurons, dropout_rate)
#         self.res2 = ResidualBlock(neurons, dropout_rate)
#         # self.res3 = ResidualBlock(neurons, dropout_rate)  # New residual block

#         # Output layer
#         self.output_bn = nn.BatchNorm1d(neurons)
#         self.output_layer = nn.Linear(neurons, 1)

#         # Initialize weights
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         # Input processing
#         x = self.input_layer(x)
#         x = self.input_bn(x)
#         x = F.relu(x)

#         # Residual blocks
#         x = self.res1(x)
#         x = self.financial_res_block(x)  # Add FinancialResidualBlock
#         x = self.res2(x)
#         # x = self.res3(x)  # New residual block

#         # Output processing
#         x = self.output_bn(x)
#         x = F.relu(x)
#         x = F.softplus(self.output_layer(x))

#         return x



# import torch
# import torch.nn as nn
# from typing import List, Optional

# class CaNNModel(nn.Module):
#     def __init__(
#         self,
#         input_features: int = 10,
#         hidden_layers: List[int] = [250, 250, 250, 250, 250, 250],
#         dropout_rate: float = 0.0,
#         activation: Optional[nn.Module] = None,
#         batch_norm: bool = True,
#         output_activation: Optional[nn.Module] = nn.Softplus()
#     ):
#         """
#          Neural Network model with configurable architecture.

#         Args:
#             input_features: Number of input features
#             hidden_layers: List of neurons for each hidden layer
#             dropout_rate: Dropout probability
#             activation: Activation function to use (defaults to ReLU)
#             batch_norm: Whether to use batch normalization
#             output_activation: Activation function for output layer
#         """
#         super().__init__()

#         self.activation = activation if activation is not None else nn.ReLU()
#         self.output_activation = output_activation

#         # Build network architecture dynamically
#         layers = []
#         prev_size = input_features

#         for hidden_size in hidden_layers:
#             layers.extend([
#                 nn.Linear(prev_size, hidden_size),
#                 nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity(),
#                 self.activation,
#                 nn.Dropout(dropout_rate)
#             ])
#             prev_size = hidden_size

#         # Output layer
#         layers.append(nn.Linear(prev_size, 1))
#         if output_activation is not None:
#             layers.append(output_activation)

#         self.network = nn.Sequential(*layers)

#         # Initialize weights
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         """Initialize network weights using He initialization."""
#         if isinstance(module, nn.Linear):
#             nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass through the network."""
#         return self.network(x)

#     def get_number_of_parameters(self) -> int:
#         """Return the total number of trainable parameters."""
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)

# import torch
# import torch.nn as nn

# class CaNNModel(nn.Module):
#     def __init__(self, dropout_rate=0.0):
#         super(CaNNModel, self).__init__()
#         input_features = 10
#         neurons = 250

#         # Layer Normalization for each layer
#         self.layer_norm_input = nn.LayerNorm(neurons)
#         self.layer_norm1 = nn.LayerNorm(neurons)
#         self.layer_norm2 = nn.LayerNorm(neurons)
#         self.layer_norm3 = nn.LayerNorm(neurons)
#         self.layer_norm4 = nn.LayerNorm(neurons)
#         self.layer_norm5 = nn.LayerNorm(neurons)
#         self.layer_norm6 = nn.LayerNorm(neurons)

#         # Linear layers
#         self.input_layer = nn.Linear(input_features, neurons)
#         self.hl1 = nn.Linear(neurons, neurons)
#         self.hl2 = nn.Linear(neurons, neurons)
#         self.hl3 = nn.Linear(neurons, neurons)
#         self.hl4 = nn.Linear(neurons, neurons)
#         self.hl5 = nn.Linear(neurons, neurons)
#         self.hl6 = nn.Linear(neurons, neurons)
#         self.output_layer = nn.Linear(neurons, 1)

#         # Dropout layers
#         self.dropout1 = nn.Dropout(dropout_rate)
#         self.dropout2 = nn.Dropout(dropout_rate)
#         self.dropout3 = nn.Dropout(dropout_rate)
#         self.dropout4 = nn.Dropout(dropout_rate)
#         self.dropout5 = nn.Dropout(dropout_rate)
#         self.dropout6 = nn.Dropout(dropout_rate)

#         # Weight initialization
#         self.reset_parameters()

#     def reset_parameters(self):
#         # Xavier/Glorot initialization for better weight initialization
#         for layer in [self.input_layer, self.hl1, self.hl2, self.hl3,
#                       self.hl4, self.hl5, self.hl6, self.output_layer]:
#             nn.init.xavier_normal_(layer.weight)
#             nn.init.zeros_(layer.bias)

#     def forward(self, x):
#         # Input layer with normalization and activation
#         x = self.input_layer(x)
#         x = self.layer_norm_input(x)
#         x = activation(x)

#         # Hidden layers with normalization, dropout, and activation
#         x = self.hl1(x)
#         x = self.layer_norm1(x)
#         x = activation(x)
#         x = self.dropout1(x)

#         x = self.hl2(x)
#         x = self.layer_norm2(x)
#         x = activation(x)
#         x = self.dropout2(x)

#         x = self.hl3(x)
#         x = self.layer_norm3(x)
#         x = activation(x)
#         x = self.dropout3(x)

#         x = self.hl4(x)
#         x = self.layer_norm4(x)
#         x = activation(x)
#         x = self.dropout4(x)

#         x = self.hl5(x)
#         x = self.layer_norm5(x)
#         x = activation(x)
#         x = self.dropout5(x)

#         x = self.hl6(x)
#         x = self.layer_norm6(x)
#         x = activation(x)
#         x = self.dropout6(x)

#         # Output layer with softplus activation
#         x = torch.nn.functional.softplus(self.output_layer(x))
#         return x


# class ResidualBlock(nn.Module):
#     def __init__(self, channels, dropout_rate):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.BatchNorm1d(channels),
#             nn.GELU(),
#             nn.Linear(channels, channels*2),
#             nn.Dropout(dropout_rate),
#             nn.Linear(channels*2, channels),
#             nn.Dropout(dropout_rate/2)
#         )

#     def forward(self, x):
#         return x + self.block(x)  # Remove residual scaling

# class CaNNModel(nn.Module):
#     def __init__(self, dropout_rate=0.4):
#         super().__init__()
#         input_features = 10
#         neurons = 256

#         self.input = nn.Sequential(
#             nn.Linear(input_features, neurons),
#             nn.BatchNorm1d(neurons),
#             nn.GELU()  # Make sure to use PyTorch's native GELU
#         )

#         self.res_blocks = nn.Sequential(
#             ResidualBlock(neurons, dropout_rate),
#             ResidualBlock(neurons, dropout_rate),
#             # ResidualBlock(neurons, dropout_rate)
#         )

#         self.output = nn.Sequential(
#             nn.BatchNorm1d(neurons),
#             nn.Linear(neurons, 1),
#             nn.Softplus()
#         )

#         # Fixed weight initialization
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         """Initialize weights for GELU compatibility"""
#         if isinstance(module, nn.Linear):
#             # Use fan_in mode with correct GELU gain approximation
#             nn.init.kaiming_normal_(
#                 module.weight,
#                 mode='fan_in',
#                 nonlinearity='leaky_relu'  # Closest supported approximation
#             )
#             nn.init.normal_(module.bias, 0, 0.01)

#     def forward(self, x):
#         x = self.input(x)
#         x = self.res_blocks(x)
#         return self.output(x)
