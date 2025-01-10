import torch
import torch.nn as nn

activation = torch.relu

class CaNNModel(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(CaNNModel, self).__init__()
        input_features = 10
        neurons = 250
        self.input_layer = nn.Linear(input_features, neurons)
        self.hl1 = nn.Linear(neurons, neurons)
        self.hl2 = nn.Linear(neurons, neurons)
        self.hl3 = nn.Linear(neurons, neurons)
        self.hl4 = nn.Linear(neurons, neurons)
        self.hl5 = nn.Linear(neurons, neurons)
        # self.hl6 = nn.Linear(neurons, neurons)
        self.output_layer = nn.Linear(neurons, 1) # just need 1 price

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.dropout5 = nn.Dropout(dropout_rate)
        # self.dropout6 = nn.Dropout(dropout_rate)


    def forward(self, x):
        x = activation(self.input_layer(x))
        x = activation(self.hl1(x))
        x = self.dropout1(x)
        x = activation(self.hl2(x))
        x = self.dropout2(x)
        x = activation(self.hl3(x))
        x = self.dropout3(x)
        x = activation(self.hl4(x))
        x = self.dropout4(x)
        x = activation(self.hl5(x))
        x = self.dropout5(x)
        # x = activation(self.hl6(x))
        # x = self.dropout6(x)
        x = torch.nn.functional.softplus(self.output_layer(x))
        return x


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
#         Improved Neural Network model with configurable architecture.

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
#     def __init__(self, dropout_rate=0.3):
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
#     #     self.reset_parameters()

#     # def reset_parameters(self):
#     #     # Xavier/Glorot initialization for better weight initialization
#     #     for layer in [self.input_layer, self.hl1, self.hl2, self.hl3,
#     #                   self.hl4, self.hl5, self.hl6, self.output_layer]:
#     #         nn.init.xavier_uniform_(layer.weight)
#     #         nn.init.zeros_(layer.bias)

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
