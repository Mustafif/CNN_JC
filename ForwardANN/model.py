import torch
import torch.nn as nn
import torch.futures
class CaNNModel(nn.Module):
    def __init__(self, input_features=17, hidden_size=200, dropout_rate=0.0, num_hidden_layers=6):
        super().__init__()
        activation = nn.Mish()
        ln = nn.LayerNorm(hidden_size)
        # Create list of layers
        layers = []

        # Input layer
        layers.extend([
            nn.Linear(input_features, hidden_size),
            ln,
            activation,
        ])

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                ln,
                activation,
                nn.Dropout(dropout_rate)
            ])

        # Combine all hidden layers into a Sequential
        self.hidden_layers = nn.Sequential(*layers)
       # self.spc = SpectralNorm(nn.Linear(hidden_size, hidden_size))
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
           # nn.init.kaiming_uniform_(module.weight)
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
    def forward(self, x):
        x = self.hidden_layers(x)
        # x = self.spc(x)
        x = torch.nn.functional.softplus(self.output_layer(x))
        return x



class NetworkOfNetworks(nn.Module):
    def __init__(self, num_child_networks, input_features=18, hidden_size=200, dropout_rate=0.0, num_hidden_layers=6):
        super().__init__()
        # Create multiple CaNNModel instances
        self.child_networks = nn.ModuleList([
            CaNNModel(input_features, hidden_size, dropout_rate, num_hidden_layers)
            for _ in range(num_child_networks)
        ])
        # Combiner layer to merge outputs from child networks
        self.combiner = nn.Linear(num_child_networks, 1)

    def forward(self, x):
            # Process all child networks in parallel using batch processing
            child_outputs = torch.cat([child(x) for child in self.child_networks], dim=1)
            # Pass through the combiner network
            final_output = self.combiner(child_outputs)
            return final_output
