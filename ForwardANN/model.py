import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for feature interaction"""
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        # Linear transformations and split into heads
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Output projection and residual connection
        output = self.w_o(context)
        return self.layer_norm(output + x)


class ResidualBlock(nn.Module):
    """Residual block with layer normalization and dropout"""
    def __init__(self, hidden_size, dropout_rate=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 2)
        self.linear2 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.Mish()

    def forward(self, x):
        # First sublayer: feedforward network
        residual = x
        x = self.layer_norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = x + residual

        return self.layer_norm2(x)


class CaNNModel(nn.Module):
    def __init__(self, input_features=38, hidden_size=200, dropout_rate=0.0, num_hidden_layers=6):
        super().__init__()

        # Input projection with layer normalization
        self.input_projection = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )

        # Feature attention for input features
        self.feature_attention = nn.Sequential(
            nn.Linear(input_features, input_features),
            nn.Sigmoid()
        )

        # Self-attention mechanism
        self.self_attention = MultiHeadAttention(hidden_size, num_heads=4, dropout=dropout_rate)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_rate)
            for _ in range(num_hidden_layers)
        ])

        # Volatility-specific layers
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier initialization for volatility prediction
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        # Apply feature attention
        attention_weights = self.feature_attention(x)
        x = x * attention_weights

        # Input projection
        x = self.input_projection(x)

        # Add sequence dimension for self-attention
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # Apply self-attention
        x = self.self_attention(x)

        # Remove sequence dimension
        x = x.squeeze(1)  # [batch_size, hidden_size]

        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Final volatility prediction with constrained output
        x = self.volatility_head(x)

        # Ensure positive volatility output with improved activation
        x = F.softplus(x) + 1e-6  # Small epsilon to avoid zero volatility

        return x


class NetworkOfNetworks(nn.Module):
    def __init__(self, num_child_networks=3, input_features=38, hidden_size=200, dropout_rate=0.0, num_hidden_layers=6):
        super().__init__()
        # Create multiple CaNNModel instances with slight variations
        # Ensure hidden sizes are divisible by attention heads (4)
        self.child_networks = nn.ModuleList([
            CaNNModel(input_features, hidden_size + (i * 12), dropout_rate, num_hidden_layers)  # 12 is divisible by 4
            for i in range(num_child_networks)
        ])

        # Enhanced combiner with attention mechanism
        self.combiner_attention = nn.Sequential(
            nn.Linear(num_child_networks, num_child_networks),
            nn.Softmax(dim=1)
        )

        # Final combination layer
        self.combiner = nn.Sequential(
            nn.Linear(num_child_networks, num_child_networks // 2 if num_child_networks > 2 else 1),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_child_networks // 2 if num_child_networks > 2 else 1, 1)
        ) if num_child_networks > 2 else nn.Linear(num_child_networks, 1)

    def forward(self, x):
        # Process all child networks in parallel
        child_outputs = torch.cat([child(x) for child in self.child_networks], dim=1)

        # Apply attention-weighted combination
        attention_weights = self.combiner_attention(child_outputs)
        weighted_outputs = child_outputs * attention_weights

        # Final combination
        final_output = self.combiner(weighted_outputs)

        # Ensure positive output for volatility
        final_output = F.softplus(final_output) + 1e-6

        return final_output
