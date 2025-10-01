"""
Transformer models for long sequence forecasting.
"""

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class TransformerForecaster(nn.Module):
        """Simple Transformer forecasting model."""

        def __init__(self, input_size, d_model, nhead, num_layers, output_size):
            super().__init__()
            self.input_projection = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.fc = nn.Linear(d_model, output_size)

        def forward(self, x):
            x = self.input_projection(x)
            transformer_out = self.transformer(x)
            output = self.fc(transformer_out[:, -1, :])
            return output

else:

    class TransformerForecaster:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for Transformer models")
