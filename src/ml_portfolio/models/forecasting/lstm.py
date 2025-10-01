"""
LSTM and Seq2Seq implementations for time series forecasting.
"""

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class LSTMForecaster(nn.Module):
        """Simple LSTM forecasting model."""

        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            output = self.fc(lstm_out[:, -1, :])
            return output

else:

    class LSTMForecaster:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LSTM models")
