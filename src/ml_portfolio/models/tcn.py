"""
Temporal Convolutional Network (TCN) for time series forecasting.
"""

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class TCNForecaster(nn.Module):
        """Simple TCN forecasting model."""

        def __init__(self, input_size, num_channels, kernel_size, output_size, dropout=0.1):
            super().__init__()
            layers = []
            num_levels = len(num_channels)

            for i in range(num_levels):
                dilation_size = 2**i
                in_channels = input_size if i == 0 else num_channels[i - 1]
                out_channels = num_channels[i]

                layers.append(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation_size,
                        padding=(kernel_size - 1) * dilation_size,
                    )
                )
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

            self.network = nn.Sequential(*layers)
            self.fc = nn.Linear(num_channels[-1], output_size)

        def forward(self, x):
            # x should be (batch, time, features) -> (batch, features, time)
            x = x.transpose(1, 2)
            tcn_out = self.network(x)
            # Take the last time step
            output = self.fc(tcn_out[:, :, -1])
            return output

else:

    class TCNForecaster:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TCN models")
