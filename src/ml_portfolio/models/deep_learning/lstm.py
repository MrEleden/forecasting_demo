"""""""""

LSTM implementation for time series forecasting using the new base model structure.

"""LSTM implementation for time series forecasting using the new base model structure.LSTM implementation for time series forecasting using the new base model structure.



import torch""""""

import torch.nn as nn

import numpy as np

from typing import Optional, Union

import pandas as pdimport torchimport torch

from sklearn.preprocessing import StandardScaler

import torch.nn as nnimport torch.nn as nn

from ..base import PyTorchForecaster

import numpy as npimport numpy as np



class LSTMCore(nn.Module):from typing import Optional, Unionfrom typing import Optional, Union

    """Core LSTM network."""

import pandas as pdimport pandas as pd

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):

        super().__init__()from sklearn.preprocessing import StandardScalerfrom sklearn.preprocessing import StandardScaler

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_size, output_size)



    def forward(self, x):from ..base import PyTorchForecasterfrom ..base import PyTorchForecaster

        lstm_out, _ = self.lstm(x)

        output = self.fc(lstm_out[:, -1, :])

        return output





class LSTMForecaster(PyTorchForecaster):class LSTMCore(nn.Module):class LSTMCore(nn.Module):

    """

    LSTM forecaster inheriting from PyTorchForecaster base class.    """Core LSTM network."""    """Core LSTM network."""



    This implementation provides a clean sklearn-compatible interface for LSTM models.

    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):

    def __init__(

        self,        super().__init__()        super().__init__()

        input_size=1,

        hidden_size=64,        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        num_layers=2,

        output_size=1,        self.fc = nn.Linear(hidden_size, output_size)        self.fc = nn.Linear(hidden_size, output_size)

        dropout=0.2,

        lr=0.001,

        epochs=100,

        batch_size=32,    def forward(self, x):    def forward(self, x):

        device='auto',

        **kwargs        lstm_out, _ = self.lstm(x)        lstm_out, _ = self.lstm(x)

    ):

        super().__init__(device=device, **kwargs)        output = self.fc(lstm_out[:, -1, :])        output = self.fc(lstm_out[:, -1, :])



        self.input_size = input_size        return output        return output

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.output_size = output_size

        self.dropout = dropout

        self.lr = lr

        self.epochs = epochsclass LSTMForecaster(PyTorchForecaster):class LSTMForecaster(PyTorchForecaster):

        self.batch_size = batch_size

            """    """

        # Initialize scaler

        self.scaler_ = StandardScaler()    LSTM forecaster inheriting from PyTorchForecaster base class.    LSTM forecaster inheriting from PyTorchForecaster base class.



    def _create_model(self):

        """Create the LSTM model."""

        model = LSTMCore(    This implementation provides a clean sklearn-compatible interface for LSTM models.    This implementation provides a clean sklearn-compatible interface for LSTM models.

            input_size=self.input_size,

            hidden_size=self.hidden_size,    """    """

            num_layers=self.num_layers,

            output_size=self.output_size,

            dropout=self.dropout

        )    def __init__(    def __init__(

        return model.to(self.device)

        self,        self,

    def fit(self, X, y):

        """Fit the LSTM model."""        input_size=1,        input_size=1,

        # Validate input

        X, y = self._validate_input(X, y)        hidden_size=64,        hidden_size=64,



        # Update input size based on actual data        num_layers=2,        num_layers=2,

        if len(X.shape) == 3:

            self.input_size = X.shape[2]        output_size=1,        output_size=1,

        else:

            # Assume 2D input, reshape to 3D        dropout=0.2,        dropout=0.2,

            X = X.reshape(X.shape[0], 1, X.shape[1])

            self.input_size = X.shape[2]        lr=0.001,        lr=0.001,



        # Scale the data        epochs=100,        epochs=100,

        X_scaled = self.scaler_.fit_transform(X.reshape(-1, X.shape[-1]))

        X_scaled = X_scaled.reshape(X.shape)        batch_size=32,        batch_size=32,



        # Create model        device='auto',        device='auto',

        self.model_ = self._create_model()

                **kwargs        **kwargs

        # Setup training

        self.criterion_ = nn.MSELoss()    ):    ):

        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

                """        """

        # Prepare data

        X_tensor, y_tensor = self._prepare_data(X_scaled, y)        Initialize LSTM forecaster.        Initialize LSTM forecaster.



        # Training loop

        self.model_.train()

        for epoch in range(self.epochs):        Args:        Args:

            # Create batches

            for i in range(0, len(X_tensor), self.batch_size):            input_size: Number of input features            input_size: Number of input features

                batch_X = X_tensor[i:i+self.batch_size]

                batch_y = y_tensor[i:i+self.batch_size]            hidden_size: LSTM hidden size            hidden_size: LSTM hidden size



                # Forward pass            num_layers: Number of LSTM layers            num_layers: Number of LSTM layers

                outputs = self.model_(batch_X)

                loss = self.criterion_(outputs, batch_y)            output_size: Number of output features            output_size: Number of output features



                # Backward pass            dropout: Dropout rate            dropout: Dropout rate

                self.optimizer_.zero_grad()

                loss.backward()            lr: Learning rate            lr: Learning rate

                self.optimizer_.step()

                        epochs: Number of training epochs            epochs: Number of training epochs

            # Print progress occasionally

            if (epoch + 1) % max(1, self.epochs // 10) == 0:            batch_size: Batch size for training            batch_size: Batch size for training

                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')

                    device: Device to use ('cpu', 'cuda', 'auto')            device: Device to use ('cpu', 'cuda', 'auto')

        self.is_fitted_ = True

        return self        """        """



    def predict(self, X):        super().__init__(device=device, **kwargs)        super().__init__(device=device, **kwargs)

        """Make predictions using the fitted LSTM model."""

        self._check_is_fitted()



        # Prepare input        self.input_size = input_size        self.input_size = input_size

        if isinstance(X, pd.DataFrame):

            X = X.values        self.hidden_size = hidden_size        self.hidden_size = hidden_size



        if len(X.shape) == 2:        self.num_layers = num_layers        self.num_layers = num_layers

            X = X.reshape(X.shape[0], 1, X.shape[1])

                self.output_size = output_size        self.output_size = output_size

        # Scale input

        X_scaled = self.scaler_.transform(X.reshape(-1, X.shape[-1]))        self.dropout = dropout        self.dropout = dropout

        X_scaled = X_scaled.reshape(X.shape)

                self.lr = lr        self.lr = lr

        # Prepare tensor

        X_tensor = self._prepare_data(X_scaled)        self.epochs = epochs        self.epochs = epochs



        # Make predictions        self.batch_size = batch_size        self.batch_size = batch_size

        self.model_.eval()

        with torch.no_grad():

            predictions = self.model_(X_tensor)

            predictions = predictions.cpu().numpy()        # Initialize scaler        # Initialize scaler



        return predictions        self.scaler_ = StandardScaler()        self.scaler_ = StandardScaler()



    def _create_model(self):    def _create_model(self):

        """Create the LSTM model."""        """Create the LSTM model."""

        model = LSTMCore(        model = LSTMCore(

            input_size=self.input_size,            input_size=self.input_size,

            hidden_size=self.hidden_size,            hidden_size=self.hidden_size,

            num_layers=self.num_layers,            num_layers=self.num_layers,

            output_size=self.output_size,            output_size=self.output_size,

            dropout=self.dropout            dropout=self.dropout

        )        )

        return model.to(self.device)        return model.to(self.device)



    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'LSTMForecaster':    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'LSTMForecaster':

        """        """

        Fit the LSTM model.        Fit the LSTM model.



        Args:        Args:

            X: Input features of shape (n_samples, seq_length, n_features)            X: Input features of shape (n_samples, seq_length, n_features)

            y: Target values of shape (n_samples, n_targets)            y: Target values of shape (n_samples, n_targets)



        Returns:        Returns:

            self: Fitted estimator            self: Fitted estimator

        """        """

        # Validate input        # Validate input

        X, y = self._validate_input(X, y)        X, y = self._validate_input(X, y)



        # Update input size based on actual data        # Update input size based on actual data

        if len(X.shape) == 3:        if len(X.shape) == 3:

            self.input_size = X.shape[2]            self.input_size = X.shape[2]

        else:        else:

            # Assume 2D input, reshape to 3D            # Assume 2D input, reshape to 3D

            X = X.reshape(X.shape[0], 1, X.shape[1])            X = X.reshape(X.shape[0], 1, X.shape[1])

            self.input_size = X.shape[2]            self.input_size = X.shape[2]



        # Scale the data        # Scale the data

        X_scaled = self.scaler_.fit_transform(X.reshape(-1, X.shape[-1]))        X_scaled = self.scaler_.fit_transform(X.reshape(-1, X.shape[-1]))

        X_scaled = X_scaled.reshape(X.shape)        X_scaled = X_scaled.reshape(X.shape)



        # Create model        # Create model

        self.model_ = self._create_model()        self.model_ = self._create_model()



        # Setup training        # Setup training

        self.criterion_ = nn.MSELoss()        self.criterion_ = nn.MSELoss()

        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.lr)        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.lr)



        # Prepare data        # Prepare data

        X_tensor, y_tensor = self._prepare_data(X_scaled, y)        X_tensor, y_tensor = self._prepare_data(X_scaled, y)



        # Training loop        # Training loop

        self.model_.train()        self.model_.train()

        for epoch in range(self.epochs):        for epoch in range(self.epochs):

            # Create batches            # Create batches

            for i in range(0, len(X_tensor), self.batch_size):            for i in range(0, len(X_tensor), self.batch_size):

                batch_X = X_tensor[i:i+self.batch_size]                batch_X = X_tensor[i:i+self.batch_size]

                batch_y = y_tensor[i:i+self.batch_size]                batch_y = y_tensor[i:i+self.batch_size]



                # Forward pass                # Forward pass

                outputs = self.model_(batch_X)                outputs = self.model_(batch_X)

                loss = self.criterion_(outputs, batch_y)                loss = self.criterion_(outputs, batch_y)



                # Backward pass                # Backward pass

                self.optimizer_.zero_grad()                self.optimizer_.zero_grad()

                loss.backward()                loss.backward()

                self.optimizer_.step()                self.optimizer_.step()



            # Print progress occasionally            # Print progress occasionally

            if (epoch + 1) % max(1, self.epochs // 10) == 0:            if (epoch + 1) % (self.epochs // 10) == 0:

                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')



        self.is_fitted_ = True        self.is_fitted_ = True

        return self        return self



    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:

        """        """

        Make predictions using the fitted LSTM model.        Make predictions using the fitted LSTM model.



        Args:        Args:

            X: Input features of shape (n_samples, seq_length, n_features)            X: Input features of shape (n_samples, seq_length, n_features)



        Returns:        Returns:

            Predictions of shape (n_samples, n_targets)            Predictions of shape (n_samples, n_targets)

        """        """

        self._check_is_fitted()        self._check_is_fitted()



        # Prepare input        # Prepare input

        if isinstance(X, pd.DataFrame):        if isinstance(X, pd.DataFrame):

            X = X.values            X = X.values



        if len(X.shape) == 2:        if len(X.shape) == 2:

            X = X.reshape(X.shape[0], 1, X.shape[1])            X = X.reshape(X.shape[0], 1, X.shape[1])



        # Scale input        # Scale input

        X_scaled = self.scaler_.transform(X.reshape(-1, X.shape[-1]))        X_scaled = self.scaler_.transform(X.reshape(-1, X.shape[-1]))

        X_scaled = X_scaled.reshape(X.shape)        X_scaled = X_scaled.reshape(X.shape)



        # Prepare tensor        # Prepare tensor

        X_tensor = self._prepare_data(X_scaled)        X_tensor = self._prepare_data(X_scaled)



        # Make predictions        # Make predictions

        self.model_.eval()        self.model_.eval()

        with torch.no_grad():        with torch.no_grad():

            predictions = self.model_(X_tensor)            predictions = self.model_(X_tensor)

            predictions = predictions.cpu().numpy()            predictions = predictions.cpu().numpy()



        return predictions        return predictions



    def forecast(self, steps: int, last_values: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> np.ndarray:    def forecast(self, steps: int, last_values: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> np.ndarray:

        """        """

        Generate multi-step ahead forecasts.        Generate multi-step ahead forecasts.



        Args:        Args:

            steps: Number of steps to forecast ahead            steps: Number of steps to forecast ahead

            last_values: Last known values to use for forecasting            last_values: Last known values to use for forecasting



        Returns:        Returns:

            Forecasts of shape (steps, n_targets)            Forecasts of shape (steps, n_targets)

        """        """

        self._check_is_fitted()        self._check_is_fitted()



        if last_values is None:        if last_values is None:

            raise ValueError("last_values must be provided for LSTM forecasting")            raise ValueError("last_values must be provided for LSTM forecasting")



        forecasts = []        forecasts = []

        current_input = last_values.copy()        current_input = last_values.copy()



        for _ in range(steps):        for _ in range(steps):

            # Make prediction for next step            # Make prediction for next step

            next_pred = self.predict(current_input.reshape(1, -1, current_input.shape[-1]))            next_pred = self.predict(current_input.reshape(1, -1, current_input.shape[-1]))

            forecasts.append(next_pred[0])            forecasts.append(next_pred[0])



            # Update input for next prediction (sliding window)            # Update input for next prediction (sliding window)

            if len(current_input.shape) == 2:            if len(current_input.shape) == 2:

                current_input = np.roll(current_input, -1, axis=0)                current_input = np.roll(current_input, -1, axis=0)

                current_input[-1] = next_pred[0]                current_input[-1] = next_pred[0]

            else:            else:

                current_input = np.roll(current_input, -1, axis=1)                current_input = np.roll(current_input, -1, axis=1)

                current_input[:, -1] = next_pred[0]                current_input[:, -1] = next_pred[0]



        return np.array(forecasts)        return np.array(forecasts)
            output_size: Number of output features (forecast horizon)
            dropout: Dropout rate
            lr: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device to run on (auto-detected if None)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        # Auto-detect device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = LSTMCore(input_size, hidden_size, num_layers, output_size, dropout)
        self.model.to(self.device)

        # Training components
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def _prepare_sequences(self, X, y=None):
        """
        Convert flattened features back to sequences for LSTM.

        Args:
            X: Flattened features (batch_size, sequence_length)
            y: Targets (batch_size, forecast_horizon) - optional

        Returns:
            X_sequences: (batch_size, sequence_length, 1)
            y_sequences: (batch_size, forecast_horizon) - if y provided
        """
        # Reshape X back to sequences
        X_sequences = X.reshape(X.shape[0], -1, self.input_size)

        if y is not None:
            return X_sequences, y
        return X_sequences

    def fit(self, X, y):
        """
        Fit the LSTM model.

        Args:
            X: Training features (batch_size, sequence_length)
            y: Training targets (batch_size, forecast_horizon)
        """
        print(f"Training LSTM on device: {self.device}")

        # Prepare data
        X_seq, y_targets = self._prepare_sequences(X, y)

        # Scale data
        X_scaled = self.scaler_X.fit_transform(X_seq.reshape(-1, self.input_size)).reshape(X_seq.shape)
        y_scaled = self.scaler_y.fit_transform(y_targets.reshape(-1, 1)).reshape(y_targets.shape)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1:3d}/{self.epochs}: Loss = {loss.item():.6f}")

        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions with the LSTM model.

        Args:
            X: Features (batch_size, sequence_length)

        Returns:
            predictions: (batch_size, forecast_horizon)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Prepare data
        X_seq = self._prepare_sequences(X)

        # Scale data
        X_scaled = self.scaler_X.transform(X_seq.reshape(-1, self.input_size)).reshape(X_seq.shape)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)

        # Convert back to numpy and inverse transform
        predictions_scaled = outputs.cpu().numpy()
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(
            predictions_scaled.shape
        )

        return predictions

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for param, value in params.items():
            setattr(self, param, value)
        return self
