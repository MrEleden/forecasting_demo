"""Statistical forecasting models module.

This module contains traditional statistical forecasting models such as:
- ARIMA (AutoRegressive Integrated Moving Average)
- Seasonal Naive and Enhanced Seasonal Naive
- ETS (Exponential Smoothing) family models
- SVD-based cross-series forecasting
- Fourier + ARIMA (Dynamic Harmonic Regression)
- STL + ARIMA decomposition models
- M5 competition-inspired ensemble approaches
"""

from .ets import ETSForecaster, SimpleETSForecaster
from .fourier_arima import AdaptiveFourierARIMA, FourierARIMAForecaster
from .seasonal_naive import MultiSeasonalNaive, SeasonalNaiveForecaster
from .statistical import ARIMAWrapper
from .stlf_arima import RobustSTLFARIMA, STLFARIMAForecaster
from .svd_models import SVDETSForecaster, SVDSTLFForecaster

# from .walmart_ensemble import WalmartEnsembleForecaster, PerSeriesForecaster

__all__ = [
    # Original models
    "ARIMAWrapper",
    # M5-inspired standalone models
    "SeasonalNaiveForecaster",
    "MultiSeasonalNaive",
    "ETSForecaster",
    "SimpleETSForecaster",
    "SVDETSForecaster",
    "SVDSTLFForecaster",
    "FourierARIMAForecaster",
    "AdaptiveFourierARIMA",
    "STLFARIMAForecaster",
    "RobustSTLFARIMA",
    # Ensemble models
    # "WalmartEnsembleForecaster",
    # "PerSeriesForecaster",
]
