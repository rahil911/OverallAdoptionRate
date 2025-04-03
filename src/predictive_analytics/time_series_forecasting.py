"""
Time Series Forecasting Module

This module provides functions for time series forecasting of adoption rate metrics.
It implements various forecasting methods including trend extrapolation, ARIMA, ETS,
and automated method selection based on data characteristics.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Union, Optional, Tuple
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

from src.data_models.metrics import OverallAdoptionRate

# Configure logging
logger = logging.getLogger(__name__)

# Available forecasting methods
FORECAST_METHODS = ["trend", "arima", "ets"]


def select_best_forecast_method(data: List[OverallAdoptionRate], metric_type: str) -> str:
    """
    Automatically select the best forecasting method based on data characteristics.
    
    Args:
        data: List of OverallAdoptionRate objects
        metric_type: Type of metric to forecast ("daily", "weekly", "monthly", "yearly")
        
    Returns:
        str: Selected forecast method ("trend", "arima", "ets")
    """
    if not data:
        logger.warning("No data provided for method selection")
        return "trend"  # Default to simplest method
    
    # Sort data by date
    sorted_data = sorted(data, key=lambda x: x.date)
    
    # Extract the appropriate time series based on metric_type
    if metric_type == "daily":
        time_series = pd.Series([item.daily_adoption_rate for item in sorted_data],
                               index=[item.date for item in sorted_data])
    elif metric_type == "weekly":
        time_series = pd.Series([item.weekly_adoption_rate for item in sorted_data],
                               index=[item.date for item in sorted_data])
    elif metric_type == "yearly":
        time_series = pd.Series([item.yearly_adoption_rate for item in sorted_data],
                               index=[item.date for item in sorted_data])
    else:  # Default to monthly
        time_series = pd.Series([item.monthly_adoption_rate for item in sorted_data],
                               index=[item.date for item in sorted_data])
    
    # Check for minimum data requirements
    if len(time_series) < 10:
        logger.info("Not enough data points, using trend extrapolation")
        return "trend"
    
    # Check for seasonality
    try:
        # Only perform seasonal decomposition if enough data points
        if len(time_series) >= 24:  # Need at least 2*seasonal_period
            # For monthly data, use 12 as seasonal period
            seasonal_period = 12 if metric_type == "monthly" else 7 if metric_type == "daily" else 4
            decomposition = seasonal_decompose(time_series, period=seasonal_period, model='additive')
            
            # Calculate strength of seasonality
            seasonal_strength = np.std(decomposition.seasonal) / np.std(time_series)
            
            if seasonal_strength > 0.3:
                logger.info(f"Strong seasonality detected (strength: {seasonal_strength:.2f}), using ETS")
                return "ets"
        
        # Check for trend vs stationarity
        # Calculate rolling mean
        rolling_mean = time_series.rolling(window=3).mean()
        rolling_std = time_series.rolling(window=3).std()
        
        # Calculate trend strength
        trend_strength = np.abs(np.corrcoef(np.arange(len(time_series)), time_series)[0, 1])
        
        if trend_strength > 0.7:
            logger.info(f"Strong trend detected (strength: {trend_strength:.2f}), using trend extrapolation")
            return "trend"
        
        # Check for autocorrelation
        acf = np.abs(pd.Series(time_series).autocorr(lag=1))
        
        if acf > 0.7:
            logger.info(f"Strong autocorrelation detected (ACF: {acf:.2f}), using ARIMA")
            return "arima"
        
        # Default to ARIMA for general cases
        logger.info("No strong pattern detected, using ARIMA as default")
        return "arima"
        
    except Exception as e:
        logger.error(f"Error in forecast method selection: {str(e)}")
        return "trend"  # Default to simplest method in case of error


def create_time_series_forecast(data: List[OverallAdoptionRate], metric_type: str = "monthly",
                               forecast_periods: int = 12, method: str = "auto") -> Dict[str, Any]:
    """
    Create a time series forecast for adoption rate data.
    
    Args:
        data: List of OverallAdoptionRate objects
        metric_type: Type of metric to forecast ("daily", "weekly", "monthly", "yearly")
        forecast_periods: Number of periods to forecast
        method: Forecasting method to use ("auto", "trend", "arima", "ets")
        
    Returns:
        dict: Dictionary containing forecast results
    """
    if not data:
        logger.warning("No data provided for forecasting")
        return {
            "forecast_values": [],
            "forecast_dates": [],
            "confidence_intervals": {"lower": [], "upper": []},
            "model_metrics": {},
            "model_info": {},
            "trend_info": {}
        }
    
    # Sort data by date
    sorted_data = sorted(data, key=lambda x: x.date)
    
    # Extract the appropriate time series based on metric_type
    if metric_type == "daily":
        values = [item.daily_adoption_rate for item in sorted_data]
        dates = [item.date for item in sorted_data]
        rate_field = "daily_adoption_rate"
        frequency = "D"
        period_days = 1
    elif metric_type == "weekly":
        values = [item.weekly_adoption_rate for item in sorted_data]
        dates = [item.date for item in sorted_data]
        rate_field = "weekly_adoption_rate"
        frequency = "W"
        period_days = 7
    elif metric_type == "yearly":
        values = [item.yearly_adoption_rate for item in sorted_data]
        dates = [item.date for item in sorted_data]
        rate_field = "yearly_adoption_rate"
        frequency = "Y"
        period_days = 365
    else:  # Default to monthly
        values = [item.monthly_adoption_rate for item in sorted_data]
        dates = [item.date for item in sorted_data]
        rate_field = "monthly_adoption_rate"
        frequency = "M"
        period_days = 30
    
    # Create pandas Series
    time_series = pd.Series(values, index=dates)
    
    # Automatically select method if specified
    if method == "auto":
        method = select_best_forecast_method(sorted_data, metric_type)
    
    # Check if method is valid
    if method not in FORECAST_METHODS:
        logger.warning(f"Invalid forecast method: {method}, falling back to trend")
        method = "trend"
    
    try:
        # Call the appropriate forecasting method
        if method == "trend":
            result = _forecast_with_trend(time_series, forecast_periods, period_days)
        elif method == "arima":
            result = _forecast_with_arima(time_series, forecast_periods, period_days)
        elif method == "ets":
            result = _forecast_with_ets(time_series, forecast_periods, period_days)
        else:
            result = _forecast_with_trend(time_series, forecast_periods, period_days)
        
        # Add trend information
        trend_info = _analyze_forecast_trend(time_series, result["forecast_values"])
        result["trend_info"] = trend_info
        
        return result
    
    except Exception as e:
        logger.error(f"Error in time series forecasting: {str(e)}")
        # Return a simple trend forecast as fallback
        return _forecast_with_trend(time_series, forecast_periods, period_days)


def _forecast_with_trend(time_series: pd.Series, forecast_periods: int, 
                        period_days: int) -> Dict[str, Any]:
    """
    Create a forecast using polynomial trend extrapolation.
    
    Args:
        time_series: Time series data as pandas Series
        forecast_periods: Number of periods to forecast
        period_days: Number of days in each period
        
    Returns:
        dict: Dictionary containing forecast results
    """
    # Convert index to numeric (days since first date)
    if isinstance(time_series.index[0], (datetime, date)):
        first_date = time_series.index[0]
        x = np.array([(d - first_date).days for d in time_series.index])
    else:
        x = np.array(range(len(time_series)))
        first_date = datetime.now().date() - timedelta(days=len(time_series) * period_days)
    
    y = time_series.values
    
    # Fit polynomial of degree 2 (quadratic) or 1 (linear) based on data size
    degree = 2 if len(time_series) >= 12 else 1
    coefficients = np.polyfit(x, y, degree)
    
    # Generate forecast dates
    last_date = time_series.index[-1] if isinstance(time_series.index[-1], (datetime, date)) else first_date + timedelta(days=x[-1])
    forecast_dates = [last_date + timedelta(days=(i+1)*period_days) for i in range(forecast_periods)]
    
    # Generate x values for forecast
    forecast_x = np.array([x[-1] + (i+1)*period_days for i in range(forecast_periods)])
    
    # Calculate forecast values
    forecast_values = np.polyval(coefficients, forecast_x)
    
    # Ensure values are within valid range (0-100%)
    forecast_values = np.clip(forecast_values, 0, 100)
    
    # Calculate confidence intervals
    residuals = y - np.polyval(coefficients, x)
    residual_std = np.std(residuals)
    
    # Calculate confidence intervals (95%)
    confidence_width = 1.96 * residual_std
    lower_bounds = forecast_values - confidence_width
    upper_bounds = forecast_values + confidence_width
    
    # Ensure bounds are within valid range
    lower_bounds = np.clip(lower_bounds, 0, 100)
    upper_bounds = np.clip(upper_bounds, 0, 100)
    
    # Calculate fit metrics
    ss_residual = np.sum(residuals**2)
    ss_total = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    
    return {
        "forecast_values": forecast_values.tolist(),
        "forecast_dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in forecast_dates],
        "confidence_intervals": {
            "lower": lower_bounds.tolist(),
            "upper": upper_bounds.tolist()
        },
        "model_metrics": {
            "r_squared": r_squared,
            "residual_std": residual_std,
            "mean_absolute_error": np.mean(np.abs(residuals))
        },
        "model_info": {
            "method": "trend",
            "degree": degree,
            "coefficients": coefficients.tolist()
        }
    }


def _forecast_with_arima(time_series: pd.Series, forecast_periods: int, 
                        period_days: int) -> Dict[str, Any]:
    """
    Create a forecast using ARIMA model.
    
    Args:
        time_series: Time series data as pandas Series
        forecast_periods: Number of periods to forecast
        period_days: Number of days in each period
        
    Returns:
        dict: Dictionary containing forecast results
    """
    from statsmodels.tsa.arima.model import ARIMA
    
    # Ensure time series is properly indexed
    if not isinstance(time_series.index, pd.DatetimeIndex):
        if isinstance(time_series.index[0], (datetime, date)):
            time_series.index = pd.DatetimeIndex(time_series.index)
        else:
            # Create a date range for the index
            first_date = datetime.now().date() - timedelta(days=len(time_series) * period_days)
            date_range = pd.date_range(start=first_date, periods=len(time_series), freq=f"{period_days}D")
            time_series.index = date_range
    
    # Choose order based on data characteristics (simplified)
    # In a more sophisticated implementation, we would use AIC/BIC for model selection
    if len(time_series) >= 24:
        order = (2, 1, 2)  # p, d, q
    elif len(time_series) >= 12:
        order = (1, 1, 1)
    else:
        order = (1, 0, 0)  # AR(1) for short series
    
    try:
        # Fit ARIMA model
        model = ARIMA(time_series, order=order)
        model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.forecast(steps=forecast_periods)
        forecast_values = forecast.values
        
        # Ensure values are within valid range (0-100%)
        forecast_values = np.clip(forecast_values, 0, 100)
        
        # Generate forecast dates
        last_date = time_series.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=period_days), 
                                     periods=forecast_periods, freq=f"{period_days}D")
        
        # Get confidence intervals (95%)
        pred_conf = model_fit.get_forecast(steps=forecast_periods).conf_int(alpha=0.05)
        lower_bounds = pred_conf.iloc[:, 0].values
        upper_bounds = pred_conf.iloc[:, 1].values
        
        # Ensure bounds are within valid range
        lower_bounds = np.clip(lower_bounds, 0, 100)
        upper_bounds = np.clip(upper_bounds, 0, 100)
        
        # Calculate model metrics
        residuals = model_fit.resid
        mean_abs_error = np.mean(np.abs(residuals))
        
        return {
            "forecast_values": forecast_values.tolist(),
            "forecast_dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
            "confidence_intervals": {
                "lower": lower_bounds.tolist(),
                "upper": upper_bounds.tolist()
            },
            "model_metrics": {
                "aic": model_fit.aic,
                "bic": model_fit.bic,
                "residual_std": np.std(residuals),
                "mean_absolute_error": mean_abs_error
            },
            "model_info": {
                "method": "arima",
                "order": order
            }
        }
        
    except Exception as e:
        logger.error(f"Error in ARIMA forecast: {str(e)}")
        # Fall back to trend forecast
        return _forecast_with_trend(time_series, forecast_periods, period_days)


def _forecast_with_ets(time_series: pd.Series, forecast_periods: int, 
                      period_days: int) -> Dict[str, Any]:
    """
    Create a forecast using Exponential Smoothing (ETS) model.
    
    Args:
        time_series: Time series data as pandas Series
        forecast_periods: Number of periods to forecast
        period_days: Number of days in each period
        
    Returns:
        dict: Dictionary containing forecast results
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    # Ensure time series is properly indexed
    if not isinstance(time_series.index, pd.DatetimeIndex):
        if isinstance(time_series.index[0], (datetime, date)):
            time_series.index = pd.DatetimeIndex(time_series.index)
        else:
            # Create a date range for the index
            first_date = datetime.now().date() - timedelta(days=len(time_series) * period_days)
            date_range = pd.date_range(start=first_date, periods=len(time_series), freq=f"{period_days}D")
            time_series.index = date_range
    
    # Determine seasonality period
    if period_days == 1:  # Daily data
        seasonal_periods = 7  # Weekly seasonality
    elif period_days == 7:  # Weekly data
        seasonal_periods = 52  # Annual seasonality
    elif period_days == 30:  # Monthly data
        seasonal_periods = 12  # Annual seasonality
    else:  # Yearly or other
        seasonal_periods = 0  # No seasonality
    
    # Check if we have enough data for seasonality
    use_seasonal = len(time_series) >= 2 * seasonal_periods and seasonal_periods > 0
    
    try:
        # Fit ETS model
        if use_seasonal:
            model = ExponentialSmoothing(
                time_series, 
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods,
                damped=True
            )
        else:
            model = ExponentialSmoothing(
                time_series,
                trend='add',
                damped=True
            )
            
        model_fit = model.fit(optimized=True)
        
        # Generate forecast
        forecast = model_fit.forecast(steps=forecast_periods)
        forecast_values = forecast.values
        
        # Ensure values are within valid range (0-100%)
        forecast_values = np.clip(forecast_values, 0, 100)
        
        # Generate forecast dates
        last_date = time_series.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=period_days), 
                                     periods=forecast_periods, freq=f"{period_days}D")
        
        # Calculate confidence intervals based on prediction errors
        residuals = model_fit.resid
        residual_std = np.std(residuals)
        
        # Calculate 95% confidence intervals
        z_value = 1.96  # 95% confidence interval
        margin = z_value * residual_std
        
        lower_bounds = forecast_values - margin
        upper_bounds = forecast_values + margin
        
        # Ensure bounds are within valid range
        lower_bounds = np.clip(lower_bounds, 0, 100)
        upper_bounds = np.clip(upper_bounds, 0, 100)
        
        # Calculate model metrics
        mean_abs_error = np.mean(np.abs(residuals))
        
        return {
            "forecast_values": forecast_values.tolist(),
            "forecast_dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
            "confidence_intervals": {
                "lower": lower_bounds.tolist(),
                "upper": upper_bounds.tolist()
            },
            "model_metrics": {
                "aic": model_fit.aic,
                "bic": model_fit.bic,
                "residual_std": residual_std,
                "mean_absolute_error": mean_abs_error
            },
            "model_info": {
                "method": "ets",
                "seasonal": use_seasonal,
                "seasonal_periods": seasonal_periods if use_seasonal else None,
                "params": {
                    "smoothing_level": model_fit.params.get('smoothing_level', None),
                    "smoothing_trend": model_fit.params.get('smoothing_trend', None),
                    "damping_trend": model_fit.params.get('damping_trend', None)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error in ETS forecast: {str(e)}")
        # Fall back to trend forecast
        return _forecast_with_trend(time_series, forecast_periods, period_days)


def _analyze_forecast_trend(historical_series: pd.Series, forecast_values: List[float]) -> Dict[str, Any]:
    """
    Analyze the trend in the forecast values.
    
    Args:
        historical_series: Historical time series data
        forecast_values: Forecasted values
        
    Returns:
        dict: Dictionary with trend analysis information
    """
    if not forecast_values:
        return {
            "direction": "unknown",
            "strength": "unknown",
            "seasonality": "unknown"
        }
    
    # Calculate overall trend direction
    first_value = forecast_values[0]
    last_value = forecast_values[-1]
    trend_pct_change = ((last_value - first_value) / first_value * 100) if first_value > 0 else 0
    
    # Determine trend direction
    if trend_pct_change > 5:
        direction = "increasing"
    elif trend_pct_change < -5:
        direction = "decreasing"
    else:
        direction = "stable"
    
    # Determine trend strength
    if abs(trend_pct_change) > 20:
        strength = "strong"
    elif abs(trend_pct_change) > 10:
        strength = "moderate"
    else:
        strength = "weak"
    
    # Check for seasonality in forecast
    # We'll use a simple approach by looking at the deviations from the linear trend
    x = np.arange(len(forecast_values))
    coeffs = np.polyfit(x, forecast_values, 1)
    trend_line = np.polyval(coeffs, x)
    deviations = np.array(forecast_values) - trend_line
    
    # Calculate autocorrelation of deviations
    if len(deviations) > 3:
        from statsmodels.tsa.stattools import acf
        try:
            acf_values = acf(deviations, nlags=min(len(deviations)//2, 10), fft=True)
            max_acf = np.max(np.abs(acf_values[1:]))  # Exclude lag 0
            
            if max_acf > 0.5:
                seasonality = "significant"
            elif max_acf > 0.3:
                seasonality = "moderate"
            else:
                seasonality = "minimal"
        except:
            seasonality = "unknown"
    else:
        seasonality = "unknown"
    
    return {
        "direction": direction,
        "strength": strength,
        "percent_change": trend_pct_change,
        "seasonality": seasonality
    } 