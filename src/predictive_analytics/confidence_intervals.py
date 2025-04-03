"""
Confidence Intervals Module

This module provides functions for calculating confidence intervals for forecasts,
including methods for different types of forecasting models.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)


def calculate_forecast_intervals(
    forecast_values: List[float],
    model_info: Dict[str, Any],
    confidence_level: float = 0.95
) -> Dict[str, List[float]]:
    """
    Calculate confidence intervals for forecast values.
    
    Args:
        forecast_values: List of forecasted values
        model_info: Dictionary with model information from the forecasting method
        confidence_level: Confidence level (0-1), defaults to 0.95 (95%)
        
    Returns:
        dict: Dictionary with lower and upper bounds for the confidence intervals
    """
    if not forecast_values:
        return {"lower": [], "upper": []}
    
    # Get forecast method
    method = model_info.get("method", "trend")
    
    try:
        # Call the appropriate interval calculation method
        if method == "trend":
            return _calculate_trend_intervals(forecast_values, model_info, confidence_level)
        elif method == "arima":
            return _calculate_arima_intervals(forecast_values, model_info, confidence_level)
        elif method == "ets":
            return _calculate_ets_intervals(forecast_values, model_info, confidence_level)
        else:
            # Default to simple intervals based on standard deviation
            return _calculate_simple_intervals(forecast_values, confidence_level)
            
    except Exception as e:
        logger.error(f"Error calculating confidence intervals: {str(e)}")
        # Fall back to simple intervals
        return _calculate_simple_intervals(forecast_values, confidence_level)


def _calculate_trend_intervals(
    forecast_values: List[float],
    model_info: Dict[str, Any],
    confidence_level: float = 0.95
) -> Dict[str, List[float]]:
    """
    Calculate confidence intervals for trend-based forecasts.
    
    Args:
        forecast_values: List of forecasted values
        model_info: Dictionary with model information from trend forecasting
        confidence_level: Confidence level (0-1)
        
    Returns:
        dict: Dictionary with lower and upper bounds
    """
    # Extract residual standard deviation from model info
    residual_std = model_info.get("residual_std", None)
    
    if residual_std is None and "model_metrics" in model_info:
        # Try to get it from model_metrics
        residual_std = model_info["model_metrics"].get("residual_std", None)
    
    if residual_std is None:
        # If still not found, use the standard deviation of the forecast values
        residual_std = np.std(forecast_values) if len(forecast_values) > 1 else 1.0
    
    # Calculate z-value for the given confidence level
    # For 95% confidence, z = 1.96
    z_value = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Calculate margin of error
    margin = z_value * residual_std
    
    # Calculate bounds
    lower_bounds = [max(0, val - margin) for val in forecast_values]
    upper_bounds = [min(100, val + margin) for val in forecast_values]
    
    return {
        "lower": lower_bounds,
        "upper": upper_bounds
    }


def _calculate_arima_intervals(
    forecast_values: List[float],
    model_info: Dict[str, Any],
    confidence_level: float = 0.95
) -> Dict[str, List[float]]:
    """
    Calculate confidence intervals for ARIMA forecasts.
    
    Args:
        forecast_values: List of forecasted values
        model_info: Dictionary with model information from ARIMA forecasting
        confidence_level: Confidence level (0-1)
        
    Returns:
        dict: Dictionary with lower and upper bounds
    """
    # For ARIMA forecasts, the error variance increases with the forecast horizon
    residual_std = model_info.get("residual_std", None)
    
    if residual_std is None and "model_metrics" in model_info:
        # Try to get it from model_metrics
        residual_std = model_info["model_metrics"].get("residual_std", None)
    
    if residual_std is None:
        # If still not found, use the standard deviation of the forecast values
        residual_std = np.std(forecast_values) if len(forecast_values) > 1 else 1.0
    
    # Calculate z-value for the given confidence level
    z_value = stats.norm.ppf((1 + confidence_level) / 2)
    
    # For ARIMA, the prediction error grows with the horizon
    # We'll use a simple approach where the standard error increases with sqrt(h)
    # where h is the forecast horizon
    horizons = np.arange(1, len(forecast_values) + 1)
    margins = z_value * residual_std * np.sqrt(horizons)
    
    # Calculate bounds
    lower_bounds = [max(0, forecast_values[i] - margins[i]) for i in range(len(forecast_values))]
    upper_bounds = [min(100, forecast_values[i] + margins[i]) for i in range(len(forecast_values))]
    
    return {
        "lower": lower_bounds,
        "upper": upper_bounds
    }


def _calculate_ets_intervals(
    forecast_values: List[float],
    model_info: Dict[str, Any],
    confidence_level: float = 0.95
) -> Dict[str, List[float]]:
    """
    Calculate confidence intervals for ETS (Exponential Smoothing) forecasts.
    
    Args:
        forecast_values: List of forecasted values
        model_info: Dictionary with model information from ETS forecasting
        confidence_level: Confidence level (0-1)
        
    Returns:
        dict: Dictionary with lower and upper bounds
    """
    # Extract residual standard deviation
    residual_std = model_info.get("residual_std", None)
    
    if residual_std is None and "model_metrics" in model_info:
        # Try to get it from model_metrics
        residual_std = model_info["model_metrics"].get("residual_std", None)
    
    if residual_std is None:
        # If still not found, use the standard deviation of the forecast values
        residual_std = np.std(forecast_values) if len(forecast_values) > 1 else 1.0
    
    # Calculate z-value for the given confidence level
    z_value = stats.norm.ppf((1 + confidence_level) / 2)
    
    # For ETS with additive errors, the prediction interval width grows with the horizon
    # For simplicity, we'll use a similar approach as ARIMA but adjust based on ETS parameters
    
    # Get parameters
    alpha = model_info.get("params", {}).get("smoothing_level", 0.3)
    beta = model_info.get("params", {}).get("smoothing_trend", 0.1)
    phi = model_info.get("params", {}).get("damping_trend", 0.9)
    
    # Use default values if parameters are missing
    if alpha is None:
        alpha = 0.3
    if beta is None:
        beta = 0.1
    if phi is None:
        phi = 0.9
    
    # Calculate growth factors for forecast horizons
    horizons = np.arange(1, len(forecast_values) + 1)
    
    # For ETS models, the forecast variance depends on the model parameters
    # This is a simplified approach
    if model_info.get("seasonal", False):
        # For seasonal models, the variance grows more slowly
        variance_factors = np.sqrt(1 + alpha**2 * horizons + beta**2 * np.sum([(phi**i)**2 for i in range(1, h+1)]) for h in horizons)
    else:
        # For non-seasonal models
        variance_factors = np.sqrt(1 + alpha**2 * horizons)
    
    # Calculate margins of error
    margins = z_value * residual_std * variance_factors
    
    # Calculate bounds
    lower_bounds = [max(0, forecast_values[i] - margins[i]) for i in range(len(forecast_values))]
    upper_bounds = [min(100, forecast_values[i] + margins[i]) for i in range(len(forecast_values))]
    
    return {
        "lower": lower_bounds,
        "upper": upper_bounds
    }


def _calculate_simple_intervals(
    forecast_values: List[float],
    confidence_level: float = 0.95
) -> Dict[str, List[float]]:
    """
    Calculate simple confidence intervals based on standard deviation.
    
    This is a fallback method when more specific interval calculations fail.
    
    Args:
        forecast_values: List of forecasted values
        confidence_level: Confidence level (0-1)
        
    Returns:
        dict: Dictionary with lower and upper bounds
    """
    # Calculate standard deviation of the forecasts
    if len(forecast_values) <= 1:
        # If only one forecast value, use a default width
        std_dev = 2.0
    else:
        std_dev = np.std(forecast_values)
    
    # Calculate z-value for the given confidence level
    z_value = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Calculate margin of error
    margin = z_value * std_dev
    
    # Calculate bounds
    lower_bounds = [max(0, val - margin) for val in forecast_values]
    upper_bounds = [min(100, val + margin) for val in forecast_values]
    
    return {
        "lower": lower_bounds,
        "upper": upper_bounds
    }


def calculate_prediction_interval_width(
    forecast_horizon: int,
    residual_std: float,
    confidence_level: float = 0.95,
    method: str = "trend"
) -> float:
    """
    Calculate the width of prediction intervals for a given forecast horizon.
    
    Args:
        forecast_horizon: Number of periods into the future
        residual_std: Standard deviation of the residuals
        confidence_level: Confidence level (0-1)
        method: Forecasting method used ("trend", "arima", "ets")
        
    Returns:
        float: Width of the prediction interval
    """
    # Calculate z-value for the given confidence level
    z_value = stats.norm.ppf((1 + confidence_level) / 2)
    
    if method == "trend":
        # For trend forecasts, interval width is constant
        width = 2 * z_value * residual_std
    elif method == "arima":
        # For ARIMA, width grows with sqrt(horizon)
        width = 2 * z_value * residual_std * np.sqrt(forecast_horizon)
    elif method == "ets":
        # For ETS, use a simplified approach
        width = 2 * z_value * residual_std * np.sqrt(1 + 0.3**2 * forecast_horizon)
    else:
        # Default
        width = 2 * z_value * residual_std
    
    return width 