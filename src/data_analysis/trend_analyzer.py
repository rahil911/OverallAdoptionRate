"""
Trend Analyzer module for identifying and analyzing trends in adoption rate data.

This module provides functions for detecting peaks and valleys, calculating trends,
and generating natural language descriptions of trends in the adoption rate data.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats
from typing import List, Dict, Tuple, Union, Optional, Any
from datetime import date, datetime, timedelta
import logging

from src.data_models.metrics import OverallAdoptionRate, MetricCollection

# Set up logging
logger = logging.getLogger(__name__)


def detect_peaks_and_valleys(
    data: List[OverallAdoptionRate],
    rate_type: str = 'monthly',
    prominence: float = 1.0,
    width: int = 1
) -> Dict[str, List[OverallAdoptionRate]]:
    """
    Detect peaks and valleys in adoption rate time series data.
    
    Uses scipy's signal processing functions to detect local maxima and minima
    in the adoption rate data.
    
    Args:
        data: List of OverallAdoptionRate objects sorted by date
        rate_type: Type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
        prominence: Minimum prominence required for a peak/valley (sensitivity parameter)
        width: Minimum width required for a peak/valley (in data points)
        
    Returns:
        Dictionary with 'peaks' and 'valleys' keys, each containing a list of OverallAdoptionRate objects
    """
    if not data:
        logger.warning("No data provided for peak/valley detection")
        return {'peaks': [], 'valleys': []}
    
    # Sort data by date to ensure proper time series
    sorted_data = sorted(data, key=lambda x: x.date)
    
    # Get the appropriate adoption rate values based on rate_type
    if rate_type == 'daily':
        values = [item.daily_adoption_rate for item in sorted_data]
    elif rate_type == 'weekly':
        values = [item.weekly_adoption_rate for item in sorted_data]
    elif rate_type == 'monthly':
        values = [item.monthly_adoption_rate for item in sorted_data]
    elif rate_type == 'yearly':
        values = [item.yearly_adoption_rate for item in sorted_data]
    else:
        raise ValueError(f"Invalid rate_type: {rate_type}")
    
    # Convert to numpy array for signal processing
    values_array = np.array(values)
    
    try:
        # Find peaks (local maxima)
        peaks, _ = signal.find_peaks(values_array, prominence=prominence, width=width)
        
        # Find valleys (local minima) by inverting the signal
        valleys, _ = signal.find_peaks(-values_array, prominence=prominence, width=width)
        
        # Return the corresponding OverallAdoptionRate objects
        result = {
            'peaks': [sorted_data[i] for i in peaks],
            'valleys': [sorted_data[i] for i in valleys]
        }
        
        logger.info(f"Detected {len(result['peaks'])} peaks and {len(result['valleys'])} valleys in {rate_type} adoption rate")
        return result
    
    except Exception as e:
        logger.error(f"Error detecting peaks and valleys: {str(e)}")
        return {'peaks': [], 'valleys': []}


def calculate_trend_line(
    data: List[OverallAdoptionRate],
    rate_type: str = 'monthly',
    degree: int = 1
) -> Dict[str, Any]:
    """
    Calculate a trend line for adoption rate data.
    
    Uses polynomial regression to fit a trend line to the adoption rate data.
    
    Args:
        data: List of OverallAdoptionRate objects sorted by date
        rate_type: Type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
        degree: Degree of the polynomial to fit (1 for linear, 2 for quadratic, etc.)
        
    Returns:
        Dictionary with trend information:
        - 'coefficients': List of polynomial coefficients
        - 'slope': Overall slope of the trend (for degree=1)
        - 'r_squared': R-squared value of the fit
        - 'trend_values': Calculated trend values for each data point
        - 'direction': Direction of the trend ('increasing', 'decreasing', 'stable')
        - 'trend_dates': List of dates corresponding to trend_values
        - 'trend_strength': Qualitative strength of the trend ('strong', 'moderate', 'weak')
    """
    if not data:
        logger.warning("No data provided for trend calculation")
        return {
            'coefficients': [],
            'slope': 0,
            'r_squared': 0,
            'trend_values': [],
            'direction': 'stable',
            'trend_dates': [],
            'trend_strength': 'weak'
        }
    
    # Sort data by date to ensure proper time series
    sorted_data = sorted(data, key=lambda x: x.date)
    
    # Get the appropriate adoption rate values based on rate_type
    if rate_type == 'daily':
        values = [item.daily_adoption_rate for item in sorted_data]
    elif rate_type == 'weekly':
        values = [item.weekly_adoption_rate for item in sorted_data]
    elif rate_type == 'monthly':
        values = [item.monthly_adoption_rate for item in sorted_data]
    elif rate_type == 'yearly':
        values = [item.yearly_adoption_rate for item in sorted_data]
    else:
        raise ValueError(f"Invalid rate_type: {rate_type}")
    
    # Use days since the first date as the x-axis
    first_date = sorted_data[0].date
    x_days = [(item.date - first_date).days for item in sorted_data]
    
    try:
        # Fit polynomial
        coefficients = np.polyfit(x_days, values, degree)
        
        # Calculate trend values
        trend_values = np.polyval(coefficients, x_days)
        
        # Calculate R-squared
        residuals = np.array(values) - trend_values
        ss_residual = np.sum(residuals**2)
        ss_total = np.sum((np.array(values) - np.mean(values))**2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        # Determine trend direction
        if degree == 1:
            slope = coefficients[0]
            # Define threshold for considering a trend significant
            threshold = 0.01  # Adjust as needed
            
            if slope > threshold:
                direction = 'increasing'
            elif slope < -threshold:
                direction = 'decreasing'
            else:
                direction = 'stable'
            
            # Determine trend strength based on R-squared
            if r_squared > 0.7:
                trend_strength = 'strong'
            elif r_squared > 0.3:
                trend_strength = 'moderate'
            else:
                trend_strength = 'weak'
        else:
            # For higher degree polynomials, look at the difference between start and end
            if trend_values[-1] > trend_values[0] + 1:
                direction = 'increasing'
            elif trend_values[-1] < trend_values[0] - 1:
                direction = 'decreasing'
            else:
                direction = 'stable'
            
            # Determine trend strength based on R-squared
            if r_squared > 0.8:  # Higher threshold for non-linear models
                trend_strength = 'strong'
            elif r_squared > 0.5:
                trend_strength = 'moderate'
            else:
                trend_strength = 'weak'
        
        return {
            'coefficients': coefficients.tolist(),
            'slope': coefficients[0] if degree == 1 else None,
            'r_squared': r_squared,
            'trend_values': trend_values.tolist(),
            'direction': direction,
            'trend_dates': [item.date for item in sorted_data],
            'trend_strength': trend_strength
        }
    
    except Exception as e:
        logger.error(f"Error calculating trend line: {str(e)}")
        return {
            'coefficients': [],
            'slope': 0,
            'r_squared': 0,
            'trend_values': [],
            'direction': 'stable',
            'trend_dates': [],
            'trend_strength': 'weak'
        }


def calculate_moving_average(
    data: List[OverallAdoptionRate], 
    rate_type: str = 'monthly',
    window: int = 3,
    ma_type: str = 'simple'
) -> List[Tuple[date, float]]:
    """
    Calculate moving average for adoption rate data.
    
    Args:
        data: List of OverallAdoptionRate objects sorted by date
        rate_type: Type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
        window: Window size for moving average
        ma_type: Type of moving average ('simple', 'weighted', 'exponential')
        
    Returns:
        List of (date, moving_average) tuples
    """
    if not data or len(data) < window:
        logger.warning(f"Insufficient data for calculating {window}-point moving average")
        return []
    
    # Sort data by date to ensure proper time series
    sorted_data = sorted(data, key=lambda x: x.date)
    
    # Get the appropriate adoption rate values based on rate_type
    if rate_type == 'daily':
        values = [item.daily_adoption_rate for item in sorted_data]
    elif rate_type == 'weekly':
        values = [item.weekly_adoption_rate for item in sorted_data]
    elif rate_type == 'monthly':
        values = [item.monthly_adoption_rate for item in sorted_data]
    elif rate_type == 'yearly':
        values = [item.yearly_adoption_rate for item in sorted_data]
    else:
        raise ValueError(f"Invalid rate_type: {rate_type}")
    
    dates = [item.date for item in sorted_data]
    
    try:
        # Create pandas Series for easy calculation of moving averages
        series = pd.Series(values, index=dates)
        
        if ma_type == 'simple':
            # Simple moving average
            ma = series.rolling(window=window, min_periods=1).mean()
        elif ma_type == 'weighted':
            # Weighted moving average (more weight to recent data)
            weights = np.arange(1, window + 1)
            ma = series.rolling(window=window, min_periods=1).apply(
                lambda x: np.sum(weights[-len(x):] * x) / np.sum(weights[-len(x):])
                if len(x) > 0 else np.nan
            )
        elif ma_type == 'exponential':
            # Exponential moving average
            ma = series.ewm(span=window, adjust=False).mean()
        else:
            raise ValueError(f"Invalid ma_type: {ma_type}")
        
        # Convert back to list of tuples (date, value)
        result = [(date, value) for date, value in zip(dates, ma.values)]
        return result
    
    except Exception as e:
        logger.error(f"Error calculating moving average: {str(e)}")
        return []


def generate_trend_description(
    data: List[OverallAdoptionRate], 
    rate_type: str = 'monthly',
    time_period: str = 'recent'
) -> str:
    """
    Generate a natural language description of the adoption rate trend.
    
    Args:
        data: List of OverallAdoptionRate objects
        rate_type: Type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
        time_period: Time period to describe ('recent', 'all')
        
    Returns:
        Natural language description of the trend
    """
    if not data:
        return "No data available to analyze trends."
    
    # Sort data by date
    sorted_data = sorted(data, key=lambda x: x.date)
    
    # For 'recent' analysis, use only the last 3 months of data if available
    if time_period == 'recent' and len(sorted_data) > 30:
        three_months_ago = sorted_data[-1].date - timedelta(days=90)
        recent_data = [item for item in sorted_data if item.date >= three_months_ago]
        if recent_data:
            sorted_data = recent_data
    
    # Calculate trend
    trend_info = calculate_trend_line(sorted_data, rate_type, degree=1)
    
    # Detect peaks and valleys
    extrema = detect_peaks_and_valleys(sorted_data, rate_type)
    
    # Get start, end, min, max values
    if rate_type == 'daily':
        start_value = sorted_data[0].daily_adoption_rate
        end_value = sorted_data[-1].daily_adoption_rate
        values = [item.daily_adoption_rate for item in sorted_data]
    elif rate_type == 'weekly':
        start_value = sorted_data[0].weekly_adoption_rate
        end_value = sorted_data[-1].weekly_adoption_rate
        values = [item.weekly_adoption_rate for item in sorted_data]
    elif rate_type == 'monthly':
        start_value = sorted_data[0].monthly_adoption_rate
        end_value = sorted_data[-1].monthly_adoption_rate
        values = [item.monthly_adoption_rate for item in sorted_data]
    elif rate_type == 'yearly':
        start_value = sorted_data[0].yearly_adoption_rate
        end_value = sorted_data[-1].yearly_adoption_rate
        values = [item.yearly_adoption_rate for item in sorted_data]
    else:
        raise ValueError(f"Invalid rate_type: {rate_type}")
    
    min_value = min(values)
    max_value = max(values)
    avg_value = sum(values) / len(values)
    
    # Format period string
    period_str = f"over the past 3 months" if time_period == 'recent' else f"from {sorted_data[0].date.strftime('%B %Y')} to {sorted_data[-1].date.strftime('%B %Y')}"
    
    # Build description
    description = f"The {rate_type} adoption rate {period_str} shows a {trend_info['trend_strength']} {trend_info['direction']} trend. "
    
    # Add change magnitude
    change = end_value - start_value
    percent_change = (change / start_value * 100) if start_value > 0 else 0
    
    if abs(percent_change) < 1:
        change_magnitude = "minimal change"
    elif abs(percent_change) < 5:
        change_magnitude = "slight " + ("increase" if change > 0 else "decrease")
    elif abs(percent_change) < 10:
        change_magnitude = "moderate " + ("increase" if change > 0 else "decrease")
    else:
        change_magnitude = "significant " + ("increase" if change > 0 else "decrease")
    
    description += f"There has been a {change_magnitude} from {start_value:.2f}% to {end_value:.2f}% ({percent_change:.1f}%). "
    
    # Add info about peaks and valleys if they exist
    if extrema['peaks']:
        peak_dates = [peak.date.strftime('%B %Y') for peak in extrema['peaks'][-2:]]
        if len(peak_dates) == 1:
            description += f"A notable peak occurred in {peak_dates[0]}. "
        elif len(peak_dates) > 1:
            description += f"Notable peaks occurred in {' and '.join(peak_dates)}. "
    
    if extrema['valleys']:
        valley_dates = [valley.date.strftime('%B %Y') for valley in extrema['valleys'][-2:]]
        if len(valley_dates) == 1:
            description += f"A notable low point occurred in {valley_dates[0]}. "
        elif len(valley_dates) > 1:
            description += f"Notable low points occurred in {' and '.join(valley_dates)}. "
    
    # Add context relative to min/max/average
    description += f"The current value ({end_value:.2f}%) is "
    
    if abs(end_value - max_value) < 0.1:
        description += "at its highest point in the period. "
    elif abs(end_value - min_value) < 0.1:
        description += "at its lowest point in the period. "
    elif end_value > avg_value:
        description += f"above the period average of {avg_value:.2f}%. "
    else:
        description += f"below the period average of {avg_value:.2f}%. "
    
    return description


def identify_significant_changes(
    data: List[OverallAdoptionRate],
    rate_type: str = 'monthly',
    threshold_percent: float = 10.0
) -> List[Dict[str, Any]]:
    """
    Identify significant changes in the adoption rate data.
    
    Args:
        data: List of OverallAdoptionRate objects
        rate_type: Type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
        threshold_percent: Percent change threshold to consider significant
        
    Returns:
        List of dictionaries containing information about significant changes:
        - 'date': Date of the change
        - 'previous_date': Date of the previous point
        - 'value': Adoption rate value
        - 'previous_value': Previous adoption rate value
        - 'percent_change': Percentage change
        - 'direction': Direction of change ('increase' or 'decrease')
    """
    if not data or len(data) < 2:
        logger.warning("Insufficient data for identifying significant changes")
        return []
    
    # Sort data by date
    sorted_data = sorted(data, key=lambda x: x.date)
    
    # Get the appropriate adoption rate values based on rate_type
    if rate_type == 'daily':
        values = [item.daily_adoption_rate for item in sorted_data]
    elif rate_type == 'weekly':
        values = [item.weekly_adoption_rate for item in sorted_data]
    elif rate_type == 'monthly':
        values = [item.monthly_adoption_rate for item in sorted_data]
    elif rate_type == 'yearly':
        values = [item.yearly_adoption_rate for item in sorted_data]
    else:
        raise ValueError(f"Invalid rate_type: {rate_type}")
    
    dates = [item.date for item in sorted_data]
    
    significant_changes = []
    
    for i in range(1, len(sorted_data)):
        current_value = values[i]
        previous_value = values[i-1]
        
        # Skip if previous value is zero to avoid division by zero
        if previous_value == 0:
            continue
        
        percent_change = ((current_value - previous_value) / previous_value) * 100
        
        if abs(percent_change) >= threshold_percent:
            significant_changes.append({
                'date': dates[i],
                'previous_date': dates[i-1],
                'value': current_value,
                'previous_value': previous_value,
                'percent_change': percent_change,
                'direction': 'increase' if percent_change > 0 else 'decrease'
            })
    
    return significant_changes 