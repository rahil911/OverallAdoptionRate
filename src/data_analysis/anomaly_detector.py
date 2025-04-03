"""
Anomaly Detector module for identifying anomalies and outliers in adoption rate data.

This module provides functions for detecting anomalies using various statistical methods 
including Z-score, modified Z-score, moving average, and adaptive thresholds.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any
from datetime import date, datetime, timedelta
import logging
from scipy import stats

from src.data_models.metrics import OverallAdoptionRate, MetricCollection

# Set up logging
logger = logging.getLogger(__name__)


def detect_anomalies_zscore(
    data: List[OverallAdoptionRate],
    rate_type: str = 'monthly',
    threshold: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Detect anomalies in adoption rate data using Z-score method.
    
    The Z-score measures how many standard deviations a value is from the mean.
    Values with a Z-score greater than the threshold are considered anomalies.
    
    Args:
        data: List of OverallAdoptionRate objects
        rate_type: Type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
        threshold: Z-score threshold for anomaly detection (default: 2.0)
        
    Returns:
        List of dictionaries containing information about anomalies:
        - 'date': Date of the anomaly
        - 'value': Adoption rate value
        - 'zscore': Z-score of the value
        - 'direction': Direction of anomaly ('high' or 'low')
    """
    if not data or len(data) < 2:
        logger.warning("Insufficient data for Z-score anomaly detection")
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
    
    # Convert to numpy array
    values_array = np.array(values)
    
    # Calculate Z-scores
    mean = np.mean(values_array)
    std = np.std(values_array)
    
    if std == 0:
        logger.warning("Standard deviation is zero, can't calculate Z-scores")
        return []
    
    zscores = [(value - mean) / std for value in values_array]
    
    # Identify anomalies
    anomalies = []
    
    for i, zscore in enumerate(zscores):
        if abs(zscore) > threshold:
            anomalies.append({
                'date': sorted_data[i].date,
                'value': values[i],
                'zscore': zscore,
                'direction': 'high' if zscore > 0 else 'low'
            })
    
    logger.info(f"Detected {len(anomalies)} anomalies using Z-score method (threshold: {threshold})")
    return anomalies


def detect_anomalies_modified_zscore(
    data: List[OverallAdoptionRate],
    rate_type: str = 'monthly',
    threshold: float = 3.5
) -> List[Dict[str, Any]]:
    """
    Detect anomalies using modified Z-score, which is more robust to outliers.
    
    Uses median and median absolute deviation (MAD) instead of mean and standard deviation.
    
    Args:
        data: List of OverallAdoptionRate objects
        rate_type: Type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
        threshold: Modified Z-score threshold (default: 3.5, based on literature)
        
    Returns:
        List of dictionaries containing information about anomalies:
        - 'date': Date of the anomaly
        - 'value': Adoption rate value
        - 'modified_zscore': Modified Z-score of the value
        - 'direction': Direction of anomaly ('high' or 'low')
    """
    if not data or len(data) < 2:
        logger.warning("Insufficient data for modified Z-score anomaly detection")
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
    
    # Convert to numpy array
    values_array = np.array(values)
    
    # Calculate median and median absolute deviation (MAD)
    median = np.median(values_array)
    mad = np.median(np.abs(values_array - median))
    
    # Constant factor for normal distribution
    c = 0.6745
    
    if mad == 0:
        logger.warning("Median absolute deviation is zero, can't calculate modified Z-scores")
        return []
    
    # Calculate modified Z-scores
    modified_zscores = [c * (value - median) / mad for value in values_array]
    
    # Identify anomalies
    anomalies = []
    
    for i, modified_zscore in enumerate(modified_zscores):
        if abs(modified_zscore) > threshold:
            anomalies.append({
                'date': sorted_data[i].date,
                'value': values[i],
                'modified_zscore': modified_zscore,
                'direction': 'high' if modified_zscore > 0 else 'low'
            })
    
    logger.info(f"Detected {len(anomalies)} anomalies using modified Z-score method (threshold: {threshold})")
    return anomalies


def detect_anomalies_iqr(
    data: List[OverallAdoptionRate],
    rate_type: str = 'monthly',
    multiplier: float = 1.5
) -> List[Dict[str, Any]]:
    """
    Detect anomalies using the Interquartile Range (IQR) method.
    
    Values outside the range [Q1 - multiplier*IQR, Q3 + multiplier*IQR] are considered anomalies.
    
    Args:
        data: List of OverallAdoptionRate objects
        rate_type: Type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
        multiplier: Multiplier for IQR (default: 1.5 for mild outliers, 3.0 for extreme outliers)
        
    Returns:
        List of dictionaries containing information about anomalies:
        - 'date': Date of the anomaly
        - 'value': Adoption rate value
        - 'boundary': Boundary that was exceeded ('upper' or 'lower')
        - 'boundary_value': Value of the boundary that was exceeded
    """
    if not data or len(data) < 4:  # Need enough data for quartiles
        logger.warning("Insufficient data for IQR-based anomaly detection")
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
    
    # Calculate quartiles and IQR
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    # Calculate boundaries
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    # Identify anomalies
    anomalies = []
    
    for i, value in enumerate(values):
        if value < lower_bound:
            anomalies.append({
                'date': sorted_data[i].date,
                'value': value,
                'boundary': 'lower',
                'boundary_value': lower_bound
            })
        elif value > upper_bound:
            anomalies.append({
                'date': sorted_data[i].date,
                'value': value,
                'boundary': 'upper',
                'boundary_value': upper_bound
            })
    
    logger.info(f"Detected {len(anomalies)} anomalies using IQR method (multiplier: {multiplier})")
    return anomalies


def detect_anomalies_moving_average(
    data: List[OverallAdoptionRate],
    rate_type: str = 'monthly',
    window: int = 5,
    std_multiplier: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Detect anomalies using moving average and standard deviation.
    
    This method is more adaptive to trends in the data.
    
    Args:
        data: List of OverallAdoptionRate objects
        rate_type: Type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
        window: Window size for moving average
        std_multiplier: Multiplier for standard deviation to set thresholds
        
    Returns:
        List of dictionaries containing information about anomalies:
        - 'date': Date of the anomaly
        - 'value': Adoption rate value
        - 'ma_value': Moving average value
        - 'deviation': Deviation from moving average
        - 'threshold': Threshold that was exceeded
    """
    if not data or len(data) < window + 1:
        logger.warning(f"Insufficient data for moving average anomaly detection (need at least {window+1} points)")
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
    
    # Create pandas Series for calculations
    series = pd.Series(values)
    
    # Calculate moving average and standard deviation
    ma = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    
    # Handle NaN values in the std
    std = std.fillna(0)
    
    # Calculate thresholds
    upper_thresholds = ma + std_multiplier * std
    lower_thresholds = ma - std_multiplier * std
    
    # Identify anomalies
    anomalies = []
    
    for i in range(len(values)):
        if i < window - 1:
            # Skip the first few points where moving average is not reliable
            continue
        
        value = values[i]
        ma_value = ma.iloc[i]
        upper_threshold = upper_thresholds.iloc[i]
        lower_threshold = lower_thresholds.iloc[i]
        
        if value > upper_threshold:
            anomalies.append({
                'date': sorted_data[i].date,
                'value': value,
                'ma_value': ma_value,
                'deviation': value - ma_value,
                'threshold': upper_threshold,
                'direction': 'high'
            })
        elif value < lower_threshold:
            anomalies.append({
                'date': sorted_data[i].date,
                'value': value,
                'ma_value': ma_value,
                'deviation': value - ma_value,
                'threshold': lower_threshold,
                'direction': 'low'
            })
    
    logger.info(f"Detected {len(anomalies)} anomalies using moving average method (window: {window}, std_multiplier: {std_multiplier})")
    return anomalies


def detect_anomalies_adaptive_threshold(
    data: List[OverallAdoptionRate],
    rate_type: str = 'monthly',
    window: int = 10,
    influence: float = 0.5,
    threshold: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Detect anomalies using an adaptive threshold based on recent history.
    
    This method is effective for time series with trends.
    
    Args:
        data: List of OverallAdoptionRate objects
        rate_type: Type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
        window: Window size for calculating mean and std
        influence: Influence of anomalies on the threshold (0-1)
        threshold: Threshold multiplier for mean
        
    Returns:
        List of dictionaries containing information about anomalies:
        - 'date': Date of the anomaly
        - 'value': Adoption rate value
        - 'adaptive_mean': Adaptive mean at this point
        - 'adaptive_std': Adaptive standard deviation at this point
        - 'direction': Direction of anomaly ('high' or 'low')
    """
    if not data or len(data) <= window:
        logger.warning(f"Insufficient data for adaptive threshold anomaly detection (need > {window} points)")
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
    
    # Initialize arrays
    signals = np.zeros(len(values))
    filtered_values = np.array(values).copy()
    adaptive_mean = np.zeros(len(values))
    adaptive_std = np.zeros(len(values))
    
    # Initialize with the first window
    adaptive_mean[window - 1] = np.mean(values[0:window])
    adaptive_std[window - 1] = np.std(values[0:window])
    
    # Process the rest of the signal
    for i in range(window, len(values)):
        # If the value is more than 'threshold' standard deviations from the mean
        if abs(values[i] - adaptive_mean[i - 1]) > threshold * adaptive_std[i - 1]:
            signals[i] = 1 if values[i] > adaptive_mean[i - 1] else -1
            
            # Update filtered series with reduced influence
            filtered_values[i] = influence * values[i] + (1 - influence) * filtered_values[i - 1]
        else:
            # No signal, update normally
            filtered_values[i] = values[i]
        
        # Update mean and std for the next iteration
        adaptive_mean[i] = np.mean(filtered_values[i - window + 1:i + 1])
        adaptive_std[i] = np.std(filtered_values[i - window + 1:i + 1])
    
    # Identify anomalies
    anomalies = []
    
    for i in range(window, len(values)):
        if signals[i] != 0:
            anomalies.append({
                'date': sorted_data[i].date,
                'value': values[i],
                'adaptive_mean': adaptive_mean[i],
                'adaptive_std': adaptive_std[i],
                'direction': 'high' if signals[i] > 0 else 'low'
            })
    
    logger.info(f"Detected {len(anomalies)} anomalies using adaptive threshold method (window: {window}, threshold: {threshold})")
    return anomalies


def detect_anomalies_ensemble(
    data: List[OverallAdoptionRate],
    rate_type: str = 'monthly',
    min_methods: int = 2
) -> List[Dict[str, Any]]:
    """
    Detect anomalies using an ensemble of multiple detection methods.
    
    Points flagged by at least min_methods detection methods are considered anomalies.
    
    Args:
        data: List of OverallAdoptionRate objects
        rate_type: Type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
        min_methods: Minimum number of methods that must flag a point as anomaly
        
    Returns:
        List of dictionaries containing information about anomalies:
        - 'date': Date of the anomaly
        - 'value': Adoption rate value
        - 'methods': List of methods that flagged this point
        - 'confidence': Percentage of methods that flagged this point
    """
    if not data:
        logger.warning("No data provided for ensemble anomaly detection")
        return []
    
    # Apply different detection methods
    zscore_anomalies = detect_anomalies_zscore(data, rate_type)
    modified_zscore_anomalies = detect_anomalies_modified_zscore(data, rate_type)
    iqr_anomalies = detect_anomalies_iqr(data, rate_type)
    
    # If we have enough data, also use moving average and adaptive methods
    if len(data) > 10:
        ma_anomalies = detect_anomalies_moving_average(data, rate_type)
        adaptive_anomalies = detect_anomalies_adaptive_threshold(data, rate_type)
        methods = 5
    else:
        ma_anomalies = []
        adaptive_anomalies = []
        methods = 3
    
    # Create sets of anomaly dates for each method
    zscore_dates = {anomaly['date'] for anomaly in zscore_anomalies}
    modified_zscore_dates = {anomaly['date'] for anomaly in modified_zscore_anomalies}
    iqr_dates = {anomaly['date'] for anomaly in iqr_anomalies}
    ma_dates = {anomaly['date'] for anomaly in ma_anomalies}
    adaptive_dates = {anomaly['date'] for anomaly in adaptive_anomalies}
    
    # Create a dictionary to track how many methods flagged each date
    anomaly_counts = {}
    
    for date in zscore_dates:
        anomaly_counts[date] = ['zscore']
    
    for date in modified_zscore_dates:
        if date in anomaly_counts:
            anomaly_counts[date].append('modified_zscore')
        else:
            anomaly_counts[date] = ['modified_zscore']
    
    for date in iqr_dates:
        if date in anomaly_counts:
            anomaly_counts[date].append('iqr')
        else:
            anomaly_counts[date] = ['iqr']
    
    for date in ma_dates:
        if date in anomaly_counts:
            anomaly_counts[date].append('moving_average')
        else:
            anomaly_counts[date] = ['moving_average']
    
    for date in adaptive_dates:
        if date in anomaly_counts:
            anomaly_counts[date].append('adaptive_threshold')
        else:
            anomaly_counts[date] = ['adaptive_threshold']
    
    # Filter dates flagged by at least min_methods
    ensemble_anomalies = []
    
    # Sort data by date
    sorted_data = sorted(data, key=lambda x: x.date)
    date_to_value = {}
    
    # Get the appropriate adoption rate values based on rate_type
    if rate_type == 'daily':
        for item in sorted_data:
            date_to_value[item.date] = item.daily_adoption_rate
    elif rate_type == 'weekly':
        for item in sorted_data:
            date_to_value[item.date] = item.weekly_adoption_rate
    elif rate_type == 'monthly':
        for item in sorted_data:
            date_to_value[item.date] = item.monthly_adoption_rate
    elif rate_type == 'yearly':
        for item in sorted_data:
            date_to_value[item.date] = item.yearly_adoption_rate
    else:
        raise ValueError(f"Invalid rate_type: {rate_type}")
    
    for date, methods_list in anomaly_counts.items():
        if len(methods_list) >= min_methods:
            ensemble_anomalies.append({
                'date': date,
                'value': date_to_value.get(date),
                'methods': methods_list,
                'confidence': len(methods_list) / methods
            })
    
    # Sort by date
    ensemble_anomalies.sort(key=lambda x: x['date'])
    
    logger.info(f"Detected {len(ensemble_anomalies)} anomalies using ensemble method (min_methods: {min_methods})")
    return ensemble_anomalies


def generate_anomaly_explanation(
    anomalies: List[Dict[str, Any]],
    data: List[OverallAdoptionRate],
    rate_type: str = 'monthly'
) -> List[Dict[str, str]]:
    """
    Generate natural language explanations for detected anomalies.
    
    Args:
        anomalies: List of anomaly dictionaries from detect_anomalies_* functions
        data: List of OverallAdoptionRate objects
        rate_type: Type of adoption rate being analyzed ('daily', 'weekly', 'monthly', 'yearly')
        
    Returns:
        List of dictionaries with:
        - 'date': Date of the anomaly
        - 'explanation': Natural language explanation
    """
    if not anomalies:
        return []
    
    # Sort data by date
    sorted_data = sorted(data, key=lambda x: x.date)
    
    # Get the appropriate adoption rate values based on rate_type
    if rate_type == 'daily':
        date_to_index = {item.date: i for i, item in enumerate(sorted_data)}
        values = [item.daily_adoption_rate for item in sorted_data]
    elif rate_type == 'weekly':
        date_to_index = {item.date: i for i, item in enumerate(sorted_data)}
        values = [item.weekly_adoption_rate for item in sorted_data]
    elif rate_type == 'monthly':
        date_to_index = {item.date: i for i, item in enumerate(sorted_data)}
        values = [item.monthly_adoption_rate for item in sorted_data]
    elif rate_type == 'yearly':
        date_to_index = {item.date: i for i, item in enumerate(sorted_data)}
        values = [item.yearly_adoption_rate for item in sorted_data]
    else:
        raise ValueError(f"Invalid rate_type: {rate_type}")
    
    # Calculate overall statistics
    overall_mean = np.mean(values)
    overall_std = np.std(values)
    
    explanations = []
    
    for anomaly in anomalies:
        date = anomaly['date']
        value = anomaly['value']
        
        # Skip if we can't find the date in our data
        if date not in date_to_index:
            continue
        
        index = date_to_index[date]
        
        # Calculate percentage deviation from mean
        pct_deviation = ((value - overall_mean) / overall_mean) * 100 if overall_mean != 0 else float('inf')
        
        # Determine if this is a high or low anomaly
        direction = 'unusually high' if value > overall_mean else 'unusually low'
        
        # Build explanation
        explanation = f"On {date.strftime('%B %d, %Y')}, the {rate_type} adoption rate was {direction} at {value:.2f}%, "
        
        # Add context about the deviation
        explanation += f"which is {abs(pct_deviation):.1f}% {direction.split()[-1]} than the average of {overall_mean:.2f}%. "
        
        # Look at surrounding points for context if possible
        window = 3
        min_index = max(0, index - window)
        max_index = min(len(sorted_data) - 1, index + window)
        
        if min_index < index:
            # There are points before this one
            prev_value = values[index - 1]
            point_change = ((value - prev_value) / prev_value) * 100 if prev_value != 0 else float('inf')
            
            if abs(point_change) > 10:  # Significant point-to-point change
                explanation += f"This represents a {abs(point_change):.1f}% {direction.split()[-1]} from the previous point. "
        
        if max_index > index:
            # There are points after this one
            next_value = values[index + 1]
            recovery = ((next_value - value) / value) * 100 if value != 0 else float('inf')
            
            if (direction == 'unusually high' and recovery < -10) or (direction == 'unusually low' and recovery > 10):
                explanation += f"The rate quickly returned to normal levels afterward. "
            elif abs(recovery) < 5:
                explanation += f"The rate remained at similar levels afterward. "
        
        # Add information about the detection method if available
        if 'methods' in anomaly:
            method_count = len(anomaly['methods'])
            if method_count > 1:
                explanation += f"This anomaly was detected by {method_count} different statistical methods, "
                explanation += f"giving it a confidence score of {anomaly['confidence']:.1%}."
            else:
                explanation += f"This anomaly was detected by the {anomaly['methods'][0]} method."
        elif 'zscore' in anomaly:
            explanation += f"The Z-score for this point is {anomaly['zscore']:.2f}, "
            explanation += f"indicating it's {abs(anomaly['zscore']):.1f} standard deviations from the mean."
        elif 'boundary' in anomaly:
            boundary_type = "upper" if anomaly['boundary'] == 'upper' else "lower"
            explanation += f"This value is beyond the {boundary_type} boundary of {anomaly['boundary_value']:.2f}% "
            explanation += f"based on the interquartile range method."
        elif 'ma_value' in anomaly:
            explanation += f"This value deviates significantly from the moving average of {anomaly['ma_value']:.2f}%."
        
        explanations.append({
            'date': date,
            'explanation': explanation
        })
    
    return explanations 