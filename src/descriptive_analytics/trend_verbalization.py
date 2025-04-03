"""
Trend Verbalization Module

This module provides functions for verbalizing trends, patterns, and anomalies
in adoption rate data.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import scipy.signal as signal

# Configure logging
logger = logging.getLogger(__name__)

def verbalize_trend(data, rate_column):
    """
    Generate a natural language description of trends in the adoption rate data.
    
    Args:
        data (pandas.DataFrame): DataFrame containing adoption rate data
        rate_column (str): Column name for the adoption rate
        
    Returns:
        dict: Dictionary containing trend description with these keys:
            - trend_direction: Overall direction of the trend (increasing, decreasing, stable)
            - trend_strength: Strength of the trend (strong, moderate, weak)
            - volatility: Description of volatility (high, moderate, low)
            - peaks_valleys: Notable peaks and valleys in the data
            - description: Consolidated natural language description
    """
    if data.empty or rate_column not in data.columns:
        return {
            "description": "No data available for trend analysis."
        }
    
    # Ensure data is sorted by date
    data = data.sort_values('Date')
    
    # Extract rate values and dates
    rates = data[rate_column].values
    dates = data['Date'].values
    
    # Calculate overall trend
    if len(rates) > 1:
        # Linear regression to get trend slope
        x = np.arange(len(rates))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, rates)
        
        # Determine trend direction and strength
        trend_direction = "stable"
        if slope > 0.05:
            trend_direction = "increasing"
        elif slope < -0.05:
            trend_direction = "decreasing"
        
        # Determine trend strength based on R-squared
        r_squared = r_value**2
        trend_strength = "weak"
        if r_squared > 0.7:
            trend_strength = "strong"
        elif r_squared > 0.3:
            trend_strength = "moderate"
        
        # Calculate volatility (coefficient of variation)
        mean_rate = np.mean(rates)
        std_dev = np.std(rates)
        cv = std_dev / mean_rate if mean_rate != 0 else 0
        
        volatility = "low"
        if cv > 0.5:
            volatility = "high"
        elif cv > 0.2:
            volatility = "moderate"
        
        # Find significant peaks and valleys
        if len(rates) >= 5:
            # Use scipy's find_peaks function to identify peaks
            peaks, _ = signal.find_peaks(rates, prominence=0.5)
            valleys, _ = signal.find_peaks(-rates, prominence=0.5)
            
            peaks_data = []
            for peak in peaks:
                peaks_data.append({
                    "date": dates[peak],
                    "value": rates[peak],
                    "description": f"Peak of {rates[peak]:.2f}% on {dates[peak].strftime('%Y-%m-%d')}"
                })
            
            valleys_data = []
            for valley in valleys:
                valleys_data.append({
                    "date": dates[valley],
                    "value": rates[valley],
                    "description": f"Valley of {rates[valley]:.2f}% on {dates[valley].strftime('%Y-%m-%d')}"
                })
            
            # Sort by value to get most significant peaks and valleys
            peaks_data = sorted(peaks_data, key=lambda x: x["value"], reverse=True)[:3]
            valleys_data = sorted(valleys_data, key=lambda x: x["value"])[:3]
        else:
            peaks_data = []
            valleys_data = []
        
        # Generate recent trend
        recent_trend = "stable"
        if len(rates) >= 3:
            recent_rates = rates[-3:]
            if recent_rates[-1] > recent_rates[-2] > recent_rates[-3]:
                recent_trend = "recently increasing"
            elif recent_rates[-1] < recent_rates[-2] < recent_rates[-3]:
                recent_trend = "recently decreasing"
            elif recent_rates[-1] > recent_rates[-3]:
                recent_trend = "fluctuating but generally increasing"
            elif recent_rates[-1] < recent_rates[-3]:
                recent_trend = "fluctuating but generally decreasing"
        
        # Generate consolidated description
        description = f"The adoption rate shows a {trend_strength} {trend_direction} trend "
        description += f"with {volatility} volatility over the analyzed period "
        description += f"({dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}). "
        
        if len(peaks_data) > 0:
            description += f"The highest peak was {peaks_data[0]['value']:.2f}% on {peaks_data[0]['date'].strftime('%Y-%m-%d')}. "
        
        if len(valleys_data) > 0:
            description += f"The lowest point was {valleys_data[0]['value']:.2f}% on {valleys_data[0]['date'].strftime('%Y-%m-%d')}. "
        
        description += f"The trend is {recent_trend} based on recent data points."
        
        return {
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "volatility": volatility,
            "slope": slope,
            "r_squared": r_squared,
            "coefficient_of_variation": cv,
            "most_recent_value": rates[-1],
            "peaks": peaks_data,
            "valleys": valleys_data,
            "recent_trend": recent_trend,
            "description": description
        }
    else:
        return {
            "description": "Insufficient data for trend analysis. At least two data points are required."
        }

def verbalize_period_comparison(period1_data, period2_data, rate_column):
    """
    Generate a natural language description comparing two time periods.
    
    Args:
        period1_data (pandas.DataFrame): DataFrame for the first period
        period2_data (pandas.DataFrame): DataFrame for the second period
        rate_column (str): Column name for the adoption rate
        
    Returns:
        dict: Dictionary containing period comparison with these keys:
            - period1_info: Summary of first period
            - period2_info: Summary of second period
            - change: Changes between periods
            - description: Consolidated natural language description
    """
    if period1_data.empty or period2_data.empty or rate_column not in period1_data.columns or rate_column not in period2_data.columns:
        return {
            "description": "Insufficient data for period comparison."
        }
    
    # Calculate key statistics for both periods
    period1_mean = period1_data[rate_column].mean()
    period2_mean = period2_data[rate_column].mean()
    
    period1_median = period1_data[rate_column].median()
    period2_median = period2_data[rate_column].median()
    
    period1_min = period1_data[rate_column].min()
    period2_min = period2_data[rate_column].min()
    
    period1_max = period1_data[rate_column].max()
    period2_max = period2_data[rate_column].max()
    
    period1_std = period1_data[rate_column].std()
    period2_std = period2_data[rate_column].std()
    
    # Calculate changes
    absolute_change = period2_mean - period1_mean
    percent_change = (absolute_change / period1_mean * 100) if period1_mean != 0 else float('inf')
    
    # Determine change direction and magnitude
    change_direction = "unchanged"
    if absolute_change > 0.5:
        change_direction = "increased"
    elif absolute_change < -0.5:
        change_direction = "decreased"
    
    change_magnitude = "slightly"
    if abs(percent_change) > 50:
        change_magnitude = "dramatically"
    elif abs(percent_change) > 20:
        change_magnitude = "significantly"
    elif abs(percent_change) > 5:
        change_magnitude = "moderately"
    
    # Generate period ranges
    period1_start = period1_data['Date'].min()
    period1_end = period1_data['Date'].max()
    period2_start = period2_data['Date'].min()
    period2_end = period2_data['Date'].max()
    
    # Generate consolidated description
    description = f"Comparing the period {period1_start.strftime('%Y-%m-%d')} to {period1_end.strftime('%Y-%m-%d')} "
    description += f"with {period2_start.strftime('%Y-%m-%d')} to {period2_end.strftime('%Y-%m-%d')}, "
    
    if change_direction == "unchanged":
        description += f"the adoption rate remained relatively stable (from {period1_mean:.2f}% to {period2_mean:.2f}%). "
    else:
        description += f"the adoption rate {change_direction} {change_magnitude} from {period1_mean:.2f}% to {period2_mean:.2f}% "
        description += f"(an absolute change of {absolute_change:.2f} percentage points or {percent_change:.1f}%). "
    
    # Add information about variability
    if period2_std > period1_std * 1.5:
        description += f"The second period showed much higher variability (std dev: {period2_std:.2f} vs {period1_std:.2f}). "
    elif period1_std > period2_std * 1.5:
        description += f"The second period showed much lower variability (std dev: {period2_std:.2f} vs {period1_std:.2f}). "
    
    # Add information about extremes
    if period2_max > period1_max:
        description += f"The highest adoption rate increased from {period1_max:.2f}% to {period2_max:.2f}%. "
    if period2_min < period1_min:
        description += f"The lowest adoption rate decreased from {period1_min:.2f}% to {period2_min:.2f}%. "
    
    return {
        "period1_info": {
            "start_date": period1_start,
            "end_date": period1_end,
            "mean": period1_mean,
            "median": period1_median,
            "min": period1_min,
            "max": period1_max,
            "std_dev": period1_std
        },
        "period2_info": {
            "start_date": period2_start,
            "end_date": period2_end,
            "mean": period2_mean,
            "median": period2_median,
            "min": period2_min,
            "max": period2_max,
            "std_dev": period2_std
        },
        "change": {
            "absolute_change": absolute_change,
            "percent_change": percent_change,
            "direction": change_direction,
            "magnitude": change_magnitude
        },
        "description": description
    }

def verbalize_anomalies(data, rate_column, threshold=2.0):
    """
    Generate a natural language description of anomalies in the adoption rate data.
    
    Args:
        data (pandas.DataFrame): DataFrame containing adoption rate data
        rate_column (str): Column name for the adoption rate
        threshold (float): Z-score threshold for anomaly detection
        
    Returns:
        dict: Dictionary containing anomaly description with these keys:
            - anomalies: List of anomalies
            - high_anomalies: Unusually high values
            - low_anomalies: Unusually low values
            - description: Consolidated natural language description
    """
    if data.empty or rate_column not in data.columns:
        return {
            "description": "No data available for anomaly detection."
        }
    
    # Ensure data is sorted by date
    data = data.sort_values('Date')
    
    # Calculate Z-scores
    mean_rate = data[rate_column].mean()
    std_dev = data[rate_column].std()
    
    if std_dev == 0:  # Avoid division by zero
        return {
            "description": "No variability in the data, no anomalies detected."
        }
    
    data['z_score'] = (data[rate_column] - mean_rate) / std_dev
    
    # Identify anomalies
    anomalies = data[abs(data['z_score']) > threshold].copy()
    high_anomalies = data[data['z_score'] > threshold].copy()
    low_anomalies = data[data['z_score'] < -threshold].copy()
    
    # Format anomalies for output
    anomaly_list = []
    for _, row in anomalies.iterrows():
        anomaly_list.append({
            "date": row['Date'],
            "value": row[rate_column],
            "z_score": row['z_score'],
            "type": "high" if row['z_score'] > 0 else "low",
            "description": f"{'Unusually high' if row['z_score'] > 0 else 'Unusually low'} value of {row[rate_column]:.2f}% on {row['Date'].strftime('%Y-%m-%d')} (z-score: {row['z_score']:.2f})"
        })
    
    # Generate consolidated description
    total_anomalies = len(anomaly_list)
    high_count = len(high_anomalies)
    low_count = len(low_anomalies)
    
    if total_anomalies == 0:
        description = f"No anomalies detected using a threshold of {threshold} standard deviations from the mean."
    else:
        description = f"Detected {total_anomalies} anomalies using a threshold of {threshold} standard deviations "
        description += f"from the mean value of {mean_rate:.2f}%. "
        
        if high_count > 0:
            description += f"There are {high_count} unusually high values, "
            max_high = high_anomalies[rate_column].max()
            max_high_date = high_anomalies.loc[high_anomalies[rate_column].idxmax(), 'Date']
            description += f"with the most extreme being {max_high:.2f}% on {max_high_date.strftime('%Y-%m-%d')}. "
        
        if low_count > 0:
            description += f"There are {low_count} unusually low values, "
            min_low = low_anomalies[rate_column].min()
            min_low_date = low_anomalies.loc[low_anomalies[rate_column].idxmin(), 'Date']
            description += f"with the most extreme being {min_low:.2f}% on {min_low_date.strftime('%Y-%m-%d')}. "
        
        description += f"These anomalies may represent significant events or data issues that warrant investigation."
    
    return {
        "anomalies": anomaly_list,
        "high_anomalies": high_anomalies[[rate_column, 'Date', 'z_score']].to_dict('records') if not high_anomalies.empty else [],
        "low_anomalies": low_anomalies[[rate_column, 'Date', 'z_score']].to_dict('records') if not low_anomalies.empty else [],
        "total_count": total_anomalies,
        "high_count": high_count,
        "low_count": low_count,
        "threshold": threshold,
        "mean": mean_rate,
        "std_dev": std_dev,
        "description": description
    }

def generate_future_outlook(data, rate_column, forecast_periods=3):
    """
    Generate a future outlook for adoption rates based on historical trends.
    
    Args:
        data (pandas.DataFrame): DataFrame containing adoption rate data
        rate_column (str): Column name for the adoption rate
        forecast_periods (int): Number of periods to forecast
        
    Returns:
        dict: Dictionary containing future outlook with these keys:
            - forecast: Predicted values
            - confidence_interval: Range of possible values
            - trend_parameters: Parameters of the trend model
            - description: Consolidated natural language description
    """
    if data.empty or rate_column not in data.columns or len(data) < 3:
        return {
            "description": "Insufficient data for forecasting. At least three data points are required."
        }
    
    # Ensure data is sorted by date
    data = data.sort_values('Date')
    
    # Extract rate values and dates
    rates = data[rate_column].values
    dates = data['Date'].values
    
    # Linear regression to predict future values
    x = np.arange(len(rates))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, rates)
    
    # Predict future values
    future_x = np.arange(len(rates), len(rates) + forecast_periods)
    future_values = slope * future_x + intercept
    
    # Calculate confidence interval
    confidence = 0.95
    n = len(rates)
    mean_x = np.mean(x)
    t_value = stats.t.ppf((1 + confidence) / 2, n - 2)
    
    # Calculate prediction intervals
    prediction_intervals = []
    for x_val in future_x:
        se_predictor = std_err * np.sqrt(1 + 1/n + (x_val - mean_x)**2 / np.sum((x - mean_x)**2))
        margin_of_error = t_value * se_predictor
        prediction_intervals.append(margin_of_error)
    
    # Determine date intervals based on data frequency
    date_diffs = []
    for i in range(1, len(dates)):
        date_diffs.append((dates[i] - dates[i-1]).days)
    
    avg_interval = int(np.mean(date_diffs)) if date_diffs else 30
    
    # Generate future dates
    future_dates = []
    last_date = dates[-1]
    for i in range(forecast_periods):
        future_date = last_date + timedelta(days=avg_interval * (i + 1))
        future_dates.append(future_date)
    
    # Build forecast data
    forecast_data = []
    for i in range(forecast_periods):
        forecast_data.append({
            "date": future_dates[i],
            "value": max(0, future_values[i]),  # Ensure non-negative values
            "lower_bound": max(0, future_values[i] - prediction_intervals[i]),
            "upper_bound": future_values[i] + prediction_intervals[i],
            "description": f"Forecast for {future_dates[i].strftime('%Y-%m-%d')}: {future_values[i]:.2f}% "
                          f"({max(0, future_values[i] - prediction_intervals[i]):.2f}% - {future_values[i] + prediction_intervals[i]:.2f}%)"
        })
    
    # Determine trend confidence
    trend_confidence = "uncertain"
    if r_value**2 > 0.7:
        trend_confidence = "high"
    elif r_value**2 > 0.3:
        trend_confidence = "moderate"
    
    # Determine trend direction
    trend_direction = "stable"
    if slope > 0.05:
        trend_direction = "increasing"
    elif slope < -0.05:
        trend_direction = "decreasing"
    
    # Generate consolidated description
    last_value = rates[-1]
    final_forecast = future_values[-1]
    
    description = f"Based on historical data, the adoption rate is expected to "
    if trend_direction == "increasing":
        description += f"increase from the current {last_value:.2f}% to approximately {final_forecast:.2f}% "
    elif trend_direction == "decreasing":
        description += f"decrease from the current {last_value:.2f}% to approximately {final_forecast:.2f}% "
    else:
        description += f"remain stable around {last_value:.2f}% to {final_forecast:.2f}% "
    
    description += f"over the next {forecast_periods} periods, ending around {future_dates[-1].strftime('%Y-%m-%d')}. "
    
    description += f"This forecast has {trend_confidence} confidence based on historical patterns "
    description += f"(R²: {r_value**2:.2f}). "
    
    if trend_confidence != "high":
        description += "Consider this a general indication rather than a precise prediction. "
    
    description += f"The forecast range for the final period is {max(0, future_values[-1] - prediction_intervals[-1]):.2f}% to {future_values[-1] + prediction_intervals[-1]:.2f}%."
    
    return {
        "forecast": forecast_data,
        "confidence_interval": prediction_intervals,
        "trend_parameters": {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err,
            "direction": trend_direction,
            "confidence": trend_confidence
        },
        "description": description
    } 