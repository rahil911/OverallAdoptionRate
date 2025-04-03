"""
Statistics Module

This module provides functions for generating summary statistics
about adoption rate data, including basic stats, period-specific
statistics, extrema identification, and trend calculations.
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.signal as signal

# Configure logging
logger = logging.getLogger(__name__)

def generate_summary_statistics(data, rate_column="MOverallAdoptionRate"):
    """
    Generate basic summary statistics for adoption rate data.
    
    Args:
        data (pandas.DataFrame): Adoption rate data
        rate_column (str): Column name for the rate data to analyze
        
    Returns:
        dict: Dictionary containing summary statistics
    """
    if data.empty or rate_column not in data.columns:
        logger.warning(f"No data provided or column {rate_column} not found")
        return {
            "description": "No data available for summary statistics."
        }
    
    # Make a copy to avoid modifying the original DataFrame
    data_copy = data.copy()
    
    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data_copy['Date']):
        data_copy['Date'] = pd.to_datetime(data_copy['Date'])
    
    # Basic statistics
    mean_value = data_copy[rate_column].mean()
    median_value = data_copy[rate_column].median()
    min_value = data_copy[rate_column].min()
    max_value = data_copy[rate_column].max()
    std_value = data_copy[rate_column].std()
    range_value = max_value - min_value
    count = data_copy[rate_column].count()
    
    # Calculate percentiles
    percentiles = [10, 25, 50, 75, 90]
    percentile_values = {f"percentile_{p}": np.percentile(data_copy[rate_column].dropna(), p) for p in percentiles}
    
    # Recent trends
    # Sort by date and get the last 90 days
    data_copy = data_copy.sort_values('Date')
    recent_data = data_copy[data_copy['Date'] >= data_copy['Date'].max() - pd.Timedelta(days=90)]
    
    if len(recent_data) >= 2:
        recent_trend = recent_data[rate_column].iloc[-1] - recent_data[rate_column].iloc[0]
        recent_trend_pct = (recent_trend / recent_data[rate_column].iloc[0] * 100) if recent_data[rate_column].iloc[0] > 0 else np.nan
        recent_mean = recent_data[rate_column].mean()
        recent_mean_vs_overall = recent_mean - mean_value
    else:
        recent_trend = np.nan
        recent_trend_pct = np.nan
        recent_mean = np.nan
        recent_mean_vs_overall = np.nan
    
    # Generate description
    metric_name = rate_column.replace('OverallAdoptionRate', ' adoption rate')
    description = f"Summary Statistics for {metric_name}: "
    
    description += f"The average rate is {mean_value:.2f}% with a median of {median_value:.2f}%. "
    description += f"Values range from {min_value:.2f}% to {max_value:.2f}%, "
    description += f"with a standard deviation of {std_value:.2f}%. "
    
    if not np.isnan(recent_trend):
        trend_direction = "increased" if recent_trend > 0 else "decreased" if recent_trend < 0 else "remained stable"
        description += f"Over the last 90 days, the rate has {trend_direction} "
        
        if abs(recent_trend) > 0.01:  # Only include the amount if there's a meaningful change
            description += f"by {abs(recent_trend):.2f}%. "
        else:
            description += ". "
    
    return {
        "mean": mean_value,
        "median": median_value,
        "min": min_value,
        "max": max_value,
        "std": std_value,
        "range": range_value,
        "count": count,
        "percentiles": percentile_values,
        "recent_trend": recent_trend,
        "recent_trend_pct": recent_trend_pct,
        "recent_mean": recent_mean,
        "recent_mean_vs_overall": recent_mean_vs_overall,
        "description": description
    }

def generate_period_statistics(data, period_type="month", rate_column="MOverallAdoptionRate"):
    """
    Generate period-specific statistics for adoption rate data.
    
    Args:
        data (pandas.DataFrame): Adoption rate data
        period_type (str): Type of period to analyze ('day', 'week', 'month', 'year')
        rate_column (str): Column name for the rate data to analyze
        
    Returns:
        dict: Dictionary containing period-specific statistics
    """
    if data.empty or rate_column not in data.columns:
        logger.warning(f"No data provided or column {rate_column} not found")
        return {
            "description": "No data available for period statistics."
        }
    
    # Make a copy to avoid modifying the original DataFrame
    data_copy = data.copy()
    
    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data_copy['Date']):
        data_copy['Date'] = pd.to_datetime(data_copy['Date'])
    
    # Create period column based on period_type
    if period_type == "day":
        data_copy['period'] = data_copy['Date'].dt.strftime('%Y-%m-%d')
    elif period_type == "week":
        data_copy['period'] = data_copy['Date'].dt.strftime('%Y-W%U')
    elif period_type == "year":
        data_copy['period'] = data_copy['Date'].dt.year
    else:  # Default to month
        data_copy['period'] = data_copy['Date'].dt.strftime('%Y-%m')
        period_type = "month"
    
    # Group by period and calculate statistics
    period_stats = data_copy.groupby('period')[rate_column].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('std', 'std'),
        ('count', 'count')
    ]).reset_index()
    
    # Calculate period-over-period changes
    period_stats = period_stats.sort_values('period')
    period_stats['prev_mean'] = period_stats['mean'].shift(1)
    period_stats['change'] = period_stats['mean'] - period_stats['prev_mean']
    period_stats['change_pct'] = (period_stats['change'] / period_stats['prev_mean'] * 100)
    
    # Find best and worst periods
    if len(period_stats) > 0:
        best_period = period_stats.loc[period_stats['mean'].idxmax()]
        worst_period = period_stats.loc[period_stats['mean'].idxmin()]
        
        # Find period with highest growth
        growth_periods = period_stats.dropna(subset=['change'])
        if len(growth_periods) > 0:
            best_growth = growth_periods.loc[growth_periods['change'].idxmax()]
            worst_growth = growth_periods.loc[growth_periods['change'].idxmin()]
        else:
            best_growth = None
            worst_growth = None
    else:
        best_period = None
        worst_period = None
        best_growth = None
        worst_growth = None
    
    # Calculate overall trend
    if len(period_stats) >= 2:
        x = np.arange(len(period_stats))
        y = period_stats['mean'].values
        trend, _, _, _, _ = np.polyfit(x, y, 1, full=True)
        trend_direction = "increasing" if trend[0] > 0 else "decreasing" if trend[0] < 0 else "stable"
        trend_strength = abs(trend[0])
    else:
        trend_direction = "unknown"
        trend_strength = 0
    
    # Generate description
    if len(period_stats) == 0:
        description = f"There is no {period_type} data available for analysis."
    else:
        description = f"Period Analysis for {period_type}s: "
        
        # Trend
        if trend_direction != "unknown":
            description += f"The overall trend is {trend_direction}. "
        
        # Best and worst periods
        if best_period is not None and worst_period is not None:
            metric_name = rate_column.replace('OverallAdoptionRate', ' adoption rate')
            description += f"The {metric_name} was highest in {best_period['period']} "
            description += f"({best_period['mean']:.2f}%) and lowest in {worst_period['period']} "
            description += f"({worst_period['mean']:.2f}%). "
        
        # Growth
        if best_growth is not None and worst_growth is not None:
            description += f"The largest increase was {best_growth['change']:.2f}% in {best_growth['period']}, "
            description += f"while the largest decrease was {worst_growth['change']:.2f}% in {worst_growth['period']}."
    
    return {
        "period_stats": period_stats.to_dict('records') if len(period_stats) > 0 else [],
        "best_period": best_period.to_dict() if best_period is not None else None,
        "worst_period": worst_period.to_dict() if worst_period is not None else None,
        "best_growth": best_growth.to_dict() if best_growth is not None else None,
        "worst_growth": worst_growth.to_dict() if worst_growth is not None else None,
        "trend_direction": trend_direction,
        "trend_strength": float(trend_strength) if isinstance(trend_strength, (int, float, np.number)) else 0,
        "period_type": period_type,
        "description": description
    }

def identify_extrema(data, rate_column="MOverallAdoptionRate", top_n=3):
    """
    Identify extreme values (peaks and valleys) in the adoption rate data.
    
    Args:
        data (pandas.DataFrame): Adoption rate data
        rate_column (str): Column name for the rate data to analyze
        top_n (int): Number of top extrema to identify
        
    Returns:
        dict: Dictionary containing identified extrema
    """
    if data.empty or rate_column not in data.columns:
        logger.warning(f"No data provided or column {rate_column} not found")
        return {
            "description": "No data available for extrema identification."
        }
    
    # Make a copy to avoid modifying the original DataFrame
    data_copy = data.copy()
    
    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data_copy['Date']):
        data_copy['Date'] = pd.to_datetime(data_copy['Date'])
    
    # Sort by date
    data_copy = data_copy.sort_values('Date')
    
    # Need at least 3 points to identify peaks and valleys
    if len(data_copy) < 3:
        return {
            "description": "Insufficient data to identify extrema. Need at least 3 data points."
        }
    
    # Extract the rate values
    values = data_copy[rate_column].values
    
    # Use signal.find_peaks to identify peaks and valleys
    # Valleys are the peaks of the negative signal
    peaks, _ = signal.find_peaks(values, prominence=np.std(values) * 0.5)
    valleys, _ = signal.find_peaks(-values, prominence=np.std(values) * 0.5)
    
    # If we couldn't find any with prominence, try without
    if len(peaks) == 0:
        peaks, _ = signal.find_peaks(values)
    if len(valleys) == 0:
        valleys, _ = signal.find_peaks(-values)
    
    # Convert peak and valley indices to DataFrames
    if len(peaks) > 0:
        peak_rows = data_copy.iloc[peaks]
        peak_rows = peak_rows.sort_values(rate_column, ascending=False).head(top_n)
        peaks_list = [
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "value": row[rate_column],
                "index": i
            }
            for i, row in peak_rows.iterrows()
        ]
    else:
        peaks_list = []
    
    if len(valleys) > 0:
        valley_rows = data_copy.iloc[valleys]
        valley_rows = valley_rows.sort_values(rate_column).head(top_n)
        valleys_list = [
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "value": row[rate_column],
                "index": i
            }
            for i, row in valley_rows.iterrows()
        ]
    else:
        valleys_list = []
    
    # Also identify all-time high and low
    all_time_high_row = data_copy.loc[data_copy[rate_column].idxmax()]
    all_time_low_row = data_copy.loc[data_copy[rate_column].idxmin()]
    
    all_time_high = {
        "date": all_time_high_row["Date"].strftime("%Y-%m-%d"),
        "value": all_time_high_row[rate_column],
        "index": all_time_high_row.name
    }
    
    all_time_low = {
        "date": all_time_low_row["Date"].strftime("%Y-%m-%d"),
        "value": all_time_low_row[rate_column],
        "index": all_time_low_row.name
    }
    
    # Generate description
    metric_name = rate_column.replace('OverallAdoptionRate', ' adoption rate')
    description = f"Extrema for {metric_name}: "
    
    description += f"The all-time high is {all_time_high['value']:.2f}% on {all_time_high['date']}. "
    description += f"The all-time low is {all_time_low['value']:.2f}% on {all_time_low['date']}. "
    
    if peaks_list:
        description += f"Identified {len(peaks_list)} significant peaks (local maxima). "
        description += f"The most notable peak is {peaks_list[0]['value']:.2f}% on {peaks_list[0]['date']}. "
    
    if valleys_list:
        description += f"Identified {len(valleys_list)} significant valleys (local minima). "
        description += f"The most notable valley is {valleys_list[0]['value']:.2f}% on {valleys_list[0]['date']}. "
    
    return {
        "peaks": peaks_list,
        "valleys": valleys_list,
        "all_time_high": all_time_high,
        "all_time_low": all_time_low,
        "description": description
    }

def calculate_statistical_trends(data, rate_column="MOverallAdoptionRate"):
    """
    Calculate statistical trends for adoption rate data.
    
    Args:
        data (pandas.DataFrame): Adoption rate data
        rate_column (str): Column name for the rate data to analyze
        
    Returns:
        dict: Dictionary containing identified trends
    """
    if data.empty or rate_column not in data.columns:
        logger.warning(f"No data provided or column {rate_column} not found")
        return {
            "description": "No data available for trend analysis."
        }
    
    # Make a copy to avoid modifying the original DataFrame
    data_copy = data.copy()
    
    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data_copy['Date']):
        data_copy['Date'] = pd.to_datetime(data_copy['Date'])
    
    # Sort by date
    data_copy = data_copy.sort_values('Date')
    
    # Need at least 2 points to calculate trends
    if len(data_copy) < 2:
        return {
            "description": "Insufficient data to calculate trends. Need at least 2 data points."
        }
    
    # Calculate simple moving averages
    window_sizes = [3, 7, 30]
    moving_averages = {}
    
    for window in window_sizes:
        if len(data_copy) >= window:
            ma_name = f"MA{window}"
            moving_averages[ma_name] = data_copy[rate_column].rolling(window=window).mean().iloc[-1]
    
    # Calculate linear regression trend
    try:
        # Use scipy.stats.linregress for linear regression
        # Convert dates to numeric (days since first date)
        days_since_start = (data_copy['Date'] - data_copy['Date'].min()).dt.days
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(days_since_start, data_copy[rate_column])
        
        # Calculate trend strength
        # r_value is the correlation coefficient
        trend_strength = abs(r_value)
        
        # Determine trend direction
        if abs(slope) < 0.0001:  # Very small slope
            trend_direction = "stable"
        else:
            trend_direction = "increasing" if slope > 0 else "decreasing"
        
        # Determine significance
        # p-value < 0.05 is considered statistically significant
        trend_significant = p_value < 0.05
        
        # Calculate recent trend (last 30 days or all data if less)
        days_to_check = min(30, len(data_copy))
        recent_data = data_copy.iloc[-days_to_check:]
        
        if len(recent_data) >= 2:
            recent_days = (recent_data['Date'] - recent_data['Date'].min()).dt.days
            recent_slope, recent_intercept, recent_r, recent_p, recent_std_err = stats.linregress(
                recent_days, recent_data[rate_column])
            
            if abs(recent_slope) < 0.0001:
                recent_trend = "stable"
            else:
                recent_trend = "increasing" if recent_slope > 0 else "decreasing"
                
            recent_significant = recent_p < 0.05
        else:
            recent_trend = "unknown"
            recent_slope = 0
            recent_r = 0
            recent_p = 1
            recent_significant = False
        
        # Generate description
        description = f"Trend Analysis: "
        
        if trend_significant:
            description += f"There is a statistically significant {trend_direction} trend "
            description += f"(p={p_value:.3f}, r²={r_value**2:.2f}). "
        else:
            description += f"There is a {trend_direction} trend, but it is not statistically significant "
            description += f"(p={p_value:.3f}, r²={r_value**2:.2f}). "
        
        if recent_trend != "unknown":
            description += f"The recent trend (last {days_to_check} days) is {recent_trend}. "
            
            if recent_trend != trend_direction:
                description += f"This suggests a potential change in direction. "
        
        # Add moving average information
        if moving_averages:
            description += "Moving averages: "
            ma_descriptions = []
            for name, value in moving_averages.items():
                ma_descriptions.append(f"{name}: {value:.2f}%")
            description += ", ".join(ma_descriptions) + ". "
        
        return {
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "trend_significant": trend_significant,
            "slope": slope,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "recent_trend": recent_trend,
            "recent_slope": recent_slope,
            "recent_r_squared": recent_r ** 2 if recent_trend != "unknown" else None,
            "recent_p_value": recent_p if recent_trend != "unknown" else None,
            "moving_averages": moving_averages,
            "description": description
        }
    
    except Exception as e:
        logger.error(f"Error calculating trends: {str(e)}")
        return {
            "description": f"Error calculating trends: {str(e)}"
        } 