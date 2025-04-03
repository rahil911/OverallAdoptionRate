"""
Statistics Module

This module provides functions for generating summary statistics about
adoption rate data, including basic stats, period-specific statistics,
extrema identification, and trend calculations.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

def generate_summary_statistics(data, rate_column):
    """
    Generate summary statistics for adoption rate data.
    
    Args:
        data (pandas.DataFrame): DataFrame containing adoption rate data
        rate_column (str): Column name for the adoption rate
        
    Returns:
        dict: Dictionary containing summary statistics with these keys:
            - mean: Average value
            - median: Median value
            - min: Minimum value
            - max: Maximum value
            - std_dev: Standard deviation
            - percentiles: Key percentiles (25%, 75%, 90%)
            - sample_size: Number of data points
            - date_range: Range of dates covered
            - description: Consolidated natural language description
    """
    if data.empty or rate_column not in data.columns:
        return {
            "description": "No data available for summary statistics."
        }
    
    # Calculate basic statistics
    mean_value = data[rate_column].mean()
    median_value = data[rate_column].median()
    min_value = data[rate_column].min()
    max_value = data[rate_column].max()
    std_dev = data[rate_column].std()
    
    # Calculate percentiles
    percentile_25 = data[rate_column].quantile(0.25)
    percentile_75 = data[rate_column].quantile(0.75)
    percentile_90 = data[rate_column].quantile(0.90)
    
    # Get date range and sample size
    date_range = (data['Date'].min(), data['Date'].max())
    sample_size = len(data)
    
    # Generate consolidated description
    description = f"The adoption rate averaged {mean_value:.2f}% with a median of {median_value:.2f}% "
    description += f"over the analyzed period ({date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}). "
    description += f"Values ranged from a minimum of {min_value:.2f}% to a maximum of {max_value:.2f}%, "
    description += f"with a standard deviation of {std_dev:.2f}%. "
    description += f"25% of values were below {percentile_25:.2f}%, while 75% were below {percentile_75:.2f}%. "
    description += f"This analysis is based on {sample_size} data points."
    
    return {
        "mean": mean_value,
        "median": median_value,
        "min": min_value,
        "max": max_value,
        "std_dev": std_dev,
        "percentiles": {
            "25%": percentile_25,
            "75%": percentile_75,
            "90%": percentile_90
        },
        "sample_size": sample_size,
        "date_range": date_range,
        "description": description
    }

def generate_period_statistics(data, rate_column, period_type="month"):
    """
    Generate statistics for adoption rate data grouped by period.
    
    Args:
        data (pandas.DataFrame): DataFrame containing adoption rate data
        rate_column (str): Column name for the adoption rate
        period_type (str): Type of period to group by (day, week, month, quarter, year)
        
    Returns:
        dict: Dictionary containing period statistics with these keys:
            - period_stats: Statistics for each period
            - best_period: Period with highest average rate
            - worst_period: Period with lowest average rate
            - description: Consolidated natural language description
    """
    if data.empty or rate_column not in data.columns:
        return {
            "description": "No data available for period statistics."
        }
    
    # Ensure data is sorted by date
    data = data.sort_values('Date')
    
    # Create period column based on period_type
    if period_type == "day":
        data['Period'] = data['Date'].dt.strftime('%Y-%m-%d')
    elif period_type == "week":
        data['Period'] = data['Date'].dt.strftime('%Y-W%U')
    elif period_type == "month":
        data['Period'] = data['Date'].dt.strftime('%Y-%m')
    elif period_type == "quarter":
        data['Period'] = data['Date'].dt.year.astype(str) + '-Q' + ((data['Date'].dt.month - 1) // 3 + 1).astype(str)
    else:  # year
        data['Period'] = data['Date'].dt.year.astype(str)
    
    # Group by period and calculate statistics
    period_stats = data.groupby('Period')[rate_column].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('std', 'std'),
        ('count', 'count')
    ]).reset_index()
    
    # Find best and worst periods
    best_period = period_stats.loc[period_stats['mean'].idxmax()]
    worst_period = period_stats.loc[period_stats['mean'].idxmin()]
    
    # Format period type for description
    period_name = period_type
    if period_type == "week":
        period_name = "weekly"
    elif period_type == "month":
        period_name = "monthly"
    elif period_type == "quarter":
        period_name = "quarterly"
    elif period_type == "year":
        period_name = "yearly"
    
    # Generate consolidated description
    description = f"On a {period_name} basis, the adoption rate varied significantly. "
    description += f"The best {period_type} was {best_period['Period']} with an average of {best_period['mean']:.2f}%, "
    description += f"while the worst was {worst_period['Period']} with an average of {worst_period['mean']:.2f}%. "
    
    # Add trend information
    if len(period_stats) >= 3:
        # Check if there's a trend in the recent periods
        last_periods = period_stats.tail(3)
        
        if last_periods['mean'].iloc[0] < last_periods['mean'].iloc[1] < last_periods['mean'].iloc[2]:
            description += f"The recent trend shows consistent improvement over the last 3 {period_type}s."
        elif last_periods['mean'].iloc[0] > last_periods['mean'].iloc[1] > last_periods['mean'].iloc[2]:
            description += f"The recent trend shows consistent decline over the last 3 {period_type}s."
        else:
            description += f"The recent trend shows mixed results over the last 3 {period_type}s."
    
    return {
        "period_stats": period_stats.to_dict('records'),
        "best_period": {
            "period": best_period['Period'],
            "mean": best_period['mean'],
            "median": best_period['median'],
            "min": best_period['min'],
            "max": best_period['max'],
            "std": best_period['std'],
            "count": best_period['count']
        },
        "worst_period": {
            "period": worst_period['Period'],
            "mean": worst_period['mean'],
            "median": worst_period['median'],
            "min": worst_period['min'],
            "max": worst_period['max'],
            "std": worst_period['std'],
            "count": worst_period['count']
        },
        "period_type": period_type,
        "description": description
    }

def identify_extrema(data, rate_column):
    """
    Identify extreme values (peaks and valleys) in adoption rate data.
    
    Args:
        data (pandas.DataFrame): DataFrame containing adoption rate data
        rate_column (str): Column name for the adoption rate
        
    Returns:
        dict: Dictionary containing extrema information with these keys:
            - peaks: Notable peaks in the data
            - valleys: Notable valleys in the data
            - all_time_high: Highest value
            - all_time_low: Lowest value
            - description: Consolidated natural language description
    """
    if data.empty or rate_column not in data.columns:
        return {
            "description": "No data available for extrema identification."
        }
    
    # Ensure data is sorted by date
    data = data.sort_values('Date')
    
    # Find global extrema
    all_time_high = data.loc[data[rate_column].idxmax()]
    all_time_low = data.loc[data[rate_column].idxmin()]
    
    # Find local extrema (peaks and valleys)
    # A point is a peak if it's higher than the 2 points before and after it
    peaks = []
    valleys = []
    
    if len(data) >= 5:  # Need at least 5 points to find meaningful local extrema
        for i in range(2, len(data) - 2):
            current = data.iloc[i][rate_column]
            
            # Check if it's a peak
            if (current > data.iloc[i-2][rate_column] and 
                current > data.iloc[i-1][rate_column] and 
                current > data.iloc[i+1][rate_column] and 
                current > data.iloc[i+2][rate_column]):
                
                peaks.append({
                    "date": data.iloc[i]['Date'],
                    "value": current,
                    "description": f"Peak of {current:.2f}% on {data.iloc[i]['Date'].strftime('%Y-%m-%d')}"
                })
            
            # Check if it's a valley
            if (current < data.iloc[i-2][rate_column] and 
                current < data.iloc[i-1][rate_column] and 
                current < data.iloc[i+1][rate_column] and 
                current < data.iloc[i+2][rate_column]):
                
                valleys.append({
                    "date": data.iloc[i]['Date'],
                    "value": current,
                    "description": f"Valley of {current:.2f}% on {data.iloc[i]['Date'].strftime('%Y-%m-%d')}"
                })
    
    # Sort peaks and valleys by magnitude
    peaks = sorted(peaks, key=lambda x: x["value"], reverse=True)
    valleys = sorted(valleys, key=lambda x: x["value"])
    
    # Generate consolidated description
    description = f"The all-time high adoption rate was {all_time_high[rate_column]:.2f}% on {all_time_high['Date'].strftime('%Y-%m-%d')}. "
    description += f"The all-time low was {all_time_low[rate_column]:.2f}% on {all_time_low['Date'].strftime('%Y-%m-%d')}. "
    
    if peaks:
        description += f"Notable peaks include: "
        for i, peak in enumerate(peaks[:3]):  # List up to top 3 peaks
            if i > 0:
                description += ", "
            description += f"{peak['value']:.2f}% on {peak['date'].strftime('%Y-%m-%d')}"
        description += ". "
    
    if valleys:
        description += f"Notable valleys include: "
        for i, valley in enumerate(valleys[:3]):  # List up to top 3 valleys
            if i > 0:
                description += ", "
            description += f"{valley['value']:.2f}% on {valley['date'].strftime('%Y-%m-%d')}"
        description += "."
    
    return {
        "peaks": peaks,
        "valleys": valleys,
        "all_time_high": {
            "date": all_time_high['Date'],
            "value": all_time_high[rate_column]
        },
        "all_time_low": {
            "date": all_time_low['Date'],
            "value": all_time_low[rate_column]
        },
        "description": description
    }

def calculate_statistical_trends(data, rate_column):
    """
    Calculate statistical trends in adoption rate data.
    
    Args:
        data (pandas.DataFrame): DataFrame containing adoption rate data
        rate_column (str): Column name for the adoption rate
        
    Returns:
        dict: Dictionary containing trend information with these keys:
            - linear_regression: Parameters of linear regression
            - correlation: Correlation with time
            - seasonality: Detected seasonality patterns
            - forecast: Forecast for next value
            - description: Consolidated natural language description
    """
    if data.empty or rate_column not in data.columns or len(data) < 3:
        return {
            "description": "Insufficient data for trend analysis. At least three data points are required."
        }
    
    # Ensure data is sorted by date
    data = data.sort_values('Date')
    
    # Extract rate values and create index for linear regression
    rates = data[rate_column].values
    x = np.arange(len(rates))
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, rates)
    
    # Calculate correlation with time
    correlation = r_value
    r_squared = r_value ** 2
    
    # Determine if trend is statistically significant
    is_significant = p_value < 0.05
    
    # Check for seasonality (if we have enough data points)
    seasonality = None
    if len(rates) >= 12:
        # Calculate autocorrelation
        autocorr = np.correlate(rates - np.mean(rates), rates - np.mean(rates), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Look for peaks in autocorrelation (ignoring the first peak at lag 0)
        # A peak in autocorrelation at lag k indicates a potential seasonality of period k
        peaks = []
        for i in range(2, min(len(autocorr) - 1, 52)):  # Look for seasonality up to 52 periods
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
                peaks.append((i, autocorr[i]))
        
        if peaks:
            # Sort peaks by correlation strength
            peaks.sort(key=lambda x: x[1], reverse=True)
            top_period, top_corr = peaks[0]
            
            seasonality = {
                "period": top_period,
                "correlation": top_corr,
                "description": f"Detected potential seasonality with period of {top_period} data points (correlation: {top_corr:.2f})"
            }
    
    # Forecast next value using linear model
    next_x = len(rates)
    forecast = slope * next_x + intercept
    
    # Generate forecast confidence interval
    confidence = 0.95
    n = len(rates)
    mean_x = np.mean(x)
    t_value = stats.t.ppf((1 + confidence) / 2, n - 2)
    s_err = std_err * np.sqrt(1 + 1/n + (next_x - mean_x)**2 / np.sum((x - mean_x)**2))
    margin = t_value * s_err
    
    # Determine trend direction and strength
    trend_direction = "stable"
    if slope > 0.05:
        trend_direction = "increasing"
    elif slope < -0.05:
        trend_direction = "decreasing"
    
    trend_strength = "weak"
    if r_squared > 0.7:
        trend_strength = "strong"
    elif r_squared > 0.3:
        trend_strength = "moderate"
    
    # Generate consolidated description
    description = f"The adoption rate shows a {trend_strength} {trend_direction} trend "
    description += f"(slope: {slope:.4f} per time unit, R²: {r_squared:.2f}"
    
    if is_significant:
        description += f", statistically significant with p-value: {p_value:.4f}"
    else:
        description += f", not statistically significant with p-value: {p_value:.4f}"
    
    description += f"). "
    
    if seasonality:
        description += f"There appears to be a seasonal pattern repeating approximately every {seasonality['period']} time units. "
    
    description += f"Based on this trend, the next value is forecasted to be {forecast:.2f}% "
    description += f"(95% confidence interval: {forecast-margin:.2f}% to {forecast+margin:.2f}%)."
    
    return {
        "linear_regression": {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "r_squared": r_squared,
            "p_value": p_value,
            "std_err": std_err,
            "is_significant": is_significant
        },
        "correlation": {
            "with_time": correlation,
            "strength": trend_strength,
            "direction": trend_direction
        },
        "seasonality": seasonality,
        "forecast": {
            "next_value": forecast,
            "lower_bound": forecast - margin,
            "upper_bound": forecast + margin,
            "confidence_level": confidence
        },
        "description": description
    } 