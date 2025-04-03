"""
Current State Analysis Module

This module provides functions for analyzing and describing the current
state of adoption rates, including comparing the current rate with historical
averages, recent trends, and providing overall context.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

def describe_current_state(data, as_of_date=None, rate_columns=None):
    """
    Generate a comprehensive description of the current state of adoption rates.
    
    Args:
        data (pandas.DataFrame): DataFrame containing adoption rate data
        as_of_date (datetime, optional): The date to consider as "current". 
                                       If None, uses the most recent date in the data.
        rate_columns (list, optional): List of rate columns to analyze.
                                      If None, analyzes all available rate columns.
    
    Returns:
        dict: Dictionary containing current state information with these keys:
            - current_state: Current values for each available rate metric
            - historical_comparison: Comparison with historical averages
            - recent_summary: Summary of recent trends
            - description: Consolidated natural language description
    """
    if data.empty:
        return {
            "description": "No data available for current state analysis."
        }
    
    # Ensure data is sorted by date
    data = data.sort_values('Date')
    
    # If as_of_date is not provided, use the most recent date in the data
    if as_of_date is None:
        as_of_date = data['Date'].max()
    else:
        # Make sure as_of_date is in the data, otherwise use closest date
        if as_of_date not in data['Date'].values:
            closest_date = data.iloc[abs(data['Date'] - as_of_date).argmin()]['Date']
            logger.info(f"Specified date {as_of_date} not found in data. Using closest date: {closest_date}")
            as_of_date = closest_date
    
    # Determine which rate columns to analyze
    if rate_columns is None:
        # Default columns that typically exist in adoption rate data
        possible_rate_columns = [
            'DOverallAdoptionRate', 'WOverallAdoptionRate', 
            'MOverallAdoptionRate', 'YOverallAdoptionRate'
        ]
        rate_columns = [col for col in possible_rate_columns if col in data.columns]
    
    # Get current values for each rate column
    current_row = data[data['Date'] == as_of_date]
    
    if current_row.empty:
        return {
            "description": f"No data available for the specified date: {as_of_date.strftime('%Y-%m-%d')}."
        }
    
    current_state = {}
    for col in rate_columns:
        current_state[col] = current_row[col].iloc[0]
    
    # Generate historical comparison
    historical_comparison = {}
    for col in rate_columns:
        # Calculate historical metrics
        historical_mean = data[col].mean()
        historical_median = data[col].median()
        
        # Calculate percentile of current value within historical distribution
        percentile = (data[col] <= current_state[col]).mean() * 100
        
        # Determine if current value is a record high or low
        is_record_high = current_state[col] >= data[col].max()
        is_record_low = current_state[col] <= data[col].min()
        
        historical_comparison[col] = {
            "current": current_state[col],
            "historical_mean": historical_mean,
            "historical_median": historical_median,
            "percentile": percentile,
            "vs_mean_absolute": current_state[col] - historical_mean,
            "vs_mean_percent": (current_state[col] / historical_mean - 1) * 100 if historical_mean > 0 else None,
            "is_record_high": is_record_high,
            "is_record_low": is_record_low
        }
    
    # Generate recent summary (last 30, 90, 365 days)
    recent_summary = {}
    
    # Define time periods for recent summary
    periods = {
        '30d': timedelta(days=30),
        '90d': timedelta(days=90),
        '365d': timedelta(days=365)
    }
    
    for period_name, period_delta in periods.items():
        period_start = as_of_date - period_delta
        period_data = data[data['Date'] >= period_start]
        
        # Skip if we don't have enough data for this period
        if len(period_data) < 2:
            continue
            
        period_summary = {}
        for col in rate_columns:
            period_mean = period_data[col].mean()
            period_trend = period_data[col].iloc[-1] - period_data[col].iloc[0]
            
            period_summary[col] = {
                "mean": period_mean,
                "min": period_data[col].min(),
                "max": period_data[col].max(),
                "trend": period_trend,
                "trend_percent": (period_trend / period_data[col].iloc[0]) * 100 if period_data[col].iloc[0] > 0 else None
            }
        
        recent_summary[period_name] = period_summary
    
    # Generate consolidated description
    # Focus on monthly rate for the main description
    main_rate_col = 'MOverallAdoptionRate' if 'MOverallAdoptionRate' in rate_columns else rate_columns[0]
    
    description = f"As of {as_of_date.strftime('%Y-%m-%d')}, "
    
    # Current state
    description += f"the {_format_rate_name(main_rate_col)} is {current_state[main_rate_col]:.2f}%, "
    
    # Historical comparison
    comparison = historical_comparison[main_rate_col]
    if comparison["is_record_high"]:
        description += "which is an all-time high. "
    elif comparison["is_record_low"]:
        description += "which is an all-time low. "
    else:
        description += f"which is in the {comparison['percentile']:.0f}th percentile historically. "
    
    description += f"This is {abs(comparison['vs_mean_absolute']):.2f}% "
    description += "above" if comparison['vs_mean_absolute'] >= 0 else "below"
    description += f" the historical average of {comparison['historical_mean']:.2f}%. "
    
    # Recent trend (90 days)
    if '90d' in recent_summary and main_rate_col in recent_summary['90d']:
        trend_90d = recent_summary['90d'][main_rate_col]['trend']
        if abs(trend_90d) > 0.01:  # Only mention if there's a meaningful change
            description += f"Over the past 90 days, the rate has "
            description += "increased" if trend_90d > 0 else "decreased"
            description += f" by {abs(trend_90d):.2f}%. "
    
    # Add information about other metrics if available
    other_rates = [col for col in rate_columns if col != main_rate_col]
    if other_rates:
        description += "Other current metrics include: "
        for i, col in enumerate(other_rates):
            if i > 0:
                description += ", "
            description += f"{_format_rate_name(col)}: {current_state[col]:.2f}%"
        description += "."
    
    return {
        "current_state": current_state,
        "historical_comparison": historical_comparison,
        "recent_summary": recent_summary,
        "as_of_date": as_of_date,
        "description": description
    }

def _format_rate_name(column_name):
    """
    Format a rate column name to a more readable form.
    
    Args:
        column_name (str): The column name to format
        
    Returns:
        str: Formatted column name
    """
    if column_name == 'DOverallAdoptionRate':
        return 'daily adoption rate'
    elif column_name == 'WOverallAdoptionRate':
        return 'weekly adoption rate'
    elif column_name == 'MOverallAdoptionRate':
        return 'monthly adoption rate'
    elif column_name == 'YOverallAdoptionRate':
        return 'yearly adoption rate'
    else:
        return column_name.replace('OverallAdoptionRate', '').lower() + ' adoption rate' 