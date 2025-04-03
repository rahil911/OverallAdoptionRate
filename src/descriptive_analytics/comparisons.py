"""
Comparisons Module

This module provides functions for comparing adoption rate data across different
time periods, benchmarks, and targets.
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

def compare_month_over_month(data, current_month=None, rate_column="MOverallAdoptionRate"):
    """
    Compare adoption rates month-over-month.
    
    Args:
        data (pandas.DataFrame): Adoption rate data with Date column
        current_month (str, optional): Month to use as reference (format: YYYY-MM)
        rate_column (str): Column name for the rate data to analyze
        
    Returns:
        dict: Dictionary containing month-over-month comparison with these keys:
            - current_month: Current month information
            - previous_month: Previous month information
            - change: Change between months
            - description: Natural language description of the comparison
    """
    if data.empty or rate_column not in data.columns:
        logger.warning(f"No data provided or column {rate_column} not found")
        return {
            "current_month": None,
            "previous_month": None,
            "description": "No adoption rate data available for month-over-month comparison."
        }
    
    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])
    
    # Add month column
    data['month'] = data['Date'].dt.strftime('%Y-%m')
    
    # Sort by date
    sorted_data = data.sort_values(by='Date')
    
    # If current_month not provided, use the most recent month
    if current_month is None:
        current_month = sorted_data['month'].iloc[-1]
    
    # Get unique months
    unique_months = sorted(sorted_data['month'].unique())
    
    # Find current and previous month
    if current_month not in unique_months:
        logger.warning(f"Current month {current_month} not found in data")
        return {
            "current_month": current_month,
            "previous_month": None,
            "description": f"No data available for month {current_month}."
        }
    
    current_idx = unique_months.index(current_month)
    if current_idx == 0:
        logger.warning(f"No previous month available for comparison with {current_month}")
        
        # Get current month data
        current_month_data = sorted_data[sorted_data['month'] == current_month]
        current_avg = current_month_data[rate_column].mean()
        
        return {
            "current_month": {
                "month": current_month,
                "average": current_avg,
                "data_points": len(current_month_data)
            },
            "previous_month": None,
            "description": f"The average {rate_column.replace('OverallAdoptionRate', ' adoption rate')} "
                         f"for {current_month} is {current_avg:.2f}%. No previous month data available for comparison."
        }
    
    previous_month = unique_months[current_idx - 1]
    
    # Get data for both months
    current_month_data = sorted_data[sorted_data['month'] == current_month]
    previous_month_data = sorted_data[sorted_data['month'] == previous_month]
    
    # Calculate averages
    current_avg = current_month_data[rate_column].mean()
    previous_avg = previous_month_data[rate_column].mean()
    
    # Calculate change
    absolute_change = current_avg - previous_avg
    percent_change = (absolute_change / previous_avg * 100) if previous_avg > 0 else float('inf')
    
    # Generate description
    direction = "increased" if absolute_change > 0 else "decreased" if absolute_change < 0 else "remained stable"
    
    metric_name = rate_column.replace('OverallAdoptionRate', ' adoption rate')
    description = f"Month-over-Month Comparison for {metric_name}: "
    
    if abs(absolute_change) < 0.1:
        description += f"The average rate for {current_month} is {current_avg:.2f}%, "
        description += f"which is virtually unchanged from {previous_avg:.2f}% in {previous_month}."
    else:
        description += f"The average rate for {current_month} is {current_avg:.2f}%, "
        description += f"which has {direction} by {abs(absolute_change):.2f} percentage points "
        
        if percent_change != float('inf'):
            description += f"({abs(percent_change):.1f}%) "
        
        description += f"from {previous_avg:.2f}% in {previous_month}."
    
    return {
        "current_month": {
            "month": current_month,
            "average": current_avg,
            "data_points": len(current_month_data)
        },
        "previous_month": {
            "month": previous_month,
            "average": previous_avg,
            "data_points": len(previous_month_data)
        },
        "change": {
            "absolute": absolute_change,
            "percent": percent_change,
            "direction": direction
        },
        "description": description
    }

def compare_quarter_over_quarter(data, current_quarter=None, rate_column="MOverallAdoptionRate"):
    """
    Compare adoption rates quarter-over-quarter.
    
    Args:
        data (pandas.DataFrame): Adoption rate data with Date column
        current_quarter (str, optional): Quarter to use as reference (format: YYYY-Q#)
        rate_column (str): Column name for the rate data to analyze
        
    Returns:
        dict: Dictionary containing quarter-over-quarter comparison with these keys:
            - current_quarter: Current quarter information
            - previous_quarter: Previous quarter information
            - change: Change between quarters
            - description: Natural language description of the comparison
    """
    if data.empty or rate_column not in data.columns:
        logger.warning(f"No data provided or column {rate_column} not found")
        return {
            "current_quarter": None,
            "previous_quarter": None,
            "description": "No adoption rate data available for quarter-over-quarter comparison."
        }
    
    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])
    
    # Add quarter column
    data['quarter'] = data['Date'].dt.year.astype(str) + "-Q" + data['Date'].dt.quarter.astype(str)
    
    # Sort by date
    sorted_data = data.sort_values(by='Date')
    
    # If current_quarter not provided, use the most recent quarter
    if current_quarter is None:
        current_quarter = sorted_data['quarter'].iloc[-1]
    
    # Get unique quarters
    unique_quarters = sorted(sorted_data['quarter'].unique())
    
    # Find current and previous quarter
    if current_quarter not in unique_quarters:
        logger.warning(f"Current quarter {current_quarter} not found in data")
        return {
            "current_quarter": current_quarter,
            "previous_quarter": None,
            "description": f"No data available for quarter {current_quarter}."
        }
    
    current_idx = unique_quarters.index(current_quarter)
    if current_idx == 0:
        logger.warning(f"No previous quarter available for comparison with {current_quarter}")
        
        # Get current quarter data
        current_quarter_data = sorted_data[sorted_data['quarter'] == current_quarter]
        current_avg = current_quarter_data[rate_column].mean()
        
        return {
            "current_quarter": {
                "quarter": current_quarter,
                "average": current_avg,
                "data_points": len(current_quarter_data)
            },
            "previous_quarter": None,
            "description": f"The average {rate_column.replace('OverallAdoptionRate', ' adoption rate')} "
                         f"for {current_quarter} is {current_avg:.2f}%. No previous quarter data available for comparison."
        }
    
    previous_quarter = unique_quarters[current_idx - 1]
    
    # Get data for both quarters
    current_quarter_data = sorted_data[sorted_data['quarter'] == current_quarter]
    previous_quarter_data = sorted_data[sorted_data['quarter'] == previous_quarter]
    
    # Calculate averages
    current_avg = current_quarter_data[rate_column].mean()
    previous_avg = previous_quarter_data[rate_column].mean()
    
    # Calculate change
    absolute_change = current_avg - previous_avg
    percent_change = (absolute_change / previous_avg * 100) if previous_avg > 0 else float('inf')
    
    # Generate description
    direction = "increased" if absolute_change > 0 else "decreased" if absolute_change < 0 else "remained stable"
    
    metric_name = rate_column.replace('OverallAdoptionRate', ' adoption rate')
    description = f"Quarter-over-Quarter Comparison for {metric_name}: "
    
    if abs(absolute_change) < 0.1:
        description += f"The average rate for {current_quarter} is {current_avg:.2f}%, "
        description += f"which is virtually unchanged from {previous_avg:.2f}% in {previous_quarter}."
    else:
        description += f"The average rate for {current_quarter} is {current_avg:.2f}%, "
        description += f"which has {direction} by {abs(absolute_change):.2f} percentage points "
        
        if percent_change != float('inf'):
            description += f"({abs(percent_change):.1f}%) "
        
        description += f"from {previous_avg:.2f}% in {previous_quarter}."
    
    return {
        "current_quarter": {
            "quarter": current_quarter,
            "average": current_avg,
            "data_points": len(current_quarter_data)
        },
        "previous_quarter": {
            "quarter": previous_quarter,
            "average": previous_avg,
            "data_points": len(previous_quarter_data)
        },
        "change": {
            "absolute": absolute_change,
            "percent": percent_change,
            "direction": direction
        },
        "description": description
    }

def compare_year_over_year(data, current_year=None, rate_column="MOverallAdoptionRate"):
    """
    Compare adoption rates year-over-year.
    
    Args:
        data (pandas.DataFrame): Adoption rate data with Date column
        current_year (str, optional): Year to use as reference (format: YYYY)
        rate_column (str): Column name for the rate data to analyze
        
    Returns:
        dict: Dictionary containing year-over-year comparison with these keys:
            - current_year: Current year information
            - previous_year: Previous year information
            - change: Change between years
            - description: Natural language description of the comparison
    """
    if data.empty or rate_column not in data.columns:
        logger.warning(f"No data provided or column {rate_column} not found")
        return {
            "current_year": None,
            "previous_year": None,
            "description": "No adoption rate data available for year-over-year comparison."
        }
    
    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])
    
    # Add year column
    data['year'] = data['Date'].dt.year.astype(str)
    
    # Sort by date
    sorted_data = data.sort_values(by='Date')
    
    # If current_year not provided, use the most recent year
    if current_year is None:
        current_year = sorted_data['year'].iloc[-1]
    elif isinstance(current_year, int):
        current_year = str(current_year)
    
    # Get unique years
    unique_years = sorted(sorted_data['year'].unique())
    
    # Find current and previous year
    if current_year not in unique_years:
        logger.warning(f"Current year {current_year} not found in data")
        return {
            "current_year": current_year,
            "previous_year": None,
            "description": f"No data available for year {current_year}."
        }
    
    current_idx = unique_years.index(current_year)
    if current_idx == 0:
        logger.warning(f"No previous year available for comparison with {current_year}")
        
        # Get current year data
        current_year_data = sorted_data[sorted_data['year'] == current_year]
        current_avg = current_year_data[rate_column].mean()
        
        return {
            "current_year": {
                "year": current_year,
                "average": current_avg,
                "data_points": len(current_year_data)
            },
            "previous_year": None,
            "description": f"The average {rate_column.replace('OverallAdoptionRate', ' adoption rate')} "
                         f"for {current_year} is {current_avg:.2f}%. No previous year data available for comparison."
        }
    
    previous_year = unique_years[current_idx - 1]
    
    # Get data for both years
    current_year_data = sorted_data[sorted_data['year'] == current_year]
    previous_year_data = sorted_data[sorted_data['year'] == previous_year]
    
    # Calculate averages
    current_avg = current_year_data[rate_column].mean()
    previous_avg = previous_year_data[rate_column].mean()
    
    # Calculate change
    absolute_change = current_avg - previous_avg
    percent_change = (absolute_change / previous_avg * 100) if previous_avg > 0 else float('inf')
    
    # Generate description
    direction = "increased" if absolute_change > 0 else "decreased" if absolute_change < 0 else "remained stable"
    
    metric_name = rate_column.replace('OverallAdoptionRate', ' adoption rate')
    description = f"Year-over-Year Comparison for {metric_name}: "
    
    if abs(absolute_change) < 0.1:
        description += f"The average rate for {current_year} is {current_avg:.2f}%, "
        description += f"which is virtually unchanged from {previous_avg:.2f}% in {previous_year}."
    else:
        description += f"The average rate for {current_year} is {current_avg:.2f}%, "
        description += f"which has {direction} by {abs(absolute_change):.2f} percentage points "
        
        if percent_change != float('inf'):
            description += f"({abs(percent_change):.1f}%) "
        
        description += f"from {previous_avg:.2f}% in {previous_year}."
    
    return {
        "current_year": {
            "year": current_year,
            "average": current_avg,
            "data_points": len(current_year_data)
        },
        "previous_year": {
            "year": previous_year,
            "average": previous_avg,
            "data_points": len(previous_year_data)
        },
        "change": {
            "absolute": absolute_change,
            "percent": percent_change,
            "direction": direction
        },
        "description": description
    }

def compare_periods(data, current_period=None, previous_period=None, period_type="month", rate_column="MOverallAdoptionRate"):
    """
    Compare adoption rates between any two periods.
    
    Args:
        data (pandas.DataFrame): Adoption rate data with Date column
        current_period (str): Current period identifier (format depends on period_type)
        previous_period (str): Previous period identifier (format depends on period_type)
        period_type (str): Type of period to compare ('day', 'month', 'quarter', 'year')
        rate_column (str): Column name for the rate data to analyze
        
    Returns:
        dict: Dictionary containing period comparison with these keys:
            - current_period: Current period information
            - previous_period: Previous period information
            - change: Change between periods
            - description: Natural language description of the comparison
    """
    if data.empty or rate_column not in data.columns:
        logger.warning(f"No data provided or column {rate_column} not found")
        return {
            "current_period": None,
            "previous_period": None,
            "description": "No adoption rate data available for period comparison."
        }
    
    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])
    
    # Create period column based on period_type
    if period_type == "day":
        data['period'] = data['Date'].dt.strftime('%Y-%m-%d')
    elif period_type == "month":
        data['period'] = data['Date'].dt.strftime('%Y-%m')
    elif period_type == "quarter":
        data['period'] = data['Date'].dt.year.astype(str) + "-Q" + data['Date'].dt.quarter.astype(str)
    elif period_type == "year":
        data['period'] = data['Date'].dt.year.astype(str)
    else:
        logger.warning(f"Invalid period_type: {period_type}. Using 'month' as default.")
        data['period'] = data['Date'].dt.strftime('%Y-%m')
        period_type = "month"
    
    # Sort by date
    sorted_data = data.sort_values(by='Date')
    
    # If current_period not provided, use the most recent period
    if current_period is None:
        current_period = sorted_data['period'].iloc[-1]
    
    # If previous_period not provided, use the second most recent period
    if previous_period is None:
        unique_periods = sorted(sorted_data['period'].unique())
        if current_period in unique_periods:
            current_idx = unique_periods.index(current_period)
            if current_idx > 0:
                previous_period = unique_periods[current_idx - 1]
    
    # Check if both periods exist in data
    if current_period not in sorted_data['period'].values:
        logger.warning(f"Current period {current_period} not found in data")
        return {
            "current_period": current_period,
            "previous_period": previous_period,
            "description": f"No data available for {period_type} {current_period}."
        }
    
    if previous_period is None or previous_period not in sorted_data['period'].values:
        logger.warning(f"Previous period {previous_period} not found in data")
        
        # Get current period data
        current_period_data = sorted_data[sorted_data['period'] == current_period]
        current_avg = current_period_data[rate_column].mean()
        
        return {
            "current_period": {
                "period": current_period,
                "average": current_avg,
                "data_points": len(current_period_data)
            },
            "previous_period": None,
            "description": f"The average {rate_column.replace('OverallAdoptionRate', ' adoption rate')} "
                         f"for {period_type} {current_period} is {current_avg:.2f}%. "
                         f"No data available for comparison period."
        }
    
    # Get data for both periods
    current_period_data = sorted_data[sorted_data['period'] == current_period]
    previous_period_data = sorted_data[sorted_data['period'] == previous_period]
    
    # Calculate averages
    current_avg = current_period_data[rate_column].mean()
    previous_avg = previous_period_data[rate_column].mean()
    
    # Calculate change
    absolute_change = current_avg - previous_avg
    percent_change = (absolute_change / previous_avg * 100) if previous_avg > 0 else float('inf')
    
    # Generate description
    direction = "increased" if absolute_change > 0 else "decreased" if absolute_change < 0 else "remained stable"
    
    metric_name = rate_column.replace('OverallAdoptionRate', ' adoption rate')
    description = f"Period Comparison for {metric_name}: "
    
    if abs(absolute_change) < 0.1:
        description += f"The average rate for {period_type} {current_period} is {current_avg:.2f}%, "
        description += f"which is virtually unchanged from {previous_avg:.2f}% in {period_type} {previous_period}."
    else:
        description += f"The average rate for {period_type} {current_period} is {current_avg:.2f}%, "
        description += f"which has {direction} by {abs(absolute_change):.2f} percentage points "
        
        if percent_change != float('inf'):
            description += f"({abs(percent_change):.1f}%) "
        
        description += f"from {previous_avg:.2f}% in {period_type} {previous_period}."
    
    return {
        "current_period": {
            "period": current_period,
            "average": current_avg,
            "data_points": len(current_period_data)
        },
        "previous_period": {
            "period": previous_period,
            "average": previous_avg,
            "data_points": len(previous_period_data)
        },
        "change": {
            "absolute": absolute_change,
            "percent": percent_change,
            "direction": direction
        },
        "description": description
    }

def compare_to_target(data, target_value, as_of_date=None, rate_column="MOverallAdoptionRate"):
    """
    Compare current adoption rate to a target value.
    
    Args:
        data (pandas.DataFrame): Adoption rate data with Date column
        target_value (float): Target adoption rate value to compare against
        as_of_date (datetime, optional): Date to use for comparison
        rate_column (str): Column name for the rate data to analyze
        
    Returns:
        dict: Dictionary containing target comparison with these keys:
            - current_value: Current adoption rate value
            - target_value: Target adoption rate value
            - gap: Gap between current and target values
            - description: Natural language description of the comparison
    """
    if data.empty or rate_column not in data.columns:
        logger.warning(f"No data provided or column {rate_column} not found")
        return {
            "current_value": None,
            "target_value": target_value,
            "description": "No adoption rate data available for target comparison."
        }
    
    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])
    
    # Sort by date
    sorted_data = data.sort_values(by='Date')
    
    # If as_of_date not provided, use the most recent date
    if as_of_date is None:
        current_value = sorted_data[rate_column].iloc[-1]
        as_of_date = sorted_data['Date'].iloc[-1]
    else:
        # Find closest date
        if not isinstance(as_of_date, pd.Timestamp):
            as_of_date = pd.to_datetime(as_of_date)
        
        closest_idx = sorted_data['Date'].searchsorted(as_of_date)
        
        if closest_idx >= len(sorted_data):
            closest_idx = len(sorted_data) - 1
        
        current_value = sorted_data[rate_column].iloc[closest_idx]
        as_of_date = sorted_data['Date'].iloc[closest_idx]
    
    # Calculate gap
    gap = current_value - target_value
    gap_percentage = (gap / target_value * 100) if target_value > 0 else float('inf')
    
    # Generate description
    metric_name = rate_column.replace('OverallAdoptionRate', ' adoption rate')
    description = f"Target Comparison for {metric_name}: "
    
    if abs(gap) < 0.1:
        description += f"The current rate of {current_value:.2f}% as of {as_of_date.strftime('%Y-%m-%d')} "
        description += f"is right at the target value of {target_value:.2f}%."
    elif gap > 0:
        description += f"The current rate of {current_value:.2f}% as of {as_of_date.strftime('%Y-%m-%d')} "
        description += f"is {abs(gap):.2f} percentage points "
        
        if gap_percentage != float('inf'):
            description += f"({abs(gap_percentage):.1f}%) "
        
        description += f"above the target value of {target_value:.2f}%."
    else:
        description += f"The current rate of {current_value:.2f}% as of {as_of_date.strftime('%Y-%m-%d')} "
        description += f"is {abs(gap):.2f} percentage points "
        
        if gap_percentage != float('inf'):
            description += f"({abs(gap_percentage):.1f}%) "
        
        description += f"below the target value of {target_value:.2f}%."
    
    # Add context about historical performance relative to target
    historical_data = sorted_data[sorted_data['Date'] < as_of_date]
    if not historical_data.empty:
        historical_avg = historical_data[rate_column].mean()
        historical_min = historical_data[rate_column].min()
        historical_max = historical_data[rate_column].max()
        
        # Count how many data points are above target
        historical_above_target = (historical_data[rate_column] >= target_value).sum()
        historical_total = len(historical_data)
        
        if historical_total > 0:
            historical_percentage = (historical_above_target / historical_total) * 100
            
            description += f" Historically, the rate has been at or above the target {historical_percentage:.1f}% of the time "
            description += f"(range: {historical_min:.2f}% to {historical_max:.2f}%, average: {historical_avg:.2f}%)."
    
    return {
        "current_value": current_value,
        "target_value": target_value,
        "as_of_date": as_of_date,
        "gap": gap,
        "gap_percentage": gap_percentage,
        "description": description
    }

def compare_performance_to_benchmark(data, benchmark_data, rate_column="MOverallAdoptionRate", benchmark_column=None):
    """
    Compare adoption rate performance to a benchmark.
    
    Args:
        data (pandas.DataFrame): Adoption rate data with Date column
        benchmark_data (pandas.DataFrame): Benchmark data with Date column
        rate_column (str): Column name for the rate data to analyze
        benchmark_column (str, optional): Column name for benchmark data
        
    Returns:
        dict: Dictionary containing benchmark comparison with these keys:
            - average_performance: Average adoption rate
            - average_benchmark: Average benchmark value
            - relative_performance: Performance relative to benchmark
            - description: Natural language description of the comparison
    """
    if benchmark_column is None:
        benchmark_column = rate_column
    
    if data.empty or rate_column not in data.columns:
        logger.warning(f"No data provided or column {rate_column} not found")
        return {
            "average_performance": None,
            "average_benchmark": None,
            "description": "No adoption rate data available for benchmark comparison."
        }
    
    if benchmark_data.empty or benchmark_column not in benchmark_data.columns:
        logger.warning(f"No benchmark data provided or column {benchmark_column} not found")
        return {
            "average_performance": data[rate_column].mean(),
            "average_benchmark": None,
            "description": "Benchmark data not available for comparison."
        }
    
    # Ensure Date columns are datetime
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])
    
    if not pd.api.types.is_datetime64_any_dtype(benchmark_data['Date']):
        benchmark_data = benchmark_data.copy()
        benchmark_data['Date'] = pd.to_datetime(benchmark_data['Date'])
    
    # Find overlapping date range
    start_date = max(data['Date'].min(), benchmark_data['Date'].min())
    end_date = min(data['Date'].max(), benchmark_data['Date'].max())
    
    # Filter data for overlapping range
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    filtered_benchmark = benchmark_data[(benchmark_data['Date'] >= start_date) & (benchmark_data['Date'] <= end_date)]
    
    if filtered_data.empty or filtered_benchmark.empty:
        logger.warning("No overlapping date range between data and benchmark")
        return {
            "average_performance": data[rate_column].mean(),
            "average_benchmark": benchmark_data[benchmark_column].mean(),
            "description": "No overlapping date range between data and benchmark for direct comparison."
        }
    
    # Calculate averages
    avg_performance = filtered_data[rate_column].mean()
    avg_benchmark = filtered_benchmark[benchmark_column].mean()
    
    # Calculate relative performance
    absolute_diff = avg_performance - avg_benchmark
    relative_diff = (absolute_diff / avg_benchmark * 100) if avg_benchmark > 0 else float('inf')
    
    # Generate description
    metric_name = rate_column.replace('OverallAdoptionRate', ' adoption rate')
    benchmark_name = benchmark_column.replace('OverallAdoptionRate', ' adoption rate')
    
    description = f"Benchmark Comparison for {metric_name}: "
    
    if abs(absolute_diff) < 0.1:
        description += f"The average rate of {avg_performance:.2f}% is virtually identical to "
        description += f"the benchmark value of {avg_benchmark:.2f}%."
    elif absolute_diff > 0:
        description += f"The average rate of {avg_performance:.2f}% is {absolute_diff:.2f} percentage points "
        
        if relative_diff != float('inf'):
            description += f"({relative_diff:.1f}%) "
        
        description += f"higher than the benchmark value of {avg_benchmark:.2f}%."
    else:
        description += f"The average rate of {avg_performance:.2f}% is {abs(absolute_diff):.2f} percentage points "
        
        if relative_diff != float('inf'):
            description += f"({abs(relative_diff):.1f}%) "
        
        description += f"lower than the benchmark value of {avg_benchmark:.2f}%."
    
    # Add trend comparison
    if len(filtered_data) >= 3 and len(filtered_benchmark) >= 3:
        # Calculate simple trends (average of last 3 points minus average of first 3 points)
        sorted_data = filtered_data.sort_values(by='Date')
        sorted_benchmark = filtered_benchmark.sort_values(by='Date')
        
        data_first_avg = sorted_data[rate_column].iloc[:3].mean()
        data_last_avg = sorted_data[rate_column].iloc[-3:].mean()
        data_trend = data_last_avg - data_first_avg
        
        benchmark_first_avg = sorted_benchmark[benchmark_column].iloc[:3].mean()
        benchmark_last_avg = sorted_benchmark[benchmark_column].iloc[-3:].mean()
        benchmark_trend = benchmark_last_avg - benchmark_first_avg
        
        if (data_trend > 0 and benchmark_trend > 0) or (data_trend < 0 and benchmark_trend < 0):
            trend_alignment = "same"
        else:
            trend_alignment = "opposite"
        
        description += f" Over the comparison period ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}), "
        
        if trend_alignment == "same":
            description += f"both the {metric_name} and the benchmark are moving in the "
            description += f"{'upward' if data_trend > 0 else 'downward'} direction, "
            
            if abs(data_trend) > abs(benchmark_trend):
                description += f"with the {metric_name} changing more rapidly."
            elif abs(data_trend) < abs(benchmark_trend):
                description += f"with the benchmark changing more rapidly."
            else:
                description += f"with similar rates of change."
        else:
            description += f"the {metric_name} is trending {'upward' if data_trend > 0 else 'downward'}, "
            description += f"while the benchmark is trending {'upward' if benchmark_trend > 0 else 'downward'}."
    
    return {
        "average_performance": avg_performance,
        "average_benchmark": avg_benchmark,
        "comparison_period": {
            "start_date": start_date,
            "end_date": end_date,
            "data_points": len(filtered_data)
        },
        "difference": {
            "absolute": absolute_diff,
            "relative": relative_diff
        },
        "description": description
    } 