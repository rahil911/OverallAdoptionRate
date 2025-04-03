"""
Period Analyzer module for calculating and comparing period-over-period changes in adoption rates.

This module provides functions to analyze changes between different time periods, 
including Month-over-Month (MoM), Quarter-over-Quarter (QoQ), and Year-over-Year (YoY) changes.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any
from datetime import date, datetime, timedelta
import logging
from collections import defaultdict
import calendar

from src.data_models.metrics import OverallAdoptionRate, MonthlyActiveUsers, DailyActiveUsers, MetricCollection

# Set up logging
logger = logging.getLogger(__name__)


def calculate_mom_change(
    data: List[OverallAdoptionRate],
    rate_type: str = 'monthly',
    months: int = 1
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Calculate Month-over-Month (MoM) changes in adoption rate.
    
    Args:
        data: List of OverallAdoptionRate objects
        rate_type: Type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
        months: Number of months to compare (1 for consecutive months)
        
    Returns:
        Dictionary with (year, month) tuples as keys and dictionaries containing:
        - 'current_rate': The adoption rate for current month
        - 'previous_rate': The adoption rate for previous month
        - 'absolute_change': Absolute change in adoption rate
        - 'percent_change': Percentage change in adoption rate
        - 'previous_year': Year of the previous month
        - 'previous_month': Previous month number
    """
    if not data:
        logger.warning("No data provided for MoM calculation")
        return {}
    
    # Group data by year and month
    monthly_data = defaultdict(list)
    
    for item in data:
        year = item.date.year
        month = item.date.month
        monthly_data[(year, month)].append(item)
    
    # Calculate average adoption rate for each month
    monthly_rates = {}
    
    for (year, month), items in monthly_data.items():
        # Get the appropriate adoption rate values based on rate_type
        if rate_type == 'daily':
            values = [item.daily_adoption_rate for item in items]
        elif rate_type == 'weekly':
            values = [item.weekly_adoption_rate for item in items]
        elif rate_type == 'monthly':
            values = [item.monthly_adoption_rate for item in items]
        elif rate_type == 'yearly':
            values = [item.yearly_adoption_rate for item in items]
        else:
            raise ValueError(f"Invalid rate_type: {rate_type}")
        
        # Calculate average for the month
        monthly_rates[(year, month)] = sum(values) / len(values)
    
    # Calculate MoM changes
    mom_changes = {}
    
    for (year, month), rate in monthly_rates.items():
        # Calculate previous month
        if month > months:
            previous_month = month - months
            previous_year = year
        else:
            previous_month = 12 - (months - month)
            previous_year = year - 1
        
        previous_key = (previous_year, previous_month)
        
        if previous_key in monthly_rates:
            previous_rate = monthly_rates[previous_key]
            absolute_change = rate - previous_rate
            percent_change = (absolute_change / previous_rate * 100) if previous_rate != 0 else 0
            
            mom_changes[(year, month)] = {
                'current_rate': rate,
                'previous_rate': previous_rate,
                'absolute_change': absolute_change,
                'percent_change': percent_change,
                'previous_year': previous_year,
                'previous_month': previous_month
            }
    
    return mom_changes


def calculate_qoq_change(
    data: List[OverallAdoptionRate],
    rate_type: str = 'monthly'
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Calculate Quarter-over-Quarter (QoQ) changes in adoption rate.
    
    Args:
        data: List of OverallAdoptionRate objects
        rate_type: Type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
        
    Returns:
        Dictionary with (year, quarter) tuples as keys and dictionaries containing:
        - 'current_rate': The adoption rate for current quarter
        - 'previous_rate': The adoption rate for previous quarter
        - 'absolute_change': Absolute change in adoption rate
        - 'percent_change': Percentage change in adoption rate
        - 'previous_year': Year of the previous quarter
        - 'previous_quarter': Previous quarter number
    """
    if not data:
        logger.warning("No data provided for QoQ calculation")
        return {}
    
    # Group data by year and quarter
    quarterly_data = defaultdict(list)
    
    for item in data:
        year = item.date.year
        # Calculate quarter (1-4)
        quarter = (item.date.month - 1) // 3 + 1
        quarterly_data[(year, quarter)].append(item)
    
    # Calculate average adoption rate for each quarter
    quarterly_rates = {}
    
    for (year, quarter), items in quarterly_data.items():
        # Get the appropriate adoption rate values based on rate_type
        if rate_type == 'daily':
            values = [item.daily_adoption_rate for item in items]
        elif rate_type == 'weekly':
            values = [item.weekly_adoption_rate for item in items]
        elif rate_type == 'monthly':
            values = [item.monthly_adoption_rate for item in items]
        elif rate_type == 'yearly':
            values = [item.yearly_adoption_rate for item in items]
        else:
            raise ValueError(f"Invalid rate_type: {rate_type}")
        
        # Calculate average for the quarter
        quarterly_rates[(year, quarter)] = sum(values) / len(values)
    
    # Calculate QoQ changes
    qoq_changes = {}
    
    for (year, quarter), rate in quarterly_rates.items():
        # Calculate previous quarter
        if quarter > 1:
            previous_quarter = quarter - 1
            previous_year = year
        else:
            previous_quarter = 4
            previous_year = year - 1
        
        previous_key = (previous_year, previous_quarter)
        
        if previous_key in quarterly_rates:
            previous_rate = quarterly_rates[previous_key]
            absolute_change = rate - previous_rate
            percent_change = (absolute_change / previous_rate * 100) if previous_rate != 0 else 0
            
            qoq_changes[(year, quarter)] = {
                'current_rate': rate,
                'previous_rate': previous_rate,
                'absolute_change': absolute_change,
                'percent_change': percent_change,
                'previous_year': previous_year,
                'previous_quarter': previous_quarter
            }
    
    return qoq_changes


def calculate_yoy_change(
    data: List[OverallAdoptionRate],
    rate_type: str = 'monthly'
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Calculate Year-over-Year (YoY) changes in adoption rate.
    
    Args:
        data: List of OverallAdoptionRate objects
        rate_type: Type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
        
    Returns:
        Dictionary with (year, month) tuples as keys and dictionaries containing:
        - 'current_rate': The adoption rate for current month/year
        - 'previous_rate': The adoption rate for same month in previous year
        - 'absolute_change': Absolute change in adoption rate
        - 'percent_change': Percentage change in adoption rate
    """
    if not data:
        logger.warning("No data provided for YoY calculation")
        return {}
    
    # Group data by year and month
    monthly_data = defaultdict(list)
    
    for item in data:
        year = item.date.year
        month = item.date.month
        monthly_data[(year, month)].append(item)
    
    # Calculate average adoption rate for each month
    monthly_rates = {}
    
    for (year, month), items in monthly_data.items():
        # Get the appropriate adoption rate values based on rate_type
        if rate_type == 'daily':
            values = [item.daily_adoption_rate for item in items]
        elif rate_type == 'weekly':
            values = [item.weekly_adoption_rate for item in items]
        elif rate_type == 'monthly':
            values = [item.monthly_adoption_rate for item in items]
        elif rate_type == 'yearly':
            values = [item.yearly_adoption_rate for item in items]
        else:
            raise ValueError(f"Invalid rate_type: {rate_type}")
        
        # Calculate average for the month
        monthly_rates[(year, month)] = sum(values) / len(values)
    
    # Calculate YoY changes
    yoy_changes = {}
    
    for (year, month), rate in monthly_rates.items():
        previous_key = (year - 1, month)
        
        if previous_key in monthly_rates:
            previous_rate = monthly_rates[previous_key]
            absolute_change = rate - previous_rate
            percent_change = (absolute_change / previous_rate * 100) if previous_rate != 0 else 0
            
            yoy_changes[(year, month)] = {
                'current_rate': rate,
                'previous_rate': previous_rate,
                'absolute_change': absolute_change,
                'percent_change': percent_change
            }
    
    return yoy_changes


def generate_period_comparison_summary(
    data: List[OverallAdoptionRate],
    rate_type: str = 'monthly',
    reference_date: Optional[date] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Generate a summary of period-over-period changes for a specific reference date.
    
    This provides MoM, QoQ, and YoY changes in a single call for the most recent or specified date.
    
    Args:
        data: List of OverallAdoptionRate objects
        rate_type: Type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
        reference_date: Date to use as reference point (defaults to most recent date in data)
        
    Returns:
        Dictionary with summary information:
        - 'mom': Month-over-Month comparison
        - 'qoq': Quarter-over-Quarter comparison
        - 'yoy': Year-over-Year comparison
        - 'reference_date': The reference date used
        Each comparison contains:
        - 'current_rate': Current adoption rate
        - 'previous_rate': Previous period's adoption rate
        - 'absolute_change': Absolute change in rate
        - 'percent_change': Percentage change in rate
        - 'direction': Direction of change ('increase', 'decrease', 'stable')
    """
    if not data:
        logger.warning("No data provided for period comparison")
        return {}
    
    # Sort data by date
    sorted_data = sorted(data, key=lambda x: x.date)
    
    # Use the most recent date if no reference date is provided
    if reference_date is None:
        reference_date = sorted_data[-1].date
    
    # Get the year, month, and quarter for the reference date
    ref_year = reference_date.year
    ref_month = reference_date.month
    ref_quarter = (reference_date.month - 1) // 3 + 1
    
    # Calculate MoM, QoQ, and YoY changes
    mom_changes = calculate_mom_change(data, rate_type)
    qoq_changes = calculate_qoq_change(data, rate_type)
    yoy_changes = calculate_yoy_change(data, rate_type)
    
    # Create summary dictionary
    summary = {}
    
    # Add MoM comparison if available
    if (ref_year, ref_month) in mom_changes:
        mom_data = mom_changes[(ref_year, ref_month)]
        direction = 'stable'
        if mom_data['absolute_change'] > 0.5:
            direction = 'increase'
        elif mom_data['absolute_change'] < -0.5:
            direction = 'decrease'
            
        summary['mom'] = {
            'current_rate': mom_data['current_rate'],
            'previous_rate': mom_data['previous_rate'],
            'absolute_change': mom_data['absolute_change'],
            'percent_change': mom_data['percent_change'],
            'direction': direction,
            'current_period': f"{calendar.month_name[ref_month]} {ref_year}",
            'previous_period': f"{calendar.month_name[mom_data['previous_month']]} {mom_data['previous_year']}"
        }
    
    # Add QoQ comparison if available
    if (ref_year, ref_quarter) in qoq_changes:
        qoq_data = qoq_changes[(ref_year, ref_quarter)]
        direction = 'stable'
        if qoq_data['absolute_change'] > 0.5:
            direction = 'increase'
        elif qoq_data['absolute_change'] < -0.5:
            direction = 'decrease'
            
        summary['qoq'] = {
            'current_rate': qoq_data['current_rate'],
            'previous_rate': qoq_data['previous_rate'],
            'absolute_change': qoq_data['absolute_change'],
            'percent_change': qoq_data['percent_change'],
            'direction': direction,
            'current_period': f"Q{ref_quarter} {ref_year}",
            'previous_period': f"Q{qoq_data['previous_quarter']} {qoq_data['previous_year']}"
        }
    
    # Add YoY comparison if available
    if (ref_year, ref_month) in yoy_changes:
        yoy_data = yoy_changes[(ref_year, ref_month)]
        direction = 'stable'
        if yoy_data['absolute_change'] > 0.5:
            direction = 'increase'
        elif yoy_data['absolute_change'] < -0.5:
            direction = 'decrease'
            
        summary['yoy'] = {
            'current_rate': yoy_data['current_rate'],
            'previous_rate': yoy_data['previous_rate'],
            'absolute_change': yoy_data['absolute_change'],
            'percent_change': yoy_data['percent_change'],
            'direction': direction,
            'current_period': f"{calendar.month_name[ref_month]} {ref_year}",
            'previous_period': f"{calendar.month_name[ref_month]} {ref_year - 1}"
        }
    
    summary['reference_date'] = reference_date
    
    return summary


def generate_period_comparison_text(summary: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate a human-readable text summary of period-over-period comparisons.
    
    Args:
        summary: Period comparison summary from generate_period_comparison_summary()
        
    Returns:
        Human-readable text summary of period-over-period changes
    """
    if not summary:
        return "No period comparison data available."
    
    text = "Period Comparison Summary:\n\n"
    
    # Add MoM comparison if available
    if 'mom' in summary:
        mom = summary['mom']
        text += f"Month-over-Month (MoM) - {mom['current_period']} vs {mom['previous_period']}:\n"
        text += f"  Current rate: {mom['current_rate']:.2f}%\n"
        text += f"  Previous rate: {mom['previous_rate']:.2f}%\n"
        text += f"  Absolute change: {mom['absolute_change']:.2f}%\n"
        text += f"  Percent change: {mom['percent_change']:.2f}%\n"
        text += f"  Direction: {mom['direction'].capitalize()}\n\n"
    
    # Add QoQ comparison if available
    if 'qoq' in summary:
        qoq = summary['qoq']
        text += f"Quarter-over-Quarter (QoQ) - {qoq['current_period']} vs {qoq['previous_period']}:\n"
        text += f"  Current rate: {qoq['current_rate']:.2f}%\n"
        text += f"  Previous rate: {qoq['previous_rate']:.2f}%\n"
        text += f"  Absolute change: {qoq['absolute_change']:.2f}%\n"
        text += f"  Percent change: {qoq['percent_change']:.2f}%\n"
        text += f"  Direction: {qoq['direction'].capitalize()}\n\n"
    
    # Add YoY comparison if available
    if 'yoy' in summary:
        yoy = summary['yoy']
        text += f"Year-over-Year (YoY) - {yoy['current_period']} vs {yoy['previous_period']}:\n"
        text += f"  Current rate: {yoy['current_rate']:.2f}%\n"
        text += f"  Previous rate: {yoy['previous_rate']:.2f}%\n"
        text += f"  Absolute change: {yoy['absolute_change']:.2f}%\n"
        text += f"  Percent change: {yoy['percent_change']:.2f}%\n"
        text += f"  Direction: {yoy['direction'].capitalize()}\n\n"
    
    # Add overall summary
    directions = []
    if 'mom' in summary:
        directions.append(summary['mom']['direction'])
    if 'qoq' in summary:
        directions.append(summary['qoq']['direction'])
    if 'yoy' in summary:
        directions.append(summary['yoy']['direction'])
    
    if directions:
        # Count occurrences of each direction
        increase_count = directions.count('increase')
        decrease_count = directions.count('decrease')
        stable_count = directions.count('stable')
        
        # Determine overall trend
        if increase_count > decrease_count and increase_count > stable_count:
            overall_trend = "improving"
        elif decrease_count > increase_count and decrease_count > stable_count:
            overall_trend = "declining"
        else:
            overall_trend = "stable"
        
        text += f"Overall Trend: The adoption rate is {overall_trend} across the analyzed time periods."
    
    return text


def compare_time_periods(
    data: List[OverallAdoptionRate],
    period1_start: date,
    period1_end: date,
    period2_start: date,
    period2_end: date,
    rate_type: str = 'monthly'
) -> Dict[str, Any]:
    """
    Compare adoption rates between two distinct time periods.
    
    Args:
        data: List of OverallAdoptionRate objects
        period1_start: Start date of first period
        period1_end: End date of first period
        period2_start: Start date of second period
        period2_end: End date of second period
        rate_type: Type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
        
    Returns:
        Dictionary with comparison information:
        - 'period1_avg': Average adoption rate for period 1
        - 'period2_avg': Average adoption rate for period 2
        - 'absolute_change': Absolute change between periods
        - 'percent_change': Percentage change between periods
        - 'period1_peak': Peak adoption rate in period 1
        - 'period2_peak': Peak adoption rate in period 2
        - 'period1_min': Minimum adoption rate in period 1
        - 'period2_min': Minimum adoption rate in period 2
        - 'period1_volatility': Standard deviation of rates in period 1
        - 'period2_volatility': Standard deviation of rates in period 2
        - 'direction': Direction of change ('increase', 'decrease', 'stable')
    """
    if not data:
        logger.warning("No data provided for time period comparison")
        return {}
    
    # Filter data for period 1
    period1_data = [
        item for item in data 
        if period1_start <= item.date <= period1_end
    ]
    
    # Filter data for period 2
    period2_data = [
        item for item in data 
        if period2_start <= item.date <= period2_end
    ]
    
    # Check if we have data for both periods
    if not period1_data or not period2_data:
        logger.warning("Insufficient data for one or both time periods")
        return {}
    
    # Get values for period 1
    if rate_type == 'daily':
        period1_values = [item.daily_adoption_rate for item in period1_data]
        period2_values = [item.daily_adoption_rate for item in period2_data]
    elif rate_type == 'weekly':
        period1_values = [item.weekly_adoption_rate for item in period1_data]
        period2_values = [item.weekly_adoption_rate for item in period2_data]
    elif rate_type == 'monthly':
        period1_values = [item.monthly_adoption_rate for item in period1_data]
        period2_values = [item.monthly_adoption_rate for item in period2_data]
    elif rate_type == 'yearly':
        period1_values = [item.yearly_adoption_rate for item in period1_data]
        period2_values = [item.yearly_adoption_rate for item in period2_data]
    else:
        raise ValueError(f"Invalid rate_type: {rate_type}")
    
    # Calculate statistics
    period1_avg = sum(period1_values) / len(period1_values)
    period2_avg = sum(period2_values) / len(period2_values)
    
    absolute_change = period2_avg - period1_avg
    percent_change = (absolute_change / period1_avg * 100) if period1_avg != 0 else 0
    
    # Determine direction
    threshold = 0.5  # Threshold for considering a change significant
    if absolute_change > threshold:
        direction = 'increase'
    elif absolute_change < -threshold:
        direction = 'decrease'
    else:
        direction = 'stable'
    
    # Calculate additional statistics
    period1_peak = max(period1_values)
    period2_peak = max(period2_values)
    period1_min = min(period1_values)
    period2_min = min(period2_values)
    
    # Calculate volatility (standard deviation)
    period1_volatility = np.std(period1_values)
    period2_volatility = np.std(period2_values)
    
    return {
        'period1_avg': period1_avg,
        'period2_avg': period2_avg,
        'absolute_change': absolute_change,
        'percent_change': percent_change,
        'period1_peak': period1_peak,
        'period2_peak': period2_peak,
        'period1_min': period1_min,
        'period2_min': period2_min,
        'period1_volatility': period1_volatility,
        'period2_volatility': period2_volatility,
        'direction': direction,
        'period1_start': period1_start,
        'period1_end': period1_end,
        'period2_start': period2_start,
        'period2_end': period2_end
    } 