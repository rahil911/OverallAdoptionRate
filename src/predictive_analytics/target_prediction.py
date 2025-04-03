"""
Target Prediction Module

This module provides functions for predicting when a target adoption rate will be achieved.
It uses forecasting methods to project future adoption rates and estimate the achievement date.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Union, Optional, Tuple

from src.data_models.metrics import OverallAdoptionRate
from src.predictive_analytics.time_series_forecasting import create_time_series_forecast

# Configure logging
logger = logging.getLogger(__name__)


def predict_target_achievement_date(
    data: List[OverallAdoptionRate],
    target_value: float,
    metric_type: str = "monthly",
    max_horizon: int = 730,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Predict when a target adoption rate will be achieved.
    
    Args:
        data: List of OverallAdoptionRate objects
        target_value: Target adoption rate to achieve
        metric_type: Type of metric to analyze ("daily", "weekly", "monthly", "yearly")
        max_horizon: Maximum number of days to forecast into the future
        confidence_level: Confidence level for prediction (0-1)
        
    Returns:
        dict: Dictionary with prediction results:
            - achievement_date: Estimated date of target achievement
            - confidence_level: Confidence in the prediction
            - earliest_date: Earliest possible achievement date
            - latest_date: Latest possible achievement date
            - forecast_values: Forecast values leading to target
            - forecast_dates: Dates corresponding to forecast values
            - explanation: Natural language explanation of the prediction
    """
    if not data:
        return {
            "explanation": "No data available for target prediction."
        }
    
    # Sort data by date
    sorted_data = sorted(data, key=lambda x: x.date)
    
    # Get the latest value
    latest_date = sorted_data[-1].date
    
    # Extract the appropriate rate values based on metric_type
    if metric_type == "daily":
        latest_value = sorted_data[-1].daily_adoption_rate
        rate_field = "daily_adoption_rate"
    elif metric_type == "weekly":
        latest_value = sorted_data[-1].weekly_adoption_rate
        rate_field = "weekly_adoption_rate"
    elif metric_type == "yearly":
        latest_value = sorted_data[-1].yearly_adoption_rate
        rate_field = "yearly_adoption_rate"
    else:  # Default to monthly
        latest_value = sorted_data[-1].monthly_adoption_rate
        rate_field = "monthly_adoption_rate"
    
    # Check if we've already achieved the target
    if latest_value >= target_value:
        return {
            "achievement_date": latest_date.strftime("%Y-%m-%d"),
            "confidence_level": 1.0,
            "earliest_date": latest_date.strftime("%Y-%m-%d"),
            "latest_date": latest_date.strftime("%Y-%m-%d"),
            "forecast_values": [],
            "forecast_dates": [],
            "explanation": f"The target of {target_value:.1f}% has already been achieved as of {latest_date.strftime('%Y-%m-%d')}."
        }
    
    # Determine the forecast period based on metric_type
    if metric_type == "daily":
        forecast_periods = max_horizon
        period_days = 1
    elif metric_type == "weekly":
        forecast_periods = max_horizon // 7
        period_days = 7
    elif metric_type == "yearly":
        forecast_periods = max_horizon // 365
        period_days = 365
    else:  # monthly
        forecast_periods = max_horizon // 30
        period_days = 30
    
    # Create forecast
    forecast_result = create_time_series_forecast(
        sorted_data,
        metric_type=metric_type,
        forecast_periods=forecast_periods,
        method="auto"
    )
    
    # Extract forecast values and dates
    forecast_values = forecast_result.get("forecast_values", [])
    forecast_dates_str = forecast_result.get("forecast_dates", [])
    
    # Convert date strings to datetime objects
    forecast_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in forecast_dates_str]
    
    # Extract confidence intervals
    confidence_intervals = forecast_result.get("confidence_intervals", {})
    lower_bounds = confidence_intervals.get("lower", [])
    upper_bounds = confidence_intervals.get("upper", [])
    
    # Find when the forecast reaches the target value
    achievement_index = None
    earliest_index = None
    latest_index = None
    
    for i, value in enumerate(forecast_values):
        # Check if main forecast reaches target
        if value >= target_value and achievement_index is None:
            achievement_index = i
        
        # Check if upper bound reaches target (earliest possible)
        if upper_bounds and upper_bounds[i] >= target_value and earliest_index is None:
            earliest_index = i
        
        # Check if lower bound reaches target (latest possible with confidence)
        if lower_bounds and lower_bounds[i] >= target_value and latest_index is None:
            latest_index = i
    
    # If we didn't find a crossing point in the forecast
    if achievement_index is None:
        return {
            "achievement_date": None,
            "confidence_level": 0.0,
            "earliest_date": None,
            "latest_date": None,
            "forecast_values": forecast_values,
            "forecast_dates": forecast_dates_str,
            "explanation": f"The target of {target_value:.1f}% is not predicted to be achieved within the forecast horizon of {forecast_periods} {metric_type} periods."
        }
    
    # Calculate achievement date
    achievement_date = forecast_dates[achievement_index]
    
    # Calculate earliest and latest possible dates
    earliest_date = forecast_dates[earliest_index] if earliest_index is not None else None
    latest_date = forecast_dates[latest_index] if latest_index is not None else None
    
    # Calculate days until achievement
    days_until_achievement = (achievement_date - latest_date).days
    
    # Calculate confidence level
    if latest_index is not None:
        # If we have lower bound crossing, high confidence
        prediction_confidence = confidence_level
    elif earliest_index is not None:
        # If only upper bound crossing, medium confidence
        prediction_confidence = 0.5
    else:
        # Low confidence if only mean forecast crosses
        prediction_confidence = 0.3
    
    # Generate explanation
    explanation = _generate_target_prediction_explanation(
        latest_value,
        target_value,
        achievement_date,
        days_until_achievement,
        prediction_confidence,
        earliest_date,
        latest_date,
        metric_type,
        forecast_result.get("model_info", {}).get("method", "")
    )
    
    return {
        "achievement_date": achievement_date.strftime("%Y-%m-%d") if achievement_date else None,
        "confidence_level": prediction_confidence,
        "earliest_date": earliest_date.strftime("%Y-%m-%d") if earliest_date else None,
        "latest_date": latest_date.strftime("%Y-%m-%d") if latest_date else None,
        "forecast_values": forecast_values,
        "forecast_dates": forecast_dates_str,
        "explanation": explanation
    }


def estimate_growth_rate_for_target(
    current_value: float,
    target_value: float,
    target_date: datetime,
    current_date: datetime = None
) -> Dict[str, Any]:
    """
    Estimate the growth rate needed to achieve a target by a certain date.
    
    Args:
        current_value: Current adoption rate
        target_value: Target adoption rate
        target_date: Date by which to achieve the target
        current_date: Current date (defaults to today)
        
    Returns:
        dict: Dictionary with growth rate information:
            - required_growth_rate: Growth rate needed (percentage points per month)
            - required_percent_increase: Percent increase needed
            - months_until_target: Number of months until target date
            - is_achievable: Whether the target is realistically achievable
            - explanation: Natural language explanation
    """
    if current_date is None:
        current_date = datetime.now().date()
    elif isinstance(current_date, datetime):
        current_date = current_date.date()
    
    if isinstance(target_date, datetime):
        target_date = target_date.date()
    
    # Check if target is already achieved
    if current_value >= target_value:
        return {
            "required_growth_rate": 0,
            "required_percent_increase": 0,
            "months_until_target": 0,
            "is_achievable": True,
            "explanation": f"The target of {target_value:.1f}% has already been achieved."
        }
    
    # Calculate time until target date
    days_until_target = (target_date - current_date).days
    
    if days_until_target <= 0:
        return {
            "required_growth_rate": float('inf'),
            "required_percent_increase": float('inf'),
            "months_until_target": 0,
            "is_achievable": False,
            "explanation": "The target date is in the past or today. No growth rate can achieve this target in time."
        }
    
    # Convert to months
    months_until_target = days_until_target / 30.0
    
    # Calculate required growth per month
    value_gap = target_value - current_value
    required_growth_rate = value_gap / months_until_target
    
    # Calculate percent increase
    required_percent_increase = (target_value / current_value - 1) * 100 if current_value > 0 else float('inf')
    
    # Assess achievability (heuristic)
    if required_growth_rate > 5:  # More than 5 percentage points per month
        is_achievable = False
        assessment = "This is unlikely to be achievable based on typical adoption rate growth patterns."
    elif required_growth_rate > 2:  # 2-5 percentage points per month
        is_achievable = False
        assessment = "This would be very challenging to achieve based on typical adoption rate growth patterns."
    elif required_growth_rate > 1:  # 1-2 percentage points per month
        is_achievable = True
        assessment = "This is ambitious but potentially achievable with focused efforts."
    else:  # Less than 1 percentage point per month
        is_achievable = True
        assessment = "This appears to be a reasonable target based on typical adoption rate growth patterns."
    
    # Generate explanation
    explanation = (
        f"To reach {target_value:.1f}% by {target_date.strftime('%Y-%m-%d')}, "
        f"the adoption rate needs to grow by {required_growth_rate:.2f} percentage points per month. "
        f"This represents a total increase of {value_gap:.1f} percentage points over {months_until_target:.1f} months. "
        f"{assessment}"
    )
    
    return {
        "required_growth_rate": required_growth_rate,
        "required_percent_increase": required_percent_increase,
        "months_until_target": months_until_target,
        "is_achievable": is_achievable,
        "explanation": explanation
    }


def _generate_target_prediction_explanation(
    current_value: float,
    target_value: float,
    achievement_date: Optional[datetime],
    days_until_achievement: int,
    confidence_level: float,
    earliest_date: Optional[datetime],
    latest_date: Optional[datetime],
    metric_type: str,
    method: str
) -> str:
    """
    Generate a natural language explanation for target prediction.
    
    Args:
        current_value: Current adoption rate
        target_value: Target adoption rate
        achievement_date: Predicted achievement date
        days_until_achievement: Days until achievement
        confidence_level: Confidence level for the prediction
        earliest_date: Earliest possible achievement date
        latest_date: Latest possible achievement date
        metric_type: Type of metric ("daily", "weekly", "monthly", "yearly")
        method: Forecasting method used
        
    Returns:
        str: Natural language explanation
    """
    if not achievement_date:
        return f"The target of {target_value:.1f}% is not predicted to be achieved within the forecast horizon."
    
    # Format dates
    achievement_date_str = achievement_date.strftime("%Y-%m-%d")
    earliest_date_str = earliest_date.strftime("%Y-%m-%d") if earliest_date else "unknown"
    latest_date_str = latest_date.strftime("%Y-%m-%d") if latest_date else "may be beyond the forecast horizon"
    
    # Calculate time differences
    years = days_until_achievement // 365
    remaining_days = days_until_achievement % 365
    months = remaining_days // 30
    remaining_days = remaining_days % 30
    
    # Format timeframe
    if years > 0:
        timeframe = f"approximately {years} years and {months} months"
    elif months > 0:
        timeframe = f"approximately {months} months and {remaining_days} days"
    else:
        timeframe = f"{days_until_achievement} days"
    
    # Format confidence level
    if confidence_level >= 0.9:
        confidence_phrase = "high confidence"
    elif confidence_level >= 0.6:
        confidence_phrase = "moderate confidence"
    else:
        confidence_phrase = "low confidence"
    
    # Generate explanation
    explanation = [
        f"Based on current {metric_type} adoption rate trends of {current_value:.1f}%, "
        f"the target of {target_value:.1f}% is predicted to be achieved by {achievement_date_str} "
        f"({timeframe} from now), with {confidence_phrase}."
    ]
    
    # Add information about confidence interval
    if earliest_date and latest_date:
        explanation.append(
            f"The earliest possible achievement date is {earliest_date_str}, and "
            f"the latest is {latest_date_str}, within a 95% confidence interval."
        )
    elif earliest_date:
        explanation.append(
            f"The earliest possible achievement date is {earliest_date_str}, but "
            f"the latest achievement date {latest_date_str}."
        )
    
    # Add information about growth required
    value_gap = target_value - current_value
    months_until = days_until_achievement / 30.0
    monthly_growth = value_gap / months_until if months_until > 0 else float('inf')
    
    explanation.append(
        f"This requires an average growth of {monthly_growth:.2f} percentage points per month "
        f"from the current rate of {current_value:.1f}%."
    )
    
    # Add caveats
    explanation.append(
        f"This prediction is based on the {method} forecasting method and assumes "
        f"that current adoption trends will continue without major changes."
    )
    
    return "\n".join(explanation) 