"""
Metric Explanation Module

This module provides functions for explaining adoption rate metrics,
including their definitions, formulas, examples, and context.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Define metric definitions
METRIC_DEFINITIONS = {
    "daily": {
        "name": "Daily Adoption Rate",
        "short_name": "Daily Rate",
        "description": "The percentage of users who actively used the system on a specific day, relative to the total number of eligible users.",
        "formula": "DAU / Total Users * 100",
        "formula_explanation": "Daily Active Users divided by Total Users, multiplied by 100 to express as a percentage.",
        "typical_range": "Typically ranges from 1% to 20% for enterprise software, with variation by industry and software type.",
        "example": "If there are 1,000 total users and 150 users actively used the system on a specific day, the daily adoption rate would be 15%."
    },
    "weekly": {
        "name": "Weekly Adoption Rate",
        "short_name": "Weekly Rate",
        "description": "The percentage of users who actively used the system at least once during a specific week, relative to the total number of eligible users.",
        "formula": "WAU / Total Users * 100",
        "formula_explanation": "Weekly Active Users divided by Total Users, multiplied by 100 to express as a percentage.",
        "typical_range": "Typically ranges from 5% to 40% for enterprise software, with variation by industry and software type.",
        "example": "If there are 1,000 total users and 350 users actively used the system at least once during a specific week, the weekly adoption rate would be 35%."
    },
    "monthly": {
        "name": "Monthly Adoption Rate",
        "short_name": "Monthly Rate",
        "description": "The percentage of users who actively used the system at least once during a specific month, relative to the total number of eligible users.",
        "formula": "MAU / Total Users * 100",
        "formula_explanation": "Monthly Active Users divided by Total Users, multiplied by 100 to express as a percentage.",
        "typical_range": "Typically ranges from 10% to 60% for enterprise software, with variation by industry and software type.",
        "example": "If there are 1,000 total users and 500 users actively used the system at least once during a specific month, the monthly adoption rate would be 50%."
    },
    "yearly": {
        "name": "Yearly Adoption Rate",
        "short_name": "Yearly Rate",
        "description": "The percentage of users who actively used the system at least once during a specific year, relative to the total number of eligible users.",
        "formula": "YAU / Total Users * 100",
        "formula_explanation": "Yearly Active Users divided by Total Users, multiplied by 100 to express as a percentage.",
        "typical_range": "Typically ranges from 30% to 90% for enterprise software, with variation by industry and software type.",
        "example": "If there are 1,000 total users and 800 users actively used the system at least once during a specific year, the yearly adoption rate would be 80%."
    }
}

# Define active user metrics
ACTIVE_USER_METRICS = {
    "DAU": {
        "name": "Daily Active Users",
        "description": "The number of unique users who performed at least one meaningful action in the system during a specific day.",
        "formula": "Count of unique users with at least one activity in a day",
        "relationship": "Determines the daily adoption rate when divided by total users and multiplied by 100."
    },
    "WAU": {
        "name": "Weekly Active Users",
        "description": "The number of unique users who performed at least one meaningful action in the system during a specific week.",
        "formula": "Count of unique users with at least one activity in a week",
        "relationship": "Determines the weekly adoption rate when divided by total users and multiplied by 100."
    },
    "MAU": {
        "name": "Monthly Active Users",
        "description": "The number of unique users who performed at least one meaningful action in the system during a specific month.",
        "formula": "Count of unique users with at least one activity in a month",
        "relationship": "Determines the monthly adoption rate when divided by total users and multiplied by 100."
    },
    "YAU": {
        "name": "Yearly Active Users",
        "description": "The number of unique users who performed at least one meaningful action in the system during a specific year.",
        "formula": "Count of unique users with at least one activity in a year",
        "relationship": "Determines the yearly adoption rate when divided by total users and multiplied by 100."
    }
}

def explain_adoption_metric(metric_type="monthly"):
    """
    Provide an explanation of what an adoption rate metric means.
    
    Args:
        metric_type (str): Type of metric to explain. 
                          Options: 'daily', 'weekly', 'monthly', 'yearly'
        
    Returns:
        dict: Dictionary containing explanation information with these keys:
            - name: Full name of the metric
            - description: Detailed description of what the metric means
            - formula: How the metric is calculated
            - typical_range: Expected range of values
            - example: Concrete example of the metric
            - related_metrics: Related metrics that provide context
            - business_impact: How this metric affects business outcomes
            - description: Consolidated natural language description
    """
    if metric_type.lower() not in METRIC_DEFINITIONS:
        logger.warning(f"Unknown metric type: {metric_type}. Using 'monthly' as default.")
        metric_type = "monthly"
    
    # Get the metric definition
    metric_info = METRIC_DEFINITIONS[metric_type.lower()]
    
    # Determine active user metric
    if metric_type.lower() == "daily":
        active_user_metric = "DAU"
    elif metric_type.lower() == "weekly":
        active_user_metric = "WAU"
    elif metric_type.lower() == "monthly":
        active_user_metric = "MAU"
    else:  # yearly
        active_user_metric = "YAU"
    
    active_user_info = ACTIVE_USER_METRICS[active_user_metric]
    
    # Build business impact description
    business_impact = f"The {metric_info['name']} is a key indicator of user engagement and product value. "
    business_impact += f"Higher rates indicate better user adoption and potentially higher ROI on the software investment. "
    
    if metric_type.lower() == "daily":
        business_impact += "Daily rates are useful for tracking immediate impact of changes and spotting short-term trends, but can be volatile."
    elif metric_type.lower() == "weekly":
        business_impact += "Weekly rates provide a balance between responsiveness and stability, making them useful for tracking mid-term trends."
    elif metric_type.lower() == "monthly":
        business_impact += "Monthly rates provide a stable view of adoption trends and are less susceptible to short-term fluctuations."
    else:  # yearly
        business_impact += "Yearly rates provide the most comprehensive view of overall adoption but are less responsive to recent changes."
    
    # Generate consolidated description
    description = f"The {metric_info['name']} measures {metric_info['description']} "
    description += f"It is calculated as {metric_info['formula_explanation']} "
    description += f"{metric_info['typical_range']} "
    description += f"For example: {metric_info['example']} "
    description += f"This metric relates to {active_user_info['name']}, which {active_user_info['relationship']} "
    description += business_impact
    
    return {
        "name": metric_info['name'],
        "short_name": metric_info['short_name'],
        "description": metric_info['description'],
        "formula": metric_info['formula'],
        "formula_explanation": metric_info['formula_explanation'],
        "typical_range": metric_info['typical_range'],
        "example": metric_info['example'],
        "active_user_metric": {
            "name": active_user_info['name'],
            "description": active_user_info['description'],
            "formula": active_user_info['formula'],
            "relationship": active_user_info['relationship']
        },
        "business_impact": business_impact,
        "consolidated_description": description
    }

def generate_metric_context(data, value, rate_column):
    """
    Generate contextual information about a metric's value.
    
    Args:
        data (pandas.DataFrame): DataFrame containing adoption rate data
        value (float): The adoption rate value to contextualize
        rate_column (str): Column name for the rate in the data
        
    Returns:
        dict: Dictionary containing contextual information with these keys:
            - value: The provided value
            - percentile: Percentile within historical data
            - interpretation: What this value means
            - comparison_to_avg: How it compares to the average
            - potential_implications: Business implications
            - description: Consolidated natural language description
    """
    if data.empty or rate_column not in data.columns:
        return {
            "description": f"No data available to provide context for the value {value:.2f}%."
        }
    
    # Calculate percentile and comparison metrics
    historical_mean = data[rate_column].mean()
    historical_median = data[rate_column].median()
    historical_min = data[rate_column].min()
    historical_max = data[rate_column].max()
    
    # Calculate percentile of the value within historical distribution
    percentile = (data[rate_column] <= value).mean() * 100
    
    # Determine if value is a record high or low
    is_record_high = value >= historical_max
    is_record_low = value <= historical_min
    
    # Interpret the value
    interpretation = ""
    if is_record_high:
        interpretation = "This is an all-time high value, indicating peak adoption."
    elif is_record_low:
        interpretation = "This is an all-time low value, indicating concerning adoption levels."
    elif percentile >= 90:
        interpretation = "This is a very high value, in the top 10% of historical data."
    elif percentile >= 75:
        interpretation = "This is a high value, in the top quartile of historical data."
    elif percentile >= 50:
        interpretation = "This is an above-average value, higher than the median."
    elif percentile >= 25:
        interpretation = "This is a below-average value, in the bottom half but not lowest quartile."
    else:
        interpretation = "This is a low value, in the bottom quartile of historical data."
    
    # Compare to average
    diff_from_avg = value - historical_mean
    diff_percentage = (diff_from_avg / historical_mean * 100) if historical_mean > 0 else float('inf')
    
    if abs(diff_from_avg) < 0.5:
        comparison = f"This value is very close to the historical average of {historical_mean:.2f}%."
    elif diff_from_avg > 0:
        comparison = f"This value is {diff_from_avg:.2f} percentage points ({diff_percentage:.1f}%) higher than the historical average of {historical_mean:.2f}%."
    else:
        comparison = f"This value is {abs(diff_from_avg):.2f} percentage points ({abs(diff_percentage):.1f}%) lower than the historical average of {historical_mean:.2f}%."
    
    # Determine implications
    implications = ""
    if value >= 80:
        implications = "This exceptional adoption rate suggests strong user engagement and high software value. Focus on maintaining this level through continued excellence."
    elif value >= 60:
        implications = "This strong adoption rate indicates good user acceptance. Consider highlighting success factors and amplifying what's working well."
    elif value >= 40:
        implications = "This moderate adoption rate shows decent engagement. There may be opportunities to improve specific features or user education."
    elif value >= 20:
        implications = "This adoption rate suggests room for improvement. Consider targeted user outreach, training programs, or feature enhancements."
    else:
        implications = "This low adoption rate indicates potential issues with the software's value proposition or user experience. A detailed review may be warranted."
    
    # Generate consolidated description
    description = f"A {rate_column.replace('OverallAdoptionRate', '').lower()} adoption rate of {value:.2f}% is at the {percentile:.0f}th percentile historically. "
    description += interpretation + " " + comparison + " " + implications
    
    return {
        "value": value,
        "percentile": percentile,
        "historical_comparison": {
            "mean": historical_mean,
            "median": historical_median,
            "min": historical_min,
            "max": historical_max,
            "vs_mean_absolute": diff_from_avg,
            "vs_mean_percent": diff_percentage,
            "is_record_high": is_record_high,
            "is_record_low": is_record_low
        },
        "interpretation": interpretation,
        "comparison_to_avg": comparison,
        "potential_implications": implications,
        "description": description
    } 