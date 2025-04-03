"""
Goal Setting Module

This module provides functionality for assisting with realistic goal setting for adoption rates
based on historical data and growth patterns.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

from src.data_models.metrics import MetricCollection
from src.data_analysis.trend_analyzer import calculate_trend_line as calculate_trend
from src.data_analysis.period_analyzer import calculate_mom_change, calculate_qoq_change, calculate_yoy_change
from src.predictive_analytics.time_series_forecasting import create_time_series_forecast as forecast_with_best_method

# Set up logging
logger = logging.getLogger(__name__)

# Define goal setting constants
GROWTH_RATE_CATEGORIES = {
    "aggressive": {
        "description": "Ambitious goals requiring significant changes",
        "percentile": 90,
        "scaling_factor": 1.5,
        "min_improvement": 50, # percent
        "typical_time_horizon": 12, # months
        "confidence_level": 0.2
    },
    "challenging": {
        "description": "Stretching but potentially achievable goals",
        "percentile": 75,
        "scaling_factor": 1.25,
        "min_improvement": 25, # percent
        "typical_time_horizon": 6, # months
        "confidence_level": 0.4
    },
    "realistic": {
        "description": "Goals achievable with focused effort",
        "percentile": 60,
        "scaling_factor": 1.1,
        "min_improvement": 10, # percent
        "typical_time_horizon": 3, # months
        "confidence_level": 0.6
    },
    "conservative": {
        "description": "Safe goals with high probability of achievement",
        "percentile": 40,
        "scaling_factor": 1.05,
        "min_improvement": 5, # percent
        "typical_time_horizon": 2, # months
        "confidence_level": 0.8
    }
}

# Timeframe definitions (in months)
TIMEFRAME_DEFINITIONS = {
    "short_term": 3,
    "medium_term": 6,
    "long_term": 12
}

def _analyze_historical_growth(data, metric_type="monthly"):
    """
    Analyze historical growth patterns to understand realistic growth rates.
    
    Args:
        data: Collection of adoption rate metrics
        metric_type: Type of metric to analyze (daily, weekly, monthly, yearly)
    
    Returns:
        dict: Dictionary with growth analysis
    """
    # Map metric type to field name
    if metric_type == "daily":
        rate_field = "daily_adoption_rate"
    elif metric_type == "weekly":
        rate_field = "weekly_adoption_rate"
    elif metric_type == "yearly":
        rate_field = "yearly_adoption_rate"
    else:  # Default to monthly
        rate_field = "monthly_adoption_rate"
    
    # Sort data by date
    sorted_data = sorted(data.overall_adoption_rates, key=lambda x: x.date)
    
    if not sorted_data or len(sorted_data) < 2:
        logger.warning("Insufficient data for historical growth analysis")
        return {
            "avg_growth_rate": 0.05,  # Default 5% growth rate assumption
            "growth_rate_percentiles": {
                "25": 0.02,
                "50": 0.05,
                "75": 0.08,
                "90": 0.15
            },
            "max_growth_rate": 0.15,
            "volatility": 0.1,
            "current_rate": 0 if not sorted_data else getattr(sorted_data[-1], rate_field),
            "growth_rates": []
        }
    
    # Calculate period-over-period growth rates
    rates = [getattr(item, rate_field) for item in sorted_data]
    growth_rates = []
    
    for i in range(1, len(rates)):
        if rates[i-1] > 0:
            growth_rate = (rates[i] - rates[i-1]) / rates[i-1]
            growth_rates.append(growth_rate)
    
    if not growth_rates:
        logger.warning("Could not calculate any growth rates from the data")
        return {
            "avg_growth_rate": 0.05,
            "growth_rate_percentiles": {
                "25": 0.02,
                "50": 0.05,
                "75": 0.08,
                "90": 0.15
            },
            "max_growth_rate": 0.15,
            "volatility": 0.1,
            "current_rate": getattr(sorted_data[-1], rate_field),
            "growth_rates": []
        }
    
    # Calculate statistics
    avg_growth_rate = np.mean(growth_rates)
    max_growth_rate = np.max(growth_rates)
    volatility = np.std(growth_rates)
    
    # Calculate percentiles
    percentiles = {}
    for p in [25, 50, 75, 90]:
        percentiles[str(p)] = np.percentile(growth_rates, p)
    
    return {
        "avg_growth_rate": avg_growth_rate,
        "growth_rate_percentiles": percentiles,
        "max_growth_rate": max_growth_rate,
        "volatility": volatility,
        "current_rate": getattr(sorted_data[-1], rate_field),
        "growth_rates": growth_rates
    }

def _calculate_growth_based_targets(current_rate, growth_analysis, goal_type, timeframe_months):
    """
    Calculate target adoption rates based on growth patterns and goal type.
    
    Args:
        current_rate: Current adoption rate
        growth_analysis: Analysis of historical growth patterns
        goal_type: Type of goal (aggressive, challenging, realistic, conservative)
        timeframe_months: Timeframe for the goal in months
    
    Returns:
        dict: Dictionary with target information
    """
    if goal_type not in GROWTH_RATE_CATEGORIES:
        goal_type = "realistic"  # Default to realistic goals
    
    goal_config = GROWTH_RATE_CATEGORIES[goal_type]
    
    # Get growth rate based on percentile from historical data
    percentile_key = str(goal_config["percentile"])
    
    if percentile_key in growth_analysis["growth_rate_percentiles"]:
        base_growth_rate = growth_analysis["growth_rate_percentiles"][percentile_key]
    else:
        # Fallback to average growth rate if percentile not available
        base_growth_rate = growth_analysis["avg_growth_rate"]
    
    # Apply scaling factor
    adjusted_growth_rate = base_growth_rate * goal_config["scaling_factor"]
    
    # Ensure minimum improvement percentage
    min_monthly_growth = goal_config["min_improvement"] / 100 / 12  # Convert to monthly rate
    adjusted_growth_rate = max(adjusted_growth_rate, min_monthly_growth)
    
    # Calculate compound growth over the timeframe
    target_rate = current_rate * (1 + adjusted_growth_rate) ** timeframe_months
    
    # Calculate absolute and percentage increase
    absolute_increase = target_rate - current_rate
    percentage_increase = (absolute_increase / current_rate) * 100 if current_rate > 0 else 0
    
    # Calculate monthly increase needed
    monthly_increase_needed = current_rate * ((target_rate / current_rate) ** (1/timeframe_months) - 1)
    
    return {
        "target_rate": target_rate,
        "absolute_increase": absolute_increase,
        "percentage_increase": percentage_increase,
        "monthly_growth_rate_needed": adjusted_growth_rate,
        "monthly_increase_needed": monthly_increase_needed,
        "confidence_level": goal_config["confidence_level"],
        "timeframe_months": timeframe_months
    }

def _validate_target_achievability(target_info, growth_analysis):
    """
    Validate if a target is achievable based on historical patterns.
    
    Args:
        target_info: Target information from _calculate_growth_based_targets
        growth_analysis: Analysis of historical growth patterns
    
    Returns:
        dict: Dictionary with validation information
    """
    # Compare monthly growth rate needed to maximum historical rate
    max_historical = growth_analysis["max_growth_rate"]
    monthly_needed = target_info["monthly_growth_rate_needed"]
    
    # Calculate ratio of needed to maximum historical
    ratio = monthly_needed / max_historical if max_historical > 0 else float('inf')
    
    # Assess achievability
    if ratio <= 0.8:
        assessment = "highly achievable"
        justification = "The required growth rate is well within historical performance."
        confidence = min(0.9, target_info["confidence_level"] + 0.2)
    elif ratio <= 1.0:
        assessment = "achievable"
        justification = "The required growth rate is close to historical maximums."
        confidence = target_info["confidence_level"]
    elif ratio <= 1.5:
        assessment = "challenging but possible"
        justification = "The required growth rate exceeds historical performance."
        confidence = max(0.1, target_info["confidence_level"] - 0.2)
    else:
        assessment = "unlikely"
        justification = "The required growth rate far exceeds historical performance."
        confidence = max(0.05, target_info["confidence_level"] - 0.4)
    
    # Calculate effort level required
    if ratio <= 0.5:
        effort = "minimal"
    elif ratio <= 0.8:
        effort = "moderate"
    elif ratio <= 1.2:
        effort = "significant"
    else:
        effort = "extraordinary"
    
    return {
        "assessment": assessment,
        "justification": justification,
        "confidence": confidence,
        "required_to_max_ratio": ratio,
        "effort_required": effort
    }

def _generate_milestones(current_rate, target_info):
    """
    Generate milestone targets to track progress toward the goal.
    
    Args:
        current_rate: Current adoption rate
        target_info: Target information from _calculate_growth_based_targets
    
    Returns:
        list: List of milestone dictionaries
    """
    timeframe = target_info["timeframe_months"]
    target_rate = target_info["target_rate"]
    
    # Determine number of milestones based on timeframe
    if timeframe <= 3:
        milestone_months = [1, 2, 3]
    elif timeframe <= 6:
        milestone_months = [1, 3, 6]
    elif timeframe <= 12:
        milestone_months = [3, 6, 9, 12]
    else:
        milestone_months = [int(timeframe * p) for p in [0.25, 0.5, 0.75, 1.0]]
    
    # Filter milestone months to not exceed timeframe
    milestone_months = [m for m in milestone_months if m <= timeframe]
    
    milestones = []
    for month in milestone_months:
        # Calculate expected rate at this milestone
        progress_ratio = month / timeframe
        
        # Use compound growth to calculate milestone target
        milestone_target = current_rate * (1 + target_info["monthly_growth_rate_needed"]) ** month
        
        # Calculate absolute and percentage increase from start
        absolute_increase = milestone_target - current_rate
        percentage_increase = (absolute_increase / current_rate) * 100 if current_rate > 0 else 0
        
        # Calculate percentage of final target
        target_percentage = (milestone_target - current_rate) / (target_rate - current_rate) * 100 if target_rate > current_rate else 0
        
        milestones.append({
            "month": month,
            "target_rate": milestone_target,
            "absolute_increase": absolute_increase,
            "percentage_increase": percentage_increase,
            "percentage_of_final_target": target_percentage
        })
    
    return milestones

def _generate_custom_goal(current_rate, target_rate, timeframe_months, growth_analysis):
    """
    Generate information for a custom goal with user-specified target rate.
    
    Args:
        current_rate: Current adoption rate
        target_rate: User-specified target rate
        timeframe_months: Timeframe for achieving the goal in months
        growth_analysis: Analysis of historical growth patterns
    
    Returns:
        dict: Dictionary with custom goal information
    """
    # Calculate required monthly growth rate
    required_monthly_growth = ((target_rate / current_rate) ** (1/timeframe_months) - 1) if current_rate > 0 else 0
    
    # Calculate absolute and percentage increase
    absolute_increase = target_rate - current_rate
    percentage_increase = (absolute_increase / current_rate) * 100 if current_rate > 0 else 0
    
    # Compare to historical growth rates to determine goal type
    percentiles = growth_analysis["growth_rate_percentiles"]
    
    # Determine goal type based on where the required growth rate falls in historical percentiles
    goal_type = "custom (unknown difficulty)"
    confidence_level = 0.5  # Default confidence
    
    if required_monthly_growth <= float(percentiles.get("25", 0)):
        goal_type = "custom (conservative)"
        confidence_level = 0.8
    elif required_monthly_growth <= float(percentiles.get("50", 0)):
        goal_type = "custom (realistic)"
        confidence_level = 0.6
    elif required_monthly_growth <= float(percentiles.get("75", 0)):
        goal_type = "custom (challenging)"
        confidence_level = 0.4
    elif required_monthly_growth <= float(percentiles.get("90", 0)):
        goal_type = "custom (aggressive)"
        confidence_level = 0.2
    else:
        goal_type = "custom (extremely aggressive)"
        confidence_level = 0.1
    
    custom_goal = {
        "target_rate": target_rate,
        "absolute_increase": absolute_increase,
        "percentage_increase": percentage_increase,
        "monthly_growth_rate_needed": required_monthly_growth,
        "monthly_increase_needed": current_rate * required_monthly_growth,
        "confidence_level": confidence_level,
        "timeframe_months": timeframe_months,
        "goal_type": goal_type
    }
    
    # Validate achievability
    achievability = _validate_target_achievability(custom_goal, growth_analysis)
    
    # Generate milestones
    milestones = _generate_milestones(current_rate, custom_goal)
    
    return {
        "goal_info": custom_goal,
        "achievability": achievability,
        "milestones": milestones
    }

def _generate_goal_explanation(goal_info, achievability, milestones, goal_type, timeframe_name):
    """
    Generate a detailed explanation of the goal setting.
    
    Args:
        goal_info: Goal information dictionary
        achievability: Achievability assessment dictionary
        milestones: List of milestone dictionaries
        goal_type: Type of goal (aggressive, challenging, realistic, conservative)
        timeframe_name: Name of timeframe (short_term, medium_term, long_term)
    
    Returns:
        str: Detailed explanation
    """
    explanation = f"**{goal_type.capitalize()} {timeframe_name.replace('_', '-')} Goal**\n\n"
    
    # Add target details
    explanation += f"Target adoption rate: {goal_info['target_rate']:.2f}%\n"
    explanation += f"Increase from current: {goal_info['absolute_increase']:.2f} percentage points "
    explanation += f"({goal_info['percentage_increase']:.1f}%)\n"
    explanation += f"Timeframe: {goal_info['timeframe_months']} months\n"
    explanation += f"Required monthly growth: {goal_info['monthly_growth_rate_needed']*100:.2f}% per month\n\n"
    
    # Add achievability assessment
    explanation += f"**Achievability Assessment: {achievability['assessment'].upper()}**\n"
    explanation += f"{achievability['justification']}\n"
    explanation += f"Confidence level: {achievability['confidence']*100:.0f}%\n"
    explanation += f"Effort required: {achievability['effort_required']}\n\n"
    
    # Add milestones
    explanation += "**Suggested Milestones:**\n"
    for milestone in milestones:
        explanation += f"Month {milestone['month']}: {milestone['target_rate']:.2f}% "
        explanation += f"({milestone['percentage_of_final_target']:.0f}% of final target)\n"
    
    return explanation

def assist_goal_setting(data, metric_type="monthly", goal_type=None, timeframe=None, custom_target=None):
    """
    Assist with setting realistic adoption rate goals based on historical data.
    
    Args:
        data: Collection of adoption rate metrics
        metric_type: Type of metric to analyze (daily, weekly, monthly, yearly)
        goal_type: Type of goal (aggressive, challenging, realistic, conservative)
        timeframe: Timeframe for the goal (short_term, medium_term, long_term)
        custom_target: Custom target adoption rate
    
    Returns:
        dict: Dictionary containing goal setting information
    """
    logger.info(f"Generating goal setting assistance for {metric_type} adoption rate")
    
    try:
        # Set defaults if not specified
        if not goal_type or goal_type not in GROWTH_RATE_CATEGORIES:
            goal_type = "realistic"
        
        if not timeframe or timeframe not in TIMEFRAME_DEFINITIONS:
            timeframe = "medium_term"
        
        # Get timeframe in months
        timeframe_months = TIMEFRAME_DEFINITIONS[timeframe]
        
        # Analyze historical growth patterns
        growth_analysis = _analyze_historical_growth(data, metric_type)
        current_rate = growth_analysis["current_rate"]
        
        if custom_target is not None:
            # Generate custom goal information
            custom_goal = _generate_custom_goal(current_rate, custom_target, timeframe_months, growth_analysis)
            
            return {
                "current_rate": current_rate,
                "custom_goal": custom_goal,
                "explanation": _generate_goal_explanation(
                    custom_goal["goal_info"], 
                    custom_goal["achievability"],
                    custom_goal["milestones"],
                    custom_goal["goal_info"]["goal_type"],
                    timeframe
                )
            }
        else:
            # Generate target options for different goal types
            target_options = {}
            explanations = {}
            
            for goal_category in GROWTH_RATE_CATEGORIES:
                # Calculate target for this goal type
                target_info = _calculate_growth_based_targets(
                    current_rate, 
                    growth_analysis, 
                    goal_category, 
                    timeframe_months
                )
                
                # Validate achievability
                achievability = _validate_target_achievability(target_info, growth_analysis)
                
                # Generate milestones
                milestones = _generate_milestones(current_rate, target_info)
                
                # Store in options
                target_options[goal_category] = {
                    "target_info": target_info,
                    "achievability": achievability,
                    "milestones": milestones
                }
                
                # Generate explanation
                explanations[goal_category] = _generate_goal_explanation(
                    target_info,
                    achievability,
                    milestones,
                    goal_category,
                    timeframe
                )
            
            # Create summary explanation
            summary = f"Based on analysis of your historical adoption rate data, we've generated the following goal options for the {timeframe.replace('_', '-')} ({timeframe_months} months):\n\n"
            
            for goal_category in ["conservative", "realistic", "challenging", "aggressive"]:
                option = target_options[goal_category]
                target = option["target_info"]["target_rate"]
                increase = option["target_info"]["percentage_increase"]
                confidence = option["achievability"]["confidence"] * 100
                summary += f"• {goal_category.capitalize()}: {target:.2f}% (+{increase:.1f}%, {confidence:.0f}% confidence)\n"
            
            # Generate focused explanation for requested goal type
            detailed_explanation = explanations[goal_type]
            
            return {
                "current_rate": current_rate,
                "growth_analysis": growth_analysis,
                "target_options": target_options,
                "recommended_option": target_options[goal_type],
                "summary": summary,
                "detailed_explanation": detailed_explanation,
                "all_explanations": explanations
            }
    
    except Exception as e:
        logger.error(f"Error in goal setting assistance: {e}")
        return {
            "explanation": f"An error occurred when generating goal setting advice: {str(e)}"
        } 