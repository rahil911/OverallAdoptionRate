"""
Intervention Impact Module

This module provides functionality for estimating the impact of various interventions
on adoption rates, helping stakeholders understand which actions will be most effective.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Union, Tuple

from src.data_models.metrics import MetricCollection
from src.predictive_analytics.time_series_forecasting import create_time_series_forecast as forecast_with_best_method

# Set up logging
logger = logging.getLogger(__name__)

# Define intervention types and their typical impacts
INTERVENTION_TYPES = {
    "product_enhancement": {
        "description": "Major improvements to product functionality or user experience",
        "baseline_impact": {
            "min": 0.05,  # 5% increase
            "max": 0.25,  # 25% increase
            "typical": 0.15  # 15% increase
        },
        "time_to_impact": {
            "min": 1,  # 1 month
            "max": 3,  # 3 months
            "typical": 2  # 2 months
        },
        "impact_duration": {
            "min": 3,  # 3 months
            "max": 12,  # 12 months
            "typical": 6  # 6 months
        },
        "decay_rate": 0.1  # 10% decay per month after peak
    },
    "onboarding_improvement": {
        "description": "Enhanced user onboarding process or materials",
        "baseline_impact": {
            "min": 0.1,  # 10% increase
            "max": 0.3,  # 30% increase
            "typical": 0.2  # 20% increase
        },
        "time_to_impact": {
            "min": 0,  # Immediate
            "max": 2,  # 2 months
            "typical": 1  # 1 month
        },
        "impact_duration": {
            "min": 2,  # 2 months
            "max": 6,  # 6 months
            "typical": 3  # 3 months
        },
        "decay_rate": 0.15  # 15% decay per month after peak
    },
    "training_program": {
        "description": "User training programs or educational content",
        "baseline_impact": {
            "min": 0.05,  # 5% increase
            "max": 0.2,  # 20% increase
            "typical": 0.1  # 10% increase
        },
        "time_to_impact": {
            "min": 1,  # 1 month
            "max": 3,  # 3 months
            "typical": 2  # 2 months
        },
        "impact_duration": {
            "min": 4,  # 4 months
            "max": 12,  # 12 months
            "typical": 6  # 6 months
        },
        "decay_rate": 0.08  # 8% decay per month after peak
    },
    "marketing_campaign": {
        "description": "Marketing or promotional campaigns",
        "baseline_impact": {
            "min": 0.05,  # 5% increase
            "max": 0.25,  # 25% increase
            "typical": 0.15  # 15% increase
        },
        "time_to_impact": {
            "min": 0,  # Immediate
            "max": 1,  # 1 month
            "typical": 0.5  # 2 weeks
        },
        "impact_duration": {
            "min": 1,  # 1 month
            "max": 3,  # 3 months
            "typical": 2  # 2 months
        },
        "decay_rate": 0.25  # 25% decay per month after peak
    },
    "feature_promotion": {
        "description": "Campaigns to highlight existing features",
        "baseline_impact": {
            "min": 0.03,  # 3% increase
            "max": 0.15,  # 15% increase
            "typical": 0.08  # 8% increase
        },
        "time_to_impact": {
            "min": 0,  # Immediate
            "max": 1,  # 1 month
            "typical": 0.5  # 2 weeks
        },
        "impact_duration": {
            "min": 1,  # 1 month
            "max": 4,  # 4 months
            "typical": 2  # 2 months
        },
        "decay_rate": 0.2  # 20% decay per month after peak
    },
    "incentive_program": {
        "description": "User incentives or rewards for adoption",
        "baseline_impact": {
            "min": 0.1,  # 10% increase
            "max": 0.3,  # 30% increase
            "typical": 0.2  # 20% increase
        },
        "time_to_impact": {
            "min": 0,  # Immediate
            "max": 1,  # 1 month
            "typical": 0.25  # 1 week
        },
        "impact_duration": {
            "min": 1,  # 1 month
            "max": 3,  # 3 months
            "typical": 2  # 2 months
        },
        "decay_rate": 0.3  # 30% decay per month after peak
    },
    "performance_optimization": {
        "description": "Improvements to system performance or reliability",
        "baseline_impact": {
            "min": 0.02,  # 2% increase
            "max": 0.15,  # 15% increase
            "typical": 0.07  # 7% increase
        },
        "time_to_impact": {
            "min": 0,  # Immediate
            "max": 2,  # 2 months
            "typical": 1  # 1 month
        },
        "impact_duration": {
            "min": 6,  # 6 months
            "max": 24,  # 24 months
            "typical": 12  # 12 months
        },
        "decay_rate": 0.05  # 5% decay per month after peak
    }
}

def _calculate_baseline_forecast(data, metric_type="monthly", forecast_periods=12):
    """
    Calculate a baseline forecast without any interventions.
    
    Args:
        data: Collection of adoption rate metrics
        metric_type: Type of metric to analyze (daily, weekly, monthly, yearly)
        forecast_periods: Number of periods to forecast
    
    Returns:
        dict: Dictionary with baseline forecast information
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
    
    if not sorted_data:
        logger.warning("No adoption rate data available for baseline forecast")
        return {
            "forecast": [0] * forecast_periods,
            "dates": [(datetime.now() + timedelta(days=30*i)).strftime("%Y-%m-%d") for i in range(forecast_periods)],
            "confidence_intervals": {
                "lower": [0] * forecast_periods,
                "upper": [0] * forecast_periods
            }
        }
    
    # Extract rates
    rates = [getattr(item, rate_field) for item in sorted_data]
    dates = [item.date for item in sorted_data]
    
    # Create a pandas Series with DatetimeIndex
    rate_series = pd.Series(rates, index=pd.DatetimeIndex(dates))
    
    # Use forecasting function from predictive analytics module
    try:
        forecast_result = forecast_with_best_method(
            sorted_data,
            metric_type=metric_type,
            forecast_periods=forecast_periods
        )
        
        # Extract relevant forecast data
        forecast_values = forecast_result.get('forecast_values', [])
        forecast_dates = forecast_result.get('forecast_dates', [])
        confidence_intervals = forecast_result.get('confidence_intervals', {})
        
        return {
            "forecast": forecast_values,
            "dates": [d.strftime("%Y-%m-%d") if isinstance(d, (datetime, date)) else d for d in forecast_dates],
            "confidence_intervals": {
                "lower": confidence_intervals.get('lower', []),
                "upper": confidence_intervals.get('upper', [])
            }
        }
    except Exception as e:
        logger.error(f"Error in baseline forecast calculation: {e}")
        # Fallback to a simple trend forecast
        return {
            "forecast": [rates[-1]] * forecast_periods,
            "dates": [(sorted_data[-1].date + timedelta(days=30*i)).strftime("%Y-%m-%d") for i in range(1, forecast_periods+1)],
            "confidence_intervals": {
                "lower": [max(0, rates[-1] * 0.9)] * forecast_periods,
                "upper": [min(100, rates[-1] * 1.1)] * forecast_periods
            }
        }

def _adjust_impact_based_on_history(intervention_type, data, metric_type="monthly"):
    """
    Adjust the typical impact values based on historical data patterns.
    
    Args:
        intervention_type: Type of intervention from INTERVENTION_TYPES
        data: Collection of adoption rate metrics
        metric_type: Type of metric to analyze
    
    Returns:
        dict: Adjusted impact values
    """
    if intervention_type not in INTERVENTION_TYPES:
        return None
    
    # Get the baseline impact for this intervention type
    baseline = INTERVENTION_TYPES[intervention_type]["baseline_impact"]
    
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
        # Not enough data to adjust, return baseline
        return {
            "min": baseline["min"],
            "max": baseline["max"],
            "typical": baseline["typical"],
            "adjusted": False
        }
    
    # Calculate historical volatility
    rates = [getattr(item, rate_field) for item in sorted_data]
    rate_changes = []
    
    for i in range(1, len(rates)):
        if rates[i-1] > 0:
            change = (rates[i] - rates[i-1]) / rates[i-1]
            rate_changes.append(change)
    
    if not rate_changes:
        # No valid changes detected
        return {
            "min": baseline["min"],
            "max": baseline["max"],
            "typical": baseline["typical"],
            "adjusted": False
        }
    
    # Calculate volatility (standard deviation of changes)
    volatility = np.std(rate_changes)
    
    # Calculate adjustment factor based on volatility
    # Higher volatility suggests more responsiveness to interventions
    if volatility < 0.05:  # Low volatility
        adjustment_factor = 0.8  # Reduce impact
    elif volatility < 0.1:  # Moderate volatility
        adjustment_factor = 1.0  # No change
    elif volatility < 0.2:  # High volatility
        adjustment_factor = 1.2  # Increase impact
    else:  # Very high volatility
        adjustment_factor = 1.5  # Significantly increase impact
    
    # Calculate average growth rate
    avg_growth = np.mean(rate_changes)
    
    # Adjust based on existing growth trend
    # If already growing fast, interventions may have less relative impact
    if avg_growth > 0.1:  # High growth
        adjustment_factor *= 0.9
    elif avg_growth < 0:  # Negative growth
        adjustment_factor *= 1.2
    
    # Apply adjustment
    return {
        "min": baseline["min"] * adjustment_factor,
        "max": baseline["max"] * adjustment_factor,
        "typical": baseline["typical"] * adjustment_factor,
        "adjusted": True,
        "adjustment_factor": adjustment_factor
    }

def _calculate_intervention_impact_curve(baseline_forecast, intervention_config, start_period=0):
    """
    Calculate the impact curve for a specific intervention over time.
    
    Args:
        baseline_forecast: Baseline forecast dictionary
        intervention_config: Configuration for the intervention
        start_period: Period index when the intervention starts
    
    Returns:
        list: Impact values for each period
    """
    forecast_periods = len(baseline_forecast["forecast"])
    impact_curve = [0] * forecast_periods
    
    if start_period >= forecast_periods:
        return impact_curve
    
    # Get intervention parameters
    intervention_type = intervention_config["type"]
    impact_level = intervention_config.get("impact_level", "typical")  # min, typical, max, or custom
    
    if intervention_type not in INTERVENTION_TYPES:
        return impact_curve
    
    # Get the type configuration
    type_config = INTERVENTION_TYPES[intervention_type]
    
    # Get adjusted impact based on data
    if "adjusted_impact" in intervention_config:
        adjusted_impact = intervention_config["adjusted_impact"]
    else:
        # Default to non-adjusted values
        adjusted_impact = {
            "min": type_config["baseline_impact"]["min"],
            "max": type_config["baseline_impact"]["max"],
            "typical": type_config["baseline_impact"]["typical"]
        }
    
    # Determine max impact percentage
    if impact_level == "min":
        max_impact = adjusted_impact["min"]
    elif impact_level == "max":
        max_impact = adjusted_impact["max"]
    elif impact_level == "custom" and "custom_impact" in intervention_config:
        max_impact = intervention_config["custom_impact"]
    else:  # Default to typical
        max_impact = adjusted_impact["typical"]
    
    # Get timing parameters (in periods)
    time_to_impact = type_config["time_to_impact"]["typical"]
    impact_duration = type_config["impact_duration"]["typical"]
    decay_rate = type_config["decay_rate"]
    
    # Override with custom values if provided
    if "custom_time_to_impact" in intervention_config:
        time_to_impact = intervention_config["custom_time_to_impact"]
    if "custom_impact_duration" in intervention_config:
        impact_duration = intervention_config["custom_impact_duration"]
    if "custom_decay_rate" in intervention_config:
        decay_rate = intervention_config["custom_decay_rate"]
    
    # Calculate impact curve
    for i in range(start_period, forecast_periods):
        periods_since_start = i - start_period
        
        if periods_since_start < time_to_impact:
            # Ramp-up phase
            impact_ratio = periods_since_start / time_to_impact if time_to_impact > 0 else 1
            impact_curve[i] = max_impact * impact_ratio
        elif periods_since_start < time_to_impact + impact_duration:
            # Full impact phase
            impact_curve[i] = max_impact
        else:
            # Decay phase
            periods_in_decay = periods_since_start - time_to_impact - impact_duration
            decay_factor = (1 - decay_rate) ** periods_in_decay
            impact_curve[i] = max_impact * decay_factor
    
    return impact_curve

def _apply_intervention_to_forecast(baseline_forecast, impact_curve):
    """
    Apply an intervention's impact to the baseline forecast.
    
    Args:
        baseline_forecast: Baseline forecast dictionary
        impact_curve: List of impact values for each period
    
    Returns:
        dict: Updated forecast with intervention impact
    """
    # Extract baseline values
    baseline_values = baseline_forecast["forecast"]
    lower_bound = baseline_forecast["confidence_intervals"]["lower"]
    upper_bound = baseline_forecast["confidence_intervals"]["upper"]
    
    # Apply impact
    forecast_with_impact = []
    lower_with_impact = []
    upper_with_impact = []
    
    for i in range(len(baseline_values)):
        # Impact is a percentage increase, e.g., 0.1 means 10% increase
        impact_factor = 1 + impact_curve[i]
        
        # Apply to forecast and confidence intervals
        new_value = baseline_values[i] * impact_factor
        new_lower = lower_bound[i] * impact_factor
        new_upper = upper_bound[i] * impact_factor
        
        forecast_with_impact.append(new_value)
        lower_with_impact.append(new_lower)
        upper_with_impact.append(new_upper)
    
    # Create updated forecast
    updated_forecast = baseline_forecast.copy()
    updated_forecast["forecast"] = forecast_with_impact
    updated_forecast["confidence_intervals"]["lower"] = lower_with_impact
    updated_forecast["confidence_intervals"]["upper"] = upper_with_impact
    
    return updated_forecast

def _combine_multiple_interventions(baseline_forecast, interventions, data, metric_type="monthly"):
    """
    Calculate the combined impact of multiple interventions.
    
    Args:
        baseline_forecast: Baseline forecast dictionary
        interventions: List of intervention configurations
        data: Collection of adoption rate metrics for impact adjustment
        metric_type: Type of metric to analyze
    
    Returns:
        dict: Updated forecast with combined intervention impacts
    """
    # Initialize with baseline forecast
    combined_forecast = baseline_forecast.copy()
    combined_impact_curve = [0] * len(baseline_forecast["forecast"])
    
    # Track individual intervention impacts
    intervention_impacts = []
    
    for intervention in interventions:
        intervention_type = intervention.get("type")
        if intervention_type not in INTERVENTION_TYPES:
            continue
        
        # Adjust impact based on historical data
        adjusted_impact = _adjust_impact_based_on_history(intervention_type, data, metric_type)
        intervention["adjusted_impact"] = adjusted_impact
        
        # Calculate impact curve
        start_period = intervention.get("start_period", 0)
        impact_curve = _calculate_intervention_impact_curve(baseline_forecast, intervention, start_period)
        
        # Store individual impact
        intervention_impacts.append({
            "intervention": intervention,
            "impact_curve": impact_curve
        })
        
        # Apply diminishing returns for overlapping interventions
        for i in range(len(combined_impact_curve)):
            # If there's already an impact in this period, the additional
            # impact is reduced to avoid unrealistic combined effects
            if combined_impact_curve[i] > 0:
                # Apply 70% of the new impact (diminishing returns)
                combined_impact_curve[i] += impact_curve[i] * 0.7
            else:
                combined_impact_curve[i] += impact_curve[i]
    
    # Apply combined impact to baseline
    combined_forecast = _apply_intervention_to_forecast(baseline_forecast, combined_impact_curve)
    
    # Add metadata
    combined_forecast["combined_impact_curve"] = combined_impact_curve
    combined_forecast["individual_impacts"] = intervention_impacts
    
    return combined_forecast

def _generate_impact_explanation(combined_forecast, interventions, baseline_forecast):
    """
    Generate a natural language explanation of the intervention impacts.
    
    Args:
        combined_forecast: Forecast with interventions applied
        interventions: List of intervention configurations
        baseline_forecast: Original baseline forecast
    
    Returns:
        str: Explanation text
    """
    if not interventions:
        return "No interventions were analyzed. The forecast shows the baseline trajectory without any changes."
    
    # Calculate overall impact
    baseline_end = baseline_forecast["forecast"][-1]
    combined_end = combined_forecast["forecast"][-1]
    
    absolute_increase = combined_end - baseline_end
    percentage_increase = (absolute_increase / baseline_end) * 100 if baseline_end > 0 else 0
    
    # Introduction
    explanation = f"The analysis estimates that the proposed interventions could increase the adoption rate "
    explanation += f"by {absolute_increase:.2f} percentage points ({percentage_increase:.1f}%) by the end of the forecast period.\n\n"
    
    # Sort interventions by impact
    intervention_impacts = combined_forecast["individual_impacts"]
    sorted_impacts = sorted(
        intervention_impacts,
        key=lambda x: max(x["impact_curve"]),
        reverse=True
    )
    
    # Describe individual interventions
    explanation += "Impact of individual interventions:\n\n"
    
    for i, impact_data in enumerate(sorted_impacts, 1):
        intervention = impact_data["intervention"]
        impact_curve = impact_data["impact_curve"]
        
        intervention_type = intervention["type"]
        description = INTERVENTION_TYPES[intervention_type]["description"]
        max_impact = max(impact_curve) * 100  # Convert to percentage
        
        # When the impact peaks
        peak_period = impact_curve.index(max(impact_curve))
        
        explanation += f"{i}. {intervention_type.replace('_', ' ').title()}: {description}\n"
        explanation += f"   Maximum impact: {max_impact:.1f}% increase in adoption rate\n"
        explanation += f"   Peak impact period: {peak_period + 1}\n"
        
        # Add notes on time to impact and duration
        time_config = INTERVENTION_TYPES[intervention_type]
        explanation += f"   Typical time to impact: {time_config['time_to_impact']['typical']} months\n"
        explanation += f"   Typical impact duration: {time_config['impact_duration']['typical']} months\n\n"
    
    # Add timing recommendations
    explanation += "Implementation recommendations:\n"
    explanation += "1. For maximum impact, consider staggering interventions rather than implementing all at once.\n"
    explanation += "2. Focus on high-impact interventions first to build momentum.\n"
    explanation += "3. Consider interventions with quick time-to-impact to show early results.\n"
    explanation += "4. Plan for ongoing measurement to validate the actual impact of each intervention.\n"
    
    return explanation

def estimate_intervention_impact(data, metric_type="monthly", interventions=None, forecast_periods=12):
    """
    Estimate the impact of various interventions on adoption rates.
    
    Args:
        data: Collection of adoption rate metrics
        metric_type: Type of metric to analyze (daily, weekly, monthly, yearly)
        interventions: List of intervention configurations
        forecast_periods: Number of periods to forecast
    
    Returns:
        dict: Dictionary with impact estimation results
    """
    logger.info(f"Estimating intervention impact for {metric_type} adoption rate")
    
    try:
        # Calculate baseline forecast
        baseline_forecast = _calculate_baseline_forecast(data, metric_type, forecast_periods)
        
        # If no interventions specified, return baseline and intervention options
        if not interventions:
            intervention_options = []
            
            for intervention_type, config in INTERVENTION_TYPES.items():
                adjusted_impact = _adjust_impact_based_on_history(intervention_type, data, metric_type)
                
                intervention_options.append({
                    "type": intervention_type,
                    "description": config["description"],
                    "adjusted_impact": adjusted_impact,
                    "time_to_impact": config["time_to_impact"],
                    "impact_duration": config["impact_duration"]
                })
            
            return {
                "baseline_forecast": baseline_forecast,
                "intervention_options": intervention_options,
                "explanation": "No interventions specified. Select one or more interventions to estimate their impact."
            }
        
        # Calculate combined impact of all interventions
        combined_forecast = _combine_multiple_interventions(baseline_forecast, interventions, data, metric_type)
        
        # Generate explanation
        explanation = _generate_impact_explanation(combined_forecast, interventions, baseline_forecast)
        
        return {
            "baseline_forecast": baseline_forecast,
            "combined_forecast": combined_forecast,
            "interventions": interventions,
            "explanation": explanation
        }
    
    except Exception as e:
        logger.error(f"Error estimating intervention impact: {e}")
        return {
            "explanation": f"An error occurred when estimating intervention impact: {str(e)}"
        } 