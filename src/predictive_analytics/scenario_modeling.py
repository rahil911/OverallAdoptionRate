"""
Scenario Modeling Module

This module provides functions for modeling different adoption rate scenarios
based on changing various parameters that might influence adoption rates.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple

from src.data_models.metrics import OverallAdoptionRate
from src.predictive_analytics.time_series_forecasting import create_time_series_forecast

# Configure logging
logger = logging.getLogger(__name__)

# Define standard scenarios
STANDARD_SCENARIOS = {
    "optimistic": {
        "description": "Optimistic scenario with accelerated adoption",
        "growth_factor": 1.5,
        "volatility_factor": 0.8
    },
    "pessimistic": {
        "description": "Pessimistic scenario with slower adoption",
        "growth_factor": 0.7,
        "volatility_factor": 1.2
    },
    "baseline": {
        "description": "Baseline scenario based on current trends",
        "growth_factor": 1.0,
        "volatility_factor": 1.0
    },
    "aggressive_growth": {
        "description": "Aggressive growth scenario with major initiatives",
        "growth_factor": 2.0,
        "volatility_factor": 1.5
    },
    "stagnation": {
        "description": "Stagnation scenario with minimal growth",
        "growth_factor": 0.3,
        "volatility_factor": 0.5
    }
}


def create_scenario_forecast(
    data: List[OverallAdoptionRate],
    scenario_name: str = "baseline",
    custom_factors: Dict[str, float] = None,
    metric_type: str = "monthly",
    forecast_periods: int = 12
) -> Dict[str, Any]:
    """
    Create a forecast for a specific adoption rate scenario.
    
    Args:
        data: List of OverallAdoptionRate objects
        scenario_name: Name of the scenario to model ("optimistic", "pessimistic", "baseline",
                       "aggressive_growth", "stagnation", or "custom")
        custom_factors: Custom factors for "custom" scenario:
                        - growth_factor: Multiplier for growth rate (default: 1.0)
                        - volatility_factor: Multiplier for volatility (default: 1.0)
        metric_type: Type of metric to forecast ("daily", "weekly", "monthly", "yearly")
        forecast_periods: Number of periods to forecast
    
    Returns:
        dict: Dictionary with scenario forecast results:
            - forecast_values: List of forecasted values
            - forecast_dates: List of forecast dates
            - confidence_intervals: Dictionary with upper and lower bounds
            - scenario_info: Information about the scenario
            - explanation: Natural language explanation of the scenario
    """
    if not data:
        return {
            "explanation": "No data available for scenario modeling."
        }
    
    # Sort data by date
    sorted_data = sorted(data, key=lambda x: x.date)
    
    # Get scenario parameters
    if scenario_name.lower() == "custom" and custom_factors:
        scenario_params = {
            "description": "Custom scenario with user-defined parameters",
            "growth_factor": custom_factors.get("growth_factor", 1.0),
            "volatility_factor": custom_factors.get("volatility_factor", 1.0)
        }
    elif scenario_name.lower() in STANDARD_SCENARIOS:
        scenario_params = STANDARD_SCENARIOS[scenario_name.lower()]
    else:
        # Default to baseline if scenario not found
        logger.warning(f"Scenario {scenario_name} not found, using baseline instead.")
        scenario_params = STANDARD_SCENARIOS["baseline"]
    
    # Create baseline forecast
    baseline_forecast = create_time_series_forecast(
        sorted_data,
        metric_type=metric_type,
        forecast_periods=forecast_periods,
        method="auto"
    )
    
    # Extract baseline forecast data
    baseline_values = baseline_forecast.get("forecast_values", [])
    forecast_dates = baseline_forecast.get("forecast_dates", [])
    
    if not baseline_values or not forecast_dates:
        return {
            "explanation": "Failed to create baseline forecast for scenario modeling."
        }
    
    # Extract confidence intervals
    baseline_intervals = baseline_forecast.get("confidence_intervals", {})
    baseline_lower = baseline_intervals.get("lower", [])
    baseline_upper = baseline_intervals.get("upper", [])
    
    # Apply scenario factors to create modified forecast
    scenario_values = []
    scenario_lower = []
    scenario_upper = []
    
    # Get the last actual value for reference
    last_actual_value = _get_rate_value(sorted_data[-1], metric_type)
    
    # Get the baseline growth characteristics
    baseline_growth_rate, baseline_volatility = _analyze_forecast_characteristics(
        baseline_values, last_actual_value
    )
    
    # Apply scenario factors
    growth_factor = scenario_params["growth_factor"]
    volatility_factor = scenario_params["volatility_factor"]
    
    # Calculate modified growth and volatility
    modified_growth_rate = baseline_growth_rate * growth_factor
    modified_volatility = baseline_volatility * volatility_factor
    
    # Generate scenario forecast values
    for i, baseline_value in enumerate(baseline_values):
        # Calculate forecasted value with modified growth
        period = i + 1  # 1-indexed period
        
        # Apply modified growth rate to the value, with increasing effect over time
        growth_adjustment = (modified_growth_rate - baseline_growth_rate) * period
        
        # Normalize the adjustment based on the last actual value
        normalized_adjustment = growth_adjustment * last_actual_value / 100.0
        
        # Calculate the scenario value
        scenario_value = baseline_value + normalized_adjustment
        
        # Ensure the value is within valid range (0-100%)
        scenario_value = max(0, min(100, scenario_value))
        
        scenario_values.append(scenario_value)
        
        # Calculate modified confidence intervals if available
        if baseline_lower and baseline_upper and i < len(baseline_lower) and i < len(baseline_upper):
            # Calculate original interval width
            original_width = baseline_upper[i] - baseline_lower[i]
            
            # Apply volatility factor to width
            modified_width = original_width * volatility_factor
            
            # Calculate new bounds centered around the scenario value
            half_width = modified_width / 2
            lower_bound = max(0, scenario_value - half_width)
            upper_bound = min(100, scenario_value + half_width)
            
            scenario_lower.append(lower_bound)
            scenario_upper.append(upper_bound)
    
    # Generate explanation
    explanation = _generate_scenario_explanation(
        scenario_name,
        scenario_params,
        last_actual_value,
        scenario_values,
        modified_growth_rate,
        baseline_growth_rate,
        metric_type
    )
    
    return {
        "forecast_values": scenario_values,
        "forecast_dates": forecast_dates,
        "confidence_intervals": {
            "lower": scenario_lower,
            "upper": scenario_upper
        },
        "scenario_info": {
            "name": scenario_name,
            "description": scenario_params["description"],
            "growth_factor": growth_factor,
            "volatility_factor": volatility_factor,
            "modified_growth_rate": modified_growth_rate,
            "baseline_growth_rate": baseline_growth_rate
        },
        "explanation": explanation
    }


def compare_scenarios(
    data: List[OverallAdoptionRate],
    scenarios: List[str] = ["baseline", "optimistic", "pessimistic"],
    metric_type: str = "monthly",
    forecast_periods: int = 12
) -> Dict[str, Any]:
    """
    Compare multiple scenarios for adoption rate forecasts.
    
    Args:
        data: List of OverallAdoptionRate objects
        scenarios: List of scenario names to compare
        metric_type: Type of metric to forecast
        forecast_periods: Number of periods to forecast
    
    Returns:
        dict: Dictionary with scenario comparison results:
            - scenarios: Dictionary with scenario forecasts
            - comparison: Comparison metrics between scenarios
            - forecast_dates: List of forecast dates
            - explanation: Natural language comparison of scenarios
    """
    if not data:
        return {
            "explanation": "No data available for scenario comparison."
        }
    
    # Results dictionary
    scenario_results = {}
    
    # Generate forecasts for each scenario
    for scenario_name in scenarios:
        scenario_forecast = create_scenario_forecast(
            data,
            scenario_name=scenario_name,
            metric_type=metric_type,
            forecast_periods=forecast_periods
        )
        
        if "forecast_values" in scenario_forecast:
            scenario_results[scenario_name] = scenario_forecast
    
    if not scenario_results:
        return {
            "explanation": "Failed to generate any scenario forecasts for comparison."
        }
    
    # Get common forecast dates
    forecast_dates = next(iter(scenario_results.values())).get("forecast_dates", [])
    
    # Generate comparison metrics
    comparison = _compare_scenario_metrics(scenario_results)
    
    # Generate explanation
    explanation = _generate_scenario_comparison_explanation(
        scenario_results, comparison, metric_type
    )
    
    return {
        "scenarios": scenario_results,
        "comparison": comparison,
        "forecast_dates": forecast_dates,
        "explanation": explanation
    }


def analyze_impact_factors(
    data: List[OverallAdoptionRate],
    metric_type: str = "monthly",
    factor_impacts: Dict[str, Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Analyze the impact of different factors on adoption rate forecasts.
    
    Args:
        data: List of OverallAdoptionRate objects
        metric_type: Type of metric to analyze
        factor_impacts: Dictionary of factors and their impact weights:
            {
                "factor_name": {
                    "description": "Description of the factor",
                    "impact": float (impact weight, positive or negative)
                }
            }
            
    Returns:
        dict: Dictionary with impact analysis results:
            - factors: List of factors and their impact
            - weighted_forecast: Forecast adjusted for all factors
            - individual_forecasts: Dictionary of forecasts for each factor
            - explanation: Natural language explanation of factor impacts
    """
    if not data or not factor_impacts:
        return {
            "explanation": "Insufficient data or factors for impact analysis."
        }
    
    # Sort data by date
    sorted_data = sorted(data, key=lambda x: x.date)
    
    # Create baseline forecast
    baseline_forecast = create_time_series_forecast(
        sorted_data,
        metric_type=metric_type,
        forecast_periods=12,  # Use 12 periods for factor analysis
        method="auto"
    )
    
    # Extract baseline forecast data
    baseline_values = baseline_forecast.get("forecast_values", [])
    forecast_dates = baseline_forecast.get("forecast_dates", [])
    
    if not baseline_values or not forecast_dates:
        return {
            "explanation": "Failed to create baseline forecast for impact analysis."
        }
    
    # Get the last actual value for reference
    last_actual_value = _get_rate_value(sorted_data[-1], metric_type)
    
    # Initialize results
    individual_forecasts = {}
    weighted_values = baseline_values.copy()
    
    # Apply each factor's impact
    for factor_name, factor_info in factor_impacts.items():
        impact = factor_info.get("impact", 0.0)
        
        # Skip factors with no impact
        if impact == 0:
            continue
        
        # Calculate factor-adjusted forecast
        factor_values = []
        for i, baseline_value in enumerate(baseline_values):
            # Increase impact over time (more impact in later periods)
            period_weight = (i + 1) / len(baseline_values)
            
            # Calculate adjustment based on factor impact
            adjustment = impact * period_weight * last_actual_value / 10.0  # Scale impact
            
            # Apply adjustment to baseline
            factor_value = baseline_value + adjustment
            
            # Ensure value is within valid range
            factor_value = max(0, min(100, factor_value))
            
            factor_values.append(factor_value)
            
            # Also update the weighted forecast
            weighted_values[i] = weighted_values[i] + (adjustment / len(factor_impacts))
        
        # Store individual factor forecast
        individual_forecasts[factor_name] = {
            "description": factor_info.get("description", ""),
            "impact": impact,
            "forecast_values": factor_values
        }
    
    # Ensure weighted values are within valid range
    weighted_values = [max(0, min(100, v)) for v in weighted_values]
    
    # Generate explanation
    explanation = _generate_impact_factors_explanation(
        factor_impacts, baseline_values, weighted_values, individual_forecasts, metric_type
    )
    
    return {
        "factors": factor_impacts,
        "weighted_forecast": {
            "forecast_values": weighted_values,
            "forecast_dates": forecast_dates
        },
        "individual_forecasts": individual_forecasts,
        "explanation": explanation
    }


def _get_rate_value(metric: OverallAdoptionRate, metric_type: str) -> float:
    """Get the rate value of the specified metric type."""
    if metric_type == "daily":
        return metric.daily_adoption_rate
    elif metric_type == "weekly":
        return metric.weekly_adoption_rate
    elif metric_type == "yearly":
        return metric.yearly_adoption_rate
    else:  # Default to monthly
        return metric.monthly_adoption_rate


def _analyze_forecast_characteristics(
    forecast_values: List[float],
    last_value: float
) -> Tuple[float, float]:
    """
    Analyze growth rate and volatility in forecast values.
    
    Returns:
        tuple: (growth_rate, volatility)
    """
    if not forecast_values or len(forecast_values) < 2:
        return 0.0, 0.0
    
    # Calculate average growth per period
    total_growth = forecast_values[-1] - last_value
    periods = len(forecast_values)
    avg_growth_rate = total_growth / periods
    
    # Calculate volatility (standard deviation of period-to-period changes)
    changes = []
    prev_value = last_value
    for value in forecast_values:
        changes.append(value - prev_value)
        prev_value = value
    
    volatility = np.std(changes) if changes else 0.0
    
    return avg_growth_rate, volatility


def _compare_scenario_metrics(
    scenario_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate comparison metrics for multiple scenarios.
    
    Args:
        scenario_results: Dictionary of scenario forecast results
        
    Returns:
        dict: Comparison metrics
    """
    if not scenario_results:
        return {}
    
    # Initialize comparison metrics
    comparison = {
        "final_values": {},
        "growth_rates": {},
        "volatilities": {},
        "max_differences": {},
        "average_differences": {}
    }
    
    # Use first scenario as reference (typically baseline)
    reference_scenario = next(iter(scenario_results.keys()))
    reference_values = scenario_results[reference_scenario]["forecast_values"]
    
    # Calculate metrics for each scenario
    for scenario_name, scenario_data in scenario_results.items():
        forecast_values = scenario_data["forecast_values"]
        
        if not forecast_values:
            continue
        
        # Final forecasted value
        comparison["final_values"][scenario_name] = forecast_values[-1]
        
        # Growth rate from first to last forecast value
        growth = forecast_values[-1] - forecast_values[0]
        comparison["growth_rates"][scenario_name] = growth
        
        # Volatility (standard deviation of period-to-period changes)
        changes = [forecast_values[i+1] - forecast_values[i] for i in range(len(forecast_values)-1)]
        volatility = np.std(changes) if changes else 0.0
        comparison["volatilities"][scenario_name] = volatility
        
        # Comparison to reference scenario
        if scenario_name != reference_scenario and len(forecast_values) == len(reference_values):
            # Maximum difference from reference
            differences = [abs(forecast_values[i] - reference_values[i]) for i in range(len(forecast_values))]
            comparison["max_differences"][scenario_name] = max(differences) if differences else 0.0
            
            # Average difference from reference
            comparison["average_differences"][scenario_name] = np.mean(differences) if differences else 0.0
    
    return comparison


def _generate_scenario_explanation(
    scenario_name: str,
    scenario_params: Dict[str, Any],
    last_value: float,
    forecast_values: List[float],
    modified_growth_rate: float,
    baseline_growth_rate: float,
    metric_type: str
) -> str:
    """
    Generate a natural language explanation for a scenario forecast.
    
    Returns:
        str: Explanation text
    """
    if not forecast_values:
        return f"No forecast data available for the {scenario_name} scenario."
    
    # Format growth rates
    growth_factor = scenario_params["growth_factor"]
    growth_difference = modified_growth_rate - baseline_growth_rate
    
    # Calculate total change over forecast period
    total_change = forecast_values[-1] - last_value
    percent_change = (forecast_values[-1] / last_value - 1) * 100 if last_value > 0 else 0
    
    # Generate explanation
    parts = [
        f"The {scenario_name} scenario ({scenario_params['description']}) "
        f"models {metric_type} adoption rates with a growth factor of {growth_factor:.1f}x "
        f"compared to the baseline trend."
    ]
    
    if growth_difference > 0:
        parts.append(
            f"This accelerated growth leads to an increase from the current {last_value:.1f}% "
            f"to {forecast_values[-1]:.1f}% by the end of the forecast period, "
            f"representing a total increase of {total_change:.1f} percentage points "
            f"({percent_change:.1f}% relative change)."
        )
    elif growth_difference < 0:
        parts.append(
            f"This reduced growth leads to a change from the current {last_value:.1f}% "
            f"to {forecast_values[-1]:.1f}% by the end of the forecast period, "
            f"representing a total change of {total_change:.1f} percentage points "
            f"({percent_change:.1f}% relative change)."
        )
    else:
        parts.append(
            f"This maintains the current growth trend, projecting from {last_value:.1f}% "
            f"to {forecast_values[-1]:.1f}% by the end of the forecast period, "
            f"representing a total change of {total_change:.1f} percentage points "
            f"({percent_change:.1f}% relative change)."
        )
    
    volatility_factor = scenario_params["volatility_factor"]
    if volatility_factor > 1.0:
        parts.append(
            f"This scenario includes higher volatility ({volatility_factor:.1f}x baseline), "
            f"resulting in wider confidence intervals and less predictable monthly changes."
        )
    elif volatility_factor < 1.0:
        parts.append(
            f"This scenario includes lower volatility ({volatility_factor:.1f}x baseline), "
            f"resulting in narrower confidence intervals and more consistent monthly changes."
        )
    
    return "\n".join(parts)


def _generate_scenario_comparison_explanation(
    scenario_results: Dict[str, Dict[str, Any]],
    comparison: Dict[str, Dict[str, float]],
    metric_type: str
) -> str:
    """
    Generate a natural language explanation comparing multiple scenarios.
    
    Returns:
        str: Explanation text
    """
    if not scenario_results or not comparison:
        return "No scenario data available for comparison."
    
    # Get list of scenarios
    scenarios = list(scenario_results.keys())
    
    # Format final values
    final_values = comparison.get("final_values", {})
    final_values_text = ", ".join([
        f"{scenario}: {value:.1f}%" for scenario, value in final_values.items()
    ])
    
    # Find the highest and lowest final values
    if final_values:
        highest_scenario = max(final_values.items(), key=lambda x: x[1])
        lowest_scenario = min(final_values.items(), key=lambda x: x[1])
        
        highest_text = f"{highest_scenario[0]} ({highest_scenario[1]:.1f}%)"
        lowest_text = f"{lowest_scenario[0]} ({lowest_scenario[1]:.1f}%)"
    else:
        highest_text = "unknown"
        lowest_text = "unknown"
    
    # Generate explanation
    parts = [
        f"Comparison of {len(scenarios)} {metric_type} adoption rate scenarios: "
        f"{', '.join(scenarios)}."
    ]
    
    parts.append(
        f"By the end of the forecast period, the final adoption rates are: {final_values_text}. "
        f"The highest final rate is in the {highest_text} scenario, while the "
        f"lowest is in the {lowest_text} scenario."
    )
    
    # Compare to baseline if present
    if "baseline" in scenarios and len(scenarios) > 1:
        baseline_final = final_values.get("baseline", 0)
        
        diff_from_baseline = {
            s: v - baseline_final 
            for s, v in final_values.items() 
            if s != "baseline"
        }
        
        # Format differences
        diff_text = ", ".join([
            f"{s}: {d:+.1f}%" for s, d in diff_from_baseline.items()
        ])
        
        parts.append(
            f"Compared to the baseline scenario, the differences in final adoption rates are: {diff_text}."
        )
    
    # Add volatility comparison if available
    volatilities = comparison.get("volatilities", {})
    if volatilities and len(volatilities) > 1:
        highest_volatility = max(volatilities.items(), key=lambda x: x[1])
        lowest_volatility = min(volatilities.items(), key=lambda x: x[1])
        
        parts.append(
            f"The {highest_volatility[0]} scenario shows the highest volatility, "
            f"while the {lowest_volatility[0]} scenario shows the most stable trend."
        )
    
    return "\n".join(parts)


def _generate_impact_factors_explanation(
    factor_impacts: Dict[str, Dict[str, Any]],
    baseline_values: List[float],
    weighted_values: List[float],
    individual_forecasts: Dict[str, Dict[str, Any]],
    metric_type: str
) -> str:
    """
    Generate a natural language explanation for factor impact analysis.
    
    Returns:
        str: Explanation text
    """
    if not factor_impacts or not individual_forecasts:
        return "No factor impact data available for analysis."
    
    # Count positive and negative factors
    positive_factors = [(name, info) for name, info in factor_impacts.items() if info.get("impact", 0) > 0]
    negative_factors = [(name, info) for name, info in factor_impacts.items() if info.get("impact", 0) < 0]
    
    # Get final values for baseline and weighted forecast
    baseline_final = baseline_values[-1] if baseline_values else 0
    weighted_final = weighted_values[-1] if weighted_values else 0
    
    # Calculate total change
    total_change = weighted_final - baseline_final
    
    # Generate explanation
    parts = [
        f"Analysis of {len(factor_impacts)} factors influencing future {metric_type} adoption rates:"
    ]
    
    # Overall impact
    if total_change > 0:
        parts.append(
            f"The combined effect of all factors is expected to increase the adoption rate "
            f"by {total_change:.1f} percentage points over the baseline projection, "
            f"resulting in a final rate of {weighted_final:.1f}% (vs. baseline {baseline_final:.1f}%)."
        )
    elif total_change < 0:
        parts.append(
            f"The combined effect of all factors is expected to decrease the adoption rate "
            f"by {abs(total_change):.1f} percentage points below the baseline projection, "
            f"resulting in a final rate of {weighted_final:.1f}% (vs. baseline {baseline_final:.1f}%)."
        )
    else:
        parts.append(
            f"The combined effect of all factors balances out, maintaining the baseline projection "
            f"of {baseline_final:.1f}% adoption rate by the end of the forecast period."
        )
    
    # Positive factors
    if positive_factors:
        pos_factors_text = ", ".join([
            f"{name} (+{info.get('impact', 0):.1f})" for name, info in positive_factors
        ])
        
        parts.append(
            f"Positive factors driving adoption higher: {pos_factors_text}."
        )
    
    # Negative factors
    if negative_factors:
        neg_factors_text = ", ".join([
            f"{name} ({info.get('impact', 0):.1f})" for name, info in negative_factors
        ])
        
        parts.append(
            f"Negative factors limiting adoption: {neg_factors_text}."
        )
    
    # Most influential factor
    if factor_impacts:
        most_influential = max(
            factor_impacts.items(), 
            key=lambda x: abs(x[1].get("impact", 0))
        )
        
        factor_name = most_influential[0]
        factor_impact = most_influential[1].get("impact", 0)
        factor_desc = most_influential[1].get("description", "")
        
        parts.append(
            f"The most influential factor is '{factor_name}' ({factor_desc}) "
            f"with an impact of {factor_impact:+.1f}, which could "
            f"{'increase' if factor_impact > 0 else 'decrease'} "
            f"the adoption rate significantly."
        )
    
    return "\n".join(parts) 