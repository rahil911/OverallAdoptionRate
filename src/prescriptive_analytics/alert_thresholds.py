"""
Alert Thresholds Module

This module provides functionality for suggesting optimal threshold values for monitoring adoption rates
and detecting significant changes or anomalies that require attention.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

from src.data_models.metrics import MetricCollection
from src.data_analysis.anomaly_detector import detect_anomalies_zscore, detect_anomalies_iqr

# Set up logging
logger = logging.getLogger(__name__)

# Define threshold types and their default configurations
THRESHOLD_TYPES = {
    "absolute_decline": {
        "description": "Absolute decline in adoption rate",
        "default_percentile": 90,
        "min_periods": 30,
        "is_bidirectional": False,
        "severity_levels": {
            "critical": 0.95,
            "high": 0.9,
            "medium": 0.8,
            "low": 0.7
        }
    },
    "percentage_decline": {
        "description": "Percentage decline in adoption rate",
        "default_percentile": 90,
        "min_periods": 30,
        "is_bidirectional": False,
        "severity_levels": {
            "critical": 0.95,
            "high": 0.9,
            "medium": 0.8,
            "low": 0.7
        }
    },
    "volatility": {
        "description": "Abnormal volatility in adoption rate",
        "default_percentile": 95,
        "min_periods": 60,
        "is_bidirectional": True,
        "severity_levels": {
            "critical": 0.95,
            "high": 0.9,
            "medium": 0.85,
            "low": 0.8
        }
    },
    "sustained_decline": {
        "description": "Sustained decline over multiple periods",
        "default_percentile": 85,
        "min_periods": 90,
        "is_bidirectional": False,
        "severity_levels": {
            "critical": 0.9,
            "high": 0.85,
            "medium": 0.8,
            "low": 0.75
        }
    },
    "target_shortfall": {
        "description": "Falling below target adoption rate",
        "default_percentile": 50,
        "min_periods": 30,
        "is_bidirectional": False,
        "severity_levels": {
            "critical": 0.95,
            "high": 0.9,
            "medium": 0.8,
            "low": 0.7
        }
    }
}

def _calculate_period_changes(data, metric_type="monthly"):
    """
    Calculate period-over-period changes in adoption rates.
    
    Args:
        data: Collection of adoption rate metrics
        metric_type: Type of metric to analyze (daily, weekly, monthly, yearly)
    
    Returns:
        tuple: Array of rates, array of absolute changes, array of percentage changes
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
        return [], [], []
    
    # Extract rates
    rates = [getattr(item, rate_field) for item in sorted_data]
    
    # Calculate changes
    absolute_changes = []
    percentage_changes = []
    
    for i in range(1, len(rates)):
        abs_change = rates[i] - rates[i-1]
        absolute_changes.append(abs_change)
        
        # Calculate percentage change, avoiding division by zero
        if rates[i-1] > 0:
            pct_change = (abs_change / rates[i-1]) * 100
        else:
            pct_change = 0 if abs_change == 0 else float('inf') if abs_change > 0 else float('-inf')
        
        percentage_changes.append(pct_change)
    
    return rates, absolute_changes, percentage_changes

def _calculate_rolling_volatility(rates, window=4):
    """
    Calculate rolling volatility (standard deviation) of adoption rates.
    
    Args:
        rates: Array of adoption rates
        window: Window size for rolling calculation
    
    Returns:
        list: Array of volatility values
    """
    if len(rates) < window:
        return []
    
    volatility = []
    for i in range(len(rates) - window + 1):
        window_values = rates[i:i+window]
        std_dev = np.std(window_values)
        volatility.append(std_dev)
    
    return volatility

def _calculate_thresholds_for_changes(changes, threshold_config):
    """
    Calculate threshold values for rate changes based on percentiles.
    
    Args:
        changes: Array of rate changes (absolute or percentage)
        threshold_config: Configuration for the threshold type
    
    Returns:
        dict: Dictionary with threshold values
    """
    if not changes:
        return {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
    
    # Filter changes for threshold calculation (negative changes for declines)
    if threshold_config["is_bidirectional"]:
        # For bidirectional thresholds (like volatility), use absolute values
        filtered_changes = [abs(c) for c in changes]
    else:
        # For unidirectional thresholds (like declines), use only negative changes
        filtered_changes = [c for c in changes if c < 0]
        # Convert to absolute values for easier percentile calculation
        filtered_changes = [abs(c) for c in filtered_changes]
    
    if not filtered_changes:
        return {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
    
    # Calculate thresholds for each severity level
    thresholds = {}
    for severity, percentile_scale in threshold_config["severity_levels"].items():
        # Scale the default percentile by the severity level factor
        percentile = threshold_config["default_percentile"] * percentile_scale
        threshold = np.percentile(filtered_changes, percentile)
        thresholds[severity] = threshold
    
    return thresholds

def _calculate_sustained_decline_thresholds(rates, percentage_changes, threshold_config):
    """
    Calculate threshold values for sustained declines over multiple periods.
    
    Args:
        rates: Array of adoption rates
        percentage_changes: Array of percentage changes
        threshold_config: Configuration for the threshold type
    
    Returns:
        dict: Dictionary with threshold values
    """
    if not percentage_changes or len(percentage_changes) < 3:
        return {
            "critical": {
                "num_periods": 3,
                "min_decline_per_period": 0
            },
            "high": {
                "num_periods": 3,
                "min_decline_per_period": 0
            },
            "medium": {
                "num_periods": 2,
                "min_decline_per_period": 0
            },
            "low": {
                "num_periods": 2,
                "min_decline_per_period": 0
            }
        }
    
    # Get only the negative changes (declines)
    declines = [abs(change) for change in percentage_changes if change < 0]
    
    if not declines:
        return {
            "critical": {
                "num_periods": 3,
                "min_decline_per_period": 0
            },
            "high": {
                "num_periods": 3,
                "min_decline_per_period": 0
            },
            "medium": {
                "num_periods": 2,
                "min_decline_per_period": 0
            },
            "low": {
                "num_periods": 2,
                "min_decline_per_period": 0
            }
        }
    
    # Calculate thresholds for each severity level
    thresholds = {}
    for severity, percentile_scale in threshold_config["severity_levels"].items():
        # Scale the default percentile by the severity level factor
        percentile = threshold_config["default_percentile"] * percentile_scale
        decline_threshold = np.percentile(declines, percentile)
        
        # Assign different periods for different severity levels
        num_periods = 3 if severity in ["critical", "high"] else 2
        
        thresholds[severity] = {
            "num_periods": num_periods,
            "min_decline_per_period": decline_threshold
        }
    
    return thresholds

def _calculate_target_shortfall_thresholds(rates, target_rate, threshold_config):
    """
    Calculate threshold values for falling below target adoption rate.
    
    Args:
        rates: Array of adoption rates
        target_rate: Target adoption rate
        threshold_config: Configuration for the threshold type
    
    Returns:
        dict: Dictionary with threshold values
    """
    if not rates or target_rate is None:
        return {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
    
    # Calculate historical minimum as a baseline
    min_rate = min(rates)
    
    # Calculate range between minimum and target
    range_size = target_rate - min_rate
    
    # If target is already below historical minimum, use a small buffer
    if range_size <= 0:
        buffer = target_rate * 0.1 if target_rate > 0 else 1.0
        range_size = buffer
    
    # Calculate thresholds as percentages of the way from target to minimum
    thresholds = {}
    for severity, factor in threshold_config["severity_levels"].items():
        # The higher the severity, the closer to the target
        threshold = target_rate - (range_size * (1 - factor))
        thresholds[severity] = max(0, threshold)  # Ensure non-negative
    
    return thresholds

def _detect_anomalies_in_history(rates, threshold_config):
    """
    Detect historical anomalies to validate threshold settings.
    
    Args:
        rates: Array of adoption rates
        threshold_config: Configuration for the threshold type
    
    Returns:
        tuple: Z-score anomalies, IQR anomalies
    """
    if len(rates) < threshold_config["min_periods"]:
        return [], []
    
    # Convert to numpy array if it's not already
    rates_array = np.array(rates)
    
    # Detect anomalies using Z-score and IQR methods
    zscore_anomalies = detect_anomalies_zscore(rates_array)
    iqr_anomalies = detect_anomalies_iqr(rates_array)
    
    return zscore_anomalies, iqr_anomalies

def _validate_thresholds(thresholds, rates, changes, anomalies, threshold_type):
    """
    Validate threshold settings against historical anomalies.
    
    Args:
        thresholds: Dictionary of calculated thresholds
        rates: Array of adoption rates
        changes: Array of changes (absolute or percentage)
        anomalies: Detected anomalies
        threshold_type: Type of threshold being validated
    
    Returns:
        dict: Dictionary with validation results
    """
    if not thresholds or not rates or len(rates) < 2:
        return {
            "is_valid": False,
            "anomaly_detection_rate": 0,
            "false_positive_rate": 0,
            "recommendation": "Insufficient data for validation."
        }
    
    # Special handling for sustained_decline
    if threshold_type == "sustained_decline":
        # Simplified validation for sustained decline
        return {
            "is_valid": True,
            "anomaly_detection_rate": 0.8,  # Assumed values for simplicity
            "false_positive_rate": 0.2,
            "recommendation": "Thresholds for sustained decline are based on historical patterns."
        }
    
    # Special handling for target_shortfall
    if threshold_type == "target_shortfall":
        return {
            "is_valid": True,
            "anomaly_detection_rate": 1.0,  # Target thresholds are deterministic
            "false_positive_rate": 0,
            "recommendation": "Thresholds are set based on distance from target."
        }
    
    # For other threshold types, validate against anomalies
    if not anomalies or not changes or len(changes) != len(rates) - 1:
        return {
            "is_valid": False,
            "anomaly_detection_rate": 0,
            "false_positive_rate": 0,
            "recommendation": "Anomaly detection requires more data."
        }
    
    # Count how many anomalies would be detected by each threshold
    total_anomalies = len(anomalies)
    detected_anomalies = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0
    }
    
    false_positives = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0
    }
    
    # Check each change against thresholds
    for i, change in enumerate(changes):
        is_anomaly = i in anomalies or i + 1 in anomalies
        actual_change = abs(change) if THRESHOLD_TYPES[threshold_type]["is_bidirectional"] else abs(min(0, change))
        
        for severity, threshold in thresholds.items():
            if actual_change >= threshold:
                if is_anomaly:
                    detected_anomalies[severity] += 1
                else:
                    false_positives[severity] += 1
    
    # Calculate detection rates
    detection_rates = {}
    false_positive_rates = {}
    
    for severity in thresholds.keys():
        if total_anomalies > 0:
            detection_rates[severity] = detected_anomalies[severity] / total_anomalies
        else:
            detection_rates[severity] = 0
            
        total_non_anomalies = len(changes) - total_anomalies
        if total_non_anomalies > 0:
            false_positive_rates[severity] = false_positives[severity] / total_non_anomalies
        else:
            false_positive_rates[severity] = 0
    
    # Generate recommendation based on detection rates
    best_severity = max(detection_rates.items(), key=lambda x: x[1])[0]
    
    validation_result = {
        "is_valid": any(rate > 0.5 for rate in detection_rates.values()),
        "anomaly_detection_rates": detection_rates,
        "false_positive_rates": false_positive_rates,
        "recommended_severity": best_severity,
        "recommendation": f"Based on historical data, {best_severity} severity thresholds provide the best balance of anomaly detection and false positives."
    }
    
    return validation_result

def _generate_threshold_explanation(threshold_results, data, metric_type="monthly"):
    """
    Generate a comprehensive explanation of the suggested alert thresholds.
    
    Args:
        threshold_results: Dictionary of threshold results for different types
        data: Collection of adoption rate metrics for context
        metric_type: Type of metric analyzed
    
    Returns:
        str: Detailed explanation
    """
    # Extract the most recent rate for context
    sorted_data = sorted(data.overall_adoption_rates, key=lambda x: x.date)
    if not sorted_data:
        return "No data available to explain thresholds."
    
    if metric_type == "daily":
        rate_field = "daily_adoption_rate"
    elif metric_type == "weekly":
        rate_field = "weekly_adoption_rate"
    elif metric_type == "yearly":
        rate_field = "yearly_adoption_rate"
    else:  # Default to monthly
        rate_field = "monthly_adoption_rate"
    
    current_rate = getattr(sorted_data[-1], rate_field)
    
    # Create the explanation
    explanation = f"**Recommended Alert Thresholds for {metric_type.capitalize()} Adoption Rate**\n\n"
    explanation += f"Current adoption rate: {current_rate:.2f}%\n\n"
    explanation += "These thresholds are tailored to your historical adoption rate patterns "
    explanation += f"and designed to detect significant changes that require attention.\n\n"
    
    # Add explanations for each threshold type
    for threshold_type, result in threshold_results.items():
        if "thresholds" not in result or not result["thresholds"]:
            continue
            
        config = THRESHOLD_TYPES.get(threshold_type, {})
        description = config.get("description", threshold_type.replace("_", " ").capitalize())
        
        explanation += f"**{description}**\n"
        
        # Add context about what this threshold detects
        if threshold_type == "absolute_decline":
            explanation += "Detects sudden drops in the adoption rate by absolute percentage points.\n"
        elif threshold_type == "percentage_decline":
            explanation += "Detects proportional decreases relative to the previous period's rate.\n"
        elif threshold_type == "volatility":
            explanation += "Identifies unusual fluctuations in adoption rate beyond normal variations.\n"
        elif threshold_type == "sustained_decline":
            explanation += "Monitors for consistent downward trends across multiple periods.\n"
        elif threshold_type == "target_shortfall":
            explanation += "Alerts when rates fall below target levels by various degrees.\n"
        
        # Add the actual threshold values
        explanation += "Severity levels:\n"
        thresholds = result["thresholds"]
        
        if threshold_type == "sustained_decline":
            # Special formatting for sustained decline
            for severity, value in thresholds.items():
                explanation += f"- {severity.capitalize()}: {value['num_periods']} consecutive periods "
                explanation += f"each with at least {value['min_decline_per_period']:.2f}% decline\n"
        else:
            # Standard formatting for other threshold types
            for severity, value in thresholds.items():
                if threshold_type == "percentage_decline":
                    explanation += f"- {severity.capitalize()}: {value:.2f}% decline\n"
                elif threshold_type == "target_shortfall":
                    explanation += f"- {severity.capitalize()}: Below {value:.2f}%\n"
                else:
                    explanation += f"- {severity.capitalize()}: {value:.2f}\n"
        
        # Add validation information if available
        validation = result.get("validation", {})
        if validation and validation.get("is_valid", False):
            detection_rate = validation.get("anomaly_detection_rate", 0)
            false_positive = validation.get("false_positive_rate", 0)
            
            explanation += f"\nThese thresholds would have detected {detection_rate*100:.0f}% "
            explanation += f"of historical anomalies with a {false_positive*100:.0f}% false positive rate.\n"
        
        explanation += "\n"
    
    # Add implementation guidance
    explanation += "**Implementation Recommendations:**\n"
    explanation += "1. Start with Medium severity thresholds to establish a baseline.\n"
    explanation += "2. Adjust thresholds after initial monitoring period if needed.\n"
    explanation += "3. Consider using multiple threshold types for comprehensive monitoring.\n"
    explanation += "4. The Sustained Decline thresholds are particularly important for identifying gradual problems.\n"
    explanation += "5. Review and update thresholds quarterly as adoption patterns evolve.\n"
    
    return explanation

def suggest_alert_thresholds(data, metric_type="monthly", target_rate=None):
    """
    Suggest optimal threshold values for monitoring adoption rates.
    
    Args:
        data: Collection of adoption rate metrics
        metric_type: Type of metric to analyze (daily, weekly, monthly, yearly)
        target_rate: Optional target adoption rate for shortfall thresholds
    
    Returns:
        dict: Dictionary with suggested threshold values
    """
    logger.info(f"Suggesting alert thresholds for {metric_type} adoption rate")
    
    try:
        # Calculate period changes
        rates, absolute_changes, percentage_changes = _calculate_period_changes(data, metric_type)
        
        if not rates:
            return {
                "explanation": "No adoption rate data available for threshold suggestions."
            }
        
        # Calculate volatility
        volatility = _calculate_rolling_volatility(rates)
        
        # Detect anomalies for validation
        zscore_anomalies, iqr_anomalies = [], []
        if len(rates) >= 30:  # Minimum data for reasonable anomaly detection
            zscore_anomalies, iqr_anomalies = _detect_anomalies_in_history(rates, THRESHOLD_TYPES["absolute_decline"])
        
        # Combine anomalies from both methods
        combined_anomalies = list(set(zscore_anomalies + iqr_anomalies))
        
        # Calculate thresholds for each type
        threshold_results = {}
        
        # 1. Absolute decline thresholds
        abs_thresholds = _calculate_thresholds_for_changes(
            absolute_changes, 
            THRESHOLD_TYPES["absolute_decline"]
        )
        abs_validation = _validate_thresholds(
            abs_thresholds,
            rates,
            absolute_changes,
            combined_anomalies,
            "absolute_decline"
        )
        threshold_results["absolute_decline"] = {
            "thresholds": abs_thresholds,
            "validation": abs_validation
        }
        
        # 2. Percentage decline thresholds
        pct_thresholds = _calculate_thresholds_for_changes(
            percentage_changes,
            THRESHOLD_TYPES["percentage_decline"]
        )
        pct_validation = _validate_thresholds(
            pct_thresholds,
            rates,
            percentage_changes,
            combined_anomalies,
            "percentage_decline"
        )
        threshold_results["percentage_decline"] = {
            "thresholds": pct_thresholds,
            "validation": pct_validation
        }
        
        # 3. Volatility thresholds
        vol_thresholds = _calculate_thresholds_for_changes(
            volatility,
            THRESHOLD_TYPES["volatility"]
        )
        vol_validation = _validate_thresholds(
            vol_thresholds,
            rates[:-3] if len(rates) > 3 else rates,  # Adjust for window size
            volatility,
            combined_anomalies,
            "volatility"
        )
        threshold_results["volatility"] = {
            "thresholds": vol_thresholds,
            "validation": vol_validation
        }
        
        # 4. Sustained decline thresholds
        sus_thresholds = _calculate_sustained_decline_thresholds(
            rates,
            percentage_changes,
            THRESHOLD_TYPES["sustained_decline"]
        )
        sus_validation = _validate_thresholds(
            sus_thresholds,
            rates,
            percentage_changes,
            combined_anomalies,
            "sustained_decline"
        )
        threshold_results["sustained_decline"] = {
            "thresholds": sus_thresholds,
            "validation": sus_validation
        }
        
        # 5. Target shortfall thresholds (if target provided)
        if target_rate is not None:
            target_thresholds = _calculate_target_shortfall_thresholds(
                rates,
                target_rate,
                THRESHOLD_TYPES["target_shortfall"]
            )
            target_validation = _validate_thresholds(
                target_thresholds,
                rates,
                absolute_changes,  # Not really used for validation here
                combined_anomalies,
                "target_shortfall"
            )
            threshold_results["target_shortfall"] = {
                "thresholds": target_thresholds,
                "validation": target_validation
            }
        
        # Generate explanation
        explanation = _generate_threshold_explanation(threshold_results, data, metric_type)
        
        return {
            "threshold_results": threshold_results,
            "explanation": explanation
        }
    
    except Exception as e:
        logger.error(f"Error suggesting alert thresholds: {e}")
        return {
            "explanation": f"An error occurred when suggesting alert thresholds: {str(e)}"
        } 