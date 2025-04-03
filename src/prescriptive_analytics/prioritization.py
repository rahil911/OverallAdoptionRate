"""
Prioritization Module

This module provides functionality for ranking and prioritizing actions based on
their impact, effort required, time to value, and other important factors.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Set up logging
logger = logging.getLogger(__name__)

# Define prioritization constants
PRIORITIZATION_FACTORS = {
    "impact": {
        "description": "Potential improvement in adoption rate",
        "weight": 0.4,  # Default weight
        "higher_is_better": True
    },
    "effort": {
        "description": "Effort required to implement",
        "weight": 0.2,  # Default weight
        "higher_is_better": False
    },
    "time_to_value": {
        "description": "Time until benefits are realized",
        "weight": 0.15,  # Default weight
        "higher_is_better": False
    },
    "cost": {
        "description": "Financial cost to implement",
        "weight": 0.15,  # Default weight
        "higher_is_better": False
    },
    "prerequisite_count": {
        "description": "Number of prerequisites needed",
        "weight": 0.1,  # Default weight
        "higher_is_better": False
    }
}

def _normalize_score(value, min_value, max_value, higher_is_better=True):
    """
    Normalize a raw score to a 0-1 scale.
    
    Args:
        value: The raw value to normalize
        min_value: Minimum value in the range
        max_value: Maximum value in the range
        higher_is_better: Whether higher values are better
    
    Returns:
        float: Normalized score
    """
    if min_value == max_value:
        return 0.5  # Default if all values are the same
    
    # Normalize to 0-1 range
    normalized = (value - min_value) / (max_value - min_value)
    
    # Flip if lower is better
    if not higher_is_better:
        normalized = 1 - normalized
    
    return normalized

def _calculate_priority_scores(actions, factor_weights=None):
    """
    Calculate priority scores for a list of actions.
    
    Args:
        actions: List of action dictionaries
        factor_weights: Dictionary of factor weights
    
    Returns:
        list: Actions with priority scores
    """
    # Use default weights if not specified
    if not factor_weights:
        factor_weights = {factor: config["weight"] for factor, config in PRIORITIZATION_FACTORS.items()}
    
    # Make a copy of factor_weights to avoid modifying it during iteration
    factor_weights = factor_weights.copy()
    
    # Ensure weights are normalized
    total_weight = sum(factor_weights.values())
    normalized_weights = {factor: weight / total_weight for factor, weight in factor_weights.items()}
    
    # Create copies of actions to avoid modifying the original
    prioritized_actions = []
    for action in actions:
        action_copy = action.copy()
        
        # Fill in missing scores with defaults
        if "impact_score" not in action_copy:
            impact_level = action_copy.get("impact", "medium")
            action_copy["impact_score"] = {"high": 3, "medium": 2, "low": 1}.get(impact_level, 2)
        
        if "effort_score" not in action_copy:
            effort_level = action_copy.get("effort", "medium")
            action_copy["effort_score"] = {"high": 3, "medium": 2, "low": 1}.get(effort_level, 2)
        
        if "time_to_value" not in action_copy:
            action_copy["time_to_value"] = 2  # Default: 2 months
        
        if "cost" not in action_copy:
            cost_level = action_copy.get("cost_level", "medium")
            action_copy["cost"] = {"high": 3, "medium": 2, "low": 1}.get(cost_level, 2)
        
        if "prerequisite_count" not in action_copy:
            action_copy["prerequisite_count"] = len(action_copy.get("prerequisites", []))
            
        prioritized_actions.append(action_copy)
    
    # Make sure we only use factors that exist in PRIORITIZATION_FACTORS and that exist in the actions
    # Creating a list of valid factors first before iterating
    valid_factors = {}
    for factor, weight in normalized_weights.items():
        if factor in PRIORITIZATION_FACTORS:
            valid_factors[factor] = weight
    
    # Find min and max values for each factor for normalization
    factor_ranges = {}
    for factor in valid_factors.keys():
        values = [action.get(factor, 0) for action in prioritized_actions]
        if values:
            factor_ranges[factor] = {
                "min": min(values),
                "max": max(values) if max(values) > min(values) else min(values) + 1,  # Ensure range is non-zero
                "higher_is_better": PRIORITIZATION_FACTORS[factor]["higher_is_better"]
            }
    
    # Calculate normalized and weighted scores
    for action in prioritized_actions:
        factor_scores = {}
        total_score = 0
        
        # Use the pre-validated factors to avoid dictionary modification during iteration
        for factor, weight in valid_factors.items():
            if factor not in factor_ranges:
                continue
                
            value = action.get(factor, 0)
            normalized = _normalize_score(
                value,
                factor_ranges[factor]["min"],
                factor_ranges[factor]["max"],
                factor_ranges[factor]["higher_is_better"]
            )
            weighted = normalized * weight
            
            factor_scores[factor] = {
                "raw": value,
                "normalized": normalized,
                "weighted": weighted
            }
            
            total_score += weighted
        
        action["factor_scores"] = factor_scores
        action["priority_score"] = total_score
    
    # Sort by priority score (descending)
    sorted_actions = sorted(prioritized_actions, key=lambda x: x.get("priority_score", 0), reverse=True)
    
    # Add priority tiers
    tier_boundaries = [0.8, 0.6, 0.4, 0.2]
    for action in sorted_actions:
        score = action.get("priority_score", 0)
        
        if score >= tier_boundaries[0]:
            action["priority_tier"] = "Critical"
        elif score >= tier_boundaries[1]:
            action["priority_tier"] = "High"
        elif score >= tier_boundaries[2]:
            action["priority_tier"] = "Medium"
        elif score >= tier_boundaries[3]:
            action["priority_tier"] = "Low"
        else:
            action["priority_tier"] = "Lowest"
    
    return sorted_actions

def _calculate_roi_metrics(actions):
    """
    Calculate ROI metrics for each action.
    
    Args:
        actions: List of prioritized action dictionaries
    
    Returns:
        list: Actions with ROI metrics
    """
    for action in actions:
        # Calculate impact to effort ratio
        impact = action.get("impact_score", 1)
        effort = action.get("effort_score", 1)
        cost = action.get("cost", 1)
        time_to_value = action.get("time_to_value", 1)
        
        # Avoid division by zero
        if effort == 0:
            effort = 1
        if cost == 0:
            cost = 1
        if time_to_value == 0:
            time_to_value = 1
        
        # Simple impact to cost ratio
        action["impact_to_cost_ratio"] = impact / cost
        
        # Simple impact to effort ratio
        action["impact_to_effort_ratio"] = impact / effort
        
        # ROI considering time
        time_discount = 1 / (1 + 0.1 * time_to_value)  # Simple time value of money
        action["time_adjusted_roi"] = (impact * time_discount) / (effort * cost)
        
        # Efficiency score (impact per unit of combined resources)
        action["efficiency_score"] = impact / (effort + cost + time_to_value)
    
    return actions

def _group_actions_by_priority(actions, num_groups=3):
    """
    Group actions by priority into implementation waves.
    
    Args:
        actions: List of prioritized action dictionaries
        num_groups: Number of priority groups to create
    
    Returns:
        dict: Dictionary of action groups
    """
    if not actions:
        return {}
    
    # Calculate the size of each group
    group_size = max(1, len(actions) // num_groups)
    
    # Create implementation waves
    waves = {}
    wave_names = []
    
    # First, create all the wave entries
    for i in range(num_groups):
        wave_name = f"Wave {i+1}"
        wave_names.append(wave_name)
        start_idx = i * group_size
        end_idx = start_idx + group_size if i < num_groups - 1 else len(actions)
        waves[wave_name] = actions[start_idx:end_idx]
    
    # Then, calculate metrics in a separate loop to avoid modifying during iteration
    metrics = {}
    for wave_name in wave_names:
        wave_actions = waves[wave_name]
        if not wave_actions:
            continue
            
        avg_impact = sum(action.get("impact_score", 0) for action in wave_actions) / len(wave_actions)
        avg_effort = sum(action.get("effort_score", 0) for action in wave_actions) / len(wave_actions)
        avg_time = sum(action.get("time_to_value", 0) for action in wave_actions) / len(wave_actions)
        
        metrics[f"{wave_name}_metrics"] = {
            "avg_impact": avg_impact,
            "avg_effort": avg_effort,
            "avg_time_to_value": avg_time,
            "action_count": len(wave_actions)
        }
    
    # Add all metrics to waves dictionary
    waves.update(metrics)
    
    return waves

def _generate_implementation_roadmap(action_waves):
    """
    Generate an implementation roadmap based on action waves.
    
    Args:
        action_waves: Dictionary of action waves
    
    Returns:
        dict: Dictionary with roadmap information
    """
    now = datetime.now()
    roadmap = {
        "phases": [],
        "timeline": {},
        "total_duration": 0
    }
    
    # Generate phases
    current_start = now
    for wave_name, actions in action_waves.items():
        if not isinstance(actions, list) or not actions or "_metrics" in wave_name:
            continue
            
        # Calculate phase duration based on average time to value
        avg_time = sum(action.get("time_to_value", 1) for action in actions) / len(actions)
        phase_duration = max(1, round(avg_time * 1.5))  # Add buffer for implementation
        
        phase_end = current_start + timedelta(days=phase_duration * 30)  # Convert months to days
        
        roadmap["phases"].append({
            "name": wave_name,
            "start_date": current_start.strftime("%Y-%m-%d"),
            "end_date": phase_end.strftime("%Y-%m-%d"),
            "duration_months": phase_duration,
            "actions": actions
        })
        
        # Update start date for next phase
        current_start = phase_end
    
    # Calculate total duration
    if roadmap["phases"]:
        start_date = datetime.strptime(roadmap["phases"][0]["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(roadmap["phases"][-1]["end_date"], "%Y-%m-%d")
        total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        roadmap["total_duration"] = total_months
    
    # Generate timeline
    if roadmap["phases"]:
        timeline = {}
        for month in range(roadmap["total_duration"] + 1):
            current_date = start_date + timedelta(days=month * 30)
            current_str = current_date.strftime("%Y-%m")
            
            timeline[current_str] = []
            for phase in roadmap["phases"]:
                phase_start = datetime.strptime(phase["start_date"], "%Y-%m-%d")
                phase_end = datetime.strptime(phase["end_date"], "%Y-%m-%d")
                
                if phase_start <= current_date <= phase_end:
                    timeline[current_str].append(phase["name"])
        
        roadmap["timeline"] = timeline
    
    return roadmap

def _generate_prioritization_explanation(actions, factor_weights, action_waves, roadmap):
    """
    Generate a detailed explanation of the prioritization analysis.
    
    Args:
        actions: List of prioritized action dictionaries
        factor_weights: Dictionary of factor weights used
        action_waves: Dictionary of action waves
        roadmap: Implementation roadmap dictionary
    
    Returns:
        str: Detailed explanation
    """
    explanation = "**Action Prioritization Analysis**\n\n"
    
    # Explain prioritization factors
    explanation += "Prioritization was based on the following factors:\n"
    for factor, weight in factor_weights.items():
        if factor in PRIORITIZATION_FACTORS:
            explanation += f"- {factor.capitalize()} ({PRIORITIZATION_FACTORS[factor]['description']}): {weight*100:.1f}% weight\n"
    
    # Top actions summary
    explanation += "\n**Top Priority Actions:**\n"
    for i, action in enumerate(actions[:5], 1):
        explanation += f"{i}. {action.get('action', f'Action {i}')} "
        explanation += f"(Score: {action['priority_score']:.2f}, Tier: {action['priority_tier']})\n"
    
    # Implementation approach
    explanation += "\n**Implementation Approach:**\n"
    explanation += "We recommend implementing actions in waves to balance rapid impact with resource constraints.\n\n"
    
    for wave_name, wave_actions in action_waves.items():
        if not isinstance(wave_actions, list) or not wave_actions or "_metrics" in wave_name:
            continue
            
        metrics_key = f"{wave_name}_metrics"
        metrics = action_waves.get(metrics_key, {})
        
        explanation += f"**{wave_name}**: {len(wave_actions)} actions\n"
        if metrics:
            explanation += f"- Average impact: {metrics.get('avg_impact', 0):.2f}\n"
            explanation += f"- Average effort: {metrics.get('avg_effort', 0):.2f}\n"
            explanation += f"- Average time to value: {metrics.get('avg_time_to_value', 0):.1f} months\n"
        
        # List top 3 actions in this wave
        for i, action in enumerate(wave_actions[:3], 1):
            explanation += f"  {i}. {action.get('action', f'Action {i}')}\n"
        
        explanation += "\n"
    
    # Timeline summary
    if roadmap["phases"]:
        explanation += "**Implementation Timeline:**\n"
        
        for phase in roadmap["phases"]:
            explanation += f"- {phase['name']}: {phase['start_date']} to {phase['end_date']} "
            explanation += f"({phase['duration_months']} months)\n"
        
        explanation += f"\nTotal implementation duration: approximately {roadmap['total_duration']} months\n"
    
    return explanation

def prioritize_actions(actions, factor_weights=None, num_implementation_waves=3):
    """
    Prioritize a list of actions based on multiple factors and generate an implementation plan.
    
    Args:
        actions: List of action dictionaries to prioritize
        factor_weights: Dictionary of weights for different factors
        num_implementation_waves: Number of implementation waves to create
    
    Returns:
        dict: Dictionary containing prioritized actions and implementation plan
    """
    logger.info(f"Prioritizing {len(actions)} actions")
    
    if not actions:
        return {
            "explanation": "No actions provided for prioritization."
        }
    
    try:
        # Use default weights if not specified
        if not factor_weights:
            factor_weights = {factor: config["weight"] for factor, config in PRIORITIZATION_FACTORS.items()}
        
        # Calculate priority scores
        prioritized_actions = _calculate_priority_scores(actions, factor_weights)
        
        # Calculate ROI metrics
        prioritized_actions = _calculate_roi_metrics(prioritized_actions)
        
        # Group actions into implementation waves
        action_waves = _group_actions_by_priority(prioritized_actions, num_implementation_waves)
        
        # Generate implementation roadmap
        roadmap = _generate_implementation_roadmap(action_waves)
        
        # Generate explanation
        explanation = _generate_prioritization_explanation(
            prioritized_actions, 
            factor_weights,
            action_waves,
            roadmap
        )
        
        return {
            "prioritized_actions": prioritized_actions,
            "action_waves": action_waves,
            "implementation_roadmap": roadmap,
            "explanation": explanation
        }
    
    except Exception as e:
        logger.error(f"Error in action prioritization: {e}")
        return {
            "explanation": f"An error occurred during action prioritization: {str(e)}"
        } 