"""
Action Suggestions Module

This module provides functionality for suggesting specific actions to improve adoption rates
based on various adoption rate metrics and patterns.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

from src.data_models.metrics import MetricCollection
from src.data_analysis.trend_analyzer import calculate_trend_line as calculate_trend, detect_peaks_and_valleys as detect_peaks_valleys
from src.data_analysis.correlation_analyzer import calculate_correlation_matrix

# Set up logging
logger = logging.getLogger(__name__)

# Define target areas and specific actions
TARGET_AREAS = {
    "onboarding": {
        "description": "User onboarding experience",
        "metrics": ["daily_adoption_rate"],
        "actions": [
            {
                "action": "Implement interactive product tour",
                "effort": "medium",
                "impact": "high",
                "prerequisites": []
            },
            {
                "action": "Add progress indicators during setup",
                "effort": "low",
                "impact": "medium",
                "prerequisites": []
            },
            {
                "action": "Create personalized onboarding paths",
                "effort": "high",
                "impact": "high",
                "prerequisites": ["User segmentation", "Personalization engine"]
            },
            {
                "action": "Add inline help and tooltips",
                "effort": "medium",
                "impact": "medium",
                "prerequisites": []
            },
            {
                "action": "Reduce steps in initial setup",
                "effort": "medium",
                "impact": "high",
                "prerequisites": ["UX review"]
            }
        ]
    },
    "retention": {
        "description": "User retention strategies",
        "metrics": ["monthly_adoption_rate"],
        "actions": [
            {
                "action": "Create automated re-engagement emails",
                "effort": "medium",
                "impact": "high",
                "prerequisites": ["Email automation system"]
            },
            {
                "action": "Implement in-app notification system",
                "effort": "high",
                "impact": "high",
                "prerequisites": []
            },
            {
                "action": "Add user success dashboards",
                "effort": "medium",
                "impact": "medium",
                "prerequisites": []
            },
            {
                "action": "Create milestone achievement system",
                "effort": "high",
                "impact": "high",
                "prerequisites": []
            },
            {
                "action": "Develop personalized usage suggestions",
                "effort": "high",
                "impact": "high",
                "prerequisites": ["Usage analytics", "Recommendation engine"]
            }
        ]
    },
    "feature_discovery": {
        "description": "Feature discovery and adoption",
        "metrics": ["weekly_adoption_rate"],
        "actions": [
            {
                "action": "Create feature spotlight series",
                "effort": "low",
                "impact": "medium",
                "prerequisites": []
            },
            {
                "action": "Implement contextual feature suggestions",
                "effort": "high",
                "impact": "high",
                "prerequisites": ["Usage patterns analysis"]
            },
            {
                "action": "Add 'What's New' highlights",
                "effort": "low",
                "impact": "medium",
                "prerequisites": []
            },
            {
                "action": "Create feature usage guides",
                "effort": "medium",
                "impact": "medium",
                "prerequisites": []
            },
            {
                "action": "Implement feature usage gamification",
                "effort": "high",
                "impact": "medium",
                "prerequisites": []
            }
        ]
    },
    "user_experience": {
        "description": "User experience improvements",
        "metrics": ["daily_adoption_rate", "weekly_adoption_rate"],
        "actions": [
            {
                "action": "Conduct UX audit and optimization",
                "effort": "high",
                "impact": "high",
                "prerequisites": []
            },
            {
                "action": "Optimize page load times",
                "effort": "medium",
                "impact": "medium",
                "prerequisites": ["Performance baseline"]
            },
            {
                "action": "Implement responsive design improvements",
                "effort": "high",
                "impact": "medium",
                "prerequisites": []
            },
            {
                "action": "Simplify navigation and information architecture",
                "effort": "high",
                "impact": "high",
                "prerequisites": ["User journey mapping"]
            },
            {
                "action": "Add keyboard shortcuts for power users",
                "effort": "low",
                "impact": "low",
                "prerequisites": []
            }
        ]
    },
    "communication": {
        "description": "User communication strategies",
        "metrics": ["monthly_adoption_rate"],
        "actions": [
            {
                "action": "Develop targeted email campaign strategy",
                "effort": "medium",
                "impact": "high",
                "prerequisites": ["User segmentation"]
            },
            {
                "action": "Create user education content calendar",
                "effort": "medium",
                "impact": "medium",
                "prerequisites": []
            },
            {
                "action": "Implement in-app messaging system",
                "effort": "high",
                "impact": "medium",
                "prerequisites": []
            },
            {
                "action": "Launch monthly product update webinars",
                "effort": "medium",
                "impact": "medium",
                "prerequisites": []
            },
            {
                "action": "Create user community forum",
                "effort": "high",
                "impact": "medium",
                "prerequisites": []
            }
        ]
    }
}

def _analyze_target_area_relevance(data, metric_type="monthly"):
    """
    Analyze which target areas are most relevant based on adoption metrics.
    
    Args:
        data: Collection of adoption rate metrics
        metric_type: Type of metric to analyze
        
    Returns:
        dict: Dictionary of target areas with relevance scores
    """
    # Set up relevance scores for each target area
    relevance_scores = {area: 0.0 for area in TARGET_AREAS}
    
    # Sort data by date
    sorted_data = sorted(data.overall_adoption_rates, key=lambda x: x.date)
    
    if not sorted_data:
        logger.warning("No data available for target area analysis")
        return relevance_scores
    
    # Map metric type to field name
    if metric_type == "daily":
        rate_field = "daily_adoption_rate"
    elif metric_type == "weekly":
        rate_field = "weekly_adoption_rate"
    elif metric_type == "yearly":
        rate_field = "yearly_adoption_rate"
    elif metric_type == "monthly":
        rate_field = "monthly_adoption_rate"
    else:
        logger.error(f"Invalid metric_type: {metric_type}")
        raise ValueError(f"Invalid metric_type: {metric_type}. Must be one of: daily, weekly, monthly, yearly")
    
    # Analyze trend
    trend = calculate_trend(sorted_data, metric_type)
    trend_direction = trend.get('direction', 'stable')
    trend_strength = trend.get('strength', 0.5)
    
    # Calculate growth rate
    first_rate = getattr(sorted_data[0], rate_field)
    last_rate = getattr(sorted_data[-1], rate_field)
    growth_rate = (last_rate - first_rate) / first_rate if first_rate > 0 else 0
    
    # Calculate volatility
    rates = [getattr(item, rate_field) for item in sorted_data]
    volatility = np.std(rates) / np.mean(rates) if len(rates) > 1 and np.mean(rates) > 0 else 0
    
    # Calculate relevance for each target area
    for area_name, area in TARGET_AREAS.items():
        # Onboarding relevance
        if trend_direction == 'decreasing' or growth_rate < 0:
            relevance_scores[area_name] = 0.8
        else:
            relevance_scores[area_name] = 0.5
        
        # Retention relevance
        if trend_direction == 'decreasing' and trend_strength > 0.6:
            relevance_scores[area_name] = 0.9
        elif trend_direction == 'decreasing':
            relevance_scores[area_name] = 0.7
        
        # Feature discovery relevance
        if growth_rate < 0.1:  # Slow growth
            relevance_scores[area_name] = 0.8
        else:
            relevance_scores[area_name] = 0.6
        
        # User experience relevance
        if volatility > 0.2:  # High volatility
            relevance_scores[area_name] = 0.8
        else:
            relevance_scores[area_name] = 0.5
        
        # Communication relevance
        if trend_direction == 'stable':
            relevance_scores[area_name] = 0.7
        else:
            relevance_scores[area_name] = 0.5
    
    return relevance_scores

def _filter_actions_by_relevance(relevance_scores, max_actions=10):
    """
    Filter actions based on target area relevance scores.
    
    Args:
        relevance_scores: Dictionary of target areas with relevance scores
        max_actions: Maximum number of actions to return
    
    Returns:
        list: List of relevant actions with metadata
    """
    # Sort target areas by relevance
    sorted_areas = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Collect actions from the most relevant areas
    all_actions = []
    
    for area_name, score in sorted_areas:
        if area_name not in TARGET_AREAS:
            continue
        
        area = TARGET_AREAS[area_name]
        
        for action in area['actions']:
            # Add area metadata to action
            action_copy = action.copy()
            action_copy['target_area'] = area_name
            action_copy['area_description'] = area['description']
            action_copy['relevance_score'] = score
            
            # Calculate a combined score for sorting
            impact_score = {'high': 3, 'medium': 2, 'low': 1}[action['impact']]
            effort_score = {'low': 3, 'medium': 2, 'high': 1}[action['effort']]
            action_copy['combined_score'] = (score * impact_score * effort_score) / (1 + len(action['prerequisites']))
            
            all_actions.append(action_copy)
    
    # Sort by combined score and limit to max_actions
    sorted_actions = sorted(all_actions, key=lambda x: x['combined_score'], reverse=True)
    return sorted_actions[:max_actions]

def _calculate_impact_estimates(actions, data, metric_type="monthly"):
    """
    Calculate estimated impact for each action.
    
    Args:
        actions: List of actions
        data: Collection of adoption rate metrics
        metric_type: Type of metric to analyze
        
    Returns:
        list: Actions with impact estimates
    """
    # Sort data by date
    sorted_data = sorted(data.overall_adoption_rates, key=lambda x: x.date)
    
    if not sorted_data:
        logger.warning("No data available for impact estimation")
        return actions
    
    # Map metric type to field name
    if metric_type == "daily":
        rate_field = "daily_adoption_rate"
    elif metric_type == "weekly":
        rate_field = "weekly_adoption_rate"
    elif metric_type == "yearly":
        rate_field = "yearly_adoption_rate"
    elif metric_type == "monthly":
        rate_field = "monthly_adoption_rate"
    else:
        logger.error(f"Invalid metric_type: {metric_type}")
        raise ValueError(f"Invalid metric_type: {metric_type}. Must be one of: daily, weekly, monthly, yearly")
    
    # Get current adoption rate
    current_rate = getattr(sorted_data[-1], rate_field) if sorted_data else 0
    
    # Calculate baseline growth rate
    first_rate = getattr(sorted_data[0], rate_field) if sorted_data else 0
    growth_rate = (current_rate - first_rate) / first_rate if first_rate > 0 and len(sorted_data) > 1 else 0
    
    # Impact multiplier based on past growth rate
    # Higher impact for negative or low growth rates
    if growth_rate < 0:
        impact_multiplier = 2.0
    elif growth_rate < 0.05:
        impact_multiplier = 1.5
    elif growth_rate < 0.1:
        impact_multiplier = 1.2
    else:
        impact_multiplier = 1.0
    
    # Calculate impact estimates for each action
    for action in actions:
        impact_level = action['impact']
        base_impact = {'high': 2.0, 'medium': 1.0, 'low': 0.5}[impact_level]
        
        # Apply multipliers
        relevance_multiplier = action['relevance_score']
        effort_discount = {'high': 0.8, 'medium': 0.9, 'low': 1.0}[action['effort']]
        prerequisites_discount = 1.0 / (1 + 0.2 * len(action['prerequisites']))
        
        # Calculate estimated percentage impact
        estimated_impact = (
            base_impact * 
            impact_multiplier * 
            relevance_multiplier * 
            effort_discount * 
            prerequisites_discount
        )
        
        # Add impact estimate to action
        action['estimated_impact'] = estimated_impact
        action['estimated_impact_percentage'] = estimated_impact
    
    return actions

def _generate_action_explanation(action):
    """
    Generate a detailed explanation for an action.
    
    Args:
        action: Action dictionary with metadata
    
    Returns:
        str: Detailed explanation
    """
    explanation = f"**{action['action']}**\n\n"
    
    # Add target area context
    explanation += f"Target Area: {action['area_description']}\n"
    
    # Add impact and effort assessment
    explanation += f"Impact: {action['impact'].upper()} | Effort: {action['effort'].upper()}\n"
    
    # Add estimated impact
    explanation += f"Estimated Improvement: {action['estimated_impact_percentage']:.2f}% increase in adoption rate\n\n"
    
    # Add implementation considerations
    explanation += "Implementation considerations:\n"
    
    if action['prerequisites']:
        explanation += "Prerequisites:\n"
        for prereq in action['prerequisites']:
            explanation += f"- {prereq}\n"
    else:
        explanation += "No specific prerequisites required.\n"
    
    # Add implementation difficulty based on effort
    if action['effort'] == 'high':
        explanation += "\nThis is a significant initiative that will require substantial resources and planning.\n"
    elif action['effort'] == 'medium':
        explanation += "\nThis is a moderate initiative that requires dedicated resources but is achievable within a typical quarterly cycle.\n"
    else:
        explanation += "\nThis is a relatively straightforward initiative that can be implemented with minimal resources.\n"
    
    return explanation

def suggest_actions(data, metric_type="monthly", target_area=None, max_suggestions=5):
    """
    Suggest specific actions to improve adoption rates.
    
    Args:
        data: Collection of adoption rate metrics
        metric_type: Type of metric to analyze
        target_area: Specific area to target (optional)
        max_suggestions: Maximum number of suggestions to return
    
    Returns:
        dict: Dictionary containing suggested actions
    """
    logger.info(f"Generating action suggestions for {metric_type} adoption rate")
    
    if not data.overall_adoption_rates:
        return {
            "explanation": "No data available for suggesting actions."
        }
    
    try:
        # Analyze target area relevance
        relevance_scores = _analyze_target_area_relevance(data, metric_type)
        
        # Filter to specific target area if requested
        if target_area and target_area in TARGET_AREAS:
            # Set the specified target area to high relevance
            filtered_scores = {area: (0.9 if area == target_area else 0.1) for area in relevance_scores}
        else:
            filtered_scores = relevance_scores
        
        # Filter actions by relevance
        actions = _filter_actions_by_relevance(filtered_scores, max_actions=max_suggestions * 2)
        
        # Calculate impact estimates
        actions_with_impact = _calculate_impact_estimates(actions, data, metric_type)
        
        # Sort by estimated impact and limit to max_suggestions
        sorted_actions = sorted(actions_with_impact, key=lambda x: x['estimated_impact'], reverse=True)
        top_actions = sorted_actions[:max_suggestions]
        
        # Generate detailed explanations for top actions
        action_explanations = {}
        for action in top_actions:
            action_explanations[action['action']] = _generate_action_explanation(action)
        
        # Sort target areas by relevance for the explanation
        sorted_target_areas = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Generate overall explanation
        overall_explanation = _generate_overall_explanation(top_actions, sorted_target_areas, metric_type)
        
        return {
            "actions": top_actions,
            "all_analyzed_actions": actions_with_impact,
            "target_area_relevance": relevance_scores,
            "detailed_explanations": action_explanations,
            "explanation": overall_explanation
        }
    
    except Exception as e:
        logger.error(f"Error suggesting actions: {e}")
        return {
            "explanation": f"An error occurred when suggesting actions: {str(e)}"
        }

def _generate_overall_explanation(actions, target_areas, metric_type):
    """
    Generate an overall explanation for the suggested actions.
    
    Args:
        actions: List of suggested actions
        target_areas: Target areas sorted by relevance
        metric_type: Type of metric analyzed
    
    Returns:
        str: Overall explanation
    """
    if not actions:
        return "No actions were suggested due to insufficient data."
    
    # Introduction
    explanation = "Based on the analysis of your adoption rate data, we've identified the following high-impact actions "
    explanation += f"to improve your {metric_type} adoption rates.\n\n"
    
    # Summarize target areas
    explanation += "Priority focus areas:\n"
    for i, (area, score) in enumerate(target_areas[:3], 1):
        if area in TARGET_AREAS:
            explanation += f"{i}. {TARGET_AREAS[area]['description']} (Relevance: {score:.2f})\n"
    
    explanation += "\nTop recommended actions:\n"
    for i, action in enumerate(actions[:5], 1):
        impact_str = {'high': 'high impact', 'medium': 'medium impact', 'low': 'low impact'}[action['impact']]
        effort_str = {'high': 'significant effort', 'medium': 'moderate effort', 'low': 'minimal effort'}[action['effort']]
        explanation += f"{i}. {action['action']} - {impact_str}, {effort_str}, estimated {action['estimated_impact_percentage']:.2f}% improvement\n"
    
    # Add implementation guidance
    explanation += "\nImplementation approach: We recommend prioritizing high-impact, low-effort actions first to "
    explanation += "achieve quick wins, while planning for more significant initiatives in parallel. "
    explanation += "Consider the prerequisites for each action when planning your implementation timeline."
    
    return explanation 