"""
Recommendation Engine Module

This module provides functionality for generating recommendations to improve
adoption rates based on historical data analysis and patterns.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

from src.data_models.metrics import MetricCollection
from src.data_analysis.trend_analyzer import calculate_trend_line as calculate_trend, detect_peaks_and_valleys as detect_peaks_valleys
from src.data_analysis.period_analyzer import calculate_mom_change, calculate_qoq_change, calculate_yoy_change
from src.data_analysis.anomaly_detector import detect_anomalies_ensemble

# Set up logging
logger = logging.getLogger(__name__)

# Define categories of recommendations
RECOMMENDATION_CATEGORIES = {
    "onboarding": {
        "description": "Improve new user onboarding experience",
        "impact_factor": 1.2,
        "time_to_value": "medium",
        "typical_actions": [
            "Simplify first-time user experience",
            "Improve product tour walkthrough",
            "Add interactive guides for key features",
            "Implement progress indicators for setup",
            "Add contextual help during onboarding"
        ]
    },
    "feature_engagement": {
        "description": "Increase engagement with core features",
        "impact_factor": 1.3,
        "time_to_value": "medium-long",
        "typical_actions": [
            "Add feature discovery tooltips",
            "Implement in-app tutorials for power features",
            "Send feature spotlight emails",
            "Create success metrics for each feature",
            "Add gamification elements for feature usage"
        ]
    },
    "retention": {
        "description": "Improve user retention",
        "impact_factor": 1.5,
        "time_to_value": "long",
        "typical_actions": [
            "Implement re-engagement emails",
            "Create personalized usage dashboards",
            "Add user success milestones",
            "Implement abandoned workflow reminders",
            "Create value-demonstrating success reports"
        ]
    },
    "performance": {
        "description": "Improve product performance",
        "impact_factor": 1.1,
        "time_to_value": "short",
        "typical_actions": [
            "Improve page load times",
            "Optimize database queries",
            "Implement caching for frequent operations",
            "Reduce API response times",
            "Address error rates in high-traffic areas"
        ]
    },
    "communication": {
        "description": "Enhance user communication",
        "impact_factor": 1.0,
        "time_to_value": "short-medium",
        "typical_actions": [
            "Implement targeted feature announcements",
            "Create usage-based communication segments",
            "Set up regular feedback collection",
            "Add in-app notification center",
            "Create user communities or forums"
        ]
    }
}

def _analyze_adoption_patterns(data, metric_type="monthly"):
    """
    Analyze patterns in adoption rate data to inform recommendations.
    
    Args:
        data: Collection of adoption rate metrics
        metric_type: Type of metric to analyze
        
    Returns:
        dict: Analysis results
    """
    # Extract adoption rate data
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
    
    # Sort data by date
    sorted_data = sorted(data.overall_adoption_rates, key=lambda x: x.date)
    
    # Extract dates and rates
    dates = [item.date for item in sorted_data]
    rates = [getattr(item, rate_field) for item in sorted_data]
    
    # Create a DataFrame for easier analysis
    df = pd.DataFrame({
        'date': dates,
        'rate': rates
    })
    
    # Calculate key metrics
    analysis = {}
    
    # Current value
    analysis['current_value'] = rates[-1] if rates else 0
    
    # Trend analysis
    analysis['trend'] = calculate_trend(sorted_data, metric_type)
    
    # Growth rates
    if len(rates) >= 2:
        analysis['growth_rate'] = (rates[-1] - rates[0]) / rates[0] if rates[0] > 0 else 0
    else:
        analysis['growth_rate'] = 0
    
    # Period-over-period changes
    period_changes = {}
    
    # Only calculate if enough data points
    if len(sorted_data) >= 2:
        # Last month change
        try:
            mom_change = calculate_mom_change(sorted_data, metric_type)
            period_changes['mom'] = mom_change
        except Exception as e:
            logger.warning(f"Error calculating MoM change: {e}")
            period_changes['mom'] = {'absolute_change': 0, 'percentage_change': 0}
    
    if len(sorted_data) >= 4:
        # Last quarter change
        try:
            qoq_change = calculate_qoq_change(sorted_data, metric_type)
            period_changes['qoq'] = qoq_change
        except Exception as e:
            logger.warning(f"Error calculating QoQ change: {e}")
            period_changes['qoq'] = {'absolute_change': 0, 'percentage_change': 0}
    
    if len(sorted_data) >= 12:
        # Last year change
        try:
            yoy_change = calculate_yoy_change(sorted_data, metric_type)
            period_changes['yoy'] = yoy_change
        except Exception as e:
            logger.warning(f"Error calculating YoY change: {e}")
            period_changes['yoy'] = {'absolute_change': 0, 'percentage_change': 0}
    
    analysis['period_changes'] = period_changes
    
    # Detect peaks and valleys
    try:
        peaks_valleys = detect_peaks_valleys(sorted_data, metric_type)
        analysis['peaks_valleys'] = peaks_valleys
    except Exception as e:
        logger.warning(f"Error detecting peaks and valleys: {e}")
        analysis['peaks_valleys'] = {'peaks': [], 'valleys': []}
    
    # Detect anomalies
    try:
        anomalies = detect_anomalies_ensemble(sorted_data, metric_type)
        analysis['anomalies'] = anomalies
    except Exception as e:
        logger.warning(f"Error detecting anomalies: {e}")
        analysis['anomalies'] = []
    
    # Calculate volatility
    if len(rates) > 1:
        analysis['volatility'] = np.std(rates) / np.mean(rates) if np.mean(rates) > 0 else 0
    else:
        analysis['volatility'] = 0
    
    # Calculate consistency score (inverse of volatility, normalized to 0-1)
    analysis['consistency'] = 1 / (1 + analysis['volatility']) if analysis['volatility'] > 0 else 1
    
    # Calculate seasonality score
    # This is a simplistic approach - a more sophisticated approach would use time series decomposition
    if len(rates) >= 12:
        # Calculate autocorrelation at lag 12 (annual seasonality)
        acf = np.correlate(rates, rates, mode='full')
        acf = acf[len(acf)//2:] / np.var(rates) / len(rates)
        if len(acf) >= 12:
            analysis['seasonality'] = abs(acf[12])
        else:
            analysis['seasonality'] = 0
    else:
        analysis['seasonality'] = 0
    
    return analysis

def _identify_improvement_areas(analysis):
    """
    Identify areas that need improvement based on data analysis.
    
    Args:
        analysis: Results from data analysis
        
    Returns:
        list: List of areas that need improvement, with scores
    """
    improvement_areas = []
    
    # Check trend
    trend_direction = analysis.get('trend', {}).get('direction', 'stable')
    if trend_direction == 'decreasing':
        improvement_areas.append({
            'area': 'retention',
            'score': 0.9,
            'reason': 'Adoption rate is trending downward, indicating retention issues.'
        })
    elif trend_direction == 'stable':
        improvement_areas.append({
            'area': 'feature_engagement',
            'score': 0.7,
            'reason': 'Adoption rate is stable but not growing, suggesting engagement opportunities.'
        })
    
    # Check growth rate
    growth_rate = analysis.get('growth_rate', 0)
    if growth_rate < 0.05:  # Less than 5% growth
        improvement_areas.append({
            'area': 'onboarding',
            'score': 0.8,
            'reason': 'Low growth rate suggests new user adoption challenges.'
        })
    
    # Check recent period changes
    period_changes = analysis.get('period_changes', {})
    
    # MoM changes
    mom_change = period_changes.get('mom', {}).get('percentage_change', 0)
    if mom_change < 0:
        improvement_areas.append({
            'area': 'communication',
            'score': 0.6,
            'reason': 'Recent decline in Month-over-Month growth suggests communication issues.'
        })
    
    # QoQ changes
    qoq_change = period_changes.get('qoq', {}).get('percentage_change', 0)
    if qoq_change < 0:
        # Add retention if not already added
        if not any(area['area'] == 'retention' for area in improvement_areas):
            improvement_areas.append({
                'area': 'retention',
                'score': 0.7,
                'reason': 'Quarter-over-Quarter decline indicates medium-term retention issues.'
            })
    
    # Check volatility
    volatility = analysis.get('volatility', 0)
    if volatility > 0.2:  # High volatility
        improvement_areas.append({
            'area': 'performance',
            'score': 0.6,
            'reason': 'High volatility in adoption rates suggests inconsistent performance.'
        })
    
    # Add default area if none identified
    if not improvement_areas:
        improvement_areas.append({
            'area': 'feature_engagement',
            'score': 0.5,
            'reason': 'Default recommendation to improve feature engagement.'
        })
    
    # Sort by score
    improvement_areas.sort(key=lambda x: x['score'], reverse=True)
    
    return improvement_areas

def _generate_specific_recommendations(improvement_areas, target_improvement, analysis):
    """
    Generate specific recommendations based on identified improvement areas.
    
    Args:
        improvement_areas: List of areas that need improvement
        target_improvement: Target percentage improvement
        analysis: Results from data analysis
        
    Returns:
        list: List of specific recommendations
    """
    recommendations = []
    
    current_value = analysis.get('current_value', 0)
    target_value = current_value * (1 + target_improvement / 100)
    
    for area_info in improvement_areas:
        area = area_info['area']
        score = area_info['score']
        reason = area_info['reason']
        
        if area not in RECOMMENDATION_CATEGORIES:
            continue
        
        category = RECOMMENDATION_CATEGORIES[area]
        typical_actions = category['typical_actions']
        impact_factor = category['impact_factor']
        time_to_value = category['time_to_value']
        
        # Calculate expected impact
        expected_impact = score * impact_factor * target_improvement / len(improvement_areas)
        
        # Generate recommendations for this area
        for action in typical_actions:
            # Create a recommendation
            recommendation = {
                'action': action,
                'category': area,
                'impact_estimate': expected_impact * (0.7 + 0.3 * np.random.random()),  # Add some variation
                'time_to_value': time_to_value,
                'rationale': reason,
                'current_value': current_value,
                'target_value': target_value
            }
            
            recommendations.append(recommendation)
    
    # Sort by impact estimate
    recommendations.sort(key=lambda x: x['impact_estimate'], reverse=True)
    
    return recommendations

def _generate_implementation_plan(recommendations, max_actions=5):
    """
    Generate an implementation plan based on recommendations.
    
    Args:
        recommendations: List of recommendations
        max_actions: Maximum number of actions to include
        
    Returns:
        dict: Implementation plan
    """
    # Take top recommendations
    top_recommendations = recommendations[:max_actions]
    
    # Group by time to value
    short_term = []
    medium_term = []
    long_term = []
    
    for rec in top_recommendations:
        ttv = rec['time_to_value']
        if ttv == 'short' or ttv == 'short-medium':
            short_term.append(rec)
        elif ttv == 'medium' or ttv == 'medium-long':
            medium_term.append(rec)
        else:
            long_term.append(rec)
    
    # Create implementation phases
    plan = {
        'phase1': {
            'timeframe': '0-30 days',
            'actions': short_term,
            'expected_impact': sum(rec['impact_estimate'] for rec in short_term)
        },
        'phase2': {
            'timeframe': '30-90 days',
            'actions': medium_term,
            'expected_impact': sum(rec['impact_estimate'] for rec in medium_term)
        },
        'phase3': {
            'timeframe': '90+ days',
            'actions': long_term,
            'expected_impact': sum(rec['impact_estimate'] for rec in long_term)
        }
    }
    
    # Calculate cumulative impact
    cumulative_impact = 0
    for phase in ['phase1', 'phase2', 'phase3']:
        cumulative_impact += plan[phase]['expected_impact']
        plan[phase]['cumulative_impact'] = cumulative_impact
    
    return plan

def generate_recommendations(data, metric_type="monthly", target_improvement=10):
    """
    Generate recommendations for improving adoption rates based on historical performance.
    
    Args:
        data: Collection of adoption rate metrics
        metric_type: Type of metric to analyze
        target_improvement: Target percentage improvement
        
    Returns:
        dict: Dictionary containing recommendations
    """
    logger.info(f"Generating recommendations for {metric_type} adoption rate with {target_improvement}% target")
    
    if not data.overall_adoption_rates:
        return {
            "explanation": "No data available for generating recommendations."
        }
    
    try:
        # Analyze adoption patterns
        analysis = _analyze_adoption_patterns(data, metric_type)
        
        # Identify improvement areas
        improvement_areas = _identify_improvement_areas(analysis)
        
        # Generate specific recommendations
        recommendations = _generate_specific_recommendations(
            improvement_areas, target_improvement, analysis
        )
        
        # Generate implementation plan
        implementation_plan = _generate_implementation_plan(recommendations)
        
        # Calculate overall expected impact
        overall_impact = sum(rec['impact_estimate'] for rec in recommendations[:5])
        
        # Generate natural language explanation
        explanation = _generate_recommendation_explanation(
            recommendations[:5], improvement_areas, analysis, target_improvement
        )
        
        return {
            "top_recommendations": recommendations[:5],
            "all_recommendations": recommendations,
            "improvement_areas": improvement_areas,
            "implementation_plan": implementation_plan,
            "expected_impact": overall_impact,
            "target_improvement": target_improvement,
            "data_analysis": analysis,
            "explanation": explanation
        }
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return {
            "explanation": f"An error occurred when generating recommendations: {str(e)}"
        }

def _generate_recommendation_explanation(recommendations, improvement_areas, analysis, target_improvement):
    """
    Generate a natural language explanation for the recommendations.
    
    Args:
        recommendations: Top recommendations
        improvement_areas: Identified improvement areas
        analysis: Results from data analysis
        target_improvement: Target percentage improvement
        
    Returns:
        str: Natural language explanation
    """
    current_value = analysis.get('current_value', 0)
    trend_direction = analysis.get('trend', {}).get('direction', 'stable')
    
    # Start with overview
    explanation = f"Based on the analysis of your adoption rate data, we've identified {len(improvement_areas)} key areas "
    explanation += f"for improvement to help you reach your target of {target_improvement}% growth.\n\n"
    
    # Add context about current state
    explanation += f"Your current {analysis.get('metric_type', 'monthly')} adoption rate is {current_value:.2f}%, "
    
    if trend_direction == 'increasing':
        explanation += "which is trending upward. Our recommendations will help accelerate this positive trend.\n\n"
    elif trend_direction == 'decreasing':
        explanation += "which is trending downward. Our recommendations focus on reversing this negative trend.\n\n"
    else:
        explanation += "which has been relatively stable. Our recommendations aim to catalyze growth from this stable base.\n\n"
    
    # Summarize top improvement areas
    explanation += "Key focus areas:\n"
    for i, area in enumerate(improvement_areas[:3], 1):
        category = RECOMMENDATION_CATEGORIES.get(area['area'], {})
        explanation += f"{i}. {category.get('description', area['area'].capitalize())}: {area['reason']}\n"
    
    explanation += "\nTop recommendations:\n"
    for i, rec in enumerate(recommendations[:3], 1):
        explanation += f"{i}. {rec['action']} - Expected impact: {rec['impact_estimate']:.2f}% increase\n"
    
    # Add implementation guidance
    explanation += "\nImplementation approach: We recommend a phased implementation, starting with "
    explanation += "quick wins that can show immediate results while laying groundwork for longer-term improvements."
    
    return explanation 