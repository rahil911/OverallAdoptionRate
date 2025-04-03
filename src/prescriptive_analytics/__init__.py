"""
Prescriptive Analytics Package

This package provides recommendation functionality for improving adoption rates,
suggesting specific actions, assisting with goal setting, and estimating the
impact of interventions.
"""

from src.prescriptive_analytics.prescriptive_analytics import PrescriptiveAnalytics
from src.prescriptive_analytics.recommendation_engine import generate_recommendations
from src.prescriptive_analytics.action_suggestions import suggest_actions
from src.prescriptive_analytics.goal_setting import assist_goal_setting
from src.prescriptive_analytics.intervention_impact import estimate_intervention_impact
from src.prescriptive_analytics.prioritization import prioritize_actions
from src.prescriptive_analytics.benchmark_recommendations import suggest_benchmark_recommendations
from src.prescriptive_analytics.alert_thresholds import suggest_alert_thresholds

__all__ = [
    'PrescriptiveAnalytics',
    'generate_recommendations',
    'suggest_actions',
    'assist_goal_setting',
    'estimate_intervention_impact',
    'prioritize_actions',
    'suggest_benchmark_recommendations',
    'suggest_alert_thresholds'
] 