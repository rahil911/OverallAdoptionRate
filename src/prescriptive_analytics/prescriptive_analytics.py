"""
Prescriptive Analytics Module

This module provides recommendations and action suggestions for improving adoption rates
based on data analysis. It serves as the main interface for all prescriptive analytics capabilities.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

from src.database.data_access import DataAccessLayer
from src.data_models.metrics import (
    MetricCollection,
    OverallAdoptionRate,
    MonthlyActiveUsers,
    DailyActiveUsers
)

# Set up logging
logger = logging.getLogger(__name__)

class PrescriptiveAnalytics:
    """
    Main class for providing prescriptive analytics about adoption rate data.
    This class integrates recommendation engines, action suggestions, goal-setting
    assistance, and intervention impact estimations.
    """
    
    def __init__(self, tenant_id=1388):
        """
        Initialize the class with a tenant ID.
        
        Args:
            tenant_id (int): Tenant ID to use for data retrieval (defaults to 1388)
        """
        self.tenant_id = tenant_id
        self.data_access = DataAccessLayer()
    
    def _get_data(self, from_date=None, to_date=None):
        """
        Retrieve adoption rate data within a specified date range.
        
        Args:
            from_date (datetime, optional): Start date for data retrieval
            to_date (datetime, optional): End date for data retrieval
            
        Returns:
            MetricCollection: Collection of all relevant metrics
        """
        # Set default date range if not provided
        if to_date is None:
            to_date = datetime.now()
        if from_date is None:
            from_date = to_date - timedelta(days=730)  # 2 years
        
        # Get data from database
        adoption_rate_data = DataAccessLayer.get_overall_adoption_rate(from_date, to_date, self.tenant_id)
        mau_data = DataAccessLayer.get_mau(from_date, to_date, self.tenant_id)
        dau_data = DataAccessLayer.get_dau(from_date, to_date, self.tenant_id)
        
        # Convert to MetricCollection
        collection = MetricCollection(tenant_id=self.tenant_id)
        
        # Convert DataFrame rows to model instances
        if not adoption_rate_data.empty:
            collection.overall_adoption_rates = [
                OverallAdoptionRate.from_db_row(row, self.tenant_id)
                for _, row in adoption_rate_data.iterrows()
            ]
        
        if not mau_data.empty:
            collection.monthly_active_users = [
                MonthlyActiveUsers.from_db_row(row, self.tenant_id)
                for _, row in mau_data.iterrows()
            ]
        
        if not dau_data.empty:
            collection.daily_active_users = [
                DailyActiveUsers.from_db_row(row, self.tenant_id)
                for _, row in dau_data.iterrows()
            ]
        
        return collection
    
    def generate_recommendations(self, from_date=None, to_date=None, metric_type="monthly", 
                                target_improvement=10):
        """
        Generate recommendations for improving adoption rates based on historical performance.
        
        Args:
            from_date (datetime, optional): Start date for analysis
            to_date (datetime, optional): End date for analysis
            metric_type (str): Type of metric to analyze ("daily", "weekly", "monthly", "yearly")
            target_improvement (float): Target percentage improvement
            
        Returns:
            dict: Dictionary containing recommendations including:
                - top_recommendations: List of top recommendations
                - expected_impact: Expected impact of implementing recommendations
                - reasoning: Explanation of reasoning behind recommendations
                - prioritized_actions: List of actions prioritized by impact
        """
        # Import the recommendation engine module
        from src.prescriptive_analytics.recommendation_engine import generate_recommendations
        
        # Get data for analysis
        collection = self._get_data(from_date, to_date)
        
        if not collection.overall_adoption_rates:
            return {
                "explanation": "No data available for generating recommendations."
            }
        
        # Generate recommendations
        recommendations = generate_recommendations(
            collection,
            metric_type=metric_type,
            target_improvement=target_improvement
        )
        
        return recommendations
    
    def suggest_actions(self, from_date=None, to_date=None, metric_type="monthly", 
                      target_area=None, max_suggestions=5):
        """
        Suggest specific actions to improve adoption rates.
        
        Args:
            from_date (datetime, optional): Start date for analysis
            to_date (datetime, optional): End date for analysis
            metric_type (str): Type of metric to analyze
            target_area (str, optional): Specific area to target (e.g., "onboarding", "retention")
            max_suggestions (int): Maximum number of suggestions to return
            
        Returns:
            dict: Dictionary containing suggested actions including:
                - actions: List of suggested actions
                - reasoning: Explanation for each action
                - expected_impact: Expected impact of each action
                - implementation_difficulty: Estimated difficulty to implement
        """
        # Import the action suggestions module
        from src.prescriptive_analytics.action_suggestions import suggest_actions
        
        # Get data for analysis
        collection = self._get_data(from_date, to_date)
        
        if not collection.overall_adoption_rates:
            return {
                "explanation": "No data available for suggesting actions."
            }
        
        # Generate action suggestions
        suggestions = suggest_actions(
            collection,
            metric_type=metric_type,
            target_area=target_area,
            max_suggestions=max_suggestions
        )
        
        return suggestions
    
    def assist_goal_setting(self, from_date=None, to_date=None, metric_type="monthly",
                          goal_type=None, timeframe=None, custom_target=None):
        """
        Provide goal-setting assistance based on historical data.
        
        Args:
            from_date (datetime, optional): Start date for analysis
            to_date (datetime, optional): End date for analysis
            metric_type (str): Type of metric to analyze
            goal_type (str, optional): Type of goal (conservative, realistic, challenging, aggressive)
            timeframe (str, optional): Timeframe for the goal (short_term, medium_term, long_term)
            custom_target (float, optional): Custom target adoption rate
            
        Returns:
            dict: Dictionary containing goal-setting recommendations:
                - recommended_goals: List of recommended goals
                - rationale: Explanation for each goal
                - achievability: Assessment of achievability
                - milestone_targets: Intermediate milestones
        """
        # Import the goal setting module
        from src.prescriptive_analytics.goal_setting import assist_goal_setting
        
        # Get data for analysis
        collection = self._get_data(from_date, to_date)
        
        if not collection.overall_adoption_rates:
            return {
                "explanation": "No data available for goal-setting assistance."
            }
        
        # Generate goal-setting recommendations
        goals = assist_goal_setting(
            collection,
            metric_type=metric_type,
            goal_type=goal_type,
            timeframe=timeframe,
            custom_target=custom_target
        )
        
        return goals
    
    def estimate_intervention_impact(self, from_date=None, to_date=None, metric_type="monthly",
                                   forecast_periods=12, interventions=None):
        """
        Estimate the impact of a specific intervention on adoption rates.
        
        Args:
            from_date (datetime, optional): Start date for analysis
            to_date (datetime, optional): End date for analysis
            metric_type (str): Type of metric to analyze
            forecast_periods (int): Number of periods to forecast
            interventions (list, optional): List of intervention configurations, each with type and strength
            
        Returns:
            dict: Dictionary containing impact estimation:
                - baseline_forecast: Forecast without intervention
                - intervention_forecast: Forecast with intervention
                - impact_percentage: Estimated percentage impact
                - recovery_time: Time to see impact
                - sensitivity_analysis: Analysis of impact sensitivity
        """
        # Import the intervention impact module
        from src.prescriptive_analytics.intervention_impact import estimate_intervention_impact
        
        # Get data for analysis
        collection = self._get_data(from_date, to_date)
        
        if not collection.overall_adoption_rates:
            return {
                "explanation": "No data available for intervention impact estimation."
            }
        
        # Generate intervention impact estimation
        impact = estimate_intervention_impact(
            collection,
            metric_type=metric_type,
            interventions=interventions,
            forecast_periods=forecast_periods
        )
        
        return impact
    
    def prioritize_actions(self, actions, from_date=None, to_date=None, 
                         criteria=None, weights=None):
        """
        Prioritize suggested actions based on multiple criteria.
        
        Args:
            actions (list): List of potential actions to prioritize
            from_date (datetime, optional): Start date for analysis
            to_date (datetime, optional): End date for analysis
            criteria (list, optional): Criteria for prioritization
            weights (dict, optional): Weights for different criteria
            
        Returns:
            dict: Dictionary containing prioritized actions:
                - prioritized_actions: List of actions in priority order
                - scores: Score for each action
                - rationale: Explanation of prioritization logic
                - implementation_plan: Suggested implementation sequence
        """
        # Import the prioritization module
        from src.prescriptive_analytics.prioritization import prioritize_actions as prioritize_actions_func
        
        # Get data for analysis
        collection = self._get_data(from_date, to_date)
        
        if not collection.overall_adoption_rates:
            return {
                "explanation": "No data available for action prioritization."
            }
        
        # Default criteria if not provided
        if criteria is None:
            criteria = ["impact", "effort", "time_to_value", "risk"]
        
        # Default weights if not provided
        if weights is None:
            weights = {
                "impact": 0.4,
                "effort": 0.2,
                "time_to_value": 0.3,
                "risk": 0.1
            }
        
        # Prioritize actions
        prioritized = prioritize_actions_func(
            actions,
            factor_weights=weights
        )
        
        return prioritized
    
    def suggest_benchmark_recommendations(self, from_date=None, to_date=None,
                                        benchmark_type="industry", metric_type="monthly"):
        """
        Generate recommendations based on industry benchmarks or internal standards.
        
        Args:
            from_date (datetime, optional): Start date for analysis
            to_date (datetime, optional): End date for analysis
            benchmark_type (str): Type of benchmark ("industry", "top_performer", "historical_best")
            metric_type (str): Type of metric to analyze
            
        Returns:
            dict: Dictionary containing benchmark-based recommendations:
                - current_performance: Current performance metrics
                - benchmark_performance: Benchmark performance metrics
                - gap_analysis: Analysis of the gap between current and benchmark
                - recommendations: Specific recommendations to close the gap
                - expected_improvement: Expected improvement from recommendations
        """
        # Import the benchmark recommendations module
        from src.prescriptive_analytics.benchmark_recommendations import suggest_benchmark_recommendations
        
        # Get data for analysis
        collection = self._get_data(from_date, to_date)
        
        if not collection.overall_adoption_rates:
            return {
                "explanation": "No data available for benchmark recommendations."
            }
        
        # Generate benchmark recommendations
        recommendations = suggest_benchmark_recommendations(
            collection,
            benchmark_type=benchmark_type,
            metric_type=metric_type
        )
        
        return recommendations
    
    def suggest_alert_thresholds(self, from_date=None, to_date=None, 
                               metric_type="monthly", sensitivity="medium"):
        """
        Suggest custom alert thresholds based on historical data patterns.
        
        Args:
            from_date (datetime, optional): Start date for analysis
            to_date (datetime, optional): End date for analysis
            metric_type (str): Type of metric to analyze
            sensitivity (str): Desired sensitivity level ("low", "medium", "high")
            
        Returns:
            dict: Dictionary containing suggested alert thresholds:
                - metric_thresholds: Suggested thresholds for different metrics
                - statistical_basis: Statistical justification for thresholds
                - false_positive_rate: Estimated false positive rate
                - detection_rate: Estimated detection rate for true anomalies
                - historical_simulation: Results of applying thresholds to historical data
        """
        # Import the alert thresholds module
        from src.prescriptive_analytics.alert_thresholds import suggest_alert_thresholds
        
        # Get data for analysis
        collection = self._get_data(from_date, to_date)
        
        if not collection.overall_adoption_rates:
            return {
                "explanation": "No data available for suggesting alert thresholds."
            }
        
        # Generate alert threshold suggestions
        thresholds = suggest_alert_thresholds(
            collection,
            metric_type=metric_type,
            sensitivity=sensitivity
        )
        
        return thresholds 