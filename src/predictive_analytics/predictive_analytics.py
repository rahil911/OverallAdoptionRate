"""
Predictive Analytics Module

This module provides forecasting capabilities for adoption rate data, implementing
time series forecasting, confidence interval calculation, and scenario-based modeling.
"""

import logging
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any, Union, Tuple

from src.database.data_access import DataAccessLayer
from src.data_models.metrics import (
    MetricCollection,
    OverallAdoptionRate,
    MonthlyActiveUsers,
    DailyActiveUsers
)
from src.data_analysis.trend_analyzer import (
    calculate_trend_line,
    calculate_moving_average
)

# Configure logging
logger = logging.getLogger(__name__)

class PredictiveAnalytics:
    """
    Main class for providing predictive analytics about adoption rate data.
    This class implements time series forecasting, confidence interval calculation,
    trend extrapolation, and what-if scenario modeling for adoption rate metrics.
    """
    
    def __init__(self, tenant_id=1388):
        """
        Initialize the class with a tenant ID.
        
        Args:
            tenant_id (int): Tenant ID to use for data retrieval (defaults to 1388)
        """
        self.tenant_id = tenant_id
        self.data_access = DataAccessLayer()
    
    def _get_data(self, from_date: Optional[date] = None, to_date: Optional[date] = None) -> 'MetricCollection':
        """
        Get adoption rate data for the specified date range.
        
        Args:
            from_date: Start date for data retrieval (defaults to 12 months ago)
            to_date: End date for data retrieval (defaults to today)
            
        Returns:
            MetricCollection containing adoption rate data
        """
        # Set default date range if not provided
        if not to_date:
            to_date = date.today()
        if not from_date:
            # Default to 12 months before to_date
            from_date = date(to_date.year - 1, to_date.month, to_date.day)
        
        # Get data from database
        df = self.data_access.execute_stored_procedure(
            "SP_OverallAdoptionRate_DWMY",
            {
                "FromDate": from_date,
                "ToDate": to_date,
                "Tenantid": self.tenant_id
            }
        )
        
        # Convert DataFrame rows to OverallAdoptionRate objects
        adoption_rates = []
        if not df.empty:
            for _, row in df.iterrows():
                try:
                    adoption_rate = OverallAdoptionRate.from_db_row(row, self.tenant_id)
                    adoption_rates.append(adoption_rate)
                except Exception as e:
                    logging.error(f"Error converting row to OverallAdoptionRate: {e}")
        
        # Create MetricCollection and return
        metrics = MetricCollection()
        metrics.overall_adoption_rates = adoption_rates
        
        return metrics
    
    def forecast_adoption_rate(self, from_date=None, to_date=None, forecast_periods=12, 
                              metric_type="monthly", method="auto"):
        """
        Forecast adoption rate into the future.
        
        Args:
            from_date (datetime, optional): Start date for historical data
            to_date (datetime, optional): End date for historical data
            forecast_periods (int): Number of periods to forecast
            metric_type (str): Type of metric to forecast ("daily", "weekly", "monthly", "yearly")
            method (str): Forecasting method to use ("auto", "trend", "arima", "ets", "prophet")
            
        Returns:
            dict: Dictionary containing forecast results:
                - forecast_values: List of predicted values
                - forecast_dates: List of dates for the forecast
                - confidence_intervals: Dictionary of lower and upper bounds
                - model_metrics: Performance metrics for the forecast model
                - forecast_explanation: Natural language explanation of the forecast
        """
        # Import the forecasting modules
        from src.predictive_analytics.time_series_forecasting import (
            create_time_series_forecast,
            select_best_forecast_method
        )
        
        # Get historical data
        collection = self._get_data(from_date, to_date)
        
        if not collection.overall_adoption_rates:
            return {
                "explanation": "No data available for forecasting."
            }
        
        # Sort data by date
        sorted_data = sorted(collection.overall_adoption_rates, key=lambda x: x.date)
        
        # Select method if auto is specified
        if method == "auto":
            method = select_best_forecast_method(sorted_data, metric_type)
            logger.info(f"Selected forecast method: {method}")
        
        # Create forecast
        forecast_result = create_time_series_forecast(
            sorted_data,
            metric_type=metric_type,
            forecast_periods=forecast_periods,
            method=method
        )
        
        # Generate explanation
        explanation = self._generate_forecast_explanation(
            sorted_data,
            forecast_result,
            metric_type,
            method
        )
        
        forecast_result["explanation"] = explanation
        return forecast_result
    
    def calculate_confidence_intervals(self, forecast_data, confidence_level=0.95):
        """
        Calculate confidence intervals for a forecast.
        
        Args:
            forecast_data: Forecast data from the forecast_adoption_rate method
            confidence_level (float): Confidence level (0-1)
            
        Returns:
            dict: Dictionary with lower and upper confidence bounds
        """
        # Import the confidence interval calculation module
        from src.predictive_analytics.confidence_intervals import (
            calculate_forecast_intervals
        )
        
        if "forecast_values" not in forecast_data:
            return {
                "explanation": "No forecast data available for calculating confidence intervals."
            }
        
        # Calculate intervals
        intervals = calculate_forecast_intervals(
            forecast_data["forecast_values"],
            forecast_data.get("model_info", {}),
            confidence_level
        )
        
        return intervals
    
    def predict_target_achievement(self, target_value, from_date=None, to_date=None,
                                  metric_type="monthly", max_horizon=730):
        """
        Predict when a target adoption rate will be achieved.
        
        Args:
            target_value (float): Target adoption rate to achieve
            from_date (datetime, optional): Start date for historical data
            to_date (datetime, optional): End date for historical data
            metric_type (str): Type of metric to analyze ("daily", "weekly", "monthly", "yearly")
            max_horizon (int): Maximum number of days to forecast into the future
            
        Returns:
            dict: Dictionary with prediction results:
                - achievement_date: Estimated date of target achievement
                - confidence_level: Confidence in the prediction
                - forecast_values: Forecast values leading to target
                - explanation: Natural language explanation
        """
        # Import the target prediction module
        from src.predictive_analytics.target_prediction import (
            predict_target_achievement_date
        )
        
        # Get historical data
        collection = self._get_data(from_date, to_date)
        
        if not collection.overall_adoption_rates:
            return {
                "explanation": "No data available for target prediction."
            }
        
        # Sort data by date
        sorted_data = sorted(collection.overall_adoption_rates, key=lambda x: x.date)
        
        # Predict target achievement
        prediction = predict_target_achievement_date(
            sorted_data,
            target_value,
            metric_type=metric_type,
            max_horizon=max_horizon
        )
        
        return prediction
    
    def create_what_if_scenario(self, scenario_params, from_date=None, to_date=None,
                               metric_type="monthly", forecast_periods=12):
        """
        Create what-if scenarios based on parameter changes.
        
        Args:
            scenario_params (dict): Dictionary of parameters for the scenario:
                - growth_rate (float): Assumed growth rate of adoption
                - user_increase (float): Assumed increase in user count
                - seasonality_factor (float): Factor to adjust for seasonality
                - event_impacts (list): List of events that might impact adoption
            from_date (datetime, optional): Start date for historical data
            to_date (datetime, optional): End date for historical data
            metric_type (str): Type of metric to forecast
            forecast_periods (int): Number of periods to forecast
            
        Returns:
            dict: Dictionary with scenario results:
                - baseline_forecast: Forecast without scenario adjustments
                - scenario_forecast: Forecast with scenario adjustments
                - comparison: Comparison between the two forecasts
                - explanation: Natural language explanation of the scenario
        """
        # Import the scenario modeling module
        from src.predictive_analytics.scenario_modeling import (
            create_scenario_forecast
        )
        
        # Get historical data
        collection = self._get_data(from_date, to_date)
        
        if not collection.overall_adoption_rates:
            return {
                "explanation": "No data available for what-if analysis."
            }
        
        # Create baseline forecast
        baseline = create_scenario_forecast(
            collection.overall_adoption_rates,
            scenario_name="baseline",
            metric_type=metric_type,
            forecast_periods=forecast_periods
        )
        
        # Create scenario forecast with custom parameters
        scenario = create_scenario_forecast(
            collection.overall_adoption_rates,
            scenario_name="custom",
            custom_factors=scenario_params,
            metric_type=metric_type,
            forecast_periods=forecast_periods
        )
        
        # Compare the two
        comparison = {
            "baseline_forecast": baseline,
            "scenario_forecast": scenario,
            "explanation": scenario.get("explanation", "")
        }
        
        return comparison
    
    def compare_scenarios(self, from_date=None, to_date=None, scenarios=None,
                         metric_type="monthly", forecast_periods=12):
        """
        Compare multiple adoption rate forecast scenarios.
        
        Args:
            from_date (datetime, optional): Start date for historical data
            to_date (datetime, optional): End date for historical data
            scenarios (list): List of scenario names to compare ("baseline", "optimistic", "pessimistic", etc.)
            metric_type (str): Type of metric to forecast ("daily", "weekly", "monthly", "yearly")
            forecast_periods (int): Number of periods to forecast
            
        Returns:
            dict: Dictionary with comparison results:
                - scenarios: Dictionary of scenario forecasts
                - comparison_metrics: Metrics comparing the scenarios
                - explanation: Natural language explanation of the comparison
        """
        # Import the scenario comparison module
        from src.predictive_analytics.scenario_modeling import (
            compare_scenarios
        )
        
        # Default scenarios if not provided
        if not scenarios:
            scenarios = ["baseline", "optimistic", "pessimistic"]
        
        # Get historical data
        collection = self._get_data(from_date, to_date)
        
        if not collection.overall_adoption_rates:
            return {
                "explanation": "No data available for scenario comparison."
            }
        
        # Compare scenarios
        comparison = compare_scenarios(
            collection.overall_adoption_rates,
            scenarios=scenarios,
            metric_type=metric_type,
            forecast_periods=forecast_periods
        )
        
        return comparison
    
    def create_what_if_scenario(self, from_date=None, to_date=None, factors=None,
                              metric_type="monthly", forecast_periods=12):
        """
        Create a what-if scenario based on impact factors.
        
        Args:
            from_date (datetime, optional): Start date for historical data
            to_date (datetime, optional): End date for historical data
            factors (dict): Dictionary of factors and their impacts:
                - factor_name: { "description": str, "impact": float }
            metric_type (str): Type of metric to analyze
            forecast_periods (int): Number of periods to forecast
            
        Returns:
            dict: Dictionary with what-if analysis results:
                - baseline_forecast: Forecast without factor impacts
                - modified_forecast: Forecast with factor impacts
                - factor_impacts: Individual impact of each factor
                - explanation: Natural language explanation of the analysis
        """
        # Import the impact factor analysis module
        from src.predictive_analytics.scenario_modeling import (
            analyze_impact_factors
        )
        
        # Default factors if not provided
        if not factors:
            factors = {
                "default_factor": {
                    "description": "Default impact factor",
                    "impact": 1.0
                }
            }
        
        # Get historical data
        collection = self._get_data(from_date, to_date)
        
        if not collection.overall_adoption_rates:
            return {
                "explanation": "No data available for what-if analysis."
            }
        
        # Analyze impact factors
        analysis = analyze_impact_factors(
            collection.overall_adoption_rates,
            metric_type=metric_type,
            factor_impacts=factors
        )
        
        return analysis
    
    def _generate_forecast_explanation(self, historical_data, forecast_result, metric_type, method):
        """Generate natural language explanation for a forecast."""
        if not forecast_result.get("forecast_values"):
            return "No forecast data available."
        
        forecast_values = forecast_result.get("forecast_values", [])
        if not forecast_values:
            return "No forecast values generated."
        
        # Get the rate field based on metric type
        if metric_type == "daily":
            rate_field = "daily_adoption_rate"
        elif metric_type == "weekly":
            rate_field = "weekly_adoption_rate"
        elif metric_type == "yearly":
            rate_field = "yearly_adoption_rate"
        else:  # Default to monthly
            rate_field = "monthly_adoption_rate"
            
        # Get the most recent historical value
        historical_values = sorted(historical_data, key=lambda x: x.date)
        if not historical_values:
            return "No historical data available for comparison."
        
        latest_value = getattr(historical_values[-1], rate_field)
        
        # Get the last forecast value
        forecast_end_value = forecast_values[-1]
        
        # Calculate percent change
        percent_change = ((forecast_end_value - latest_value) / latest_value) * 100 if latest_value > 0 else 0
        
        # Determine direction
        direction = "increase" if percent_change > 0 else "decrease" if percent_change < 0 else "remain stable"
        
        # Generate explanation
        explanation = [
            f"Based on {len(historical_values)} historical data points and using the {method} method, " +
            f"the {metric_type} adoption rate is forecast to {direction} by " +
            f"{abs(percent_change):.1f}% over the next {len(forecast_values)} periods."
        ]
        
        # Add information about the trend
        trends = forecast_result.get("trend_info", {})
        if trends:
            explanation.append(
                f"The forecast indicates a {trends.get('direction', 'stable')} " +
                f"trend with {trends.get('seasonality', 'no')} seasonality detected."
            )
        
        # Add confidence information
        confidence = forecast_result.get("confidence_intervals", {})
        if confidence and "lower" in confidence and "upper" in confidence:
            lower = confidence["lower"][-1]
            upper = confidence["upper"][-1]
            range_width = upper - lower
            explanation.append(
                f"With 95% confidence, the final forecast value will be between " +
                f"{lower:.1f}% and {upper:.1f}% (a range of {range_width:.1f}%)."
            )
        
        return "\n".join(explanation) 