"""
Descriptive Analytics Module

This module provides a comprehensive suite for analyzing adoption rate data,
integrating various components to provide insights into current rates,
historical trends, statistical patterns, and business context.
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any, Union

from src.database.connection import db_connection
from src.database.data_access import DataAccessLayer

from src.descriptive_analytics.current_state import (
    describe_current_state
)
from src.descriptive_analytics.descriptive_statistics import (
    generate_summary_statistics,
    generate_period_statistics,
    identify_extrema,
    calculate_statistical_trends
)
from src.descriptive_analytics.metric_explanation import (
    explain_adoption_metric,
    generate_metric_context
)
from src.descriptive_analytics.trend_verbalization import (
    verbalize_trend,
    verbalize_period_comparison,
    verbalize_anomalies,
    generate_future_outlook
)
from src.descriptive_analytics.comparisons import (
    compare_month_over_month,
    compare_quarter_over_quarter,
    compare_year_over_year,
    compare_periods,
    compare_to_target,
    compare_performance_to_benchmark
)

# Configure logging
logger = logging.getLogger(__name__)

class DescriptiveAnalytics:
    """
    Main class for providing descriptive analytics about adoption rate data.
    This class integrates all the descriptive analytics capabilities and provides
    a unified interface for analyzing adoption rate data.
    """
    
    def __init__(self, tenant_id=1388):
        """
        Initialize the class with a tenant ID.
        
        Args:
            tenant_id (int): Tenant ID to use for data retrieval (defaults to 1388)
        """
        self.tenant_id = tenant_id
        self.data_access = DataAccessLayer()
    
    def _get_data(self, from_date=None, to_date=None, metric_type="monthly"):
        """
        Retrieve adoption rate data within a specified date range.
        
        Args:
            from_date (datetime, optional): Start date for data retrieval
            to_date (datetime, optional): End date for data retrieval
            metric_type (str): Type of metric to analyze ("daily", "weekly", "monthly", "yearly")
            
        Returns:
            pandas.DataFrame: Adoption rate data
        """
        # Set default date range if not provided
        if to_date is None:
            to_date = datetime.now()
        if from_date is None:
            from_date = to_date - timedelta(days=730)  # 2 years
        
        # Get data from database using the static method
        data = DataAccessLayer.get_overall_adoption_rate(from_date, to_date, self.tenant_id)
        
        # Convert to DataFrame if not already
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        return data
    
    def describe_current_state(self, as_of_date=None):
        """
        Describe the current state of adoption rates.
        
        Args:
            as_of_date (datetime, optional): The date to consider as "current", defaults to most recent
            
        Returns:
            dict: Dictionary containing current state information
        """
        # Get data
        data = self._get_data()
        
        if data.empty:
            return {
                "description": "No data available for current state analysis."
            }
        
        # Use the most recent date if not specified
        if as_of_date is None and not data.empty:
            as_of_date = data["Date"].max()
        
        # Determine which columns to use based on data availability
        rate_columns = [
            "DOverallAdoptionRate",
            "WOverallAdoptionRate", 
            "MOverallAdoptionRate",
            "YOverallAdoptionRate"
        ]
        
        # Filter to only include columns that exist in the data
        available_rate_columns = [col for col in rate_columns if col in data.columns]
        
        # Import the current state description function
        from src.descriptive_analytics.current_state import describe_current_state
        
        # Generate current state description
        result = describe_current_state(data, as_of_date, available_rate_columns)
        
        return result
    
    def generate_summary_statistics(self, from_date=None, to_date=None, metric_type="monthly"):
        """
        Generate summary statistics for adoption rate data.
        
        Args:
            from_date (datetime, optional): Start date for the data range
            to_date (datetime, optional): End date for the data range
            metric_type (str, optional): Type of metric to analyze (daily, weekly, monthly, yearly)
            
        Returns:
            dict: Dictionary containing summary statistics including basic stats, period stats,
                  extrema, and trends.
        """
        # Get the adoption rate data
        data = self._get_data(from_date, to_date)
        
        if data.empty:
            return {
                "description": "No data available for summary statistics."
            }
        
        # Ensure Date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['Date']):
            data = data.copy()
            data['Date'] = pd.to_datetime(data['Date'])
        
        # Determine which rate column to use based on metric_type
        if metric_type == "daily":
            rate_column = "DOverallAdoptionRate"
        elif metric_type == "weekly":
            rate_column = "WOverallAdoptionRate"
        elif metric_type == "yearly":
            rate_column = "YOverallAdoptionRate"
        else:  # Default to monthly
            rate_column = "MOverallAdoptionRate"
            metric_type = "monthly"
        
        # Check if the selected rate column exists
        if rate_column not in data.columns:
            available_columns = [col for col in ["DOverallAdoptionRate", "WOverallAdoptionRate", 
                                               "MOverallAdoptionRate", "YOverallAdoptionRate"] 
                               if col in data.columns]
            
            if not available_columns:
                return {
                    "description": f"No adoption rate data available. The requested {metric_type} metric does not exist in the data."
                }
            
            # Use the first available column
            rate_column = available_columns[0]
            logger.warning(f"Requested {metric_type} metric not found. Using {rate_column} instead.")
        
        # Import the statistics functions
        from src.descriptive_analytics.statistics import (
            generate_summary_statistics,
            generate_period_statistics,
            identify_extrema,
            calculate_statistical_trends
        )
        
        # Generate basic summary statistics
        basic_stats = generate_summary_statistics(data, rate_column)
        
        # Generate period-specific statistics
        period_type = "month"  # Default to monthly periods for clarity
        if metric_type == "daily":
            period_type = "week"  # Group daily data by week
        elif metric_type == "weekly":
            period_type = "month"  # Group weekly data by month
        elif metric_type == "yearly":
            period_type = "year"  # Group yearly data by year
        
        period_stats = generate_period_statistics(data, rate_column, period_type)
        
        # Identify extrema (peaks and valleys)
        extrema = identify_extrema(data, rate_column)
        
        # Calculate statistical trends
        trends = calculate_statistical_trends(data, rate_column)
        
        # Combine all statistics into a comprehensive result
        result = {
            "basic_stats": basic_stats,
            "period_stats": period_stats,
            "extrema": extrema,
            "trends": trends,
            "metric_type": metric_type,
            "rate_column": rate_column,
            "date_range": (data['Date'].min().strftime('%Y-%m-%d'), data['Date'].max().strftime('%Y-%m-%d')),
            "description": basic_stats.get("description", "") + " " + 
                          period_stats.get("description", "") + " " + 
                          extrema.get("description", "") + " " + 
                          trends.get("description", "")
        }
        
        return result
    
    def verbalize_trend(self, from_date=None, to_date=None, metric_type="monthly"):
        """
        Generate a natural language description of trends in the adoption rate data.
        
        Args:
            from_date (datetime, optional): Start date for the data range
            to_date (datetime, optional): End date for the data range
            metric_type (str, optional): Type of metric to analyze (daily, weekly, monthly, yearly)
            
        Returns:
            dict: Dictionary containing trend information including direction, strength,
                 volatility, notable points, and a descriptive summary.
        """
        # Get the adoption rate data
        data = self._get_data(from_date, to_date)
        
        if data.empty:
            return {
                "description": "No data available for trend analysis."
            }
        
        # Determine which rate column to use based on metric_type
        if metric_type == "daily":
            rate_column = "DOverallAdoptionRate"
        elif metric_type == "weekly":
            rate_column = "WOverallAdoptionRate"
        elif metric_type == "yearly":
            rate_column = "YOverallAdoptionRate"
        else:  # Default to monthly
            rate_column = "MOverallAdoptionRate"
        
        # Check if the selected rate column exists
        if rate_column not in data.columns:
            available_columns = [col for col in ["DOverallAdoptionRate", "WOverallAdoptionRate", 
                                               "MOverallAdoptionRate", "YOverallAdoptionRate"] 
                               if col in data.columns]
            
            if not available_columns:
                return {
                    "description": f"No adoption rate data available. The requested {metric_type} metric does not exist in the data."
                }
            
            # Use the first available column
            rate_column = available_columns[0]
            logger.warning(f"Requested {metric_type} metric not found. Using {rate_column} instead.")
        
        # Import the trend verbalization function
        from src.descriptive_analytics.trend_verbalization import verbalize_trend
        
        # Generate trend verbalization
        result = verbalize_trend(data, rate_column)
        
        return result
    
    def compare_periods(self, period1_start, period1_end, period2_start, period2_end, metric_type="monthly"):
        """
        Compare adoption rates between two time periods.
        
        Args:
            period1_start (datetime): Start date for the first period
            period1_end (datetime): End date for the first period
            period2_start (datetime): Start date for the second period
            period2_end (datetime): End date for the second period
            metric_type (str, optional): Type of metric to analyze (daily, weekly, monthly, yearly)
            
        Returns:
            dict: Dictionary containing comparison information including average rates,
                 change metrics, and a descriptive summary.
        """
        # Get data for both periods
        period1_data = self._get_data(period1_start, period1_end)
        period2_data = self._get_data(period2_start, period2_end)
        
        if period1_data.empty or period2_data.empty:
            return {
                "description": "Insufficient data for period comparison. One or both periods have no data."
            }
        
        # Determine which rate column to use based on metric_type
        if metric_type == "daily":
            rate_column = "DOverallAdoptionRate"
        elif metric_type == "weekly":
            rate_column = "WOverallAdoptionRate"
        elif metric_type == "yearly":
            rate_column = "YOverallAdoptionRate"
        else:  # Default to monthly
            rate_column = "MOverallAdoptionRate"
        
        # Check if the selected rate column exists in both datasets
        common_columns = set(period1_data.columns).intersection(set(period2_data.columns))
        if rate_column not in common_columns:
            available_columns = [col for col in ["DOverallAdoptionRate", "WOverallAdoptionRate", 
                                               "MOverallAdoptionRate", "YOverallAdoptionRate"] 
                               if col in common_columns]
            
            if not available_columns:
                return {
                    "description": f"No common adoption rate data available across both periods."
                }
            
            # Use the first available column
            rate_column = available_columns[0]
            logger.warning(f"Requested {metric_type} metric not found in both periods. Using {rate_column} instead.")
        
        # Import the period comparison function
        from src.descriptive_analytics.comparisons import compare_periods
        
        # Determine the period type based on metric_type
        period_type = "month"  # Default
        if metric_type == "daily":
            period_type = "day"
        elif metric_type == "weekly":
            period_type = "week"
        elif metric_type == "yearly":
            period_type = "year"
        
        # Format the period identifiers
        period1_id = period1_end.strftime("%Y-%m")
        period2_id = period2_end.strftime("%Y-%m")
        
        if period_type == "day":
            period1_id = period1_end.strftime("%Y-%m-%d")
            period2_id = period2_end.strftime("%Y-%m-%d")
        elif period_type == "week":
            # ISO week format
            period1_id = f"{period1_end.year}-W{period1_end.isocalendar()[1]}"
            period2_id = f"{period2_end.year}-W{period2_end.isocalendar()[1]}"
        elif period_type == "year":
            period1_id = str(period1_end.year)
            period2_id = str(period2_end.year)
        elif period_type == "month":
            period1_id = period1_end.strftime("%Y-%m")
            period2_id = period2_end.strftime("%Y-%m")
        
        # Generate period comparison
        result = compare_periods(
            pd.concat([period1_data, period2_data]),
            current_period=period1_id,
            previous_period=period2_id,
            period_type=period_type,
            rate_column=rate_column
        )
        
        return result
    
    def compare_to_target(self, target_value, as_of_date=None, metric_type="monthly"):
        """
        Compare the current adoption rate to a target value.
        
        Args:
            target_value (float): Target adoption rate value to compare against
            as_of_date (datetime, optional): The date to consider as "current"
                                          If None, uses the most recent date in the data
            metric_type (str, optional): Type of metric to analyze (daily, weekly, monthly, yearly)
            
        Returns:
            dict: Dictionary containing comparison information including current rate,
                 target, gap, and a descriptive summary.
        """
        # Get the adoption rate data
        data = self._get_data()
        
        if data.empty:
            return {
                "description": "No data available for target comparison."
            }
        
        # Determine which rate column to use based on metric_type
        if metric_type == "daily":
            rate_column = "DOverallAdoptionRate"
        elif metric_type == "weekly":
            rate_column = "WOverallAdoptionRate"
        elif metric_type == "yearly":
            rate_column = "YOverallAdoptionRate"
        else:  # Default to monthly
            rate_column = "MOverallAdoptionRate"
        
        # Check if the selected rate column exists
        if rate_column not in data.columns:
            available_columns = [col for col in ["DOverallAdoptionRate", "WOverallAdoptionRate", 
                                               "MOverallAdoptionRate", "YOverallAdoptionRate"] 
                               if col in data.columns]
            
            if not available_columns:
                return {
                    "description": f"No adoption rate data available. The requested {metric_type} metric does not exist in the data."
                }
            
            # Use the first available column
            rate_column = available_columns[0]
            logger.warning(f"Requested {metric_type} metric not found. Using {rate_column} instead.")
        
        # If as_of_date is not provided, use the most recent date in the data
        if as_of_date is None:
            as_of_date = data['Date'].max()
        else:
            # Make sure as_of_date is in the data, otherwise use closest date
            if as_of_date not in data['Date'].values:
                closest_date = data.iloc[abs(data['Date'] - as_of_date).argmin()]['Date']
                logger.info(f"Specified date {as_of_date} not found in data. Using closest date: {closest_date}")
                as_of_date = closest_date
        
        # Import the target comparison function
        from src.descriptive_analytics.comparisons import compare_to_target
        
        # Generate target comparison
        result = compare_to_target(data, target_value, as_of_date, rate_column)
        
        return result
    
    def explain_adoption_metric(self, metric_type="monthly"):
        """
        Provide an explanation of what an adoption rate metric means.
        
        Args:
            metric_type (str, optional): Type of metric to explain (daily, weekly, monthly, yearly)
            
        Returns:
            dict: Dictionary containing explanation information including definition,
                 formula, example, typical range, and a descriptive summary.
        """
        # Import the metric explanation function
        from src.descriptive_analytics.metric_explanation import explain_adoption_metric
        
        # Generate metric explanation
        result = explain_adoption_metric(metric_type)
        
        return result
    
    def generate_metric_context(self, value, metric_type="monthly"):
        """
        Generate contextual information about a metric's value.
        
        Args:
            value (float): The adoption rate value to contextualize
            metric_type (str, optional): Type of metric to contextualize (daily, weekly, monthly, yearly)
            
        Returns:
            dict: Dictionary containing contextual information including interpretation,
                 potential implications, and a descriptive summary.
        """
        # Get the adoption rate data for historical context
        data = self._get_data()
        
        if data.empty:
            return {
                "description": "No data available for metric contextualization."
            }
        
        # Determine which rate column to use based on metric_type
        if metric_type == "daily":
            rate_column = "DOverallAdoptionRate"
        elif metric_type == "weekly":
            rate_column = "WOverallAdoptionRate"
        elif metric_type == "yearly":
            rate_column = "YOverallAdoptionRate"
        else:  # Default to monthly
            rate_column = "MOverallAdoptionRate"
        
        # Check if the selected rate column exists
        if rate_column not in data.columns:
            available_columns = [col for col in ["DOverallAdoptionRate", "WOverallAdoptionRate", 
                                               "MOverallAdoptionRate", "YOverallAdoptionRate"] 
                               if col in data.columns]
            
            if not available_columns:
                return {
                    "description": f"No adoption rate data available. Cannot generate context for the {metric_type} metric."
                }
            
            # Use the first available column
            rate_column = available_columns[0]
            logger.warning(f"Requested {metric_type} metric not found. Using {rate_column} instead.")
        
        # Import the metric context function
        from src.descriptive_analytics.metric_explanation import generate_metric_context
        
        # Generate metric context
        result = generate_metric_context(data, value, rate_column)
        
        return result
    
    def detect_anomalies(self, from_date=None, to_date=None, metric_type="monthly", threshold=2.0):
        """
        Detect and describe anomalies in the adoption rate data.
        
        Args:
            from_date (datetime, optional): Start date for the data range
            to_date (datetime, optional): End date for the data range
            metric_type (str, optional): Type of metric to analyze (daily, weekly, monthly, yearly)
            threshold (float, optional): Threshold for anomaly detection (Z-score)
            
        Returns:
            dict: Dictionary containing anomaly information including high and low anomalies,
                 anomaly count, and a descriptive summary.
        """
        # Get the adoption rate data
        data = self._get_data(from_date, to_date)
        
        if data.empty:
            return {
                "description": "No data available for anomaly detection."
            }
        
        # Determine which rate column to use based on metric_type
        if metric_type == "daily":
            rate_column = "DOverallAdoptionRate"
        elif metric_type == "weekly":
            rate_column = "WOverallAdoptionRate"
        elif metric_type == "yearly":
            rate_column = "YOverallAdoptionRate"
        else:  # Default to monthly
            rate_column = "MOverallAdoptionRate"
        
        # Check if the selected rate column exists
        if rate_column not in data.columns:
            available_columns = [col for col in ["DOverallAdoptionRate", "WOverallAdoptionRate", 
                                               "MOverallAdoptionRate", "YOverallAdoptionRate"] 
                               if col in data.columns]
            
            if not available_columns:
                return {
                    "description": f"No adoption rate data available. The requested {metric_type} metric does not exist in the data."
                }
            
            # Use the first available column
            rate_column = available_columns[0]
            logger.warning(f"Requested {metric_type} metric not found. Using {rate_column} instead.")
        
        # Import the anomaly detection function
        from src.descriptive_analytics.trend_verbalization import verbalize_anomalies
        
        # Generate anomaly detection
        result = verbalize_anomalies(data, rate_column, threshold)
        
        return result
    
    def generate_future_outlook(self, from_date=None, to_date=None, metric_type="monthly", forecast_periods=3):
        """
        Generate a future outlook for adoption rates based on historical trends.
        
        Args:
            from_date (datetime, optional): Start date for the data range
            to_date (datetime, optional): End date for the data range
            metric_type (str, optional): Type of metric to analyze (daily, weekly, monthly, yearly)
            forecast_periods (int, optional): Number of periods to forecast into the future
            
        Returns:
            dict: Dictionary containing forecast information including predicted values,
                 confidence intervals, and a descriptive summary.
        """
        # Get the adoption rate data
        data = self._get_data(from_date, to_date)
        
        if data.empty:
            return {
                "description": "No data available for future outlook generation."
            }
        
        # Determine which rate column to use based on metric_type
        if metric_type == "daily":
            rate_column = "DOverallAdoptionRate"
        elif metric_type == "weekly":
            rate_column = "WOverallAdoptionRate"
        elif metric_type == "yearly":
            rate_column = "YOverallAdoptionRate"
        else:  # Default to monthly
            rate_column = "MOverallAdoptionRate"
        
        # Check if the selected rate column exists
        if rate_column not in data.columns:
            available_columns = [col for col in ["DOverallAdoptionRate", "WOverallAdoptionRate", 
                                               "MOverallAdoptionRate", "YOverallAdoptionRate"] 
                               if col in data.columns]
            
            if not available_columns:
                return {
                    "description": f"No adoption rate data available. The requested {metric_type} metric does not exist in the data."
                }
            
            # Use the first available column
            rate_column = available_columns[0]
            logger.warning(f"Requested {metric_type} metric not found. Using {rate_column} instead.")
        
        # Import the future outlook function
        from src.descriptive_analytics.trend_verbalization import generate_future_outlook
        
        # Generate future outlook
        result = generate_future_outlook(data, rate_column, forecast_periods)
        
        return result 