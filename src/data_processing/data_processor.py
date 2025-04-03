"""
Data Processor module for aggregating and transforming metric data.

This module provides functions for processing metric data, including aggregating
data for different time periods, formatting dates, and calculating statistics.
"""

import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from calendar import monthrange

from src.data_models.metrics import (
    DailyActiveUsers,
    MonthlyActiveUsers, 
    OverallAdoptionRate,
    MetricCollection
)

# Set up logging
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processes metric data for analysis and visualization.
    
    This class provides methods for aggregating metrics across different time
    periods, calculating statistics, and formatting data for visualization.
    """
    
    def __init__(self):
        """Initialize the DataProcessor."""
        pass
    
    def aggregate_daily_metrics_to_weekly(
        self, 
        daily_metrics: List[DailyActiveUsers]
    ) -> Dict[str, int]:
        """
        Aggregate daily metrics to weekly buckets.
        
        Args:
            daily_metrics: List of daily active user metrics
            
        Returns:
            Dictionary mapping week strings (YYYY-WW) to aggregated user counts
        """
        if not daily_metrics:
            return {}
        
        # Sort metrics by date
        sorted_metrics = sorted(daily_metrics, key=lambda x: x.date)
        
        # Group by calendar week
        weekly_aggregates = {}
        
        for metric in sorted_metrics:
            # Get the week number and year
            year, week, _ = metric.date.isocalendar()
            week_key = f"{year}-{week:02d}"
            
            # Add to the aggregate
            if week_key not in weekly_aggregates:
                weekly_aggregates[week_key] = 0
            
            weekly_aggregates[week_key] += metric.active_users
        
        return weekly_aggregates
    
    def aggregate_daily_metrics_to_monthly(
        self, 
        daily_metrics: List[DailyActiveUsers]
    ) -> Dict[str, int]:
        """
        Aggregate daily metrics to monthly buckets.
        
        Args:
            daily_metrics: List of daily active user metrics
            
        Returns:
            Dictionary mapping month strings (YY-MM) to aggregated user counts
        """
        if not daily_metrics:
            return {}
        
        # Sort metrics by date
        sorted_metrics = sorted(daily_metrics, key=lambda x: x.date)
        
        # Group by calendar month
        monthly_aggregates = {}
        
        for metric in sorted_metrics:
            # Get the month as YY-MM
            month_key = f"{str(metric.date.year)[-2:]}-{metric.date.month:02d}"
            
            # Add to the aggregate
            if month_key not in monthly_aggregates:
                monthly_aggregates[month_key] = 0
            
            monthly_aggregates[month_key] += metric.active_users
        
        return monthly_aggregates
    
    def format_monthly_active_users_for_chart(
        self, 
        mau_metrics: List[MonthlyActiveUsers]
    ) -> Tuple[List[str], List[int]]:
        """
        Format monthly active users data for chart display.
        
        Args:
            mau_metrics: List of monthly active user metrics
            
        Returns:
            Tuple of (labels, values) where labels are month strings (YY-MM)
            and values are user counts
        """
        if not mau_metrics:
            return [], []
        
        # Sort metrics by year and month
        sorted_metrics = sorted(mau_metrics, key=lambda x: (x.year, x.month))
        
        # Extract labels and values
        labels = [metric.year_month for metric in sorted_metrics]
        values = [metric.active_users for metric in sorted_metrics]
        
        return labels, values
    
    def format_overall_adoption_rate_for_chart(
        self, 
        oar_metrics: List[OverallAdoptionRate],
        rate_type: str = 'monthly'  # Options: 'daily', 'weekly', 'monthly', 'yearly'
    ) -> Tuple[List[str], List[float]]:
        """
        Format overall adoption rate data for chart display.
        
        Args:
            oar_metrics: List of overall adoption rate metrics
            rate_type: The type of adoption rate to return ('daily', 'weekly', 'monthly', 'yearly')
            
        Returns:
            Tuple of (labels, values) where labels are date strings and values are adoption rates
        """
        if not oar_metrics:
            return [], []
        
        # Sort metrics by date
        sorted_metrics = sorted(oar_metrics, key=lambda x: x.date)
        
        # Extract labels and values based on the rate type
        labels = []
        values = []
        
        for metric in sorted_metrics:
            # Format the date label based on the rate type
            if rate_type == 'daily' or rate_type == 'weekly':
                # For daily and weekly, use the actual date
                label = metric.date.strftime('%Y-%m-%d')
            elif rate_type == 'monthly':
                # For monthly, use the month in YY-MM format
                label = f"{str(metric.date.year)[-2:]}-{metric.date.month:02d}"
            elif rate_type == 'yearly':
                # For yearly, just use the year
                label = str(metric.date.year)
            else:
                raise ValueError(f"Invalid rate type: {rate_type}")
            
            # Get the rate value based on the rate type
            if rate_type == 'daily':
                rate = metric.daily_adoption_rate
            elif rate_type == 'weekly':
                rate = metric.weekly_adoption_rate
            elif rate_type == 'monthly':
                rate = metric.monthly_adoption_rate
            elif rate_type == 'yearly':
                rate = metric.yearly_adoption_rate
            
            # Only add if we haven't seen this label before (for aggregation)
            if label not in labels:
                labels.append(label)
                values.append(rate)
            else:
                # Update the existing value with the latest rate
                index = labels.index(label)
                values[index] = rate
        
        return labels, values
    
    def calculate_adoption_rate_statistics(
        self, 
        oar_metrics: List[OverallAdoptionRate],
        rate_type: str = 'monthly'
    ) -> Dict[str, float]:
        """
        Calculate statistics for adoption rate metrics.
        
        Args:
            oar_metrics: List of overall adoption rate metrics
            rate_type: The type of adoption rate to analyze ('daily', 'weekly', 'monthly', 'yearly')
            
        Returns:
            Dictionary with statistics: average, median, min, max, latest, trend
        """
        if not oar_metrics:
            return {
                'average': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'latest': 0.0,
                'trend': 0.0
            }
        
        # Sort metrics by date
        sorted_metrics = sorted(oar_metrics, key=lambda x: x.date)
        
        # Extract rates based on the rate type
        rates = []
        for metric in sorted_metrics:
            if rate_type == 'daily':
                rates.append(metric.daily_adoption_rate)
            elif rate_type == 'weekly':
                rates.append(metric.weekly_adoption_rate)
            elif rate_type == 'monthly':
                rates.append(metric.monthly_adoption_rate)
            elif rate_type == 'yearly':
                rates.append(metric.yearly_adoption_rate)
            else:
                raise ValueError(f"Invalid rate type: {rate_type}")
        
        # Calculate statistics
        avg_rate = np.mean(rates) if rates else 0.0
        median_rate = np.median(rates) if rates else 0.0
        min_rate = min(rates) if rates else 0.0
        max_rate = max(rates) if rates else 0.0
        latest_rate = rates[-1] if rates else 0.0
        
        # Calculate trend (positive or negative) over the last 3 data points
        if len(rates) >= 3:
            trend = rates[-1] - rates[-3]
        else:
            trend = 0.0
        
        return {
            'average': float(avg_rate),
            'median': float(median_rate),
            'min': float(min_rate),
            'max': float(max_rate),
            'latest': float(latest_rate),
            'trend': float(trend)
        }
    
    def format_metric_collection_for_chart(
        self, 
        collection: MetricCollection,
        metric_type: str,
        rate_type: str = 'monthly'
    ) -> Dict[str, Any]:
        """
        Format a MetricCollection for chart display.
        
        Args:
            collection: The MetricCollection to format
            metric_type: The type of metric to format ('dau', 'mau', 'adoption_rate')
            rate_type: For adoption rate, the type of rate ('daily', 'weekly', 'monthly', 'yearly')
            
        Returns:
            Dictionary with data for chart display
        """
        result = {
            'tenant_id': collection.tenant_id,
            'metric_type': metric_type,
            'labels': [],
            'values': []
        }
        
        if metric_type == 'dau':
            # Format DAU data
            if collection.daily_active_users:
                sorted_dau = sorted(collection.daily_active_users, key=lambda x: x.date)
                result['labels'] = [dau.date.strftime('%Y-%m-%d') for dau in sorted_dau]
                result['values'] = [dau.active_users for dau in sorted_dau]
        
        elif metric_type == 'mau':
            # Format MAU data
            if collection.monthly_active_users:
                result['labels'], result['values'] = self.format_monthly_active_users_for_chart(
                    collection.monthly_active_users
                )
        
        elif metric_type == 'adoption_rate':
            # Format adoption rate data
            if collection.overall_adoption_rates:
                result['rate_type'] = rate_type
                result['labels'], result['values'] = self.format_overall_adoption_rate_for_chart(
                    collection.overall_adoption_rates, rate_type
                )
                
                # Add statistics
                result['statistics'] = self.calculate_adoption_rate_statistics(
                    collection.overall_adoption_rates, rate_type
                )
        
        else:
            raise ValueError(f"Invalid metric type: {metric_type}")
        
        return result
    
    def get_formatted_trend_description(
        self, 
        statistics: Dict[str, float]
    ) -> str:
        """
        Get a formatted description of the trend in adoption rate.
        
        Args:
            statistics: Dictionary with statistics from calculate_adoption_rate_statistics
            
        Returns:
            A string describing the trend
        """
        if 'trend' not in statistics:
            return "No trend data available"
        
        trend = statistics['trend']
        latest = statistics.get('latest', 0.0)
        
        # Format the trend as a percentage change
        if latest > 0:
            percentage_change = (trend / latest) * 100
        else:
            percentage_change = 0.0
        
        if trend > 0.05:
            return f"Strong positive trend (+{percentage_change:.1f}%)"
        elif trend > 0.01:
            return f"Slight positive trend (+{percentage_change:.1f}%)"
        elif trend < -0.05:
            return f"Strong negative trend ({percentage_change:.1f}%)"
        elif trend < -0.01:
            return f"Slight negative trend ({percentage_change:.1f}%)"
        else:
            return "Stable trend" 