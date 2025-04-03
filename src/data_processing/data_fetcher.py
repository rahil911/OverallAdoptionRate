"""
Data Fetcher module for retrieving and processing metric data from the database.

This module provides a high-level interface for fetching adoption rate metrics
and converting the raw database data into our data models.
"""

import logging
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Tuple, Union, Any, Type
import pandas as pd

from src.database.data_access import DataAccessLayer
from src.data_models.metrics import (
    DailyActiveUsers,
    MonthlyActiveUsers,
    OverallAdoptionRate,
    MetricCollection
)

# Set up logging
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    High-level interface for fetching and processing metric data from the database.
    
    This class serves as a facade over the data access layer, providing more
    use-case focused methods and handling the conversion between raw database
    data and our data models.
    """
    
    def __init__(self, data_access_layer: Type[DataAccessLayer]):
        """
        Initialize the DataFetcher with a data access layer.
        
        Args:
            data_access_layer: The DataAccessLayer class with static methods for database access
        """
        self.dal = data_access_layer
        self._cache = {}  # Simple in-memory cache for queries
    
    def _get_cache_key(self, method_name: str, from_date: date, to_date: date, tenant_id: int) -> str:
        """
        Generate a cache key for a query.
        
        Args:
            method_name: The name of the method being called
            from_date: Start date for the query
            to_date: End date for the query
            tenant_id: Tenant ID for the query
            
        Returns:
            A string cache key
        """
        return f"{method_name}:{from_date.isoformat()}:{to_date.isoformat()}:{tenant_id}"
    
    def clear_cache(self):
        """Clear the in-memory cache"""
        self._cache = {}
    
    def get_daily_active_users(
        self, 
        from_date: date, 
        to_date: date, 
        tenant_id: Optional[int] = None
    ) -> List[DailyActiveUsers]:
        """
        Get daily active users for a date range and tenant.
        
        Args:
            from_date: Start date for the query
            to_date: End date for the query
            tenant_id: Tenant ID for the query, or None for default tenant
            
        Returns:
            A list of DailyActiveUsers objects
        """
        tenant = tenant_id or self.dal.DEFAULT_TENANT_ID
        cache_key = self._get_cache_key("get_daily_active_users", from_date, to_date, tenant)
        
        if cache_key in self._cache:
            logger.info(f"Using cached DAU data for {cache_key}")
            return self._cache[cache_key]
        
        try:
            # Convert dates to datetime for the data access layer
            from_datetime = datetime.combine(from_date, datetime.min.time())
            to_datetime = datetime.combine(to_date, datetime.min.time())
            
            # Get data from database
            df = self.dal.get_dau(from_datetime, to_datetime, tenant_id)
            
            if df.empty:
                logger.warning(f"No DAU data found for date range {from_date} to {to_date} and tenant {tenant_id}")
                return []
            
            # Convert to our data model
            result = [
                DailyActiveUsers.from_db_row(row, tenant) 
                for _, row in df.iterrows()
            ]
            
            # Validate the data
            valid_results = [r for r in result if r.validate()]
            
            if len(valid_results) != len(result):
                logger.warning(f"Some DAU data failed validation ({len(result) - len(valid_results)} rows)")
            
            # Cache the result
            self._cache[cache_key] = valid_results
            
            return valid_results
        
        except Exception as e:
            logger.error(f"Error fetching DAU data: {str(e)}", exc_info=True)
            return []
    
    def get_monthly_active_users(
        self, 
        from_date: date, 
        to_date: date, 
        tenant_id: Optional[int] = None
    ) -> List[MonthlyActiveUsers]:
        """
        Get monthly active users for a date range and tenant.
        
        Args:
            from_date: Start date for the query
            to_date: End date for the query
            tenant_id: Tenant ID for the query, or None for default tenant
            
        Returns:
            A list of MonthlyActiveUsers objects
        """
        tenant = tenant_id or self.dal.DEFAULT_TENANT_ID
        cache_key = self._get_cache_key("get_monthly_active_users", from_date, to_date, tenant)
        
        if cache_key in self._cache:
            logger.info(f"Using cached MAU data for {cache_key}")
            return self._cache[cache_key]
        
        try:
            # Convert dates to datetime for the data access layer
            from_datetime = datetime.combine(from_date, datetime.min.time())
            to_datetime = datetime.combine(to_date, datetime.min.time())
            
            # Get data from database
            df = self.dal.get_mau(from_datetime, to_datetime, tenant_id)
            
            if df.empty:
                logger.warning(f"No MAU data found for date range {from_date} to {to_date} and tenant {tenant_id}")
                return []
            
            # Convert to our data model
            result = [
                MonthlyActiveUsers.from_db_row(row, tenant) 
                for _, row in df.iterrows()
            ]
            
            # Validate the data
            valid_results = [r for r in result if r.validate()]
            
            if len(valid_results) != len(result):
                logger.warning(f"Some MAU data failed validation ({len(result) - len(valid_results)} rows)")
            
            # Cache the result
            self._cache[cache_key] = valid_results
            
            return valid_results
        
        except Exception as e:
            logger.error(f"Error fetching MAU data: {str(e)}", exc_info=True)
            return []
    
    def get_overall_adoption_rate(
        self, 
        from_date: date, 
        to_date: date, 
        tenant_id: Optional[int] = None
    ) -> List[OverallAdoptionRate]:
        """
        Get overall adoption rate data for a date range and tenant.
        
        Args:
            from_date: Start date for the query
            to_date: End date for the query
            tenant_id: Tenant ID for the query, or None for default tenant
            
        Returns:
            A list of OverallAdoptionRate objects
        """
        tenant = tenant_id or self.dal.DEFAULT_TENANT_ID
        cache_key = self._get_cache_key("get_overall_adoption_rate", from_date, to_date, tenant)
        
        if cache_key in self._cache:
            logger.info(f"Using cached Overall Adoption Rate data for {cache_key}")
            return self._cache[cache_key]
        
        try:
            # Convert dates to datetime for the data access layer
            from_datetime = datetime.combine(from_date, datetime.min.time())
            to_datetime = datetime.combine(to_date, datetime.min.time())
            
            # Get data from database
            df = self.dal.get_overall_adoption_rate(
                from_datetime, 
                to_datetime, 
                tenant_id
            )
            
            if df.empty:
                logger.warning(
                    f"No Overall Adoption Rate data found for date range {from_date} to {to_date} and tenant {tenant_id}"
                )
                return []
            
            # Convert to our data model
            result = [
                OverallAdoptionRate.from_db_row(row, tenant) 
                for _, row in df.iterrows()
            ]
            
            # Validate the data
            valid_results = [r for r in result if r.validate()]
            
            if len(valid_results) != len(result):
                logger.warning(f"Some Overall Adoption Rate data failed validation ({len(result) - len(valid_results)} rows)")
            
            # Cache the result
            self._cache[cache_key] = valid_results
            
            return valid_results
        
        except Exception as e:
            logger.error(f"Error fetching Overall Adoption Rate data: {str(e)}", exc_info=True)
            return []
    
    def get_all_metrics(
        self,
        from_date: date,
        to_date: date,
        tenant_id: Optional[int] = None
    ) -> MetricCollection:
        """
        Get all metrics for a date range and tenant.
        
        This fetches DAU, MAU, and Overall Adoption Rate data and returns them
        in a single MetricCollection object.
        
        Args:
            from_date: Start date for the query
            to_date: End date for the query
            tenant_id: Tenant ID for the query, or None for default tenant
            
        Returns:
            A MetricCollection object containing all metrics
        """
        tenant = tenant_id or self.dal.DEFAULT_TENANT_ID
        
        # Fetch all metrics
        dau_data = self.get_daily_active_users(from_date, to_date, tenant_id)
        mau_data = self.get_monthly_active_users(from_date, to_date, tenant_id)
        oar_data = self.get_overall_adoption_rate(
            from_date, 
            to_date, 
            tenant_id
        )
        
        # Create a MetricCollection
        collection = MetricCollection(
            daily_active_users=dau_data,
            monthly_active_users=mau_data,
            overall_adoption_rates=oar_data,
            tenant_id=tenant
        )
        
        return collection
    
    def get_date_range_for_period(self, period: str, reference_date: Optional[date] = None) -> Tuple[date, date]:
        """
        Get a date range for a specific period.
        
        Args:
            period: The period to get a date range for ('day', 'week', 'month', 'year', 'all')
            reference_date: The reference date to use, or None for today
            
        Returns:
            A tuple of (from_date, to_date)
        """
        if reference_date is None:
            reference_date = date.today()
        
        if period == 'day':
            return reference_date, reference_date
        elif period == 'week':
            start_of_week = reference_date - timedelta(days=reference_date.weekday())
            return start_of_week, reference_date
        elif period == 'month':
            start_of_month = date(reference_date.year, reference_date.month, 1)
            return start_of_month, reference_date
        elif period == 'year':
            start_of_year = date(reference_date.year, 1, 1)
            return start_of_year, reference_date
        elif period == 'all':
            # Use a date far in the past as start date
            return date(2000, 1, 1), reference_date
        else:
            raise ValueError(f"Invalid period: {period}")
    
    def get_metrics_for_period(
        self,
        period: str,
        reference_date: Optional[date] = None,
        tenant_id: Optional[int] = None
    ) -> MetricCollection:
        """
        Get all metrics for a specific period.
        
        Args:
            period: The period to get metrics for ('day', 'week', 'month', 'year', 'all')
            reference_date: The reference date to use, or None for today
            tenant_id: The tenant ID to get metrics for, or None for default tenant
            
        Returns:
            A MetricCollection object containing metrics for the period
        """
        from_date, to_date = self.get_date_range_for_period(period, reference_date)
        return self.get_all_metrics(
            from_date, 
            to_date, 
            tenant_id
        ) 