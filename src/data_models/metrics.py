"""
Data models for representing metric data from the database.

This module defines the core data structures used to represent the three main metrics:
1. Daily Active Users (DAU)
2. Monthly Active Users (MAU)
3. Overall Adoption Rate (combines DAU, WAU, MAU, YAU with their respective adoption rates)

These models provide type-safe access to the data and include validation methods to
ensure data consistency.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, List, Dict, Union
import numpy as np
import pandas as pd


@dataclass
class DailyActiveUsers:
    """
    Represents Daily Active Users data for a specific date.
    
    Attributes:
        date (date): The date for this DAU measurement
        active_users (int): Number of active users on this date
        tenant_id (int): The tenant ID this data belongs to
    """
    date: date
    active_users: int
    tenant_id: int
    
    def to_dict(self) -> Dict[str, any]:
        """
        Convert this instance to a dictionary representation.
        
        Returns:
            Dict[str, any]: Dictionary representation of this instance
        """
        return {
            "date": self.date.isoformat(),
            "active_users": self.active_users,
            "tenant_id": self.tenant_id
        }
    
    @classmethod
    def from_db_row(cls, row: Dict[str, any], tenant_id: int) -> 'DailyActiveUsers':
        """
        Create a DailyActiveUsers instance from a database row.
        
        Args:
            row: Dictionary containing database row with 'Date' and 'TotalActiveUsers' keys
            tenant_id: The tenant ID this data belongs to
            
        Returns:
            A new DailyActiveUsers instance
        """
        return cls(
            date=row['Date'].date() if isinstance(row['Date'], datetime) else row['Date'],
            active_users=int(row['TotalActiveUsers']),
            tenant_id=tenant_id
        )
    
    def validate(self) -> bool:
        """
        Validate the data in this model.
        
        Returns:
            bool: True if the data is valid, False otherwise
        """
        return (
            isinstance(self.date, date) and
            isinstance(self.active_users, int) and 
            self.active_users >= 0 and
            isinstance(self.tenant_id, int) and
            self.tenant_id > 0
        )


@dataclass
class MonthlyActiveUsers:
    """
    Represents Monthly Active Users data for a specific year-month.
    
    Attributes:
        year (int): The year for this MAU measurement
        month (int): The month (1-12) for this MAU measurement
        active_users (int): Number of active users in this month
        tenant_id (int): The tenant ID this data belongs to
    """
    year: int
    month: int
    active_users: int
    tenant_id: int
    
    def to_dict(self) -> Dict[str, any]:
        """
        Convert this instance to a dictionary representation.
        
        Returns:
            Dict[str, any]: Dictionary representation of this instance
        """
        return {
            "year": self.year,
            "month": self.month,
            "year_month": self.year_month,
            "active_users": self.active_users,
            "tenant_id": self.tenant_id
        }
    
    @property
    def year_month(self) -> str:
        """
        Get the year-month in 'YY-MM' format as displayed in the chart.
        
        Returns:
            str: Year-month in 'YY-MM' format
        """
        return f"{str(self.year)[-2:]}-{self.month:02d}"
    
    @classmethod
    def from_db_row(cls, row: Dict[str, any], tenant_id: int) -> 'MonthlyActiveUsers':
        """
        Create a MonthlyActiveUsers instance from a database row.
        
        Args:
            row: Dictionary containing database row with 'Year_MonthNo' and 'TotalActiveUsers' keys
            tenant_id: The tenant ID this data belongs to
            
        Returns:
            A new MonthlyActiveUsers instance
        """
        # Year_MonthNo format is expected to be 'yyyy-MM'
        year_month = str(row['Year_MonthNo'])
        year, month = year_month.split('-')
        
        return cls(
            year=int(year),
            month=int(month),
            active_users=int(row['TotalActiveUsers']),
            tenant_id=tenant_id
        )
    
    def validate(self) -> bool:
        """
        Validate the data in this model.
        
        Returns:
            bool: True if the data is valid, False otherwise
        """
        return (
            isinstance(self.year, int) and self.year > 0 and
            isinstance(self.month, int) and 1 <= self.month <= 12 and
            isinstance(self.active_users, int) and self.active_users >= 0 and
            isinstance(self.tenant_id, int) and self.tenant_id > 0
        )


@dataclass
class OverallAdoptionRate:
    """
    Represents Overall Adoption Rate data for a specific date.
    
    This includes adoption rates at different time intervals (daily, weekly, monthly, yearly)
    along with the active user counts for each interval.
    
    Attributes:
        date (date): The date for this measurement
        daily_active_users (int): Number of active users on this date (DAU)
        daily_adoption_rate (float): Adoption rate for the day (0-100%)
        weekly_active_users (int): Number of active users in the week (WAU)
        weekly_adoption_rate (float): Adoption rate for the week (0-100%)
        monthly_active_users (int): Number of active users in the month (MAU)
        monthly_adoption_rate (float): Adoption rate for the month (0-100%)
        yearly_active_users (int): Number of active users in the year (YAU)
        yearly_adoption_rate (float): Adoption rate for the year (0-100%)
        tenant_id (int): The tenant ID this data belongs to
    """
    date: date
    daily_active_users: int
    daily_adoption_rate: float
    weekly_active_users: int
    weekly_adoption_rate: float
    monthly_active_users: int
    monthly_adoption_rate: float
    yearly_active_users: int
    yearly_adoption_rate: float
    tenant_id: int
    
    def to_dict(self) -> Dict[str, any]:
        """
        Convert this instance to a dictionary representation.
        
        Returns:
            Dict[str, any]: Dictionary representation of this instance
        """
        return {
            "date": self.date.isoformat(),
            "daily_active_users": self.daily_active_users,
            "daily_adoption_rate": self.daily_adoption_rate,
            "weekly_active_users": self.weekly_active_users,
            "weekly_adoption_rate": self.weekly_adoption_rate,
            "monthly_active_users": self.monthly_active_users,
            "monthly_adoption_rate": self.monthly_adoption_rate,
            "yearly_active_users": self.yearly_active_users,
            "yearly_adoption_rate": self.yearly_adoption_rate,
            "tenant_id": self.tenant_id
        }
    
    @classmethod
    def from_db_row(cls, row: Dict[str, any], tenant_id: int) -> 'OverallAdoptionRate':
        """
        Create an OverallAdoptionRate instance from a database row.
        
        Args:
            row: Dictionary containing database row with adoption rate data
            tenant_id: The tenant ID this data belongs to
            
        Returns:
            A new OverallAdoptionRate instance
        """
        # Helper function to safely convert to int, handling NaN values
        def safe_int(value):
            if pd.isna(value):
                return 0
            return int(value)
        
        # Helper function to safely convert to float, handling NaN values
        def safe_float(value):
            if pd.isna(value):
                return 0.0
            return float(value)
        
        return cls(
            date=row['Date'].date() if isinstance(row['Date'], datetime) else row['Date'],
            daily_active_users=safe_int(row['DAU']),
            daily_adoption_rate=safe_float(row['DOverallAdoptionRate']),
            weekly_active_users=safe_int(row['WAU']),
            weekly_adoption_rate=safe_float(row['WOverallAdoptionRate']),
            monthly_active_users=safe_int(row['MAU']),
            monthly_adoption_rate=safe_float(row['MOverallAdoptionRate']),
            yearly_active_users=safe_int(row['YAU']),
            yearly_adoption_rate=safe_float(row['YOverallAdoptionRate']),
            tenant_id=tenant_id
        )
    
    def validate(self) -> bool:
        """
        Validate the data in this model.
        
        Returns:
            bool: True if the data is valid, False otherwise
        """
        return (
            isinstance(self.date, date) and
            isinstance(self.daily_active_users, int) and self.daily_active_users >= 0 and
            isinstance(self.daily_adoption_rate, float) and 0.0 <= self.daily_adoption_rate <= 100.0 and
            isinstance(self.weekly_active_users, int) and self.weekly_active_users >= 0 and
            isinstance(self.weekly_adoption_rate, float) and 0.0 <= self.weekly_adoption_rate <= 100.0 and
            isinstance(self.monthly_active_users, int) and self.monthly_active_users >= 0 and
            isinstance(self.monthly_adoption_rate, float) and 0.0 <= self.monthly_adoption_rate <= 100.0 and
            isinstance(self.yearly_active_users, int) and self.yearly_active_users >= 0 and
            isinstance(self.yearly_adoption_rate, float) and 0.0 <= self.yearly_adoption_rate <= 100.0 and
            isinstance(self.tenant_id, int) and self.tenant_id > 0
        )


@dataclass
class MetricCollection:
    """
    A collection of metrics for analysis, containing multiple instances of 
    the same metric type across different dates.
    
    This serves as a container for metric data that can be used for analysis and visualization.
    
    Attributes:
        daily_active_users (List[DailyActiveUsers]): Collection of DAU metrics
        monthly_active_users (List[MonthlyActiveUsers]): Collection of MAU metrics
        overall_adoption_rates (List[OverallAdoptionRate]): Collection of overall adoption rate metrics
        tenant_id (int): The tenant ID this data belongs to
    """
    daily_active_users: List[DailyActiveUsers] = None
    monthly_active_users: List[MonthlyActiveUsers] = None
    overall_adoption_rates: List[OverallAdoptionRate] = None
    tenant_id: int = None
    
    def __post_init__(self):
        """Initialize empty lists for None attributes"""
        if self.daily_active_users is None:
            self.daily_active_users = []
        if self.monthly_active_users is None:
            self.monthly_active_users = []
        if self.overall_adoption_rates is None:
            self.overall_adoption_rates = []
    
    def get_sorted_daily_active_users(self) -> List[DailyActiveUsers]:
        """
        Get daily active users sorted by date (ascending).
        
        Returns:
            List[DailyActiveUsers]: Sorted list of daily active users
        """
        return sorted(self.daily_active_users, key=lambda x: x.date)
    
    def get_sorted_monthly_active_users(self) -> List[MonthlyActiveUsers]:
        """
        Get monthly active users sorted by year and month (ascending).
        
        Returns:
            List[MonthlyActiveUsers]: Sorted list of monthly active users
        """
        return sorted(self.monthly_active_users, key=lambda x: (x.year, x.month))
    
    def get_sorted_overall_adoption_rates(self) -> List[OverallAdoptionRate]:
        """
        Get overall adoption rates sorted by date (ascending).
        
        Returns:
            List[OverallAdoptionRate]: Sorted list of overall adoption rates
        """
        return sorted(self.overall_adoption_rates, key=lambda x: x.date) 