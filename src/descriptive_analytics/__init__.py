"""
Descriptive Analytics Package

This package provides a comprehensive suite of tools for analyzing
adoption rate data, including current state analysis, summary statistics,
trend verbalization, comparisons, and metric explanations.
"""

# Import the main DescriptiveAnalytics class
from src.descriptive_analytics.descriptive_analytics import DescriptiveAnalytics

# Import functions from current_state
from src.descriptive_analytics.current_state import describe_current_state

# Import functions from descriptive_statistics
from src.descriptive_analytics.descriptive_statistics import (
    generate_summary_statistics,
    generate_period_statistics,
    identify_extrema,
    calculate_statistical_trends
)

# Import functions from trend_verbalization
from src.descriptive_analytics.trend_verbalization import (
    verbalize_trend,
    verbalize_period_comparison,
    verbalize_anomalies,
    generate_future_outlook
)

# Import functions from comparisons
from src.descriptive_analytics.comparisons import (
    compare_month_over_month,
    compare_quarter_over_quarter,
    compare_year_over_year,
    compare_periods,
    compare_to_target,
    compare_performance_to_benchmark
)

# Import functions from metric_explanation
from src.descriptive_analytics.metric_explanation import (
    explain_adoption_metric,
    generate_metric_context
)

# Define what should be available to users when they import this package
__all__ = [
    # Main class
    'DescriptiveAnalytics',
    
    # Current state functions
    'describe_current_state',
    
    # Descriptive statistics functions
    'generate_summary_statistics',
    'generate_period_statistics',
    'identify_extrema',
    'calculate_statistical_trends',
    
    # Trend verbalization functions
    'verbalize_trend',
    'verbalize_period_comparison',
    'verbalize_anomalies',
    'generate_future_outlook',
    
    # Comparison functions
    'compare_month_over_month',
    'compare_quarter_over_quarter',
    'compare_year_over_year',
    'compare_periods',
    'compare_to_target',
    'compare_performance_to_benchmark',
    
    # Metric explanation functions
    'explain_adoption_metric',
    'generate_metric_context'
] 