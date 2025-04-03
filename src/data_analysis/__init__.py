"""
Data Analysis package for Overall Adoption Rate Chatbot.

This package provides advanced analytical capabilities for analyzing adoption rate data:
- Trend Analysis: Identify trends, peaks, valleys in adoption rate data
- Period-over-Period Analysis: Calculate MoM, QoQ, YoY changes
- Anomaly Detection: Identify outliers and anomalies in the data
- Correlation Analysis: Examine relationships between metrics
"""

# Import key functions from trend_analyzer
from src.data_analysis.trend_analyzer import (
    detect_peaks_and_valleys,
    calculate_trend_line,
    calculate_moving_average,
    generate_trend_description,
    identify_significant_changes
)

# Import key functions from period_analyzer
from src.data_analysis.period_analyzer import (
    calculate_mom_change,
    calculate_qoq_change,
    calculate_yoy_change,
    generate_period_comparison_summary,
    generate_period_comparison_text,
    compare_time_periods
)

# Import key functions from anomaly_detector
from src.data_analysis.anomaly_detector import (
    detect_anomalies_zscore,
    detect_anomalies_modified_zscore,
    detect_anomalies_iqr,
    detect_anomalies_moving_average,
    detect_anomalies_ensemble,
    generate_anomaly_explanation
)

# Import key functions from correlation_analyzer
from src.data_analysis.correlation_analyzer import (
    calculate_correlation_matrix,
    calculate_metric_correlation,
    perform_lead_lag_analysis,
    analyze_metric_correlations,
    generate_correlation_summary
)

__all__ = [
    # Trend Analysis
    'detect_peaks_and_valleys',
    'calculate_trend_line',
    'calculate_moving_average',
    'generate_trend_description',
    'identify_significant_changes',
    
    # Period Analysis
    'calculate_mom_change',
    'calculate_qoq_change',
    'calculate_yoy_change',
    'generate_period_comparison_summary',
    'generate_period_comparison_text',
    'compare_time_periods',
    
    # Anomaly Detection
    'detect_anomalies_zscore',
    'detect_anomalies_modified_zscore',
    'detect_anomalies_iqr',
    'detect_anomalies_moving_average',
    'detect_anomalies_ensemble',
    'generate_anomaly_explanation',
    
    # Correlation Analysis
    'calculate_correlation_matrix',
    'calculate_metric_correlation',
    'perform_lead_lag_analysis',
    'analyze_metric_correlations',
    'generate_correlation_summary'
] 