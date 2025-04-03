"""
Correlation Analyzer module for examining relationships between different metrics.

This module provides functions to analyze correlations between DAU, MAU, and adoption rates,
as well as lead/lag analysis to identify potential causal relationships.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any
from datetime import date, datetime, timedelta
import logging
from scipy import stats

from src.data_models.metrics import OverallAdoptionRate, MonthlyActiveUsers, DailyActiveUsers, MetricCollection

# Set up logging
logger = logging.getLogger(__name__)


def calculate_correlation_matrix(
    data: List[OverallAdoptionRate], 
    min_periods: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Calculate correlation matrix between different metrics.
    
    Args:
        data: List of OverallAdoptionRate objects
        min_periods: Minimum number of valid data points required for calculation
        
    Returns:
        Dictionary of dictionaries containing correlation coefficients between metrics
    """
    if not data or len(data) < min_periods:
        logger.warning(f"Insufficient data for correlation analysis (need at least {min_periods} points)")
        return {}
    
    # Sort data by date
    sorted_data = sorted(data, key=lambda x: x.date)
    
    # Extract metrics into lists
    dates = [item.date for item in sorted_data]
    daily_rates = [item.daily_adoption_rate for item in sorted_data]
    weekly_rates = [item.weekly_adoption_rate for item in sorted_data]
    monthly_rates = [item.monthly_adoption_rate for item in sorted_data]
    yearly_rates = [item.yearly_adoption_rate for item in sorted_data]
    daily_users = [item.daily_active_users for item in sorted_data]
    weekly_users = [item.weekly_active_users for item in sorted_data]
    monthly_users = [item.monthly_active_users for item in sorted_data]
    yearly_users = [item.yearly_active_users for item in sorted_data]
    
    # Create dataframe for correlation calculation
    df = pd.DataFrame({
        'date': dates,
        'daily_rate': daily_rates,
        'weekly_rate': weekly_rates,
        'monthly_rate': monthly_rates,
        'yearly_rate': yearly_rates,
        'daily_users': daily_users,
        'weekly_users': weekly_users,
        'monthly_users': monthly_users,
        'yearly_users': yearly_users
    })
    
    # Set date as index
    df.set_index('date', inplace=True)
    
    # Calculate correlation matrix
    correlation_matrix = df.corr(min_periods=min_periods).to_dict()
    
    logger.info(f"Calculated correlation matrix for {len(df)} data points")
    return correlation_matrix


def calculate_metric_correlation(
    metric1_data: List[Union[OverallAdoptionRate, DailyActiveUsers, MonthlyActiveUsers]],
    metric2_data: List[Union[OverallAdoptionRate, DailyActiveUsers, MonthlyActiveUsers]],
    metric1_attribute: str = 'value',
    metric2_attribute: str = 'value',
    date_alignment: str = 'exact'
) -> Dict[str, Any]:
    """
    Calculate correlation between two specific metrics.
    
    Args:
        metric1_data: List of data objects for first metric
        metric2_data: List of data objects for second metric
        metric1_attribute: Attribute to extract from first metric objects
        metric2_attribute: Attribute to extract from second metric objects
        date_alignment: How to align dates ('exact', 'nearest', 'interpolate')
        
    Returns:
        Dictionary with correlation information:
        - 'pearson_r': Pearson correlation coefficient
        - 'pearson_p': P-value for Pearson correlation
        - 'spearman_r': Spearman rank correlation coefficient
        - 'spearman_p': P-value for Spearman correlation
        - 'kendall_tau': Kendall's tau correlation coefficient
        - 'kendall_p': P-value for Kendall's tau
        - 'data_points': Number of data points used
        - 'strength': Qualitative strength of correlation ('strong', 'moderate', 'weak', 'none')
        - 'direction': Direction of correlation ('positive', 'negative', 'none')
    """
    if not metric1_data or not metric2_data:
        logger.warning("Empty data provided for correlation analysis")
        return {
            'pearson_r': 0,
            'pearson_p': 1,
            'spearman_r': 0,
            'spearman_p': 1,
            'kendall_tau': 0,
            'kendall_p': 1,
            'data_points': 0,
            'strength': 'none',
            'direction': 'none'
        }
    
    # Extract dates and values
    dates1 = []
    values1 = []
    for item in metric1_data:
        dates1.append(item.date)
        values1.append(getattr(item, metric1_attribute, 0))
    
    dates2 = []
    values2 = []
    for item in metric2_data:
        dates2.append(item.date)
        values2.append(getattr(item, metric2_attribute, 0))
    
    # Create dataframes
    df1 = pd.DataFrame({'date': dates1, 'value': values1})
    df2 = pd.DataFrame({'date': dates2, 'value': values2})
    
    # Set date as index
    df1.set_index('date', inplace=True)
    df2.set_index('date', inplace=True)
    
    # Align dates based on specified method
    if date_alignment == 'exact':
        # Only use exact date matches
        combined = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
        values1 = combined['value_x'].values
        values2 = combined['value_y'].values
    elif date_alignment == 'nearest':
        # For each date in metric1, find nearest date in metric2
        aligned_values2 = []
        for date in dates1:
            nearest_idx = min(range(len(dates2)), key=lambda i: abs((dates2[i] - date).days))
            aligned_values2.append(values2[nearest_idx])
        values1 = np.array(values1)
        values2 = np.array(aligned_values2)
    elif date_alignment == 'interpolate':
        # Interpolate values for metric2 at metric1 dates
        df2_reindexed = df2.reindex(df1.index)
        df2_interpolated = df2_reindexed.interpolate(method='time')
        values1 = df1['value'].values
        values2 = df2_interpolated['value'].values
        # Remove NaN values
        valid_indices = ~np.isnan(values2)
        values1 = values1[valid_indices]
        values2 = values2[valid_indices]
    else:
        raise ValueError(f"Invalid date_alignment: {date_alignment}")
    
    # Check if we have enough data points
    if len(values1) < 2:
        logger.warning("Insufficient data points for correlation analysis after alignment")
        return {
            'pearson_r': 0,
            'pearson_p': 1,
            'spearman_r': 0,
            'spearman_p': 1,
            'kendall_tau': 0,
            'kendall_p': 1,
            'data_points': len(values1),
            'strength': 'none',
            'direction': 'none'
        }
    
    # Calculate correlations
    pearson_r, pearson_p = stats.pearsonr(values1, values2)
    spearman_r, spearman_p = stats.spearmanr(values1, values2)
    kendall_tau, kendall_p = stats.kendalltau(values1, values2)
    
    # Determine strength and direction
    abs_r = abs(pearson_r)
    if abs_r > 0.7:
        strength = 'strong'
    elif abs_r > 0.3:
        strength = 'moderate'
    elif abs_r > 0.1:
        strength = 'weak'
    else:
        strength = 'none'
    
    if pearson_r > 0.1:
        direction = 'positive'
    elif pearson_r < -0.1:
        direction = 'negative'
    else:
        direction = 'none'
    
    logger.info(f"Calculated correlation between metrics using {len(values1)} data points")
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p,
        'data_points': len(values1),
        'strength': strength,
        'direction': direction
    }


def perform_lead_lag_analysis(
    metric1_data: List[Union[OverallAdoptionRate, DailyActiveUsers, MonthlyActiveUsers]],
    metric2_data: List[Union[OverallAdoptionRate, DailyActiveUsers, MonthlyActiveUsers]],
    metric1_attribute: str = 'value',
    metric2_attribute: str = 'value',
    max_lag: int = 10  # Maximum number of days to lag
) -> Dict[str, Any]:
    """
    Analyze lead/lag relationship between two metrics to identify potential causal relationships.
    
    Args:
        metric1_data: List of data objects for first metric
        metric2_data: List of data objects for second metric
        metric1_attribute: Attribute to extract from first metric objects
        metric2_attribute: Attribute to extract from second metric objects
        max_lag: Maximum number of days to lag
        
    Returns:
        Dictionary with lead/lag information:
        - 'best_lag': Lag with highest correlation
        - 'best_correlation': Correlation coefficient at best lag
        - 'lag_correlations': Dictionary mapping lags to correlations
        - 'lead_metric': Which metric leads the other (1, 2, or None)
        - 'causal_evidence': Qualitative assessment of causal evidence ('strong', 'moderate', 'weak', 'none')
    """
    if not metric1_data or not metric2_data or max_lag < 1:
        logger.warning("Invalid input for lead/lag analysis")
        return {
            'best_lag': 0,
            'best_correlation': 0,
            'lag_correlations': {},
            'lead_metric': None,
            'causal_evidence': 'none'
        }
    
    try:
        # Extract dates and values
        dates1 = []
        values1 = []
        for item in metric1_data:
            dates1.append(item.date)
            values1.append(getattr(item, metric1_attribute, 0))
        
        dates2 = []
        values2 = []
        for item in metric2_data:
            dates2.append(item.date)
            values2.append(getattr(item, metric2_attribute, 0))
        
        # Create dataframes
        df1 = pd.DataFrame({'date': dates1, 'value': values1})
        df2 = pd.DataFrame({'date': dates2, 'value': values2})
        
        # Set date as index
        df1.set_index('date', inplace=True)
        df2.set_index('date', inplace=True)
        
        # Sort by date
        df1.sort_index(inplace=True)
        df2.sort_index(inplace=True)
        
        # Calculate correlation at different lags
        lag_correlations = {}
        
        # Convert index to list for manual shifting if needed
        df1_dates = df1.index.tolist()
        df1_values = df1['value'].tolist()
        df2_dates = df2.index.tolist()
        df2_values = df2['value'].tolist()
        
        # Check if we can use pandas shift with freq (requires DatetimeIndex)
        use_pandas_shift = isinstance(df1.index, pd.DatetimeIndex) and isinstance(df2.index, pd.DatetimeIndex)
        
        # Positive lags (metric1 leads metric2)
        for lag in range(-max_lag, max_lag + 1):
            if use_pandas_shift:
                # Use pandas shift which requires DatetimeIndex
                if lag > 0:
                    # Positive lag means metric1 leads metric2
                    df2_shifted = df2.shift(periods=lag, freq='D')
                elif lag < 0:
                    # Negative lag means metric2 leads metric1
                    df2_shifted = df2.shift(periods=lag, freq='D')
                else:
                    # Zero lag means no shift
                    df2_shifted = df2
                
                # Merge shifted dataframes
                combined = pd.merge(df1, df2_shifted, left_index=True, right_index=True, how='inner')
                
                if len(combined) < 2:
                    # Not enough data points for this lag
                    lag_correlations[lag] = 0
                    continue
                
                # Calculate correlation
                values1 = combined['value_x'].values
                values2 = combined['value_y'].values
            else:
                # Manual approach for non-DatetimeIndex (just shift by position)
                # For integer lags, we'll match dates manually
                df1_dict = dict(zip(df1_dates, df1_values))
                df2_dict = dict(zip(df2_dates, df2_values))
                
                # Find common dates with the lag applied
                pairs = []
                if lag > 0:
                    # Look for date pairs where date1 is lag days before date2
                    for date1 in df1_dates:
                        try:
                            target_date = date1 + timedelta(days=lag)
                            if target_date in df2_dict:
                                pairs.append((date1, target_date))
                        except (TypeError, ValueError):
                            # Handle non-date objects gracefully
                            continue
                elif lag < 0:
                    # Look for date pairs where date1 is |lag| days after date2
                    for date1 in df1_dates:
                        try:
                            target_date = date1 + timedelta(days=lag)
                            if target_date in df2_dict:
                                pairs.append((date1, target_date))
                        except (TypeError, ValueError):
                            # Handle non-date objects gracefully
                            continue
                else:
                    # Zero lag - just find common dates
                    common_dates = set(df1_dates).intersection(set(df2_dates))
                    pairs = [(date, date) for date in common_dates]
                
                if len(pairs) < 2:
                    # Not enough data points for this lag
                    lag_correlations[lag] = 0
                    continue
                
                # Extract values for correlation
                values1 = [df1_dict[pair[0]] for pair in pairs]
                values2 = [df2_dict[pair[1]] for pair in pairs]
            
            if len(values1) < 2 or len(values2) < 2:
                lag_correlations[lag] = 0
                continue
                
            try:
                # Calculate correlation
                correlation, _ = stats.pearsonr(values1, values2)
                lag_correlations[lag] = correlation if not np.isnan(correlation) else 0
            except Exception as e:
                logger.error(f"Error calculating correlation at lag {lag}: {str(e)}")
                lag_correlations[lag] = 0
        
        # Find lag with highest absolute correlation
        if not lag_correlations:
            logger.warning("No valid lag correlations found")
            return {
                'best_lag': 0,
                'best_correlation': 0,
                'lag_correlations': {},
                'lead_metric': None,
                'causal_evidence': 'none'
            }
            
        best_lag = max(lag_correlations.keys(), key=lambda x: abs(lag_correlations[x]))
        best_correlation = lag_correlations[best_lag]
        
        # Determine which metric leads
        if best_lag > 0 and abs(best_correlation) > 0.3:
            lead_metric = 1  # Metric1 leads Metric2
        elif best_lag < 0 and abs(best_correlation) > 0.3:
            lead_metric = 2  # Metric2 leads Metric1
        else:
            lead_metric = None  # No clear lead relationship
        
        # Assess causal evidence
        abs_corr = abs(best_correlation)
        if abs_corr > 0.7 and lead_metric is not None:
            causal_evidence = 'strong'
        elif abs_corr > 0.5 and lead_metric is not None:
            causal_evidence = 'moderate'
        elif abs_corr > 0.3 and lead_metric is not None:
            causal_evidence = 'weak'
        else:
            causal_evidence = 'none'
        
        logger.info(f"Performed lead/lag analysis with max lag of {max_lag} days")
        
        return {
            'best_lag': best_lag,
            'best_correlation': best_correlation,
            'lag_correlations': lag_correlations,
            'lead_metric': lead_metric,
            'causal_evidence': causal_evidence
        }
        
    except Exception as e:
        logger.error(f"Error performing lead/lag analysis: {str(e)}")
        return {
            'best_lag': 0,
            'best_correlation': 0,
            'lag_correlations': {},
            'lead_metric': None,
            'causal_evidence': 'none',
            'error': str(e)
        }


def analyze_metric_correlations(
    metric_collection: MetricCollection
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all correlations between metrics in a collection.
    
    Args:
        metric_collection: MetricCollection containing all metrics
        
    Returns:
        Dictionary with correlation analyses:
        - 'dau_vs_mau': Correlation between DAU and MAU
        - 'dau_vs_daily_rate': Correlation between DAU and daily adoption rate
        - 'mau_vs_monthly_rate': Correlation between MAU and monthly adoption rate
        - 'daily_rate_vs_monthly_rate': Correlation between daily and monthly adoption rates
        - 'lead_lag_dau_mau': Lead/lag analysis between DAU and MAU
        - 'lead_lag_dau_rate': Lead/lag analysis between DAU and adoption rate
    """
    results = {}
    
    # Check if we have data
    if (not metric_collection.daily_active_users or 
        not metric_collection.monthly_active_users or 
        not metric_collection.overall_adoption_rates):
        logger.warning("Incomplete data in metric collection for correlation analysis")
        return results
    
    # Convert DAU list to adoption rate format for easier comparison
    dau_as_adoption_rate = []
    for dau in metric_collection.daily_active_users:
        # Find matching adoption rate for this date
        matching_rate = next((rate for rate in metric_collection.overall_adoption_rates 
                             if rate.date == dau.date), None)
        
        if matching_rate:
            rate_obj = type('DauRate', (), {
                'date': dau.date,
                'value': dau.active_users,
                'daily_adoption_rate': matching_rate.daily_adoption_rate
            })
            dau_as_adoption_rate.append(rate_obj)
    
    # Convert MAU to compatible format
    mau_as_adoption_rate = []
    for mau in metric_collection.monthly_active_users:
        # Create date from year and month
        mau_date = date(mau.year, mau.month, 1)
        
        # Find matching adoption rates for this month
        month_rates = [rate for rate in metric_collection.overall_adoption_rates
                       if rate.date.year == mau.year and rate.date.month == mau.month]
        
        if month_rates:
            # Use the average monthly adoption rate for this month
            avg_monthly_rate = sum(rate.monthly_adoption_rate for rate in month_rates) / len(month_rates)
            
            rate_obj = type('MauRate', (), {
                'date': mau_date,
                'value': mau.active_users,
                'monthly_adoption_rate': avg_monthly_rate
            })
            mau_as_adoption_rate.append(rate_obj)
    
    # DAU vs MAU correlation
    try:
        results['dau_vs_mau'] = calculate_metric_correlation(
            dau_as_adoption_rate,
            mau_as_adoption_rate,
            metric1_attribute='value',
            metric2_attribute='value',
            date_alignment='nearest'
        )
    except Exception as e:
        logger.error(f"Error calculating DAU vs MAU correlation: {str(e)}")
    
    # DAU vs daily rate
    try:
        results['dau_vs_daily_rate'] = calculate_metric_correlation(
            dau_as_adoption_rate,
            metric_collection.overall_adoption_rates,
            metric1_attribute='value',
            metric2_attribute='daily_adoption_rate',
            date_alignment='exact'
        )
    except Exception as e:
        logger.error(f"Error calculating DAU vs daily rate correlation: {str(e)}")
    
    # MAU vs monthly rate
    try:
        results['mau_vs_monthly_rate'] = calculate_metric_correlation(
            mau_as_adoption_rate,
            metric_collection.overall_adoption_rates,
            metric1_attribute='value',
            metric2_attribute='monthly_adoption_rate',
            date_alignment='nearest'
        )
    except Exception as e:
        logger.error(f"Error calculating MAU vs monthly rate correlation: {str(e)}")
    
    # Daily rate vs monthly rate
    try:
        results['daily_rate_vs_monthly_rate'] = calculate_metric_correlation(
            metric_collection.overall_adoption_rates,
            metric_collection.overall_adoption_rates,
            metric1_attribute='daily_adoption_rate',
            metric2_attribute='monthly_adoption_rate',
            date_alignment='exact'
        )
    except Exception as e:
        logger.error(f"Error calculating daily rate vs monthly rate correlation: {str(e)}")
    
    # Lead/lag analysis for DAU vs MAU
    try:
        results['lead_lag_dau_mau'] = perform_lead_lag_analysis(
            dau_as_adoption_rate,
            mau_as_adoption_rate,
            metric1_attribute='value',
            metric2_attribute='value',
            max_lag=30  # Look for up to 30 days of lag
        )
    except Exception as e:
        logger.error(f"Error performing lead/lag analysis for DAU vs MAU: {str(e)}")
    
    # Lead/lag analysis for DAU vs adoption rate
    try:
        results['lead_lag_dau_rate'] = perform_lead_lag_analysis(
            dau_as_adoption_rate,
            metric_collection.overall_adoption_rates,
            metric1_attribute='value',
            metric2_attribute='daily_adoption_rate',
            max_lag=14  # Look for up to 14 days of lag
        )
    except Exception as e:
        logger.error(f"Error performing lead/lag analysis for DAU vs adoption rate: {str(e)}")
    
    logger.info(f"Completed correlation analysis for metric collection")
    return results


def generate_correlation_summary(analysis_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate a human-readable summary of correlation analysis results.
    
    Args:
        analysis_results: Results from analyze_metric_correlations()
        
    Returns:
        Human-readable summary text
    """
    if not analysis_results:
        return "No correlation analysis results available."
    
    summary = "Correlation Analysis Summary:\n\n"
    
    # Add DAU vs MAU correlation if available
    if 'dau_vs_mau' in analysis_results:
        result = analysis_results['dau_vs_mau']
        summary += "DAU vs MAU Correlation:\n"
        summary += f"  Correlation: {result['pearson_r']:.3f} (p-value: {result['pearson_p']:.4f})\n"
        summary += f"  Strength: {result['strength'].capitalize()}\n"
        summary += f"  Direction: {result['direction'].capitalize()}\n"
        summary += f"  Based on {result['data_points']} data points\n\n"
    
    # Add DAU vs daily rate
    if 'dau_vs_daily_rate' in analysis_results:
        result = analysis_results['dau_vs_daily_rate']
        summary += "DAU vs Daily Adoption Rate Correlation:\n"
        summary += f"  Correlation: {result['pearson_r']:.3f} (p-value: {result['pearson_p']:.4f})\n"
        summary += f"  Strength: {result['strength'].capitalize()}\n"
        summary += f"  Direction: {result['direction'].capitalize()}\n"
        summary += f"  Based on {result['data_points']} data points\n\n"
    
    # Add MAU vs monthly rate
    if 'mau_vs_monthly_rate' in analysis_results:
        result = analysis_results['mau_vs_monthly_rate']
        summary += "MAU vs Monthly Adoption Rate Correlation:\n"
        summary += f"  Correlation: {result['pearson_r']:.3f} (p-value: {result['pearson_p']:.4f})\n"
        summary += f"  Strength: {result['strength'].capitalize()}\n"
        summary += f"  Direction: {result['direction'].capitalize()}\n"
        summary += f"  Based on {result['data_points']} data points\n\n"
    
    # Add daily rate vs monthly rate
    if 'daily_rate_vs_monthly_rate' in analysis_results:
        result = analysis_results['daily_rate_vs_monthly_rate']
        summary += "Daily vs Monthly Adoption Rate Correlation:\n"
        summary += f"  Correlation: {result['pearson_r']:.3f} (p-value: {result['pearson_p']:.4f})\n"
        summary += f"  Strength: {result['strength'].capitalize()}\n"
        summary += f"  Direction: {result['direction'].capitalize()}\n"
        summary += f"  Based on {result['data_points']} data points\n\n"
    
    # Add lead/lag analysis for DAU vs MAU
    if 'lead_lag_dau_mau' in analysis_results:
        result = analysis_results['lead_lag_dau_mau']
        summary += "Lead/Lag Analysis - DAU vs MAU:\n"
        summary += f"  Best lag: {abs(result['best_lag'])} days "
        
        if result['lead_metric'] == 1:
            summary += "(DAU leads MAU)\n"
        elif result['lead_metric'] == 2:
            summary += "(MAU leads DAU)\n"
        else:
            summary += "(no clear lead relationship)\n"
        
        summary += f"  Correlation at best lag: {result['best_correlation']:.3f}\n"
        summary += f"  Causal evidence: {result['causal_evidence'].capitalize()}\n\n"
    
    # Add lead/lag analysis for DAU vs adoption rate
    if 'lead_lag_dau_rate' in analysis_results:
        result = analysis_results['lead_lag_dau_rate']
        summary += "Lead/Lag Analysis - DAU vs Adoption Rate:\n"
        summary += f"  Best lag: {abs(result['best_lag'])} days "
        
        if result['lead_metric'] == 1:
            summary += "(DAU leads adoption rate)\n"
        elif result['lead_metric'] == 2:
            summary += "(adoption rate leads DAU)\n"
        else:
            summary += "(no clear lead relationship)\n"
        
        summary += f"  Correlation at best lag: {result['best_correlation']:.3f}\n"
        summary += f"  Causal evidence: {result['causal_evidence'].capitalize()}\n\n"
    
    # Add overall summary
    summary += "Overall Insights:\n"
    has_insights = False
    
    if 'dau_vs_mau' in analysis_results and analysis_results['dau_vs_mau']['strength'] != 'none':
        has_insights = True
        result = analysis_results['dau_vs_mau']
        summary += f"  • DAU and MAU show a {result['strength']} {result['direction']} correlation.\n"
    
    if 'lead_lag_dau_mau' in analysis_results and analysis_results['lead_lag_dau_mau']['lead_metric'] is not None:
        has_insights = True
        result = analysis_results['lead_lag_dau_mau']
        if result['lead_metric'] == 1:
            summary += f"  • Changes in DAU appear to precede changes in MAU by approximately {abs(result['best_lag'])} days.\n"
        else:
            summary += f"  • Changes in MAU appear to precede changes in DAU by approximately {abs(result['best_lag'])} days.\n"
    
    if 'dau_vs_daily_rate' in analysis_results and 'mau_vs_monthly_rate' in analysis_results:
        dau_result = analysis_results['dau_vs_daily_rate']
        mau_result = analysis_results['mau_vs_monthly_rate']
        
        if dau_result['strength'] != 'none' and mau_result['strength'] != 'none':
            has_insights = True
            if abs(dau_result['pearson_r']) > abs(mau_result['pearson_r']):
                summary += f"  • DAU has a stronger relationship with adoption rate than MAU does.\n"
            else:
                summary += f"  • MAU has a stronger relationship with adoption rate than DAU does.\n"
    
    if not has_insights:
        summary += "  • No significant correlations or lead/lag relationships were found in the data.\n"
    
    return summary 