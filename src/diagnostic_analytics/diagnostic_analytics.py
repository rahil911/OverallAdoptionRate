"""
Diagnostic Analytics Module

This module provides diagnostic analytics capabilities for analyzing adoption rate data,
focusing on explaining changes, identifying correlations, and suggesting root causes
for observed patterns.
"""

import logging
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any, Union

from src.database.data_access import DataAccessLayer
from src.data_models.metrics import (
    MetricCollection,
    OverallAdoptionRate,
    MonthlyActiveUsers,
    DailyActiveUsers
)
from src.data_analysis.correlation_analyzer import (
    calculate_correlation_matrix,
    calculate_metric_correlation,
    perform_lead_lag_analysis,
    analyze_metric_correlations
)
from src.data_analysis.anomaly_detector import (
    detect_anomalies_zscore,
    detect_anomalies_modified_zscore,
    detect_anomalies_iqr,
    detect_anomalies_moving_average,
    detect_anomalies_adaptive_threshold,
    detect_anomalies_ensemble,
    generate_anomaly_explanation
)
from src.data_analysis.trend_analyzer import (
    detect_peaks_and_valleys,
    calculate_trend_line,
    calculate_moving_average,
    generate_trend_description,
    identify_significant_changes
)

# Configure logging
logger = logging.getLogger(__name__)

class DiagnosticAnalytics:
    """
    Main class for providing diagnostic analytics about adoption rate data.
    This class integrates correlation analysis, anomaly detection, and trend analysis
    to explain changes and patterns in adoption rate data.
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
    
    def analyze_rate_changes(self, from_date=None, to_date=None, metric_type="monthly"):
        """
        Analyze changes in adoption rates to identify potential causes.
        
        Args:
            from_date (datetime, optional): Start date for analysis
            to_date (datetime, optional): End date for analysis
            metric_type (str): Type of metric to analyze ("daily", "weekly", "monthly", "yearly")
            
        Returns:
            dict: Dictionary containing analysis results including:
                - significant_changes: List of significant rate changes
                - correlations: Correlation analysis with other metrics
                - anomalies: Any anomalies detected during changes
                - trends: Trend analysis around change points
                - explanation: Natural language explanation of findings
        """
        # Get the data
        collection = self._get_data(from_date, to_date)
        
        if not collection.overall_adoption_rates:
            return {
                "explanation": "No adoption rate data available for analysis."
            }
        
        # Identify significant changes
        significant_changes = identify_significant_changes(
            collection.overall_adoption_rates,
            rate_type=metric_type,
            threshold_percent=10.0  # Consider changes of 10% or more significant
        )
        
        # For each significant change, analyze potential causes
        change_analyses = []
        for change in significant_changes:
            # Get data around the change point
            change_date = change['date']
            window_start = change_date - timedelta(days=30)
            window_end = change_date + timedelta(days=30)
            
            # Analyze correlations around the change
            window_collection = self._get_data(window_start, window_end)
            correlations = analyze_metric_correlations(window_collection)
            
            # Look for anomalies around the change
            anomalies = detect_anomalies_ensemble(
                collection.overall_adoption_rates,
                rate_type=metric_type,
                min_methods=2
            )
            
            # Analyze trends before and after
            before_trend = calculate_trend_line(
                [x for x in collection.overall_adoption_rates if window_start <= x.date < change_date],
                rate_type=metric_type
            )
            
            after_trend = calculate_trend_line(
                [x for x in collection.overall_adoption_rates if change_date <= x.date <= window_end],
                rate_type=metric_type
            )
            
            # Compile analysis for this change
            change_analyses.append({
                "date": change_date,
                "change": change,
                "correlations": correlations,
                "anomalies": anomalies,
                "before_trend": before_trend,
                "after_trend": after_trend
            })
        
        # Generate overall explanation
        explanation = self._generate_change_explanation(change_analyses)
        
        return {
            "significant_changes": significant_changes,
            "change_analyses": change_analyses,
            "explanation": explanation
        }
    
    def identify_correlated_metrics(self, from_date=None, to_date=None):
        """
        Identify correlations between different metrics.
        
        Args:
            from_date (datetime, optional): Start date for analysis
            to_date (datetime, optional): End date for analysis
            
        Returns:
            dict: Dictionary containing correlation analysis results including:
                - correlation_matrix: Matrix of correlations between metrics
                - strong_correlations: List of strongly correlated metric pairs
                - lead_lag_relationships: Analysis of leading/lagging relationships
                - explanation: Natural language explanation of findings
        """
        # Get the data
        collection = self._get_data(from_date, to_date)
        
        if not collection.overall_adoption_rates:
            return {
                "explanation": "No data available for correlation analysis."
            }
        
        # Calculate correlation matrix
        correlation_matrix = calculate_correlation_matrix(collection.overall_adoption_rates)
        
        # Analyze metric correlations
        metric_correlations = analyze_metric_correlations(collection)
        
        # Identify strong correlations
        strong_correlations = []
        for metric1 in correlation_matrix:
            for metric2 in correlation_matrix[metric1]:
                if metric1 != metric2:
                    corr = correlation_matrix[metric1][metric2]
                    if abs(corr) > 0.7:  # Strong correlation threshold
                        strong_correlations.append({
                            "metric1": metric1,
                            "metric2": metric2,
                            "correlation": corr
                        })
        
        # Analyze lead/lag relationships for strongly correlated metrics
        lead_lag_relationships = []
        for corr in strong_correlations:
            lead_lag = perform_lead_lag_analysis(
                collection.overall_adoption_rates,
                collection.overall_adoption_rates,
                metric1_attribute=corr["metric1"],
                metric2_attribute=corr["metric2"]
            )
            lead_lag_relationships.append({
                "metrics": (corr["metric1"], corr["metric2"]),
                "analysis": lead_lag
            })
        
        # Generate explanation
        explanation = self._generate_correlation_explanation(
            strong_correlations,
            lead_lag_relationships,
            metric_correlations
        )
        
        return {
            "correlation_matrix": correlation_matrix,
            "strong_correlations": strong_correlations,
            "lead_lag_relationships": lead_lag_relationships,
            "metric_correlations": metric_correlations,
            "explanation": explanation
        }
    
    def suggest_root_causes(self, input_date, metric_type="monthly", window_days=30):
        """
        Suggest potential root causes for adoption rate changes at a specific date.
        
        Args:
            input_date (datetime): Date to analyze
            metric_type (str): Type of metric to analyze
            window_days (int): Number of days to analyze before and after the date
            
        Returns:
            dict: Dictionary containing root cause analysis including:
                - changes: Significant changes around the date
                - correlations: Correlated metrics at the time
                - anomalies: Related anomalies
                - patterns: Identified patterns
                - suggestions: List of potential root causes
                - explanation: Natural language explanation
        """
        # Get data for the analysis window
        window_start = input_date - timedelta(days=window_days)
        window_end = input_date + timedelta(days=window_days)
        collection = self._get_data(window_start, window_end)
        
        if not collection.overall_adoption_rates:
            return {
                "explanation": "No data available for root cause analysis."
            }
        
        # Identify significant changes
        changes = identify_significant_changes(
            collection.overall_adoption_rates,
            rate_type=metric_type
        )
        
        # Analyze correlations
        correlations = analyze_metric_correlations(collection)
        
        # Detect anomalies
        anomalies = detect_anomalies_ensemble(
            collection.overall_adoption_rates,
            rate_type=metric_type
        )
        
        # Analyze trends
        trend = calculate_trend_line(
            collection.overall_adoption_rates,
            rate_type=metric_type
        )
        
        # Generate suggestions based on the analysis
        suggestions = self._generate_root_cause_suggestions(
            input_date,
            changes,
            correlations,
            anomalies,
            trend
        )
        
        # Generate explanation
        explanation = self._generate_root_cause_explanation(
            input_date,
            suggestions,
            changes,
            correlations,
            anomalies,
            trend
        )
        
        return {
            "changes": changes,
            "correlations": correlations,
            "anomalies": anomalies,
            "trend": trend,
            "suggestions": suggestions,
            "explanation": explanation
        }
    
    def _generate_change_explanation(self, change_analyses):
        """Generate natural language explanation for adoption rate changes."""
        if not change_analyses:
            return "No significant changes detected in the adoption rate."
        
        explanation = []
        
        # Sort changes by date
        sorted_analyses = sorted(change_analyses, key=lambda x: x["date"])
        
        # Generate overall summary
        explanation.append(f"Analyzed {len(sorted_analyses)} significant changes in adoption rate.")
        
        # Describe each major change
        for analysis in sorted_analyses:
            change = analysis["change"]
            date_str = change["date"].strftime("%B %d, %Y")
            
            # Describe the change
            if change["direction"] == "increase":
                change_desc = f"increased by {abs(change['percent_change']):.1f}%"
            else:
                change_desc = f"decreased by {abs(change['percent_change']):.1f}%"
            
            explanation.append(f"\nOn {date_str}, the adoption rate {change_desc}.")
            
            # Add correlation insights
            if analysis["correlations"]:
                strong_corr = [k for k, v in analysis["correlations"].items() 
                             if v.get("strength") == "strong"]
                if strong_corr:
                    explanation.append("This change strongly correlated with changes in: " + 
                                    ", ".join(strong_corr) + ".")
            
            # Add anomaly insights
            relevant_anomalies = [a for a in analysis["anomalies"] 
                                if abs((a["date"] - change["date"]).days) <= 7]
            if relevant_anomalies:
                explanation.append(f"Detected {len(relevant_anomalies)} related anomalies " +
                                "within a week of this change.")
            
            # Add trend insights
            if analysis["before_trend"]["direction"] != analysis["after_trend"]["direction"]:
                explanation.append(f"This change marked a shift from a {analysis['before_trend']['direction']} " +
                                f"trend to a {analysis['after_trend']['direction']} trend.")
        
        return "\n".join(explanation)
    
    def _generate_correlation_explanation(self, strong_correlations, lead_lag_relationships, metric_correlations):
        """Generate natural language explanation for correlation analysis."""
        if not strong_correlations:
            return "No strong correlations detected between metrics."
        
        explanation = []
        
        # Overall summary
        explanation.append(f"Found {len(strong_correlations)} strong correlations between metrics.")
        
        # Describe strongest correlations
        sorted_correlations = sorted(strong_correlations, 
                                   key=lambda x: abs(x["correlation"]), 
                                   reverse=True)
        
        for corr in sorted_correlations[:3]:  # Top 3 strongest correlations
            explanation.append(f"\n{corr['metric1']} and {corr['metric2']} show a " +
                            f"strong {'positive' if corr['correlation'] > 0 else 'negative'} " +
                            f"correlation (r={corr['correlation']:.2f}).")
            
            # Add lead/lag insight if available
            for ll in lead_lag_relationships:
                if set(ll["metrics"]) == {corr["metric1"], corr["metric2"]}:
                    if ll["analysis"]["lead_metric"] is not None:
                        lead = ll["metrics"][ll["analysis"]["lead_metric"] - 1]
                        lag = ll["metrics"][1 if ll["analysis"]["lead_metric"] == 0 else 0]
                        explanation.append(f"Changes in {lead} tend to precede changes in {lag} " +
                                        f"by {abs(ll['analysis']['best_lag'])} days.")
        
        # Add insights from metric correlations
        if metric_correlations:
            if "dau_vs_mau" in metric_correlations:
                dau_mau = metric_correlations["dau_vs_mau"]
                if dau_mau["strength"] != "none":
                    explanation.append(f"\nDAU and MAU show a {dau_mau['strength']} " +
                                    f"{dau_mau['direction']} correlation.")
            
            if "lead_lag_dau_rate" in metric_correlations:
                ll_dau = metric_correlations["lead_lag_dau_rate"]
                if ll_dau["causal_evidence"] != "none":
                    explanation.append(f"\nFound {ll_dau['causal_evidence']} evidence that " +
                                    "changes in DAU may influence adoption rate.")
        
        return "\n".join(explanation)
    
    def _generate_root_cause_suggestions(self, input_date, changes, correlations, anomalies, trend):
        """Generate root cause suggestions based on analysis results."""
        suggestions = []
        
        # Ensure date is datetime.datetime for comparisons
        if isinstance(input_date, date) and not isinstance(input_date, datetime):
            # Convert date to datetime
            input_date = datetime.combine(input_date, datetime.min.time())
        
        # Look for changes around the date
        relevant_changes = []
        for c in changes:
            change_date = c["date"]
            # Convert date to same type for comparison
            if isinstance(change_date, date) and not isinstance(change_date, datetime):
                change_date = datetime.combine(change_date, datetime.min.time())
            elif isinstance(change_date, datetime) and isinstance(input_date, date):
                input_date = datetime.combine(input_date, datetime.min.time())
            
            try:
                # Now compare the dates
                if abs((change_date - input_date).days) <= 7:
                    relevant_changes.append(c)
            except TypeError as e:
                logger.error(f"Type error comparing dates: {type(change_date)} vs {type(input_date)}")
                # Skip this change
                continue
        
        if relevant_changes:
            for change in relevant_changes:
                suggestions.append({
                    "type": "change",
                    "confidence": "high",
                    "description": f"Significant {change['direction']} in adoption rate " +
                                 f"of {abs(change['percent_change']):.1f}%"
                })
        
        # Look for correlated metric changes
        if correlations:
            for metric, corr in correlations.items():
                if corr.get("strength") == "strong" and corr.get("direction") != "none":
                    suggestions.append({
                        "type": "correlation",
                        "confidence": "medium",
                        "description": f"Strong {corr['direction']} correlation with {metric}"
                    })
        
        # Look for anomalies
        relevant_anomalies = []
        for a in anomalies:
            anomaly_date = a["date"]
            # Convert date to same type for comparison
            if isinstance(anomaly_date, date) and not isinstance(anomaly_date, datetime):
                anomaly_date = datetime.combine(anomaly_date, datetime.min.time())
            elif isinstance(anomaly_date, datetime) and isinstance(input_date, date):
                input_date = datetime.combine(input_date, datetime.min.time())
            
            try:
                # Now compare the dates
                if abs((anomaly_date - input_date).days) <= 7:
                    relevant_anomalies.append(a)
            except TypeError as e:
                logger.error(f"Type error comparing dates: {type(anomaly_date)} vs {type(input_date)}")
                # Skip this anomaly
                continue
        
        if relevant_anomalies:
            for anomaly in relevant_anomalies:
                suggestions.append({
                    "type": "anomaly",
                    "confidence": "high" if anomaly.get("confidence", 0) > 0.8 else "medium",
                    "description": f"Detected {anomaly['direction']} anomaly " +
                                 f"({anomaly.get('value', 0):.1f}%)"
                })
        
        # Look for trend changes
        if trend:
            if trend["trend_strength"] == "strong":
                suggestions.append({
                    "type": "trend",
                    "confidence": "medium",
                    "description": f"Part of a strong {trend['direction']} trend"
                })
        
        return suggestions
    
    def _generate_root_cause_explanation(self, input_date, suggestions, changes, correlations, anomalies, trend):
        """
        Generate a natural language explanation of root cause analysis results.
        """
        if not suggestions:
            return "No clear root causes identified for the specified date."
        
        explanation = []
        date_str = input_date.strftime("%B %d, %Y")
        
        # Overall summary
        explanation.append(f"Analysis of adoption rate changes on {date_str} " +
                         f"identified {len(suggestions)} potential root causes.")
        
        # Group suggestions by confidence
        high_confidence = [s for s in suggestions if s["confidence"] == "high"]
        medium_confidence = [s for s in suggestions if s["confidence"] == "medium"]
        
        # Describe high confidence findings
        if high_confidence:
            explanation.append("\nHigh confidence findings:")
            for finding in high_confidence:
                explanation.append(f"- {finding['description']}")
        
        # Describe medium confidence findings
        if medium_confidence:
            explanation.append("\nOther potential factors:")
            for finding in medium_confidence:
                explanation.append(f"- {finding['description']}")
        
        # Add trend context
        if trend and trend["trend_strength"] != "weak":
            explanation.append(f"\nThis occurred during a {trend['trend_strength']} " +
                            f"{trend['direction']} trend in the adoption rate.")
        
        return "\n".join(explanation)
    
    def explain_anomalies(self, from_date=None, to_date=None, metric_type="monthly"):
        """
        Detect and explain anomalies in adoption rate data within a specified date range.
        
        Args:
            from_date (datetime, optional): Start date for analysis
            to_date (datetime, optional): End date for analysis
            metric_type (str): Type of metric to analyze ("daily", "weekly", "monthly", "yearly")
            
        Returns:
            list: List of dictionaries containing anomaly information:
                - date: Date of the anomaly
                - value: Adoption rate value
                - explanation: Natural language explanation of the anomaly
                - confidence: Confidence score of the anomaly detection
                - direction: 'high' or 'low' indicating the direction of the anomaly
        """
        # Get the data
        collection = self._get_data(from_date, to_date)
        
        if not collection.overall_adoption_rates:
            return []
        
        # Detect anomalies
        anomalies = detect_anomalies_ensemble(
            collection.overall_adoption_rates,
            rate_type=metric_type,
            min_methods=2
        )
        
        # Generate explanations for each anomaly
        result = []
        for anomaly in anomalies:
            # Get explanation for this anomaly
            explanation = generate_anomaly_explanation(
                anomalies=[anomaly], 
                data=collection.overall_adoption_rates,
                rate_type=metric_type
            )
            
            # Add to result list
            result.append({
                "date": anomaly["date"].strftime("%Y-%m-%d") if isinstance(anomaly["date"], (datetime, date)) else anomaly["date"],
                "value": anomaly["value"],
                "explanation": explanation[0] if explanation else f"Unusual {'increase' if anomaly.get('direction') == 'high' else 'decrease'} in {metric_type} adoption rate",
                "confidence": anomaly.get("confidence", 0.8),
                "direction": anomaly.get("direction", "high")
            })
        
        return result 