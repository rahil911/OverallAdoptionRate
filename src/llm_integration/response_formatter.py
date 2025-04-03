"""
Response Formatter module for the Adoption Rate Chatbot.

This module provides functions for formatting LLM responses consistently,
adding references to charts and visualizations, and ensuring responses
follow the desired format and style.
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)

# Constants
CHARTS_DIR = "plots/analysis"


def format_response(response: str, add_references: bool = True) -> str:
    """
    Format the LLM response for consistency and improved readability.
    
    Args:
        response: Raw response from the LLM
        add_references: Whether to add references to relevant visualizations
    
    Returns:
        str: Formatted response
    """
    # Clean up extra spaces and newlines
    formatted_response = re.sub(r'\n{3,}', '\n\n', response)
    
    # Add chart references if requested
    if add_references:
        formatted_response = add_chart_references(formatted_response)
    
    # Format bullet points consistently
    formatted_response = re.sub(r'(?<=\n)[-*] ', '• ', formatted_response)
    
    # Format percentage values consistently (ensure one decimal place)
    formatted_response = re.sub(
        r'(\d+)%', 
        lambda m: f"{int(m.group(1))}%", 
        formatted_response
    )
    formatted_response = re.sub(
        r'(\d+\.\d)%', 
        lambda m: f"{float(m.group(1)):.1f}%", 
        formatted_response
    )
    formatted_response = re.sub(
        r'(\d+\.\d{2,})%', 
        lambda m: f"{float(m.group(1)):.1f}%", 
        formatted_response
    )
    
    return formatted_response


def add_chart_references(response: str) -> str:
    """
    Add references to relevant charts and visualizations based on the response content.
    
    Args:
        response: Response text from the LLM
    
    Returns:
        str: Response with added chart references
    """
    keywords_to_charts = {
        # Map keywords to chart files
        "trend analysis": "trend_analysis.png",
        "peaks": "trend_analysis.png",
        "valleys": "trend_analysis.png",
        "trend": "trend_analysis.png",
        "period-over-period": "period_over_period.png",
        "month-over-month": "period_over_period.png",
        "quarter-over-quarter": "period_over_period.png",
        "year-over-year": "period_over_period.png",
        "MoM": "period_over_period.png",
        "QoQ": "period_over_period.png",
        "YoY": "period_over_period.png",
        "anomalies": "anomaly_detection.png",
        "outliers": "anomaly_detection.png",
        "unusual": "anomaly_detection.png",
        "correlation": "correlation_matrix.png",
        "correlations": "correlation_matrix.png",
        "relationship": "correlation_matrix.png",
        "related": "correlation_matrix.png"
    }
    
    # Check for chart mentions already in the response
    chart_mentions = set()
    for line in response.split('\n'):
        if "chart" in line.lower() or "visualization" in line.lower() or "graph" in line.lower():
            for chart_name in ["trend_analysis.png", "period_over_period.png", 
                              "anomaly_detection.png", "correlation_matrix.png"]:
                if chart_name in line:
                    chart_mentions.add(chart_name)
    
    # Determine which charts to reference based on response content
    charts_to_add = set()
    lower_response = response.lower()
    
    for keyword, chart in keywords_to_charts.items():
        if keyword.lower() in lower_response and chart not in chart_mentions:
            charts_to_add.add(chart)
    
    # If charts should be referenced, add them
    if charts_to_add:
        chart_references = "\n\n**Related Visualizations:**\n"
        
        chart_descriptions = {
            "trend_analysis.png": "Trend Analysis - Shows the adoption rate trend over time, with peaks, valleys, and trend lines.",
            "period_over_period.png": "Period Comparison - Compares adoption rates across different time periods (MoM, QoQ, YoY).",
            "anomaly_detection.png": "Anomaly Detection - Highlights unusual values and outliers in the adoption rate data.",
            "correlation_matrix.png": "Correlation Matrix - Shows relationships between different metrics."
        }
        
        for chart in sorted(charts_to_add):
            if chart in chart_descriptions:
                chart_path = os.path.join(CHARTS_DIR, chart)
                chart_references += f"- {chart_descriptions[chart]} ({chart_path})\n"
        
        return response + chart_references
    
    return response


def add_chart_reference_to_response(
    response: str,
    chart_name: str,
    chart_description: str,
    chart_dir: str = CHARTS_DIR
) -> str:
    """
    Add a specific chart reference to the response.
    
    Args:
        response: Original response text
        chart_name: File name of the chart
        chart_description: Description of what the chart shows
        chart_dir: Directory where charts are stored
    
    Returns:
        str: Response with added chart reference
    """
    chart_path = os.path.join(chart_dir, chart_name)
    
    # Check if chart is already referenced
    if chart_name in response or chart_path in response:
        return response
    
    # Add chart reference
    chart_reference = f"\n\n**Visualization:**\n- {chart_description} ({chart_path})"
    return response + chart_reference


def format_numbers(response: str) -> str:
    """
    Format numbers consistently in the response.
    
    Args:
        response: Response text
    
    Returns:
        str: Response with consistently formatted numbers
    """
    # Format large numbers with commas
    formatted_response = re.sub(
        r'(\d{1,3})(\d{3})(\d{3})',
        r'\1,\2,\3',
        response
    )
    formatted_response = re.sub(
        r'(\d{1,3})(\d{3})([^\d])',
        r'\1,\2\3',
        formatted_response
    )
    
    # Ensure consistent decimal places for percentages
    formatted_response = re.sub(
        r'(\d+\.\d{2,})%',
        lambda m: f"{float(m.group(1)):.1f}%",
        formatted_response
    )
    
    return formatted_response


def format_key_metrics(response: str) -> str:
    """
    Highlight key metrics in the response.
    
    Args:
        response: Response text
    
    Returns:
        str: Response with highlighted key metrics
    """
    # Highlight key metrics with bold formatting
    metric_patterns = [
        (r'(\d+(\.\d+)?%\s+(increase|decrease))', r'**\1**'),
        (r'(adoption rate of \d+(\.\d+)?%)', r'**\1**'),
        (r'(highest|lowest)(\s+value\s+of\s+\d+(\.\d+)?%)', r'\1**\2**'),
        (r'(correlation\s+coefficient\s+of\s+[+-]?\d+(\.\d+)?)', r'**\1**')
    ]
    
    formatted_response = response
    for pattern, replacement in metric_patterns:
        formatted_response = re.sub(pattern, replacement, formatted_response, flags=re.IGNORECASE)
    
    return formatted_response


def format_anomalies(response: str) -> str:
    """
    Format anomalies consistently in the response.
    
    Args:
        response: Response text
    
    Returns:
        str: Response with consistently formatted anomalies
    """
    # Look for anomaly descriptions and format them consistently
    anomaly_sections = re.findall(r'(anomal[^\n.]*[.\n])', response, re.IGNORECASE)
    
    formatted_response = response
    for section in anomaly_sections:
        # Bold the anomaly date and value
        highlighted_section = re.sub(
            r'(\d{4}-\d{2}-\d{2}|\w+ \d{4})(\s+with\s+value\s+of\s+)(\d+(\.\d+)?%)',
            r'**\1\2\3**',
            section
        )
        formatted_response = formatted_response.replace(section, highlighted_section)
    
    return formatted_response


def format_date_references(response: str) -> str:
    """
    Format date references consistently in the response.
    
    Args:
        response: Response text
    
    Returns:
        str: Response with consistently formatted dates
    """
    # Format ISO dates to more readable format
    formatted_response = re.sub(
        r'(\d{4}-\d{2}-\d{2})',
        lambda m: format_date(m.group(1)),
        response
    )
    
    return formatted_response


def format_date(iso_date: str) -> str:
    """
    Format an ISO date (YYYY-MM-DD) to a more readable format.
    
    Args:
        iso_date: Date in ISO format (YYYY-MM-DD)
    
    Returns:
        str: Formatted date
    """
    try:
        year, month, day = iso_date.split('-')
        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        month_name = month_names[int(month) - 1]
        
        return f"{month_name} {int(day)}, {year}"
    except:
        # If parsing fails, return the original date
        return iso_date 