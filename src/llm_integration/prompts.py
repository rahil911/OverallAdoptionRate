"""
Prompt Templates module for the Adoption Rate Chatbot.

This module provides templates for different types of prompts to be used with the LLM.
These include:
- System prompts that define the AI's role and capabilities
- Descriptive prompts for asking about what has happened
- Diagnostic prompts for asking why something happened
- Predictive prompts for asking what will happen
- Prescriptive prompts for asking what should be done
"""

from typing import Dict, Any, List, Optional
import json

# System prompt template
SYSTEM_PROMPT = """
You are an Adoption Rate Analysis Assistant for the Opus product. Your primary goal is to help users understand 
adoption rate metrics and trends. You have access to historical data about overall adoption rates (daily, weekly, monthly, 
and yearly), as well as active user counts.

You can:
1. Describe adoption rate trends over different time periods
2. Analyze peaks, valleys, and anomalies in the adoption rate data
3. Compare adoption rates across different time periods (MoM, QoQ, YoY)
4. Identify correlations between different metrics
5. Provide visualizations of the data when relevant
6. Suggest potential actions to improve adoption rates

Always base your responses on the actual data provided. If you don't have enough information to answer a question,
say so and suggest what additional information would be helpful. Be precise, clear, and actionable.
"""

# Descriptive prompt template (what happened)
DESCRIPTIVE_PROMPT_TEMPLATE = """
Provide a detailed description of the adoption rate {metric_type} for {time_period}.
Include key statistics, notable trends, and any significant changes compared to previous periods.
If relevant, reference any visualizations that might help illustrate the data.

Latest available data:
{data_summary}

User query: {user_query}
"""

# Diagnostic prompt template (why something happened)
DIAGNOSTIC_PROMPT_TEMPLATE = """
Analyze the adoption rate data for {time_period} to determine why {analysis_subject} occurred.
Focus on identifying potential factors that contributed to this trend or anomaly.
Consider correlations with other metrics and any external factors that might be relevant.

Latest available data:
{data_summary}

User query: {user_query}
"""

# Predictive prompt template (what will happen)
PREDICTIVE_PROMPT_TEMPLATE = """
Based on historical adoption rate data up to {current_date}, predict what we might expect to see
in terms of {prediction_subject} for {future_period}.
Support your prediction with trend analysis, seasonality patterns, and growth rates observed in the data.
Be clear about the confidence level of your prediction and what factors might cause deviations.

Historical data summary:
{data_summary}

User query: {user_query}
"""

# Prescriptive prompt template (what should be done)
PRESCRIPTIVE_PROMPT_TEMPLATE = """
Based on the adoption rate analysis for {time_period}, provide recommendations for {recommendation_subject}.
Your recommendations should be specific, actionable, and directly tied to insights from the data.
Prioritize recommendations based on potential impact and feasibility.

Data analysis summary:
{data_summary}

User query: {user_query}
"""


def get_system_prompt() -> str:
    """
    Get the standard system prompt for the Adoption Rate Chatbot.
    
    Returns:
        str: System prompt
    """
    return SYSTEM_PROMPT


def get_descriptive_prompt_template(
    metric_type: str,
    time_period: str,
    data_summary: str,
    user_query: str
) -> str:
    """
    Get a descriptive prompt template filled with the provided values.
    
    Args:
        metric_type: Type of metric (e.g., "overall", "daily", "monthly")
        time_period: Time period to analyze (e.g., "last 30 days", "March 2025")
        data_summary: Summary of the relevant data
        user_query: Original user query
    
    Returns:
        str: Filled descriptive prompt template
    """
    return DESCRIPTIVE_PROMPT_TEMPLATE.format(
        metric_type=metric_type,
        time_period=time_period,
        data_summary=data_summary,
        user_query=user_query
    )


def get_diagnostic_prompt_template(
    time_period: str,
    analysis_subject: str,
    data_summary: str,
    user_query: str
) -> str:
    """
    Get a diagnostic prompt template filled with the provided values.
    
    Args:
        time_period: Time period to analyze (e.g., "last 30 days", "March 2025")
        analysis_subject: Subject of the analysis (e.g., "the drop in adoption rate", "the spike in active users")
        data_summary: Summary of the relevant data
        user_query: Original user query
    
    Returns:
        str: Filled diagnostic prompt template
    """
    return DIAGNOSTIC_PROMPT_TEMPLATE.format(
        time_period=time_period,
        analysis_subject=analysis_subject,
        data_summary=data_summary,
        user_query=user_query
    )


def get_predictive_prompt_template(
    current_date: str,
    prediction_subject: str,
    future_period: str,
    data_summary: str,
    user_query: str
) -> str:
    """
    Get a predictive prompt template filled with the provided values.
    
    Args:
        current_date: Current date (e.g., "2025-03-31")
        prediction_subject: Subject of the prediction (e.g., "adoption rates", "active users")
        future_period: Future period to predict (e.g., "next month", "Q2 2025")
        data_summary: Summary of the relevant historical data
        user_query: Original user query
    
    Returns:
        str: Filled predictive prompt template
    """
    return PREDICTIVE_PROMPT_TEMPLATE.format(
        current_date=current_date,
        prediction_subject=prediction_subject,
        future_period=future_period,
        data_summary=data_summary,
        user_query=user_query
    )


def get_prescriptive_prompt_template(
    time_period: str,
    recommendation_subject: str,
    data_summary: str,
    user_query: str
) -> str:
    """
    Get a prescriptive prompt template filled with the provided values.
    
    Args:
        time_period: Time period being analyzed (e.g., "last quarter", "2024 to present")
        recommendation_subject: Subject of the recommendations (e.g., "improving adoption rates", "reducing churn")
        data_summary: Summary of the relevant data analysis
        user_query: Original user query
    
    Returns:
        str: Filled prescriptive prompt template
    """
    return PRESCRIPTIVE_PROMPT_TEMPLATE.format(
        time_period=time_period,
        recommendation_subject=recommendation_subject,
        data_summary=data_summary,
        user_query=user_query
    )


def determine_prompt_type(user_query: str) -> str:
    """
    Determine the most appropriate prompt type based on the user query.
    
    Args:
        user_query: The user's query string
    
    Returns:
        str: The prompt type ('descriptive', 'diagnostic', 'predictive', or 'prescriptive')
    """
    query = user_query.lower()
    
    # Check for predictive intent (future-oriented)
    if any(term in query for term in ['predict', 'forecast', 'will', 'future', 'expect', 'next']):
        return 'predictive'
    
    # Check for diagnostic intent (why something happened)
    elif any(term in query for term in ['why', 'reason', 'cause', 'explain', 'understand']):
        return 'diagnostic'
    
    # Check for prescriptive intent (what should be done)
    elif any(term in query for term in ['recommend', 'suggest', 'how to improve', 'action', 'should', 'could']):
        return 'prescriptive'
    
    # Default to descriptive (what happened)
    else:
        return 'descriptive' 