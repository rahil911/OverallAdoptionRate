"""
Function Calling module for the Adoption Rate Chatbot.

This module provides functionality for implementing function calling
with LLMs, allowing the models to request specific data or analyses
through predefined functions.
"""

import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Callable
import inspect
import pandas as pd

# Import data processing classes
from src.data_processing.data_fetcher import DataFetcher
from src.database.data_access import DataAccessLayer

# Import data analysis functions
from src.data_analysis.trend_analyzer import (
    detect_peaks_and_valleys,
    calculate_trend_line,
    generate_trend_description
)
from src.data_analysis.period_analyzer import (
    calculate_mom_change,
    calculate_qoq_change,
    calculate_yoy_change,
    generate_period_comparison_summary,
    compare_time_periods
)
from src.data_analysis.anomaly_detector import (
    detect_anomalies_zscore,
    detect_anomalies_ensemble,
    generate_anomaly_explanation
)
from src.data_analysis.correlation_analyzer import (
    calculate_correlation_matrix,
    generate_correlation_summary
)

# Set up logging
logger = logging.getLogger(__name__)

# Create an instance of the DataFetcher
data_fetcher = DataFetcher(DataAccessLayer)

# Function Definitions
FUNCTION_DEFINITIONS = [
    {
        "name": "get_overall_adoption_rate_data",
        "description": "Get overall adoption rate data for a specified time period",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format"
                },
                "tenant_id": {
                    "type": "integer",
                    "description": "Tenant ID to retrieve data for (default: 1388)"
                }
            },
            "required": ["start_date", "end_date"]
        }
    },
    {
        "name": "get_mau_data",
        "description": "Get Monthly Active Users (MAU) data for a specified time period",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format"
                },
                "tenant_id": {
                    "type": "integer",
                    "description": "Tenant ID to retrieve data for (default: 1388)"
                }
            },
            "required": ["start_date", "end_date"]
        }
    },
    {
        "name": "get_dau_data",
        "description": "Get Daily Active Users (DAU) data for a specified time period",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format"
                },
                "tenant_id": {
                    "type": "integer",
                    "description": "Tenant ID to retrieve data for (default: 1388)"
                }
            },
            "required": ["start_date", "end_date"]
        }
    },
    {
        "name": "analyze_adoption_rate_trend",
        "description": "Analyze the trend of adoption rate data",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format"
                },
                "metric_type": {
                    "type": "string",
                    "description": "Type of adoption rate to analyze (daily, weekly, monthly, yearly)",
                    "enum": ["daily", "weekly", "monthly", "yearly"]
                },
                "tenant_id": {
                    "type": "integer",
                    "description": "Tenant ID to retrieve data for (default: 1388)"
                }
            },
            "required": ["start_date", "end_date", "metric_type"]
        }
    },
    {
        "name": "compare_periods",
        "description": "Compare adoption rates between different time periods",
        "parameters": {
            "type": "object",
            "properties": {
                "period1_start": {
                    "type": "string",
                    "description": "Period 1 start date in YYYY-MM-DD format"
                },
                "period1_end": {
                    "type": "string",
                    "description": "Period 1 end date in YYYY-MM-DD format"
                },
                "period2_start": {
                    "type": "string",
                    "description": "Period 2 start date in YYYY-MM-DD format"
                },
                "period2_end": {
                    "type": "string",
                    "description": "Period 2 end date in YYYY-MM-DD format"
                },
                "comparison_type": {
                    "type": "string",
                    "description": "Type of comparison to perform",
                    "enum": ["mom", "qoq", "yoy", "custom"]
                },
                "tenant_id": {
                    "type": "integer",
                    "description": "Tenant ID to retrieve data for (default: 1388)"
                }
            },
            "required": ["period1_start", "period1_end", "period2_start", "period2_end", "comparison_type"]
        }
    },
    {
        "name": "detect_anomalies",
        "description": "Detect anomalies in adoption rate data",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format"
                },
                "metric_type": {
                    "type": "string",
                    "description": "Type of adoption rate to analyze (daily, weekly, monthly, yearly)",
                    "enum": ["daily", "weekly", "monthly", "yearly"]
                },
                "method": {
                    "type": "string",
                    "description": "Anomaly detection method to use",
                    "enum": ["zscore", "iqr", "moving_average", "ensemble"]
                },
                "tenant_id": {
                    "type": "integer",
                    "description": "Tenant ID to retrieve data for (default: 1388)"
                }
            },
            "required": ["start_date", "end_date", "metric_type", "method"]
        }
    },
    {
        "name": "analyze_correlations",
        "description": "Analyze correlations between different metrics",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format"
                },
                "metrics": {
                    "type": "array",
                    "description": "List of metrics to analyze correlations between",
                    "items": {
                        "type": "string",
                        "enum": [
                            "daily_adoption_rate", 
                            "weekly_adoption_rate", 
                            "monthly_adoption_rate", 
                            "yearly_adoption_rate",
                            "dau",
                            "mau"
                        ]
                    }
                },
                "tenant_id": {
                    "type": "integer",
                    "description": "Tenant ID to retrieve data for (default: 1388)"
                }
            },
            "required": ["start_date", "end_date", "metrics"]
        }
    }
]


def get_function_definitions() -> List[Dict[str, Any]]:
    """
    Get the list of function definitions for function calling.
    
    Returns:
        List[Dict[str, Any]]: List of function definitions
    """
    return FUNCTION_DEFINITIONS


def format_function_call(response: Dict[str, Any], provider: str = "openai") -> Optional[Dict[str, Any]]:
    """
    Extract function call information from the LLM response.
    
    Args:
        response: Response from the LLM API or direct response text
        provider: LLM provider ("openai" or "anthropic")
    
    Returns:
        Optional[Dict[str, Any]]: Function call information or None
    """
    try:
        if isinstance(response, dict) or hasattr(response, "choices"):
            # This is a direct response from the API
            if provider == "openai":
                # Extract function call from OpenAI response
                if hasattr(response, "choices") and response.choices:
                    message = response.choices[0].message
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        tool_call = message.tool_calls[0]
                        return {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments)
                        }
            elif provider == "anthropic":
                # Extract function call from Anthropic response
                if hasattr(response, "content") and response.content:
                    for content_block in response.content:
                        if content_block.type == "tool_use":
                            return {
                                "name": content_block.name,
                                "arguments": content_block.input
                            }
        
        return None
    except Exception as e:
        logger.error(f"Error formatting function call: {e}")
        return None


def execute_function(function_call: Dict[str, Any]) -> Any:
    """
    Execute a function call based on the LLM's request.
    
    Args:
        function_call: Function call information (name, arguments)
    
    Returns:
        Any: Result of the function call
    """
    try:
        function_name = function_call["name"]
        arguments = function_call["arguments"]
        
        # Map function names to actual functions
        function_map = {
            "get_overall_adoption_rate_data": _execute_get_overall_adoption_rate_data,
            "get_mau_data": _execute_get_mau_data,
            "get_dau_data": _execute_get_dau_data,
            "analyze_adoption_rate_trend": _execute_analyze_adoption_rate_trend,
            "compare_periods": _execute_compare_periods,
            "detect_anomalies": _execute_detect_anomalies,
            "analyze_correlations": _execute_analyze_correlations
        }
        
        if function_name not in function_map:
            raise ValueError(f"Unknown function: {function_name}")
        
        # Execute the function with the provided arguments
        return function_map[function_name](**arguments)
    
    except Exception as e:
        logger.error(f"Error executing function {function_call['name']}: {e}")
        return {"error": str(e)}


class DateTimeEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for handling datetime objects and model objects
    """
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
        return super(DateTimeEncoder, self).default(obj)


def handle_function_response(result: Any, function_name: str) -> Dict[str, Any]:
    """
    Handle the response from a function call.
    
    Args:
        result: The result from executing the function
        function_name: The name of the function that was called
        
    Returns:
        Dict[str, Any]: Properly formatted function response
    """
    try:
        if isinstance(result, list):
            # Handle list results (like database query results)
            if result and hasattr(result[0], 'to_dict'):
                # If items have to_dict method, convert them
                serialized_result = [item.to_dict() for item in result]
            else:
                # Otherwise, use the items directly
                serialized_result = result
                
            # Convert result to JSON string with custom encoder for dates
            result_str = json.dumps(serialized_result, cls=DateTimeEncoder)
        elif hasattr(result, 'to_dict'):
            # Handle single object with to_dict method
            result_str = json.dumps(result.to_dict(), cls=DateTimeEncoder)
        else:
            # Handle primitive types or dictionaries
            result_str = json.dumps(result, cls=DateTimeEncoder)
        
        # Create the function response
        function_response = {
            "role": "function",
            "name": function_name,
            "content": result_str
        }
        
        return function_response
    except Exception as e:
        logger.error(f"Error handling function response: {str(e)}")
        # Return simple error message as content
        return {
            "role": "function",
            "name": function_name,
            "content": f"Error processing function result: {str(e)}"
        }


# Function implementations
def _execute_get_overall_adoption_rate_data(
    start_date: str,
    end_date: str,
    tenant_id: int = 1388
) -> Dict[str, Any]:
    """Execute get_overall_adoption_rate_data function"""
    try:
        # Convert string dates to datetime objects
        start_date_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Call the data fetcher method
        data_objects = data_fetcher.get_overall_adoption_rate(
            from_date=start_date_dt,
            to_date=end_date_dt,
            tenant_id=tenant_id
        )
        
        # Convert data objects to dictionaries
        result = [obj.to_dict() for obj in data_objects]
        
        return result
    
    except Exception as e:
        logger.error(f"Error fetching overall adoption rate data: {e}")
        return {"error": str(e)}


def _execute_get_mau_data(
    start_date: str,
    end_date: str,
    tenant_id: int = 1388
) -> Dict[str, Any]:
    """Execute get_mau_data function"""
    try:
        # Convert string dates to datetime objects
        start_date_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Call the data fetcher method
        data_objects = data_fetcher.get_monthly_active_users(
            from_date=start_date_dt,
            to_date=end_date_dt,
            tenant_id=tenant_id
        )
        
        # Convert data objects to dictionaries
        result = [obj.to_dict() for obj in data_objects]
        
        return result
    
    except Exception as e:
        logger.error(f"Error fetching MAU data: {e}")
        return {"error": str(e)}


def _execute_get_dau_data(
    start_date: str,
    end_date: str,
    tenant_id: int = 1388
) -> Dict[str, Any]:
    """Execute get_dau_data function"""
    try:
        # Convert string dates to datetime objects
        start_date_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Call the data fetcher method
        data_objects = data_fetcher.get_daily_active_users(
            from_date=start_date_dt,
            to_date=end_date_dt,
            tenant_id=tenant_id
        )
        
        # Convert data objects to dictionaries
        result = [obj.to_dict() for obj in data_objects]
        
        return result
    
    except Exception as e:
        logger.error(f"Error fetching DAU data: {e}")
        return {"error": str(e)}


def _execute_analyze_adoption_rate_trend(
    start_date: str,
    end_date: str,
    metric_type: str,
    tenant_id: int = 1388
) -> Dict[str, Any]:
    """Execute analyze_adoption_rate_trend function"""
    try:
        # Convert string dates to datetime objects
        start_date_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Get adoption rate data
        data_objects = data_fetcher.get_overall_adoption_rate(
            from_date=start_date_dt,
            to_date=end_date_dt,
            tenant_id=tenant_id
        )
        
        # Convert to DataFrame for analysis
        data_dicts = [obj.to_dict() for obj in data_objects]
        df = pd.DataFrame(data_dicts)
        df.set_index('date', inplace=True)
        
        # Map metric type to column name
        metric_column_map = {
            "daily": "daily_adoption_rate",
            "weekly": "weekly_adoption_rate",
            "monthly": "monthly_adoption_rate",
            "yearly": "yearly_adoption_rate"
        }
        
        if metric_type not in metric_column_map:
            raise ValueError(f"Invalid metric type: {metric_type}")
        
        column = metric_column_map[metric_type]
        
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in data")
        
        # Analyze trend
        peaks_valleys = detect_peaks_and_valleys(data_objects, rate_type=metric_type)
        trend_info = calculate_trend_line(data_objects, rate_type=metric_type)
        trend_description = generate_trend_description(data_objects, rate_type=metric_type)
        
        # Return combined results
        return {
            "peaks_valleys": peaks_valleys,
            "trend_info": trend_info,
            "trend_description": trend_description,
            "data_summary": {
                "start_date": start_date,
                "end_date": end_date,
                "metric_type": metric_type,
                "data_points": len(df),
                "average": df[column].mean(),
                "min": df[column].min(),
                "max": df[column].max(),
                "latest_value": df[column].iloc[-1] if not df.empty else None
            }
        }
    
    except Exception as e:
        logger.error(f"Error analyzing adoption rate trend: {e}")
        return {"error": str(e)}


def _execute_compare_periods(
    period1_start: str,
    period1_end: str,
    period2_start: str,
    period2_end: str,
    comparison_type: str,
    tenant_id: int = 1388
) -> Dict[str, Any]:
    """Execute compare_periods function"""
    try:
        # Convert string dates to datetime objects
        period1_start_dt = datetime.datetime.strptime(period1_start, "%Y-%m-%d").date()
        period1_end_dt = datetime.datetime.strptime(period1_end, "%Y-%m-%d").date()
        period2_start_dt = datetime.datetime.strptime(period2_start, "%Y-%m-%d").date()
        period2_end_dt = datetime.datetime.strptime(period2_end, "%Y-%m-%d").date()
        
        # Get data for both periods
        period1_data_objects = data_fetcher.get_overall_adoption_rate(
            from_date=period1_start_dt,
            to_date=period1_end_dt,
            tenant_id=tenant_id
        )
        
        period2_data_objects = data_fetcher.get_overall_adoption_rate(
            from_date=period2_start_dt,
            to_date=period2_end_dt,
            tenant_id=tenant_id
        )
        
        if not period1_data_objects or not period2_data_objects:
            return {"error": "One or both periods have no data"}
        
        # Default to monthly rates
        rate_type = 'monthly'
        
        # Perform comparison based on type
        if comparison_type == "mom":
            comparison_result = calculate_mom_change(period2_data_objects, rate_type=rate_type)
        elif comparison_type == "qoq":
            comparison_result = calculate_qoq_change(period2_data_objects, rate_type=rate_type)
        elif comparison_type == "yoy":
            comparison_result = calculate_yoy_change(period2_data_objects, rate_type=rate_type)
        elif comparison_type == "custom":
            # For custom comparison, compare the two periods directly
            comparison_result = compare_time_periods(
                data=period1_data_objects + period2_data_objects,  # Combine data
                period1_start=period1_start_dt,
                period1_end=period1_end_dt,
                period2_start=period2_start_dt,
                period2_end=period2_end_dt,
                rate_type=rate_type
            )
        else:
            raise ValueError(f"Invalid comparison type: {comparison_type}")
        
        # Return comparison results
        return {
            "comparison_type": comparison_type,
            "period1": {
                "start_date": period1_start,
                "end_date": period1_end,
                "data_points": len(period1_data_objects)
            },
            "period2": {
                "start_date": period2_start,
                "end_date": period2_end,
                "data_points": len(period2_data_objects)
            },
            "comparison_result": comparison_result
        }
    
    except Exception as e:
        logger.error(f"Error comparing periods: {e}")
        return {"error": str(e)}


def _execute_detect_anomalies(
    start_date: str,
    end_date: str,
    metric_type: str,
    method: str,
    tenant_id: int = 1388
) -> Dict[str, Any]:
    """Execute detect_anomalies function"""
    try:
        # Convert string dates to datetime objects
        start_date_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Get adoption rate data
        data_objects = data_fetcher.get_overall_adoption_rate(
            from_date=start_date_dt,
            to_date=end_date_dt,
            tenant_id=tenant_id
        )
        
        # Map metric type to rate_type parameter
        rate_type = metric_type  # Most functions use the same name for this parameter
        
        # Detect anomalies based on method
        if method == "zscore":
            anomalies = detect_anomalies_zscore(data_objects, rate_type=rate_type)
        elif method == "ensemble":
            anomalies = detect_anomalies_ensemble(data_objects, rate_type=rate_type)
        else:
            # Default to ensemble method if not implemented
            anomalies = detect_anomalies_ensemble(data_objects, rate_type=rate_type)
        
        # Generate explanations for anomalies
        anomaly_explanations = generate_anomaly_explanation(anomalies, data_objects, rate_type=rate_type)
        
        # Convert to DataFrame for summary statistics
        data_dicts = [obj.to_dict() for obj in data_objects]
        df = pd.DataFrame(data_dicts)
        
        # Map metric type to column name for summary statistics
        metric_column_map = {
            "daily": "daily_adoption_rate",
            "weekly": "weekly_adoption_rate",
            "monthly": "monthly_adoption_rate",
            "yearly": "yearly_adoption_rate"
        }
        
        column = metric_column_map[metric_type]
        
        # Return anomaly detection results
        return {
            "method": method,
            "metric_type": metric_type,
            "data_summary": {
                "start_date": start_date,
                "end_date": end_date,
                "data_points": len(data_objects),
                "average": df[column].mean() if column in df.columns else None,
                "std_dev": df[column].std() if column in df.columns else None
            },
            "anomalies_detected": len(anomalies),
            "anomaly_details": anomaly_explanations
        }
    
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        return {"error": str(e)}


def _execute_analyze_correlations(
    start_date: str,
    end_date: str,
    metrics: List[str],
    tenant_id: int = 1388
) -> Dict[str, Any]:
    """Execute analyze_correlations function"""
    try:
        # Convert string dates to datetime objects
        start_date_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Get adoption rate data
        data_objects = data_fetcher.get_overall_adoption_rate(
            from_date=start_date_dt,
            to_date=end_date_dt,
            tenant_id=tenant_id
        )
        
        if not data_objects:
            return {"error": "No data found for the specified period"}
        
        # Calculate correlation matrix
        correlation_matrix = calculate_correlation_matrix(data_objects)
        
        # Generate correlation summary
        correlation_summary = generate_correlation_summary(correlation_matrix)
        
        # Return correlation analysis results
        return {
            "metrics_analyzed": metrics,
            "data_summary": {
                "start_date": start_date,
                "end_date": end_date,
                "data_points": len(data_objects)
            },
            "correlation_matrix": correlation_matrix,
            "correlation_summary": correlation_summary
        }
    
    except Exception as e:
        logger.error(f"Error analyzing correlations: {e}")
        return {"error": str(e)} 