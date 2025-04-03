"""
API Endpoints for the Overall Adoption Rate Chatbot UI

This module provides API endpoints for accessing adoption rate data and analytics.
"""

import logging
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import plotly.graph_objects as go
import plotly.utils
import random  # For generating fallback data
import math

# Import components from our application
from src.database.data_access import DataAccessLayer
from src.data_processing.data_fetcher import DataFetcher
from src.data_analysis.trend_analyzer import detect_peaks_and_valleys, calculate_trend_line, generate_trend_description
from src.data_analysis.anomaly_detector import detect_anomalies_ensemble, generate_anomaly_explanation
from src.data_analysis.period_analyzer import calculate_mom_change, calculate_qoq_change, calculate_yoy_change
from src.predictive_analytics.predictive_analytics import PredictiveAnalytics
from src.descriptive_analytics.descriptive_analytics import DescriptiveAnalytics
from src.diagnostic_analytics.diagnostic_analytics import DiagnosticAnalytics
from src.prescriptive_analytics.prescriptive_analytics import PrescriptiveAnalytics
from src.data_models.metrics import OverallAdoptionRate

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the blueprint
api_bp = Blueprint('api', __name__)

# Initialize data components
data_access = DataAccessLayer()
data_fetcher = DataFetcher(data_access)
default_tenant_id = 1388  # Default tenant ID for testing

# Initialize analytics components
predictive = PredictiveAnalytics()
descriptive = DescriptiveAnalytics()
diagnostic = DiagnosticAnalytics()
prescriptive = PrescriptiveAnalytics()

# Helper function to convert DataFrame data to proper format for analysis functions
def convert_df_to_adoption_rate_objects(df, rate_field='value'):
    """
    Convert a DataFrame with date and value columns to a list of OverallAdoptionRate objects
    
    Args:
        df: DataFrame with 'date' and value columns
        rate_field: The name of the column containing the rate values
        
    Returns:
        List of OverallAdoptionRate objects
    """
    result = []
    for _, row in df.iterrows():
        # Convert string date to datetime if needed
        if isinstance(row['date'], str):
            date_obj = datetime.strptime(row['date'], '%Y-%m-%d')
        else:
            date_obj = row['date']
            
        # Create an OverallAdoptionRate object
        # Set the rate based on which field we're analyzing
        obj = OverallAdoptionRate(
            date=date_obj,
            daily_adoption_rate=row[rate_field] if rate_field == 'daily_adoption_rate' else 0,
            weekly_adoption_rate=row[rate_field] if rate_field == 'weekly_adoption_rate' else 0,
            monthly_adoption_rate=row[rate_field] if rate_field == 'monthly_adoption_rate' else 0,
            yearly_adoption_rate=row[rate_field] if rate_field == 'yearly_adoption_rate' else 0
        )
        result.append(obj)
    
    return result

@api_bp.route('/chart_data', methods=['GET'])
def get_chart_data():
    """
    Get data for the adoption rate chart
    
    Query parameters:
    - time_range: The time range to fetch (1m, 3m, 6m, 1y, 2y, all)
    - metric_type: The type of metric to fetch (monthly, daily, weekly, yearly)
    """
    try:
        # Get query parameters
        time_range = request.args.get('time_range', '1y')
        metric_type = request.args.get('metric_type', 'monthly')
        
        logger.info(f"[DEBUG] Chart data request: time_range={time_range}, metric_type={metric_type}")
        
        # Calculate date range
        to_date = datetime.now()
        
        if time_range == '1m':
            from_date = to_date - timedelta(days=30)
        elif time_range == '3m':
            from_date = to_date - timedelta(days=90)
        elif time_range == '6m':
            from_date = to_date - timedelta(days=180)
        elif time_range == '1y':
            from_date = to_date - timedelta(days=365)
        elif time_range == '2y':
            from_date = to_date - timedelta(days=730)
        else:  # 'all'
            from_date = datetime(2022, 1, 1)  # Starting from beginning of available data
        
        logger.info(f"[DEBUG] Date range calculated: from_date={from_date}, to_date={to_date}")
        
        # Try to fetch real data first
        try:
            logger.info(f"[DEBUG] Fetching chart data for {metric_type} from {from_date} to {to_date}, tenant: {default_tenant_id}")
            data = data_fetcher.get_overall_adoption_rate(from_date, to_date, default_tenant_id)
            
            # Log detailed data information
            logger.info(f"[DEBUG] Data type returned: {type(data)}")
            logger.info(f"[DEBUG] Data length: {len(data) if data is not None else 'None'}")
            if data and len(data) > 0:
                logger.info(f"[DEBUG] First row sample: {data[0].__dict__ if hasattr(data[0], '__dict__') else data[0]}")
            
            if data is None or (hasattr(data, 'empty') and data.empty) or (isinstance(data, list) and len(data) == 0):
                logger.warning("[DEBUG] No adoption rate data returned from data_fetcher, using fallback data")
                # Use fallback data
                chart_data = generate_fallback_chart_data(from_date, to_date, metric_type)
                logger.info(f"[DEBUG] Generated fallback data: {len(chart_data)} points")
                return jsonify({
                    'chart_data': chart_data,
                    'trends': [],
                    'time_range': time_range,
                    'metric_type': metric_type,
                    'is_fallback': True
                })
            
            # Format data for the chart
            chart_data = []
            
            # Map metric_type to the appropriate database field
            field_mapping = {
                'daily': 'daily_adoption_rate',
                'weekly': 'weekly_adoption_rate',
                'monthly': 'monthly_adoption_rate',
                'yearly': 'yearly_adoption_rate'
            }
            
            rate_field = field_mapping.get(metric_type)
            if not rate_field:
                logger.error(f"[DEBUG] Invalid metric_type: {metric_type}")
                return jsonify({'error': f'Invalid metric_type: {metric_type}'}), 400
            
            logger.info(f"[DEBUG] Processing data using rate field: {rate_field}")
            
            # Process the data into chart format
            for item in data:
                try:
                    # Get the value based on the metric type
                    if rate_field == 'daily_adoption_rate':
                        value = item.daily_adoption_rate
                    elif rate_field == 'weekly_adoption_rate':
                        value = item.weekly_adoption_rate
                    elif rate_field == 'monthly_adoption_rate':
                        value = item.monthly_adoption_rate
                    elif rate_field == 'yearly_adoption_rate':
                        value = item.yearly_adoption_rate
                    else:
                        value = 0
                    
                    # Format date to string
                    date_str = item.date.strftime('%Y-%m-%d') if hasattr(item.date, 'strftime') else str(item.date)
                    
                    chart_data.append({
                        'date': date_str,
                        'value': value
                    })
                except Exception as e:
                    logger.error(f"[DEBUG] Error processing chart data item: {str(e)}")
            
            logger.info(f"[DEBUG] Processed {len(chart_data)} chart data points")
            
            if not chart_data:
                logger.warning("[DEBUG] No chart data could be extracted, using fallback data")
                chart_data = generate_fallback_chart_data(from_date, to_date, metric_type)
                return jsonify({
                    'chart_data': chart_data,
                    'trends': [],
                    'time_range': time_range,
                    'metric_type': metric_type,
                    'is_fallback': True
                })
            
            # Log a sample of the processed chart data
            if chart_data:
                logger.info(f"[DEBUG] Chart data sample (first 3 points): {chart_data[:3]}")
            
            # Add trend analysis (simplified for now until we debug the core chart functionality)
            trends = []
            
            # Return the data
            result = {
                'chart_data': chart_data,
                'trends': trends,
                'time_range': time_range,
                'metric_type': metric_type,
                'is_fallback': False
            }
            
            logger.info(f"[DEBUG] Successfully returning chart data with {len(chart_data)} points, is_fallback=False")
            return jsonify(result)
        
        except Exception as db_error:
            logger.error(f"[DEBUG] Error fetching from database: {str(db_error)}", exc_info=True)
            # Use fallback data
            chart_data = generate_fallback_chart_data(from_date, to_date, metric_type)
            logger.info(f"[DEBUG] Generated fallback data after exception: {len(chart_data)} points")
            return jsonify({
                'chart_data': chart_data,
                'trends': [],
                'time_range': time_range,
                'metric_type': metric_type,
                'is_fallback': True
            })
        
    except Exception as e:
        logger.error(f"[DEBUG] Error in chart_data endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def generate_fallback_chart_data(from_date, to_date, metric_type):
    """
    Generate fallback chart data when database data is not available
    
    Args:
        from_date: Start date for the data
        to_date: End date for the data
        metric_type: Type of metric (daily, weekly, monthly, yearly)
        
    Returns:
        List of dictionaries with date and value keys
    """
    logger.info(f"Generating fallback data for {metric_type} from {from_date} to {to_date}")
    
    # Determine date increment based on metric type
    if metric_type == 'daily':
        date_increment = timedelta(days=1)
        date_count = (to_date - from_date).days + 1
    elif metric_type == 'weekly':
        date_increment = timedelta(days=7)
        date_count = ((to_date - from_date).days // 7) + 1
    elif metric_type == 'monthly':
        # Approximate monthly increment
        date_count = ((to_date.year - from_date.year) * 12 + to_date.month - from_date.month) + 1
        date_increment = timedelta(days=30)  # Approximation
    else:  # yearly
        date_count = to_date.year - from_date.year + 1
        date_increment = timedelta(days=365)  # Approximation
    
    # Cap at a reasonable number
    date_count = min(date_count, 300)
    
    # Generate dates
    dates = []
    current_date = from_date
    for _ in range(date_count):
        dates.append(current_date)
        current_date += date_increment
    
    # Generate realistic adoption rate data (example pattern with growth)
    chart_data = []
    base_value = 5.0  # Starting value around 5%
    
    # Add some realistic patterns
    values = []
    
    # Create a rising trend with some variations
    for i in range(len(dates)):
        # Base trend: slight increase over time
        trend = base_value + (i / len(dates)) * 10
        
        # Add some cyclical pattern
        cyclical = 2 * math.sin(i / 10)
        
        # Add some random noise
        noise = random.uniform(-1, 1)
        
        # Combine components
        value = max(0, min(100, trend + cyclical + noise))
        values.append(value)
    
    # Create the chart data
    for i, date in enumerate(dates):
        chart_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'value': values[i]
        })
    
    return chart_data

@api_bp.route('/trend_analysis', methods=['GET'])
def get_trend_analysis():
    """
    Get trend analysis for adoption rate data
    
    Query parameters:
    - from_date: Start date (YYYY-MM-DD)
    - to_date: End date (YYYY-MM-DD)
    - metric_type: The type of metric to analyze (monthly, daily, weekly, yearly)
    """
    try:
        # Get query parameters
        from_date_str = request.args.get('from_date')
        to_date_str = request.args.get('to_date')
        metric_type = request.args.get('metric_type', 'monthly')
        
        # Parse dates
        if from_date_str:
            from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
        else:
            from_date = datetime.now() - timedelta(days=365)
            
        if to_date_str:
            to_date = datetime.strptime(to_date_str, '%Y-%m-%d')
        else:
            to_date = datetime.now()
            
        # Get descriptive analysis
        trend_description = descriptive.verbalize_trend(from_date, to_date, metric_type)
        summary_stats = descriptive.generate_summary_statistics(from_date, to_date, metric_type)
        
        return jsonify({
            'trend_description': trend_description,
            'summary_statistics': summary_stats,
            'from_date': from_date.strftime('%Y-%m-%d'),
            'to_date': to_date.strftime('%Y-%m-%d'),
            'metric_type': metric_type
        })
        
    except Exception as e:
        logger.error(f"Error generating trend analysis: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@api_bp.route('/anomaly_detection', methods=['GET'])
def get_anomalies():
    """
    Get anomalies in adoption rate data
    
    Query parameters:
    - from_date: Start date (YYYY-MM-DD)
    - to_date: End date (YYYY-MM-DD)
    - metric_type: The type of metric to analyze (monthly, daily, weekly, yearly)
    """
    try:
        # Get query parameters
        from_date_str = request.args.get('from_date')
        to_date_str = request.args.get('to_date')
        metric_type = request.args.get('metric_type', 'monthly')
        
        # Parse dates
        if from_date_str:
            from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
        else:
            from_date = datetime.now() - timedelta(days=365)
            
        if to_date_str:
            to_date = datetime.strptime(to_date_str, '%Y-%m-%d')
        else:
            to_date = datetime.now()
            
        # Get anomalies using diagnostic analytics
        anomalies = diagnostic.explain_anomalies(from_date, to_date, metric_type)
        
        return jsonify({
            'anomalies': anomalies,
            'from_date': from_date.strftime('%Y-%m-%d'),
            'to_date': to_date.strftime('%Y-%m-%d'),
            'metric_type': metric_type
        })
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@api_bp.route('/forecasts', methods=['GET'])
def get_forecasts():
    """
    Get adoption rate forecasts
    
    Query parameters:
    - from_date: Start date (YYYY-MM-DD)
    - to_date: End date (YYYY-MM-DD)
    - forecast_periods: Number of periods to forecast
    - metric_type: The type of metric to forecast (monthly, daily, weekly, yearly)
    """
    try:
        # Get query parameters
        from_date_str = request.args.get('from_date')
        to_date_str = request.args.get('to_date')
        forecast_periods = int(request.args.get('forecast_periods', '6'))
        metric_type = request.args.get('metric_type', 'monthly')
        
        # Parse dates
        if from_date_str:
            from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
        else:
            from_date = datetime.now() - timedelta(days=365)
            
        if to_date_str:
            to_date = datetime.strptime(to_date_str, '%Y-%m-%d')
        else:
            to_date = datetime.now()
            
        # Get forecast
        forecast_result = predictive.forecast_adoption_rate(
            from_date, 
            to_date, 
            forecast_periods, 
            metric_type
        )
        
        # Format data for response
        forecast_data = []
        if 'forecast_values' in forecast_result:
            for date, value in forecast_result['forecast_values'].items():
                forecast_data.append({
                    'date': date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date,
                    'value': value
                })
        
        # Format confidence intervals for response
        confidence_intervals = []
        if 'confidence_intervals' in forecast_result:
            for date, (lower, upper) in forecast_result['confidence_intervals'].items():
                confidence_intervals.append({
                    'date': date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date,
                    'lower': lower,
                    'upper': upper
                })
        
        return jsonify({
            'forecast_data': forecast_data,
            'confidence_intervals': confidence_intervals,
            'explanation': forecast_result.get('explanation', ''),
            'from_date': from_date.strftime('%Y-%m-%d'),
            'to_date': to_date.strftime('%Y-%m-%d'),
            'forecast_periods': forecast_periods,
            'metric_type': metric_type
        })
        
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@api_bp.route('/recommendations', methods=['GET'])
def get_recommendations():
    """
    Get recommendations to improve adoption rate
    
    Query parameters:
    - from_date: Start date (YYYY-MM-DD)
    - to_date: End date (YYYY-MM-DD)
    - metric_type: The type of metric to analyze (monthly, daily, weekly, yearly)
    """
    try:
        # Get query parameters
        from_date_str = request.args.get('from_date')
        to_date_str = request.args.get('to_date')
        metric_type = request.args.get('metric_type', 'monthly')
        
        # Parse dates
        if from_date_str:
            from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
        else:
            from_date = datetime.now() - timedelta(days=365)
            
        if to_date_str:
            to_date = datetime.strptime(to_date_str, '%Y-%m-%d')
        else:
            to_date = datetime.now()
            
        # Get recommendations
        recommendations = prescriptive.generate_recommendations(from_date, to_date, metric_type)
        
        # Get action suggestions
        actions = prescriptive.suggest_actions(from_date, to_date, metric_type)
        
        return jsonify({
            'recommendations': recommendations,
            'actions': actions,
            'from_date': from_date.strftime('%Y-%m-%d'),
            'to_date': to_date.strftime('%Y-%m-%d'),
            'metric_type': metric_type
        })
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@api_bp.route('/goals', methods=['GET'])
def get_goals():
    """
    Get goal setting assistance
    
    Query parameters:
    - from_date: Start date (YYYY-MM-DD)
    - to_date: End date (YYYY-MM-DD)
    - metric_type: The type of metric to analyze (monthly, daily, weekly, yearly)
    - goal_type: Type of goal (short_term, medium_term, long_term, custom)
    - timeframe: Timeframe in months for the goal
    - custom_target: Custom target value (if goal_type is 'custom')
    """
    try:
        # Get query parameters
        from_date_str = request.args.get('from_date')
        to_date_str = request.args.get('to_date')
        metric_type = request.args.get('metric_type', 'monthly')
        goal_type = request.args.get('goal_type', 'medium_term')
        timeframe = int(request.args.get('timeframe', '6'))
        custom_target = request.args.get('custom_target')
        
        if custom_target:
            custom_target = float(custom_target)
        
        # Parse dates
        if from_date_str:
            from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
        else:
            from_date = datetime.now() - timedelta(days=365)
            
        if to_date_str:
            to_date = datetime.strptime(to_date_str, '%Y-%m-%d')
        else:
            to_date = datetime.now()
            
        # Get goal setting assistance
        goals = prescriptive.assist_goal_setting(
            from_date, 
            to_date, 
            metric_type, 
            goal_type, 
            timeframe, 
            custom_target
        )
        
        return jsonify({
            'goals': goals,
            'from_date': from_date.strftime('%Y-%m-%d'),
            'to_date': to_date.strftime('%Y-%m-%d'),
            'metric_type': metric_type,
            'goal_type': goal_type,
            'timeframe': timeframe
        })
        
    except Exception as e:
        logger.error(f"Error generating goals: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500 