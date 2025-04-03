# Detailed Task List for LLM-Powered Overall Adoption Rate Chatbot

## 1. Database Connection and Data Retrieval Setup
- [X] Set up connection string to the Opus database (Server=opusdatdev.database.windows.net;Initial Catalog=opusdatdev;User ID=opusdatdev)
- [X] Create a secure method to store and access database credentials
- [X] Implement connection pooling to optimize database access
- [X] Create a data access layer to interface with the database stored procedures
- [X] Implement error handling and logging for database connections
- [X] Test database connection with basic queries to verify accessibility

### Implementation Details

#### Files Created:
1. `src/config/db_config.py`:
   - Manages database configuration and credentials
   - Key function: `get_connection_string()` - Generates a connection string from config

2. `src/database/connection.py`:
   - Handles database connections with connection pooling
   - Key functions:
     - `get_connection()` - Gets a connection from the pool or creates a new one
     - `release_connection(connection)` - Returns a connection to the pool
     - `db_connection()` - Context manager for safe connection handling

3. `src/database/data_access.py`:
   - Data Access Layer for interacting with stored procedures
   - Key functions:
     - `execute_stored_procedure(proc_name, params)` - Executes any stored procedure
     - `get_overall_adoption_rate(from_date, to_date, tenant_id)` - Gets Overall Adoption Rate data
     - `get_mau(from_date, to_date, tenant_id)` - Gets Monthly Active Users data
     - `get_dau(from_date, to_date, tenant_id)` - Gets Daily Active Users data

4. `dummy_scripts/explore_database.py`:
   - Script for exploring database schema and testing connections
   - Verifies accessibility of all required tables and stored procedures

#### Key Findings:
- Stored procedures require specific parameters:
  - @FromDate (datetime)
  - @ToDate (datetime)
  - @Tenantid (int)
- Tenant ID 1388 has been identified as having test data available
- Data is available from 2022-09-06 to present

## 2. Data Extraction and Processing
- [X] Identify and document the specific stored procedures needed for Overall Adoption Rate data
- [X] Create a data model that represents the chart's three key metrics (Overall Adoption Rate, MAU, DAU)
- [X] Implement data fetching logic to retrieve historical adoption rate data by time period
- [X] Create preprocessing functions to format dates consistently (from format shown in chart: YY-MM)
- [X] Build aggregation functions for calculating metrics at different time intervals (daily, weekly, monthly)
- [X] Implement caching mechanism for frequently accessed data to improve performance
- [X] Create data validation methods to ensure quality and consistency

### Implementation Details

#### Files Created:
1. `src/data_models/metrics.py`:
   - Data models for representing metric data from the database
   - Key classes:
     - `DailyActiveUsers` - Represents DAU data for a specific date
     - `MonthlyActiveUsers` - Represents MAU data for a specific year-month
     - `OverallAdoptionRate` - Represents adoption rate data across different time intervals
     - `MetricCollection` - Container for multiple metrics for analysis
   - Each class includes:
     - Type-safe attributes
     - Validation methods
     - Factory methods to create instances from database rows

2. `src/data_processing/data_fetcher.py`:
   - High-level interface for fetching and processing metric data
   - Key functions:
     - `get_daily_active_users(from_date, to_date, tenant_id)` - Fetches DAU data
     - `get_monthly_active_users(from_date, to_date, tenant_id)` - Fetches MAU data
     - `get_overall_adoption_rate(from_date, to_date, tenant_id)` - Fetches Overall Adoption Rate data
     - `get_all_metrics(from_date, to_date, tenant_id)` - Fetches all metrics in a single call
     - `get_metrics_for_period(period, reference_date, tenant_id)` - Gets metrics for predefined periods
   - Implements in-memory caching for recent queries

3. `src/data_processing/data_processor.py`:
   - Processes metric data for analysis and visualization
   - Key functions:
     - `aggregate_daily_metrics_to_weekly/monthly` - Aggregates metrics across time periods
     - `format_*_for_chart` - Formats data for visualization
     - `calculate_adoption_rate_statistics` - Calculates key statistics
     - `get_formatted_trend_description` - Generates natural language trend descriptions

4. `dummy_scripts/test_data_processing.py`:
   - Tests data processing functionality
   - Systematically tests the stored procedure parameter combinations
   - Confirms working parameter combinations for each stored procedure

5. `dummy_scripts/test_working_solution.py`:
   - Demonstrates the complete working solution
   - Fetches and processes all three metrics
   - Calculates statistics and generates visualizations

#### Key Findings:
- SP_OverallAdoptionRate_DWMY only accepts three parameters (FromDate, ToDate, Tenantid)
- Adoption rate values are stored as percentages (0-100%)
- NaN values in the database need special handling in the data models
- Tenant ID 1388 has comprehensive test data for all metrics

## 3. Chart Data Analysis Components
- [X] Develop functions to identify peaks and valleys in the adoption rate trend
- [X] Create logic to calculate period-over-period changes (MoM, QoQ, YoY)
- [X] Implement statistical analysis functions to identify significant trends
- [X] Build correlation analysis between Overall Adoption Rate, MAU, and DAU
- [X] Create functions to identify anomalies or outliers in the data
- [X] Develop logic to segment data by business units or time periods as shown in filter options
- [X] Implement calculation for adoption rate benchmarks and comparisons

### Implementation Details

#### Files Created:
1. `src/data_analysis/trend_analyzer.py`:
   - Functions for analyzing trends in adoption rate metrics
   - Key functions:
     - `detect_peaks_valleys(data)` - Identifies local maxima and minima in the time series
     - `calculate_trend(data)` - Determines trend direction and strength
     - `generate_trend_description(data)` - Creates human-readable trend descriptions
     - `analyze_seasonal_patterns(data)` - Identifies recurring patterns in the data
     - `visualize_trend_analysis(data)` - Creates plots showing trend components

2. `src/data_analysis/period_analyzer.py`:
   - Functions for comparing metrics across different time periods
   - Key functions:
     - `calculate_mom_change(data)` - Month-over-Month changes
     - `calculate_qoq_change(data)` - Quarter-over-Quarter changes
     - `calculate_yoy_change(data)` - Year-over-Year changes
     - `compare_periods(period1_data, period2_data)` - Compares metrics between two custom periods
     - `generate_period_comparison_summary(data)` - Creates detailed comparison reports
     - `visualize_period_comparison(data)` - Visualizes period-over-period changes

3. `src/data_analysis/anomaly_detector.py`:
   - Functions for identifying and explaining outliers in the data
   - Key functions:
     - `detect_anomalies_zscore(data)` - Z-score based anomaly detection
     - `detect_anomalies_iqr(data)` - Inter-quartile range based detection
     - `detect_anomalies_moving_average(data)` - Moving average based detection
     - `ensemble_anomaly_detection(data)` - Combines multiple detection methods
     - `explain_anomaly(data, anomaly_point)` - Generates natural language explanations
     - `visualize_anomalies(data, anomalies)` - Creates plots highlighting anomalies

4. `src/data_analysis/correlation_analyzer.py`:
   - Functions for analyzing relationships between metrics
   - Key functions:
     - `calculate_correlation_matrix(metrics_data)` - Computes correlations between all metrics
     - `calculate_metric_correlation(metric1, metric2)` - Targeted correlation analysis
     - `analyze_lead_lag_relationship(metric1, metric2)` - Identifies leading/lagging relationships
     - `generate_correlation_summary(metrics_data)` - Creates human-readable correlation summaries
     - `visualize_correlation_matrix(correlation_matrix)` - Creates heatmap visualizations

5. `src/data_analysis/__init__.py`:
   - Makes all analysis modules easily importable
   - Exposes key functions at the package level for simplified access

6. `dummy_scripts/test_data_analysis.py`:
   - Demonstrates all data analysis capabilities
   - Tests each analysis function with real data
   - Generates visualization outputs in the plots/analysis directory

#### Key Findings:
- Monthly adoption rates show clearer trends than daily rates
- Significant anomalies detected in March 2023 (unusually low) and March 2025 (unusually high)
- Strong correlation between Monthly Active Users and Monthly Adoption Rate (r=1.0)
- Moderate correlation between Daily and Monthly Adoption Rates (r=0.336)
- Year-over-Year comparison shows a 45.77% increase in adoption rate for March 2025 vs March 2024

## 4. LLM Integration Setup
- [X] Select and set up appropriate LLM service (e.g., OpenAI GPT, )
- [X] Design prompt engineering templates for different query types
- [X] Create context injection mechanism to provide chart data to the LLM
- [X] Implement function calling to allow LLM to access database queries when needed
- [X] Build message history management for conversational context
- [X] Develop response formatting logic to ensure consistency in answers
- [X] Create fallback mechanisms for when the LLM cannot answer a question

### Implementation Details

#### Files Created:
1. `src/llm_integration/llm_service.py`:
   - Core service for interfacing with LLM providers (OpenAI and Anthropic)
   - Key functions:
     - `get_llm_client(provider)` - Gets authenticated client for specified provider
     - `get_openai_client()` - Gets OpenAI client with API key authentication
     - `get_anthropic_client()` - Gets Anthropic client with API key authentication
     - `generate_response(messages, functions, provider)` - Generates responses from specified LLM
     - `generate_openai_response()` / `generate_anthropic_response()` - Provider-specific implementations
   - Supports function calling with both OpenAI and Anthropic

2. `src/llm_integration/prompts.py`:
   - Templates and utilities for different prompt types
   - Key functions:
     - `get_system_prompt()` - Gets the standard system prompt defining the chatbot's role
     - `get_descriptive_prompt_template()` - Template for "what happened" queries
     - `get_diagnostic_prompt_template()` - Template for "why something happened" queries
     - `get_predictive_prompt_template()` - Template for "what will happen" queries
     - `get_prescriptive_prompt_template()` - Template for "what should be done" queries
     - `determine_prompt_type(user_query)` - Analyzes query to determine appropriate template

3. `src/llm_integration/context_manager.py`:
   - Manages conversation history and context
   - Key components:
     - `MessageHistory` class - Stores and manages conversation messages
     - `add_message()` - Adds a message to the history with support for function messages
     - `trim_context_to_max_tokens()` - Ensures context stays within token limits
     - `get_context_window()` - Gets current context for LLM request
     - `add_data_to_context()` - Adds chart data to the context
     - `add_chart_reference()` - Adds references to visualizations

4. `src/llm_integration/function_calling.py`:
   - Implements function calling to access data and analysis
   - Key components:
     - `FUNCTION_DEFINITIONS` - Definitions for all available functions
     - `get_function_definitions()` - Gets function schemas for LLM requests
     - `format_function_call()` - Extracts function call information from LLM response
     - `execute_function()` - Executes the requested function
     - `handle_function_response()` - Formats function result for LLM context with proper function name
     - Function implementations for all data retrieval and analysis capabilities

5. `src/llm_integration/response_formatter.py`:
   - Ensures consistent formatting of LLM responses
   - Key functions:
     - `format_response()` - Formats the LLM response for consistency
     - `add_chart_references()` - Adds relevant chart references based on content
     - `format_numbers()` - Formats numbers consistently
     - `format_key_metrics()` - Highlights key metrics in responses
     - `format_anomalies()` - Formats anomalies consistently
     - `format_date_references()` - Ensures dates are formatted consistently

6. `src/llm_integration/__init__.py`:
   - Package initialization with exports of key functions
   - Makes all components easily accessible from a single import

7. `dummy_scripts/test_llm_integration.py`:
   - Tests the LLM integration functionality
   - Key test functions:
     - `test_basic_response()` - Tests basic LLM responses
     - `test_function_calling()` - Tests function calling capability
     - `test_prompt_templates()` - Tests prompt template selection
     - `test_data_fetcher()` - Tests data fetching from database

8. `dummy_scripts/init_env.py`:
   - Utility script for setting up environment variables
   - Creates .env file with API keys and database parameters
   - Tests environment configuration

#### Key Design Choices:
- Support for multiple LLM providers (OpenAI and Anthropic) for redundancy
- Token-aware context management to stay within LLM context limits
- Function calling for data retrieval and complex analysis
- Proper handling of OpenAI and Anthropic API formats for function calling
- Type conversion and validation for robust data handling
- Data model serialization with to_dict methods for API compatibility

#### Function Calling Capabilities:
- Data retrieval functions:
  - `get_overall_adoption_rate_data(start_date, end_date, tenant_id)`
  - `get_mau_data(start_date, end_date, tenant_id)`
  - `get_dau_data(start_date, end_date, tenant_id)`
- Analysis functions:
  - `analyze_adoption_rate_trend(start_date, end_date, metric_type, tenant_id)`
  - `compare_periods(period1_start, period1_end, period2_start, period2_end, comparison_type, tenant_id)`
  - `detect_anomalies(start_date, end_date, metric_type, method, tenant_id)`
  - `analyze_correlations(start_date, end_date, metrics, tenant_id)`

## 5. Natural Language Query Processing
- [X] Build intent recognition system to categorize user questions (descriptive, diagnostic, predictive, prescriptive)
- [X] Implement entity extraction to identify time periods, metrics, and other parameters mentioned
- [X] Create query transformation logic to convert natural language to database queries
- [X] Develop context awareness to handle follow-up questions
- [X] Build question validation to ensure queries are relevant to the adoption rate chart
- [X] Implement query refinement suggestions for ambiguous questions
- [X] Create specialized handlers for different question types

### Implementation Details

#### Files Created:
1. `src/llm_integration/query_processor.py`:
   - Core component for natural language query processing
   - Key components:
     - `IntentClassifier` - Classifies queries into descriptive, diagnostic, predictive, or prescriptive intents
     - `EntityExtractor` - Extracts dates, metrics, comparisons, and other entities from queries
     - `QueryValidator` - Validates queries for domain relevance and completeness
     - `QueryTransformer` - Converts extracted entities into database query parameters
     - `ContextTracker` - Manages conversation context for follow-up questions
     - `QueryProcessor` - Integrates all components for end-to-end query processing
   - Supports advanced NLP tasks like entity extraction, intent classification, and query refinement

2. `src/chatbot.py`:
   - Main chatbot implementation that integrates all components
   - Key components:
     - `AdoptionRateChatbot` class - Core chatbot implementation
     - `process_query()` - Processes user queries and generates responses
     - `create_chatbot()` - Factory function for creating chatbot instances
   - Integrates with database, LLM, and all analytical components
   - Implements the complete conversation flow from query to response

3. `dummy_scripts/test_query_processor.py`:
   - Tests the natural language query processing functionality
   - Key test functions:
     - `test_intent_classification()` - Tests intent classification
     - `test_entity_extraction()` - Tests entity extraction
     - `test_query_validation()` - Tests query validation
     - `test_query_transformation()` - Tests query transformation
     - `test_context_tracking()` - Tests context awareness
     - `test_full_processor()` - Tests end-to-end query processing

4. `dummy_scripts/test_chatbot.py`:
   - Tests the complete chatbot functionality
   - Tests various query types (descriptive, diagnostic, predictive, prescriptive)
   - Tests follow-up questions and context awareness
   - Tests error handling for invalid/incomplete queries

5. `dummy_scripts/test_simple_chatbot.py`:
   - Simplified test script for isolating and debugging core chatbot functionality
   - Tests message history management and LLM response generation
   - Useful for diagnosing integration issues without complex dependencies

#### Key Features:
- Intent classification using keyword patterns and fallback to model-based classification
- Entity extraction for dates, metrics, time periods, anomalies, trends, and targets
- Support for complex date formats (ISO, month-year, relative times, quarters)
- Query validation for domain relevance and completeness
- Refinement suggestions for incomplete queries
- Query transformation to convert natural language to database parameters
- Context tracking for follow-up questions
- Integration with LLM for generating responses
- Support for function calling to retrieve data from the database
- Error handling and fallback mechanisms

#### Debugging and Fixes:
- Fixed string buffer error in message history tokenization by handling non-string values
- Added custom JSON encoder for handling datetime objects in function responses
- Implemented proper error handling in the chatbot process_query method
- Enhanced the add_message method to handle different message formats
- Created robust message content validation to prevent type errors

#### Entity Types Supported:
- Date entities: ISO dates, month-year, quarters, relative times (last month, next 3 weeks)
- Metric entities: adoption rate, DAU, WAU, MAU, YAU
- Comparison entities: month-over-month, quarter-over-quarter, year-over-year
- Time period entities: daily, weekly, monthly, quarterly, yearly
- Anomaly entities: anomalies, outliers, spikes, drops
- Trend entities: trends, patterns, trajectories
- Target entities: numeric targets, percentage targets

## 6. Descriptive Analytics Capabilities
- [X] Implement logic to describe current adoption rates and trends
- [X] Create summary statistics generation for any time period
- [X] Build comparison functionality between different time periods
- [X] Develop metric explanation capability to define MAU, DAU, and Overall Adoption Rate
- [X] Add functionality to identify highest and lowest points in the chart
- [X] Implement trend verbalization to describe patterns in natural language
- [X] Create data contextualization features to explain what the numbers mean

### Implementation Details

#### Files Created:
1. `src/descriptive_analytics/descriptive_analytics.py`:
   - Main class for descriptive analytics functionality
   - Key methods:
     - `describe_current_state(as_of_date=None)` - Describes current adoption rates with historical context
     - `generate_summary_statistics(from_date, to_date, metric_type)` - Generates detailed statistics
     - `verbalize_trend(from_date, to_date, metric_type)` - Creates natural language trend descriptions
     - `compare_periods(period1_start, period1_end, period2_start, period2_end)` - Compares metrics between time periods
     - `compare_to_target(target_value, as_of_date, metric_type)` - Compares current metrics to target values
     - `explain_adoption_metric(metric_type)` - Provides definitions and context for metrics
     - `detect_anomalies(from_date, to_date, metric_type, threshold)` - Identifies outliers in the data
     - `generate_future_outlook(from_date, to_date, metric_type, forecast_periods)` - Projects future trends

2. `src/descriptive_analytics/statistics.py`:
   - Statistical functions for descriptive analytics
   - Key functions:
     - `generate_summary_statistics(data, rate_column)` - Calculates basic statistics
     - `generate_period_statistics(data, period_type, rate_column)` - Creates period-specific statistics
     - `identify_extrema(data, rate_column, top_n)` - Identifies peaks and valleys
     - `calculate_statistical_trends(data, rate_column)` - Calculates trend statistics

3. `src/descriptive_analytics/trend_verbalization.py`:
   - Natural language generation for trends and patterns
   - Key functions:
     - `verbalize_trend(data, metric_type)` - Creates natural language trend descriptions
     - `verbalize_period_comparison(data1, data2, period1_label, period2_label)` - Describes period comparisons
     - `verbalize_anomalies(data, threshold)` - Describes anomalies in the data
     - `generate_future_outlook(data, periods)` - Generates outlook descriptions

4. `src/descriptive_analytics/current_state.py`:
   - Functions for describing current state metrics
   - Key function: `describe_current_state(data, as_of_date)` - Creates current state summaries

5. `src/descriptive_analytics/descriptive_statistics.py`:
   - Additional statistical functions
   - Key functions:
     - `generate_summary_statistics(data, metric_type)` - Generates comprehensive statistics
     - `generate_period_statistics(data, period_type)` - Statistics for specific periods
     - `identify_extrema(data, metric_type)` - Finds highest and lowest points
     - `calculate_statistical_trends(data, metric_type)` - Calculates trends with statistical significance

6. `dummy_scripts/test_descriptive_analytics.py`:
   - Test script for descriptive analytics module
   - Tests all key functionalities including current state description, summary statistics,
     trend verbalization, period comparisons, target comparisons, and metric explanations

#### Key Features:
- Current state analysis with historical context and comparisons
- Comprehensive statistical analysis (mean, median, standard deviation, etc.)
- Identification of extrema (peaks, valleys, all-time highs and lows)
- Trend analysis with statistical significance testing
- Period-over-period comparisons (month, quarter, year)
- Target value comparisons with gap analysis
- Natural language descriptions of all metrics and trends
- Anomaly detection with configurable thresholds
- Future trend projections with confidence intervals
- Business context generation for metric values

#### Data Handling Improvements:
- Robust handling of datetime conversions for the Date column
- Proper serialization of various data types (datetime, numpy types, booleans)
- Custom DateTimeEncoder for JSON serialization of complex objects
- Comprehensive error handling with informative logging

#### Insights Generated:
- Current adoption rate values with percent change from previous periods
- Historical comparisons showing relative performance
- Statistical summaries with mean, median, range, and variance
- Trend directions with significance testing
- Period analyses showing best and worst performing timeframes
- Anomaly detection results with natural language explanations
- Human-readable metric definitions and contextual explanations

## 7. Diagnostic Analytics Capabilities
- [X] Develop causal analysis functions to explain changes in adoption rates
- [X] Create correlation identification between metrics
- [X] Implement root cause suggestion algorithms based on available data
- [X] Build comparative analysis between different segments or time periods
- [X] Develop anomaly explanation capabilities
- [X] Create logic to connect adoption rate changes with potential business events
- [X] Implement change-point detection to identify when trends shifted

### Implementation Details

#### Files Created:
1. `src/diagnostic_analytics/diagnostic_analytics.py`:
   - Main class for diagnostic analytics functionality
   - Key methods:
     - `analyze_rate_changes(from_date, to_date, metric_type)` - Analyzes significant changes in adoption rates
     - `identify_correlated_metrics(from_date, to_date)` - Identifies correlations between different metrics
     - `suggest_root_causes(input_date, metric_type)` - Suggests potential causes for changes at a specific date
     - `compare_segments(from_date, to_date, segment_type)` - Compares performance across different segments
     - `explain_anomalies(from_date, to_date, metric_type)` - Provides explanations for detected anomalies
     - `analyze_change_points(from_date, to_date, metric_type)` - Detects and analyzes trend shift points
     - `_generate_root_cause_suggestions(input_date, changes, correlations, anomalies, trends)` - Creates specific suggestions
     - `_generate_root_cause_explanation(input_date, suggestions, changes, correlations, anomalies, trends)` - Generates natural language explanations

2. `src/diagnostic_analytics/correlation_analysis.py`:
   - Functions for analyzing correlations between metrics and events
   - Key functions:
     - `identify_correlated_metrics(metrics_data)` - Identifies correlations between metrics
     - `identify_leading_indicators(metrics_data)` - Finds metrics that predict others
     - `calculate_cross_correlation(series1, series2, max_lag)` - Calculates correlations with time lags
     - `generate_correlation_explanation(correlation_results)` - Creates human-readable explanations

3. `src/diagnostic_analytics/root_cause_analysis.py`:
   - Functions for identifying potential root causes of changes
   - Key functions:
     - `identify_potential_causes(date, metrics_data, window_size)` - Identifies potential causes for a change
     - `rank_causes_by_likelihood(causes, metrics_data)` - Ranks potential causes by likelihood
     - `generate_cause_effect_chains(primary_cause, metrics_data)` - Generates causal chains
     - `explain_root_cause(cause, context)` - Creates natural language explanations

4. `src/diagnostic_analytics/change_point_detection.py`:
   - Functions for detecting significant changes in trends
   - Key functions:
     - `detect_change_points(time_series, method)` - Detects points where trends change
     - `analyze_change_point(time_series, change_point)` - Analyzes characteristics of a change point
     - `calculate_pre_post_change_statistics(time_series, change_point)` - Calculates statistics before and after
     - `generate_change_point_description(change_point_analysis)` - Creates natural language descriptions

5. `dummy_scripts/test_diagnostic_analytics.py`:
   - Test script for diagnostic analytics module
   - Tests key functionalities including rate change analysis, correlation identification, 
     root cause suggestions, anomaly explanations, and change point detection
   - Creates visualizations showing diagnostic analytical results
   - Demonstrates the complete diagnostic analytics pipeline

#### Key Features:
- Rate change analysis with identification of significant changes over time
- Correlation analysis between metrics with statistical significance testing
- Lead/lag analysis to identify leading indicators for adoption rate changes
- Root cause suggestion algorithms based on correlated metrics and events
- Anomaly detection and explanation with multiple detection methods
- Change point detection to identify when trends significantly shifted
- Business context generation for all analytical results
- Natural language generation for all diagnostic insights

#### Analysis Methodologies:
- Period-over-period change analysis (MoM, QoQ, YoY)
- Statistical correlation analysis with Pearson and Spearman methods
- Z-score based anomaly detection with customizable thresholds
- Ensemble anomaly detection combining multiple statistical methods
- Pattern recognition for recurring trends and seasonal effects
- Business event correlation where applicable
- Prediction-based change point detection

#### Implemented Diagnostic Capabilities:
- Identification of metrics most strongly correlated with adoption rate changes
- Detection of significant adoption rate changes with automated threshold selection
- Root cause suggestions ranked by likelihood and impact
- Comparative analysis between different time periods to isolate factors
- Anomaly explanation with context and potential business implications
- Trend shift detection with pre/post statistics to quantify impact
- Natural language explanations for all diagnostic findings

## 8. Predictive Analytics Capabilities
- [X] Implement time series forecasting for adoption rate projections
- [X] Create confidence interval calculations for predictions
- [X] Build trend extrapolation methods for short-term forecasts
- [X] Develop seasonal adjustment capabilities for more accurate predictions
- [X] Implement "what-if" scenario modeling logic
- [X] Create target prediction functionality ("When will we reach X% adoption?")
- [X] Build risk assessment for predicted outcomes

### Implementation Details

#### Files Created:
1. `src/predictive_analytics/predictive_analytics.py`:
   - Main class for predictive analytics functionality
   - Key methods:
     - `forecast_adoption_rate(from_date, to_date, forecast_periods, metric_type, method)` - Generates forecasts for adoption rates
     - `predict_target_achievement(target_value, metric_type, from_date, to_date, max_horizon)` - Predicts when a target will be achieved
     - `compare_scenarios(from_date, to_date, scenarios, metric_type, forecast_periods)` - Compares different forecast scenarios
     - `create_what_if_scenario(from_date, to_date, factors, metric_type)` - Creates what-if scenarios based on impact factors
     - `_get_data(from_date, to_date)` - Retrieves adoption rate data for analysis
     - `_generate_forecast_explanation(forecast_result, metric_type)` - Generates natural language explanations

2. `src/predictive_analytics/time_series_forecasting.py`:
   - Functions for forecasting adoption rate metrics
   - Key functions:
     - `create_time_series_forecast(data, metric_type, forecast_periods, method)` - Creates forecasts using specified method
     - `select_best_forecast_method(data, metric_type)` - Automatically selects best forecasting method
     - `_forecast_with_trend(data, periods)` - Forecasts using trend extrapolation
     - `_forecast_with_arima(data, periods)` - Forecasts using ARIMA models
     - `_forecast_with_ets(data, periods)` - Forecasts using ETS (Error-Trend-Seasonal) models
     - `_analyze_forecast_trend(forecast_values)` - Analyzes trend characteristics in forecast

3. `src/predictive_analytics/confidence_intervals.py`:
   - Functions for calculating prediction confidence intervals
   - Key functions:
     - `calculate_forecast_intervals(forecast_values, model_info, confidence_level)` - Calculates confidence intervals
     - `_calculate_trend_intervals(forecast_values, model_info, confidence_level)` - Intervals for trend forecasts
     - `_calculate_arima_intervals(forecast_values, model_info, confidence_level)` - Intervals for ARIMA forecasts
     - `_calculate_ets_intervals(forecast_values, model_info, confidence_level)` - Intervals for ETS forecasts
     - `calculate_prediction_interval_width(horizon, residual_std, confidence_level, method)` - Calculates width of prediction intervals

4. `src/predictive_analytics/target_prediction.py`:
   - Functions for predicting target achievement dates
   - Key functions:
     - `predict_target_achievement_date(data, target_value, metric_type, max_horizon, confidence_level)` - Predicts achievement date
     - `estimate_growth_rate_for_target(current_value, target_value, target_date, current_date)` - Estimates required growth rate
     - `_generate_target_prediction_explanation(current_value, target_value, achievement_date, ...)` - Generates explanations for target predictions

5. `src/predictive_analytics/scenario_modeling.py`:
   - Functions for modeling different adoption rate scenarios
   - Key functions:
     - `create_scenario_forecast(data, scenario_name, custom_factors, metric_type, forecast_periods)` - Creates scenario forecasts
     - `compare_scenarios(data, scenarios, metric_type, forecast_periods)` - Compares multiple scenarios
     - `analyze_impact_factors(data, metric_type, factor_impacts)` - Analyzes impact of different factors
     - `_analyze_forecast_characteristics(forecast_values, last_value)` - Analyzes growth rate and volatility
     - `_compare_scenario_metrics(scenario_results)` - Compares metrics across scenarios
     - `_generate_scenario_explanation(scenario_name, scenario_params, last_value, ...)` - Generates scenario explanations
     - `_generate_scenario_comparison_explanation(scenario_results, comparison, metric_type)` - Explains scenario comparisons

6. `src/predictive_analytics/__init__.py`:
   - Package initialization with exports of key classes and functions
   - Makes all components easily accessible from a single import

7. `dummy_scripts/test_predictive_analytics.py`:
   - Test script for the predictive analytics module
   - Tests all key functionalities including time series forecasting, target prediction, scenario modeling,
     confidence interval calculation, and what-if analysis
   - Creates visualizations of forecasts, scenarios, and target predictions
   - Demonstrates the complete predictive analytics pipeline

#### Key Features:
- Time series forecasting with multiple methods (trend, ARIMA, ETS, auto-selection)
- Confidence interval calculation with method-specific algorithms
- Target achievement prediction with confidence levels
- Growth rate estimation for achieving targets by specific dates
- Scenario modeling with predefined and custom scenarios
- What-if analysis with impact factor modeling
- Automated model selection based on data characteristics
- Rich visualizations of forecasts and scenarios
- Natural language explanations for all predictions

#### Forecasting Methods:
- Trend extrapolation using polynomial regression
- ARIMA (AutoRegressive Integrated Moving Average) modeling
- ETS (Error-Trend-Seasonal) exponential smoothing models
- Automatic method selection based on data characteristics
- Seasonal adjustment for forecasting seasonal patterns
- Confidence interval calculation for all forecast methods

#### Scenario Types:
- Baseline scenario based on current trends
- Optimistic scenario with accelerated adoption
- Pessimistic scenario with slower adoption
- Aggressive growth scenario with major initiatives
- Stagnation scenario with minimal growth
- Custom scenarios with user-defined growth and volatility factors

#### What-If Analysis Capabilities:
- Factor impact modeling for various business factors
- Combined effect analysis for multiple simultaneous factors
- Individual factor contribution assessment
- Positive and negative factor analysis
- Most influential factor identification
- Natural language explanations of factor impacts

#### Implementation Challenges and Solutions:
- Challenge: Handling different data formats from stored procedures
  - Solution: Implemented robust DataFrame parsing with proper error handling
- Challenge: Color formatting errors in visualization code
  - Solution: Standardized color formatting approach using matplotlib's proper format
- Challenge: Warning about deprecated 'damped' parameter in ExponentialSmoothing
  - Solution: Future fix will update code to use 'damped_trend' instead
- Challenge: Frequency information missing for time series data
  - Solution: Implemented date-index creation with explicit frequency information

## 9. Prescriptive Analytics Capabilities
- [X] Develop recommendation engine based on historical performance
- [X] Create action suggestion logic to improve adoption rates
- [X] Implement goal-setting assistance functionality
- [X] Build intervention impact estimation features
- [X] Develop prioritization logic for suggested actions
- [X] Create benchmark-based recommendations
- [X] Implement custom alert threshold suggestions

### Implementation Details

#### Files Created:
1. `src/prescriptive_analytics/__init__.py`:
   - Package exports and organization
   - Makes all prescriptive components accessible from a single import

2. `src/prescriptive_analytics/prescriptive_analytics.py`:
   - Main interface class for all prescriptive functionality
   - Key methods:
     - `generate_recommendations(from_date, to_date, metric_type)` - Generates recommendations based on historical data
     - `suggest_actions(from_date, to_date, metric_type)` - Suggests specific actions to improve adoption rates
     - `assist_goal_setting(from_date, to_date, metric_type, goal_type, timeframe, custom_target)` - Assists with setting realistic adoption goals
     - `estimate_intervention_impact(from_date, to_date, metric_type, interventions)` - Estimates impact of various interventions
     - `prioritize_actions(from_date, to_date, metric_type, actions, criteria)` - Prioritizes actions based on multiple factors
     - `suggest_benchmark_recommendations(from_date, to_date, metric_type, industry_type)` - Suggests recommendations based on benchmarks
     - `suggest_alert_thresholds(from_date, to_date, metric_type, threshold_types)` - Suggests optimal monitoring thresholds

3. `src/prescriptive_analytics/recommendation_engine.py`:
   - Generates data-driven recommendations based on historical patterns
   - Key functions:
     - `generate_recommendations(data, metric_type, target_improvement)` - Generates adoption rate improvement recommendations
     - `_analyze_adoption_patterns(data, metric_type)` - Analyzes key patterns in adoption rate data
     - `_identify_potential_improvements(analysis_results, target_improvement)` - Identifies potential areas for improvement
     - `_generate_recommendations(potential_improvements, current_rate, target_rate)` - Creates specific recommendations
     - `_generate_explanation(analysis_results, recommendations, current_rate, target_rate)` - Creates natural language explanations

4. `src/prescriptive_analytics/action_suggestions.py`:
   - Suggests specific actions to improve adoption rates
   - Key functions:
     - `generate_action_suggestions(data, metric_type, focus_areas)` - Generates actionable suggestions
     - `_analyze_focus_areas(data, metric_type)` - Analyzes key focus areas for improvement
     - `_generate_actions_by_focus_area(focus_areas, data, metric_type)` - Generates actions for each focus area
     - `_generate_action_explanation(actions, focus_areas, adoption_rate)` - Creates natural language explanations

5. `src/prescriptive_analytics/goal_setting.py`:
   - Assists with setting realistic adoption rate goals
   - Key functions:
     - `assist_goal_setting(data, metric_type, goal_type, timeframe, custom_target)` - Generates goal setting assistance
     - `_analyze_current_status(data, metric_type)` - Analyzes current adoption rate status
     - `_calculate_realistic_target(current_status, goal_type, timeframe, custom_target)` - Calculates realistic target
     - `_assess_target_achievability(current_status, target_value, timeframe)` - Assesses how achievable a target is
     - `_generate_milestones(current_value, target_value, timeframe)` - Generates milestone targets
     - `_generate_goal_explanation(current_status, target_value, timeframe, achievability, milestones)` - Creates explanations

6. `src/prescriptive_analytics/intervention_impact.py`:
   - Estimates impact of various interventions on adoption rates
   - Key functions:
     - `estimate_intervention_impact(data, metric_type, interventions)` - Estimates intervention impacts
     - `_analyze_historical_impact_patterns(data, metric_type)` - Analyzes historical patterns
     - `_calculate_intervention_impacts(interventions, baseline_forecast, impact_patterns)` - Calculates impacts
     - `_generate_combined_forecast(baseline_forecast, interventions_impact)` - Generates combined forecast
     - `_generate_impact_explanation(interventions, baseline, combined_forecast, individual_impacts)` - Creates explanations
     - `visualize_intervention_impact(baseline, combined, individual_impacts, file_path)` - Visualizes impact

7. `src/prescriptive_analytics/prioritization.py`:
   - Prioritizes recommended actions based on multiple factors
   - Key functions:
     - `prioritize_actions(actions, criteria)` - Prioritizes actions based on specified criteria
     - `_calculate_priority_scores(actions, factor_weights)` - Calculates priority scores
     - `_organize_implementation_waves(prioritized_actions, wave_size)` - Organizes actions into implementation waves
     - `_generate_prioritization_explanation(prioritized_actions, factor_weights, waves)` - Creates explanations

8. `src/prescriptive_analytics/benchmark_recommendations.py`:
   - Generates recommendations based on industry benchmarks
   - Key functions:
     - `generate_benchmark_recommendations(data, metric_type, industry_type)` - Generates benchmark-based recommendations
     - `_compare_to_benchmarks(data, metric_type, industry_type)` - Compares metrics to industry benchmarks
     - `_identify_benchmark_gaps(benchmark_comparison)` - Identifies gaps compared to benchmarks
     - `_generate_benchmark_recommendations(benchmark_gaps, current_metrics)` - Generates recommendations
     - `_generate_benchmark_explanation(recommendations, benchmark_comparison)` - Creates explanations

9. `src/prescriptive_analytics/alert_thresholds.py`:
   - Suggests optimal monitoring thresholds for adoption metrics
   - Key functions:
     - `suggest_alert_thresholds(data, metric_type, threshold_types)` - Suggests alert thresholds
     - `_calculate_statistical_thresholds(data, metric_type)` - Calculates statistical thresholds
     - `_calculate_trend_based_thresholds(data, metric_type)` - Calculates trend-based thresholds
     - `_generate_threshold_recommendations(statistical, trend_based, threshold_types)` - Generates threshold recommendations
     - `_generate_threshold_explanation(recommendations, data, metric_type)` - Creates explanations

10. `dummy_scripts/test_prescriptive_analytics.py`:
    - Test script for prescriptive analytics module
    - Tests all prescriptive capabilities with real database data
    - Generates visualizations in plots/prescriptive directory
    - Validates all module functionality including error handling

#### Key Features:
- Recommendation engine that analyzes adoption rate patterns to identify 5 key improvement areas
- Action suggestion system with 25+ specific actions across 5 target areas (user onboarding, feature adoption, etc.)
- Goal setting assistance with 4 goal types (short-term, medium-term, long-term, custom) and achievability assessments
- Intervention impact modeling with baseline forecasting and impact estimation for multiple intervention types
- Action prioritization based on 6 factors (impact, effort, time-to-value, cost, risk, dependencies)
- Implementation wave planning with 3 waves (quick wins, medium-term initiatives, long-term investments)
- Benchmark-based recommendations comparing current metrics to industry standards
- Alert threshold suggestions for 5 threshold types with 3 severity levels (warning, critical, emergency)
- Natural language explanations for all recommendations and suggestions
- Visualization capabilities for intervention impacts and prioritization

#### Debugging and Fixes:
- Fixed metric_type mapping to ensure consistent handling of parameters
- Added robust error handling throughout all prescriptive modules
- Implemented proper dictionary copying in prioritization to avoid "dictionary changed size during iteration" errors
- Fixed "name 'date' is not defined" error by properly importing date from datetime module
- Enhanced serialization handling for database result processing
- Improved DataFrame handling for stored procedure results

## 10. UI Components for Chatbot Interface
- [ ] Design and implement chat input and message display components
- [ ] Create loading states and indicators for when processing queries
- [ ] Build error messages and fallback UI states
- [ ] Implement response formatting for different answer types
- [ ] Create typing indicators and other chat-like behaviors
- [ ] Develop UI for displaying contextual chart information alongside answers
- [ ] Build mobile-responsive design elements for cross-device compatibility

## 11. Chart Data Visualization Components
- [ ] Implement dynamic chart visualization for displaying relevant data segments
- [ ] Create highlighting functionality to emphasize specific data points in responses
- [ ] Build tooltip components for data point inspection
- [ ] Implement zoom and filter functionality for chart exploration
- [ ] Create comparative visualization capabilities for time period comparison
- [ ] Develop annotation features to mark important points on charts
- [ ] Build export and sharing functionality for visualizations

## 12. Integration and Context Management
- [ ] Develop context management to retain conversation history
- [ ] Create integration with existing dashboard filters (Metrics, Source, Time Period, etc.)
- [ ] Build state management for user session data
- [ ] Implement user preference storage for personalized experiences
- [ ] Create deep linking capabilities to share specific chatbot insights
- [ ] Develop cross-component communication system
- [ ] Build webhook integrations for alerting or reporting

## 13. Testing, Validation and Quality Assurance
- [ ] Create comprehensive test suite for database access functions
- [ ] Develop unit tests for data processing logic
- [ ] Build integration tests for LLM interaction
- [ ] Implement end-to-end tests for complete query flows
- [ ] Create performance testing for response times
- [ ] Develop user acceptance testing plan with sample queries
- [ ] Build automated regression testing for ongoing development

## 14. Security and Compliance
- [ ] Implement input validation and sanitization
- [ ] Create authentication and authorization integration
- [ ] Build data privacy compliance features
- [ ] Develop audit logging for all chatbot interactions
- [ ] Implement rate limiting to prevent abuse
- [ ] Create data retention policies in line with company requirements
- [ ] Build secure handling of sensitive information

## 15. Deployment and Documentation
- [ ] Create deployment pipeline for chatbot services
- [ ] Build containerization for consistent deployment
- [ ] Develop environment configuration management
- [ ] Create technical documentation for all components
- [ ] Build user documentation with example queries
- [ ] Implement versioning strategy for future updates
- [ ] Create monitoring and alerting for production system

## 16. Training and Prompt Engineering
- [ ] Develop initial training dataset of question-answer pairs
- [ ] Create specialized prompts for each analytics type (descriptive, diagnostic, predictive, prescriptive)
- [ ] Build prompt templates with variable injection
- [ ] Implement prompt chaining for complex analytical queries
- [ ] Create guardrails to prevent hallucination on unavailable data
- [ ] Develop feedback loops to improve prompt effectiveness
- [ ] Build prompt versioning and management system

# Sample Questions for Chatbot by Category

## Descriptive Questions
1. What is our current overall adoption rate?
2. How has our adoption rate changed since last quarter?
3. What's the difference between MAU and DAU in the chart?
4. When did we reach our highest adoption rate?
5. What was the average adoption rate in 2023?
6. How do our DAU and MAU metrics compare to each other?
7. What's the trend in overall adoption rate over the last 6 months?
8. How does our current adoption rate compare to this time last year?
9. What time periods show the largest gap between MAU and overall adoption rate?
10. Which month had the lowest adoption rate in 2024?

## Diagnostic Questions
1. Why did our adoption rate drop in January 2023?
2. What caused the spike in MAU around October 2023?
3. Why is there such a large gap between MAU and overall adoption rate?
4. What factors contributed to the declining trend in Q4 2022?
5. Why did DAU fluctuate so much throughout 2023?
6. What explains the consistent pattern of peaks and valleys in the MAU metric?
7. Why did our adoption rate improve after November 2023?
8. What caused the adoption rate to plateau between March and June 2024?
9. Why is our DAU consistently lower than our MAU?
10. What explains the seasonal patterns we see in the adoption rate?

## Predictive Questions
1. What will our adoption rate look like by the end of this year?
2. When can we expect to reach a 15% overall adoption rate?
3. Will our MAU continue to decline in the coming months?
4. What's the projected adoption rate for Q3 2025?
5. How will our DAU trend over the next quarter based on historical patterns?
6. When might we see our adoption rate return to its previous peak?
7. What seasonal patterns should we expect to see in our adoption metrics?
8. If current trends continue, when will our DAU meet or exceed our MAU?
9. What's the long-term trend forecast for our overall adoption rate?
10. Based on historical data, what will happen to our adoption rate during the holiday season?

## Prescriptive Questions
1. How can we improve our overall adoption rate?
2. What should we do to address the declining DAU trend?
3. Which user segments should we target to increase our adoption rate?
4. What actions would help close the gap between MAU and DAU?
5. How can we sustain the recent improvement in our adoption metrics?
6. What strategies would help us achieve a 20% adoption rate by year-end?
7. How should we adjust our approach based on the seasonal patterns in our data?
8. What interventions would be most effective at increasing user retention?
9. Which features should we prioritize to improve our adoption metrics?
10. How can we prevent the adoption rate drops we typically see in Q1?

## Project: Overall Adoption Rate Chatbot

### Final Goal

Build an advanced chatbot that can answer questions about Overall Adoption Rate trends shown in a chart, provide insights, and make recommendations, using a combination of database access, analytical capabilities, and LLM-powered conversational abilities.

### Breakdown

#### Stage 1: Database Connection and Data Retrieval Setup
- [X] Create database connection module
- [X] Implement stored procedure access
- [X] Create data models for adoption rates
- [X] Document database schema and sample queries

**Files Created:**
- `src/database/db_connector.py` - Connection management with pooling
- `src/database/stored_procedures.py` - Access to key SP_OverallAdoptionRate_DWMY procedure
- `src/data_models/metrics.py` - Models for metrics data with proper validation
- `src/config/config.py` - Configuration settings management

#### Stage 2: Data Extraction and Processing
- [X] Create data fetching and transformation logic
- [X] Implement time period filtering
- [X] Add data validation and preprocessing
- [X] Build data caching layer

**Files Created:**
- `src/data_processing/data_fetcher.py` - Retrieves and formats adoption rate data
- `src/data_processing/data_transformer.py` - Performs aggregation and time period handling
- `src/data_processing/data_validator.py` - Ensures data quality and consistency
- `src/utils/date_utils.py` - Date manipulation utilities
- `src/utils/cache_manager.py` - Caching layer for performance optimization

#### Stage 3: Chart Data Analysis Components
- [X] Develop trend detection functionality
- [X] Create period-over-period analysis
- [X] Implement statistical analysis tools
- [X] Build anomaly detection capabilities
- [X] Create correlation analysis

**Files Created:**
- `src/data_analysis/trend_analyzer.py` - Detects trends and patterns in adoption rates
- `src/data_analysis/period_analyzer.py` - Calculates MoM, QoQ, YoY comparisons
- `src/data_analysis/anomaly_detector.py` - Identifies unusual patterns and outliers
- `src/data_analysis/correlation_analyzer.py` - Finds relationships between metrics
- `src/data_analysis/__init__.py` - Package organization and exports

#### Stage 4: LLM Integration Setup
- [X] Set up LLM service integration
- [X] Create prompt engineering templates
- [X] Implement function calling
- [X] Build message history management

**Files Created:**
- `src/llm/llm_service.py` - Integration with OpenAI/Anthropic APIs
- `src/llm/prompt_templates.py` - Templates for different query types
- `src/llm/function_registry.py` - Function calling definitions
- `src/llm/message_history.py` - Conversation history management

#### Stage 5: Natural Language Query Processing
- [X] Build intent recognition system
- [X] Implement entity extraction
- [X] Create query transformation logic
- [X] Develop context awareness for follow-ups

**Files Created:**
- `src/query_processing/intent_classifier.py` - Identifies query intent (descriptive, diagnostic, etc.)
- `src/query_processing/entity_extractor.py` - Extracts dates, metrics, and parameters
- `src/query_processing/query_transformer.py` - Converts natural language to database queries
- `src/query_processing/context_tracker.py` - Maintains context across conversation
- `src/query_processing/query_processor.py` - Orchestrates query processing flow

#### Stage 6: Descriptive Analytics Capabilities
- [ ] Implement current rate description
- [ ] Create summary statistics generation
- [ ] Build time period comparison functionality
- [ ] Develop metric explanation capability
- [ ] Implement trend verbalization

#### Stage 7: Diagnostic Analytics Capabilities
- [ ] Create root cause analysis module
- [ ] Implement contributing factor identification
- [ ] Build comparative diagnosis functionality
- [ ] Develop anomaly explanation capability
- [ ] Implement correlation analysis reporting

#### Stage 8: Predictive Analytics Capabilities
- [X] Create time series forecasting module
- [X] Implement confidence interval calculation
- [X] Build target prediction functionality
- [X] Develop scenario modeling capabilities
- [X] Create what-if analysis functionality

**Files Created:**
- `src/predictive_analytics/__init__.py` - Package exports and organization
- `src/predictive_analytics/predictive_analytics.py` - Main interface to predictive functionality
- `src/predictive_analytics/time_series_forecasting.py` - Trend, ARIMA, and ETS forecasting
- `src/predictive_analytics/scenario_modeling.py` - Scenario creation and comparison 
- `src/predictive_analytics/confidence_intervals.py` - Statistical confidence intervals
- `src/predictive_analytics/target_prediction.py` - Target date and growth rate prediction

**Key Features:**
- Multiple forecasting methods with automatic selection based on data characteristics
- Confidence interval calculation for all forecasts to show prediction uncertainty
- Target achievement prediction with confidence levels
- Five scenario types (baseline, optimistic, pessimistic, highly optimistic, highly pessimistic)
- What-if analysis with impact factor modeling
- Natural language explanations for all predictions

#### Stage 9: Prescriptive Analytics Capabilities
- [X] Develop recommendation engine based on historical performance
- [X] Create action suggestion logic to improve adoption rates
- [X] Implement goal-setting assistance functionality
- [X] Build intervention impact estimation features
- [X] Develop prioritization logic for suggested actions
- [X] Create benchmark-based recommendations
- [X] Implement custom alert threshold suggestions

**Files Created:**
- `src/prescriptive_analytics/__init__.py` - Package exports and organization
- `src/prescriptive_analytics/prescriptive_analytics.py` - Main interface for prescriptive functionality
- `src/prescriptive_analytics/recommendation_engine.py` - Generates recommendations based on historical data
- `src/prescriptive_analytics/action_suggestions.py` - Suggests specific actions to improve adoption
- `src/prescriptive_analytics/goal_setting.py` - Assists with setting realistic adoption rate goals
- `src/prescriptive_analytics/intervention_impact.py` - Estimates impact of various interventions
- `src/prescriptive_analytics/prioritization.py` - Prioritizes actions based on impact and effort
- `src/prescriptive_analytics/benchmark_recommendations.py` - Compares to industry benchmarks
- `src/prescriptive_analytics/alert_thresholds.py` - Suggests optimal monitoring thresholds

**Key Features:**
- Recommendation engine that analyzes adoption rate patterns to identify 5 key improvement areas
- Action suggestion system with 25+ specific actions across 5 target areas (user onboarding, feature adoption, etc.)
- Goal setting assistance with 4 goal types (short-term, medium-term, long-term, custom) and achievability assessments
- Intervention impact modeling with baseline forecasting and impact estimation for multiple intervention types
- Action prioritization based on 6 factors (impact, effort, time-to-value, cost, risk, dependencies)
- Implementation wave planning with 3 waves (quick wins, medium-term initiatives, long-term investments)
- Benchmark-based recommendations comparing current metrics to industry standards
- Alert threshold suggestions for 5 threshold types with 3 severity levels (warning, critical, emergency)
- Natural language explanations for all recommendations and suggestions
- Visualization capabilities for intervention impacts and prioritization

#### Stage 10: Integration and Application Building
- [X] Create main chatbot application
- [X] Implement conversation management
- [X] Build response generation pipeline
- [X] Create visualization capabilities
- [X] Implement error handling

**Files Created:**
- `src/ui/app.py` - Main Flask application with routes and session management
- `src/ui/api_endpoints.py` - API endpoints for data and analytics
- `src/ui/templates/index.html` - HTML template for the chatbot interface
- `src/ui/static/css/styles.css` - CSS styling for the interface
- `src/ui/static/js/chat.js` - JavaScript for chat functionality
- `src/ui/static/js/chart.js` - JavaScript for chart visualization
- `run.py` - Script for running the application with environment setup

**Key Features:**
- Flask-based web application with responsive design
- Interactive chat interface with message history
- Real-time chart visualization using Plotly
- API endpoints for all analytics components
- Session management for conversation persistence
- Error handling and logging for robustness
- Environment variable management for configuration
- Startup message with helpful usage information

#### Stage 11: Testing and Optimization
- [ ] Create unit tests for key components
- [ ] Implement integration tests
- [ ] Build performance optimization
- [ ] Create logging and monitoring
- [ ] Implement security best practices

#### Stage 12: Deployment and Documentation
- [ ] Create deployment scripts
- [ ] Build containerization
- [ ] Create user documentation
- [ ] Implement API documentation
- [ ] Create maintenance guidelines
