# Observable Pattern Detection System: Functional Specification

## 1. System Overview

The Observable Pattern Detection System analyzes Overall Adoption Rate metrics to identify significant behavioral trends and patterns. This document specifies the detailed functionality of each system component, including inputs, processing operations, outputs, and available visualizations.

## 2. Data Input Sources

### 2.1 Database Connection
- **Data Source**: Opus SQL database containing adoption metrics
- **Access Method**: SQL stored procedures
- **Primary Stored Procedures**:
  - `SP_OverallAdoptionRate_DWMY`: Returns adoption rates at daily, weekly, monthly, and yearly intervals
  - `SP_DAU`: Returns Daily Active User counts
  - `SP_MAU`: Returns Monthly Active User counts
  
### 2.2 Required Parameters
- **FromDate (datetime)**: Start date for data range analysis
- **ToDate (datetime)**: End date for data range analysis
- **TenantId (int)**: Organization identifier (typically 1388 for testing)

### 2.3 Data Schema
- **Adoption Rate Data Fields**:
  - `Date`: Timestamp of the data point
  - `DAU`: Daily Active Users count
  - `DOverallAdoptionRate`: Daily Overall Adoption Rate percentage (0-100%)
  - `WAU`: Weekly Active Users count
  - `WOverallAdoptionRate`: Weekly Overall Adoption Rate percentage (0-100%)
  - `MAU`: Monthly Active Users count
  - `MOverallAdoptionRate`: Monthly Overall Adoption Rate percentage (0-100%)
  - `YAU`: Yearly Active Users count
  - `YOverallAdoptionRate`: Yearly Overall Adoption Rate percentage (0-100%)

## 3. Pattern Detection Components

### 3.1 Trend Analysis

#### 3.1.1 Trend Slope Calculator
- **Function**: Calculates the slope of adoption rates over time to determine trend direction
- **Inputs**:
  - Time series data (from database or preprocessed cache)
  - Rate type ('daily', 'weekly', 'monthly', 'yearly')
  - Time period (date range)
- **Outputs**:
  - Trend direction ('increasing', 'decreasing', 'stable')
  - Trend strength (numerical value)
  - Statistical significance (p-value)
- **Visualization**: `trend_analysis.png`

#### 3.1.2 Peak and Valley Detector
- **Function**: Identifies local maxima (peaks) and minima (valleys) in adoption rate data
- **Inputs**:
  - Time series data (adoption rates)
  - Prominence threshold (minimum peak/valley significance, default 0.5)
  - Width parameter (minimum distance between peaks/valleys)
- **Outputs**:
  - List of peak timestamps with corresponding adoption rate values
  - List of valley timestamps with corresponding adoption rate values
- **Visualization**: `trend_analysis.png` (same as above, with peaks and valleys marked)

#### 3.1.3 Volatility Calculator
- **Function**: Measures the stability or variability of adoption rates over time
- **Inputs**:
  - Time series data (adoption rates)
  - Rate type ('daily', 'weekly', 'monthly', 'yearly')
  - Window size for calculation
- **Outputs**:
  - Volatility score (0-1, where 0 is completely stable and 1 is highly volatile)
  - Volatility classification ('low', 'medium', 'high')
  - Comparison to historical volatility
- **Visualization**: Included in `trend_analysis.png`

### 3.2 Anomaly Detection

#### 3.2.1 Z-Score Anomaly Detector
- **Function**: Identifies data points that deviate significantly from the mean
- **Inputs**:
  - Time series data (adoption rates)
  - Threshold parameter (default 2.0 standard deviations)
  - Rate type ('daily', 'weekly', 'monthly', 'yearly')
- **Outputs**:
  - List of anomalous timestamps
  - Z-score values for each anomaly
  - Direction ('high' or 'low')
- **Visualization**: `anomaly_detection.png`

#### 3.2.2 IQR-Based Anomaly Detector
- **Function**: Identifies outliers based on interquartile range
- **Inputs**:
  - Time series data (adoption rates)
  - IQR multiplier (default 1.5)
  - Rate type ('daily', 'weekly', 'monthly', 'yearly')
- **Outputs**:
  - List of anomalous timestamps
  - IQR distance values
  - Direction ('high' or 'low')
- **Visualization**: Used in ensemble anomaly detection, shown in `anomaly_detection.png`

#### 3.2.3 Adaptive Threshold Anomaly Detector
- **Function**: Detects anomalies using an adaptive threshold that adjusts based on recent data
- **Inputs**:
  - Time series data (adoption rates)
  - Window size (default 10 data points)
  - Influence parameter (0-1, default 0.5)
  - Threshold multiplier (default 2.0)
  - Rate type ('daily', 'weekly', 'monthly', 'yearly')
- **Outputs**:
  - List of anomalous timestamps
  - Adaptive mean at each anomaly point
  - Adaptive standard deviation at each anomaly point
  - Direction ('high' or 'low')
- **Visualization**: Used in ensemble anomaly detection, shown in `anomaly_detection.png`

#### 3.2.4 Ensemble Anomaly Detector
- **Function**: Combines multiple anomaly detection methods for higher confidence
- **Inputs**:
  - Time series data (adoption rates)
  - Minimum methods parameter (default 2)
  - Rate type ('daily', 'weekly', 'monthly', 'yearly')
- **Outputs**:
  - List of anomalous timestamps
  - Confidence score for each anomaly (percentage of methods that flagged it)
  - Contributing methods for each anomaly
- **Visualization**: `anomaly_detection.png`

### 3.3 Period-over-Period Analysis

#### 3.3.1 Month-over-Month Calculator
- **Function**: Calculates changes in adoption rates between consecutive months
- **Inputs**:
  - Time series data (adoption rates)
  - Rate type ('daily', 'weekly', 'monthly', 'yearly')
- **Outputs**:
  - Absolute change in adoption rate (percentage points)
  - Relative change in adoption rate (percentage)
  - Comparison to historical MoM changes
- **Visualization**: `period_over_period.png`

#### 3.3.2 Quarter-over-Quarter Calculator
- **Function**: Calculates changes in adoption rates between consecutive quarters
- **Inputs**:
  - Time series data (adoption rates)
  - Rate type ('daily', 'weekly', 'monthly', 'yearly')
- **Outputs**:
  - Absolute change in adoption rate (percentage points)
  - Relative change in adoption rate (percentage)
  - Comparison to historical QoQ changes
- **Visualization**: `period_over_period.png`

#### 3.3.3 Year-over-Year Calculator
- **Function**: Calculates changes in adoption rates between the same period in consecutive years
- **Inputs**:
  - Time series data (adoption rates)
  - Rate type ('daily', 'weekly', 'monthly', 'yearly')
- **Outputs**:
  - Absolute change in adoption rate (percentage points)
  - Relative change in adoption rate (percentage)
  - Comparison to historical YoY changes
- **Visualization**: `period_over_period.png`

### 3.4 Correlation Analysis

#### 3.4.1 Metric Correlation Calculator
- **Function**: Calculates correlations between different adoption metrics and user counts
- **Inputs**:
  - Multiple time series data (DAU, WAU, MAU, YAU, DOverallAdoptionRate, etc.)
  - Time period (date range)
- **Outputs**:
  - Correlation matrix with correlation coefficients
  - Statistical significance (p-values)
  - Identification of strong, moderate, and weak correlations
- **Visualization**: `correlation_matrix.png`

#### 3.4.2 Segment Correlation Analyzer
- **Function**: Analyzes correlations between different user segments
- **Inputs**:
  - Segmented adoption rate data
  - Segment identifiers
  - Metric type ('daily', 'weekly', 'monthly', 'yearly')
- **Outputs**:
  - Inter-segment correlation coefficients
  - Segment alignment/divergence scores
  - Identification of significantly deviant segments
- **Visualization**: Component of `correlation_matrix.png`

## 4. Pattern Recognition and Classification

### 4.1 Flat Adoption Detector

- **Function**: Identifies periods of minimal change in adoption rates
- **Inputs**:
  - Time series data (adoption rates)
  - Threshold for flatness (default 5% deviation)
  - Minimum duration (default 14 days)
  - Rate type ('daily', 'weekly', 'monthly', 'yearly')
- **Outputs**:
  - Start and end timestamps of flat periods
  - Duration of flatness
  - Average adoption rate during flat period
  - Historical context (how common is this flat period)
- **Visualization**: Highlighted segments in `trend_analysis.png`

### 4.2 Sudden Drop Detector

- **Function**: Identifies rapid significant decreases in adoption rates
- **Inputs**:
  - Time series data (adoption rates)
  - Threshold for drop significance (default 10%)
  - Maximum time window for drop (default 14 days)
  - Rate type ('daily', 'weekly', 'monthly', 'yearly')
- **Outputs**:
  - Start and end timestamps of drop
  - Magnitude of drop (absolute and percentage)
  - Rate of change (how quickly the drop occurred)
  - Severity classification ('minor', 'moderate', 'severe', 'critical')
- **Visualization**: Highlighted in `anomaly_detection.png`

### 4.3 Spike-followed-by-Drop Detector

- **Function**: Identifies temporary increases followed by decreases
- **Inputs**:
  - Time series data (adoption rates)
  - Peak and valley data (from Peak and Valley Detector)
  - Maximum time between peak and valley (default 30 days)
  - Minimum magnitude (default 5% difference)
  - Rate type ('daily', 'weekly', 'monthly', 'yearly')
- **Outputs**:
  - Peak timestamp and value
  - Valley timestamp and value
  - Duration of pattern
  - Magnitude of spike and subsequent drop
  - Pattern significance score
- **Visualization**: Highlighted sequences in `period_over_period.png`

### 4.4 Segment Gap Analyzer

- **Function**: Identifies significant differences between user segments
- **Inputs**:
  - Segmented adoption rate data
  - Segment identifiers
  - Threshold for gap significance (default 15% difference)
  - Rate type ('daily', 'weekly', 'monthly', 'yearly')
- **Outputs**:
  - Segment pairs with significant gaps
  - Gap magnitude for each pair
  - Consistency of gap over time
  - Historical gap context
- **Visualization**: `correlation_matrix.png` with segment differences highlighted

## 5. Pattern Verbalization

### 5.1 Trend Verbalization Engine

- **Function**: Generates natural language descriptions of adoption rate trends
- **Inputs**:
  - Trend analysis results
  - Peak and valley data
  - Volatility metrics
  - Time period of interest
- **Outputs**:
  - Trend direction statement
  - Trend strength characterization
  - Volatility assessment
  - Peak and valley descriptions
  - Overall trend summary paragraph
- **Sample Output**: "The adoption rate shows a weak stable trend with low volatility over the analyzed period from 2023-04-03 to 2025-04-01. Three significant peaks were detected, with the highest value of 12.40% occurring on 2025-01-16."

### 5.2 Anomaly Description Generator

- **Function**: Creates human-readable explanations of detected anomalies
- **Inputs**:
  - Anomaly detection results
  - Historical context data
  - Statistical parameters (mean, standard deviation, thresholds)
- **Outputs**:
  - Count of anomalies by type (high/low)
  - Description of most significant anomalies
  - Statistical context for each anomaly
  - Potential causes based on timing
- **Sample Output**: "Detected 90 anomalies using a threshold of 1.5 standard deviations from the mean. 62 high anomalies and 28 low anomalies were identified. The most significant high anomaly occurred on 2025-01-16 with a value of 12.40% (1.86 standard deviations above the mean)."

### 5.3 Period Comparison Narrator

- **Function**: Describes period-over-period changes in a narrative format
- **Inputs**:
  - Period-over-period analysis results (MoM, QoQ, YoY)
  - Historical comparison data
- **Outputs**:
  - Narrative description of recent period changes
  - Contextual interpretation of changes
  - Relative performance assessment
- **Sample Output**: "March 2025 showed a substantial Month-over-Month increase of 9.09 percentage points (169.27% relative growth), which represents the largest monthly improvement in the past 12 months."

### 5.4 Current State Summarizer

- **Function**: Generates comprehensive summaries of the current adoption rate state
- **Inputs**:
  - Latest adoption rate data
  - Historical percentile information
  - Recent period statistics
  - Trend direction
- **Outputs**:
  - Current rate values for all time periods
  - Historical context (percentile rankings)
  - Recent performance summary
  - Trend direction and outlook
- **Sample Output**: "The yearly adoption rate currently stands at 16.12%, which falls in the 12th percentile of historical performance and sits 26.97% below the historical average of 22.07%. Over the past 90 days, the monthly adoption rate has averaged 10.92% with a range of 5.37% to 14.46%."

## 6. Data Processing and Transformation

### 6.1 Data Fetcher

- **Function**: Retrieves data from database and prepares it for analysis
- **Inputs**:
  - Date range parameters
  - Tenant ID
  - Metric types required
  - Optional filtering parameters
- **Outputs**:
  - Structured data objects ready for analysis
  - Missing data indicators
  - Data quality metrics
- **Processing Steps**:
  1. Connect to database
  2. Execute appropriate stored procedures
  3. Convert database results to structured objects
  4. Apply data cleaning and validation
  5. Return prepared dataset

### 6.2 Data Aggregator

- **Function**: Performs temporal aggregations of adoption rate data
- **Inputs**:
  - Raw time series data
  - Desired aggregation level ('weekly', 'monthly', 'quarterly', 'yearly')
  - Aggregation method ('mean', 'median', 'max', 'min')
- **Outputs**:
  - Aggregated time series at requested level
  - Aggregation quality metrics (e.g., data coverage)
  - Timestamps for aggregated periods
- **Processing Steps**:
  1. Group data by requested time period
  2. Apply aggregation method to each group
  3. Generate timestamps for aggregated periods
  4. Return aggregated series with quality indicators

### 6.3 Missing Data Handler

- **Function**: Addresses gaps in time series data
- **Inputs**:
  - Time series data with potential gaps
  - Handling strategy ('interpolate', 'fill_zeros', 'previous', 'ignore')
  - Maximum gap size to handle
- **Outputs**:
  - Complete time series without gaps
  - Gap locations and sizes
  - Confidence metrics for filled values
- **Processing Steps**:
  1. Identify gaps in time series
  2. Apply selected handling strategy to each gap
  3. Tag filled data points with confidence indicators
  4. Return completed series with metadata

### 6.4 Outlier Preprocessor

- **Function**: Pre-processes extreme values before main analysis
- **Inputs**:
  - Raw time series data
  - Outlier threshold (default 3.0 standard deviations)
  - Treatment method ('winsorize', 'remove', 'tag', 'ignore')
- **Outputs**:
  - Preprocessed time series
  - List of identified outliers
  - Treatment applied to each outlier
- **Processing Steps**:
  1. Identify outliers using robust statistics
  2. Apply selected treatment method
  3. Document all modifications
  4. Return processed data with outlier metadata

## 7. Visualization Components

### 7.1 Primary Visualizations

#### 7.1.1 Adoption Rate Chart
- **File Name**: `adoption_rates_chart.png`
- **Content**: Overall adoption rates across all time periods (daily, weekly, monthly, yearly)
- **Features**: Multiple line series, color-coded by time period, with legend

#### 7.1.2 Trend Analysis Visualization
- **File Name**: `trend_analysis.png`
- **Content**: Adoption rate data with trend indicators
- **Features**: Raw data, trendline, peaks/valleys markers, moving average

#### 7.1.3 Anomaly Detection Chart
- **File Name**: `anomaly_detection.png`
- **Content**: Adoption rate data with anomalies highlighted
- **Features**: Normal vs. anomalous points, directional indicators, confidence scores

#### 7.1.4 Period Comparison Chart
- **File Name**: `period_over_period.png`
- **Content**: Visualization of MoM, QoQ, YoY changes
- **Features**: Bar chart of changes, reference lines for historical averages

#### 7.1.5 Correlation Matrix
- **File Name**: `correlation_matrix.png`
- **Content**: Heatmap of correlations between different metrics
- **Features**: Color-coded correlation coefficients, hierarchical clustering

### 7.2 Detailed Metric Visualizations

#### 7.2.1 Daily Active Users Chart
- **File Name**: `dau_chart.png`
- **Content**: Daily Active Users over time
- **Features**: Daily data points, trend indicators, reference lines

#### 7.2.2 Monthly Active Users Chart
- **File Name**: `mau_chart.png`
- **Content**: Monthly Active Users over time
- **Features**: Monthly data points, trend indicators, reference lines

### 7.3 Predictive Visualizations

#### 7.3.1 Forecast Charts
- **File Names**:
  - `forecast_trend.png`: Simple trend-based forecast
  - `forecast_arima.png`: ARIMA model forecast
  - `forecast_ets.png`: Exponential smoothing forecast
  - `forecast_auto.png`: Auto-selected best model forecast
- **Content**: Future adoption rate projections
- **Features**: Historical data, forecast line, confidence intervals

#### 7.3.2 Scenario Analysis Charts
- **File Names**:
  - `scenario_baseline.png`: Expected scenario
  - `scenario_optimistic.png`: Optimistic growth scenario
  - `scenario_pessimistic.png`: Pessimistic decline scenario
  - `scenario_aggressive_growth.png`: Accelerated growth scenario
  - `scenario_stagnation.png`: No growth scenario
  - `scenario_custom.png`: User-defined custom scenario
- **Content**: Multiple potential future scenarios
- **Features**: Multiple scenario lines, historical data, scenario probability

#### 7.3.3 Factor Impact Analysis
- **File Names**:
  - `factor_impact_analysis.png`: Overall impact of multiple factors
  - `individual_factor_impacts.png`: Breakdown of individual factor effects
- **Content**: Analysis of factors affecting adoption rates
- **Features**: Factor contribution bars, impact quantification, confidence intervals

## 8. Input/Output Specifications

### 8.1 Common Function Input Parameters

- **from_date**: Start date for analysis (datetime, optional)
- **to_date**: End date for analysis (datetime, optional)
- **tenant_id**: Organization identifier (integer, default 1388)
- **metric_type**: Type of metric to analyze ('daily', 'weekly', 'monthly', 'yearly', default 'monthly')
- **threshold**: Sensitivity parameter for various detection algorithms (float, varies by function)
- **window_size**: Number of data points to include in rolling calculations (integer, varies by function)
- **aggregation_level**: Time period for data aggregation ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')

### 8.2 Common Function Output Structures

- **JSON**: All functions return JSON-serializable objects for easy API integration
- **Time Series Format**: 
  ```
  {
    "dates": [datetime objects],
    "values": [numeric values],
    "metadata": {
      "metric_type": string,
      "tenant_id": integer,
      "from_date": datetime,
      "to_date": datetime,
      "data_points": integer
    }
  }
  ```
- **Pattern Detection Format**:
  ```
  {
    "pattern_type": string,
    "instances": [
      {
        "start_date": datetime,
        "end_date": datetime,
        "magnitude": number,
        "significance": number,
        "description": string
      }
    ],
    "metadata": {
      "detection_parameters": object,
      "total_instances": integer
    }
  }
  ```
- **Anomaly Detection Format**:
  ```
  {
    "anomalies": [
      {
        "date": datetime,
        "value": number,
        "expected_value": number,
        "deviation": number,
        "direction": string,
        "confidence": number,
        "methods": [strings]
      }
    ],
    "high_anomalies": [...],
    "low_anomalies": [...],
    "metadata": {
      "threshold": number,
      "total_count": integer,
      "high_count": integer,
      "low_count": integer,
      "mean": number,
      "std_dev": number
    }
  }
  ```

## 9. System Integration Points

### 9.1 Web API Endpoints

- **GET /api/adoption-rates**: Retrieves raw adoption rate data
- **GET /api/adoption-analysis/trend**: Returns trend analysis results
- **GET /api/adoption-analysis/anomalies**: Returns anomaly detection results
- **GET /api/adoption-analysis/period-comparison**: Returns period comparison analysis
- **GET /api/adoption-analysis/patterns**: Returns identified patterns
- **GET /api/adoption-analysis/current-state**: Returns current state summary

### 9.2 Chat Interface Functions

- **analyze_trend**: Analyzes and describes the adoption rate trend
- **detect_anomalies**: Identifies and explains anomalies in the data
- **compare_periods**: Compares adoption rates across different time periods
- **describe_current_state**: Provides a summary of the current adoption state
- **forecast_adoption**: Generates forecasts for future adoption rates
- **explain_pattern**: Provides detailed explanation of a specific pattern

## 10. Functional Usage Examples

### 10.1 Trend Analysis

**Input**:
```
analyze_trend(
  from_date="2023-01-01",
  to_date="2025-04-01",
  metric_type="monthly"
)
```

**Output**:
```
{
  "trend_direction": "stable",
  "trend_strength": "weak",
  "volatility": "low",
  "recent_trend": "stable",
  "peaks": [
    {"date": "2025-01-16", "value": 12.40, "significance": "high"},
    {"date": "2024-02-15", "value": 11.57, "significance": "medium"},
    {"date": "2024-05-16", "value": 10.33, "significance": "medium"}
  ],
  "valleys": [
    {"date": "2025-02-14", "value": 5.37, "significance": "high"},
    {"date": "2023-06-15", "value": 6.61, "significance": "medium"},
    {"date": "2024-04-15", "value": 6.61, "significance": "medium"}
  ],
  "description": "The adoption rate shows a weak stable trend with low volatility over the analyzed period from 2023-01-01 to 2025-04-01. Three significant peaks were detected, with the highest value of 12.40% occurring on 2025-01-16."
}
```

### 10.2 Anomaly Detection

**Input**:
```
detect_anomalies(
  from_date="2023-01-01",
  to_date="2025-04-01",
  metric_type="monthly",
  threshold=1.5
)
```

**Output**:
```
{
  "anomalies": [
    {"date": "2025-01-16", "value": 12.40, "expected_value": 8.53, "deviation": 1.86, "direction": "high", "confidence": 0.8},
    {"date": "2025-02-14", "value": 5.37, "expected_value": 8.53, "deviation": -1.52, "direction": "low", "confidence": 0.8},
    ...
  ],
  "high_anomalies": [...],
  "low_anomalies": [...],
  "total_count": 90,
  "high_count": 62,
  "low_count": 28,
  "threshold": 1.5,
  "mean": 8.53,
  "std_dev": 2.08,
  "description": "Detected 90 anomalies using a threshold of 1.5 standard deviations."
}
```

### 10.3 Period Comparison

**Input**:
```
compare_periods(
  from_date="2023-01-01",
  to_date="2025-04-01",
  metric_type="monthly"
)
```

**Output**:
```
{
  "mom": {
    "current_value": 14.46,
    "previous_value": 5.37,
    "absolute_change": 9.09,
    "percentage_change": 169.27,
    "period": "March 2025 vs February 2025"
  },
  "qoq": {
    "current_value": 10.73,
    "previous_value": 7.87,
    "absolute_change": 2.86,
    "percentage_change": 36.35,
    "period": "Q1 2025 vs Q4 2024"
  },
  "yoy": {
    "current_value": 14.46,
    "previous_value": 9.92,
    "absolute_change": 4.54,
    "percentage_change": 45.77,
    "period": "March 2025 vs March 2024"
  },
  "description": "March 2025 showed a substantial Month-over-Month increase of 9.09 percentage points (169.27% relative growth), which represents the largest monthly improvement in the past 12 months."
}
``` 