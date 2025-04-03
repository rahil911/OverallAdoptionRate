# Database Schema Documentation

This document provides a comprehensive overview of the database schema relevant to the Overall Adoption Rate chatbot project.

## Tables

### StagingEventLogs_20250221
This table contains user activity events with 1,334,850 rows.

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| id | varchar | Unique identifier for the event |
| eventId | bigint | Event identifier |
| type | varchar | Type of event (e.g., "Activity") |
| source | varchar | Source of the event (e.g., "PortalWeb", "AppAdvisor", "ImpersonateWeb") |
| targetAppId | bigint | Target application identifier |
| targetAppName | varchar | Name of the target application |
| tenantId | int | Tenant identifier |
| userId | bigint | User identifier |
| userActivationTime | varchar | Time when the user was activated |
| ipAddress | varchar | IP address of the user |
| dateTime | datetime | Date and time when the event occurred |
| spanId | varchar | Span identifier for tracing |
| action | varchar | Action performed (e.g., "LogIn", "LogOut", "Activity") |
| actionType | varchar | Type of action (e.g., "StoreLoginToken", "Logout:") |
| actionId | bigint | Action identifier |
| actionDetails | varchar | Details about the action |
| actionResult | bigint | Result of the action |
| sourceId | bigint | Source identifier |
| doctype | int | Document type |
| documentId | varchar | Document identifier |
| title | varchar | Title |
| document | varchar | Document content |
| step | varchar | Step in a process |
| url | varchar | URL associated with the event |
| questionId | varchar | Question identifier |
| question | varchar | Question content |
| attemptStatus | varchar | Status of an attempt |
| correctResponses | int | Number of correct responses |
| scorePercentage | float | Percentage score |
| passingThreshold | int | Threshold for passing |
| maxIncorrectAllowed | int | Maximum incorrect responses allowed |

**Date Range in Data**: 2022-09-06 to 2025-03-26

### OpusTenantsUsers_20250227
This table maps users to tenants with 83,715 rows.

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| UserId | int | User identifier |
| TenantId | int | Tenant identifier |
| Status | nvarchar(50) | User status (e.g., "Active", "Inactive", "Incomplete") |

### TenantIDLookUp
This table contains tenant lookup information.

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| TenantID | int | Tenant identifier |
| Customer | nvarchar | Customer name |

## Stored Procedures

### SP_OverallAdoptionRate_DWMY
Returns overall adoption rate metrics (daily, weekly, monthly, yearly).

**Parameters**:
- @FromDate (datetime) - Start date for the data range
- @ToDate (datetime) - End date for the data range
- @Tenantid (int) - Tenant identifier

**Return Columns**:
- Date: The date of the metrics
- DAU: Daily Active Users count
- DOverallAdoptionRate: Daily Overall Adoption Rate
- WAU: Weekly Active Users count
- WOverallAdoptionRate: Weekly Overall Adoption Rate
- MAU: Monthly Active Users count
- MOverallAdoptionRate: Monthly Overall Adoption Rate
- YAU: Yearly Active Users count
- YOverallAdoptionRate: Yearly Overall Adoption Rate

**Sample Data**:
```
         Date  DAU  DOverallAdoptionRate  WAU  WOverallAdoptionRate  MAU  MOverallAdoptionRate   YAU  YOverallAdoptionRate
0  2022-01-01  NaN                   NaN  NaN                   NaN  NaN                   NaN  23.0                   9.5
1  2022-01-02  NaN                   NaN  NaN                   NaN  NaN                   NaN  23.0                   9.5
```

### SP_MAU
Returns Monthly Active Users data.

**Parameters**:
- @FromDate (datetime) - Start date for the data range
- @ToDate (datetime) - End date for the data range
- @Tenantid (int) - Tenant identifier

**Return Columns**:
- Year_MonthNo: The year and month (format: YYYY-MM)
- TotalActiveUsers: Count of monthly active users

**Sample Data**:
```
  Year_MonthNo  TotalActiveUsers
0      2022-10              15.0
1      2022-11              15.0
```

### SP_DAU
Returns Daily Active Users data.

**Parameters**:
- @FromDate (datetime) - Start date for the data range
- @ToDate (datetime) - End date for the data range
- @Tenantid (int) - Tenant identifier

**Return Columns**:
- Date: The date of the metrics
- TotalActiveUsers: Count of daily active users

**Sample Data**:
```
         Date  TotalActiveUsers
0  2022-09-06               5.0
1  2022-09-07               7.0
```

## Key Relationships

- StagingEventLogs_20250221.userId connects to OpusTenantsUsers_20250227.UserId
- StagingEventLogs_20250221.tenantId connects to OpusTenantsUsers_20250227.TenantId

## Test Data

Tenant ID 1388 has been verified to have test data available for all metrics:
- Overall Adoption Rate: 1,156 rows
- MAU: 31 rows
- DAU: 722 rows
- Event count: 662,936 events
- User count: 242 users

## Data Formats

### Date Formats

The "Date" column in the Overall Adoption Rate data and DAU data are in the format YYYY-MM-DD.

The "Year_MonthNo" column in the MAU data is in the format YYYY-MM.

### Rate Calculation

The Overall Adoption Rate appears to be calculated as the percentage of active users out of the total users for a given time period:
- DOverallAdoptionRate = (DAU / Total Users) * 100
- WOverallAdoptionRate = (WAU / Total Users) * 100
- MOverallAdoptionRate = (MAU / Total Users) * 100
- YOverallAdoptionRate = (YAU / Total Users) * 100

## Filtering Capabilities

The stored procedures support filtering by:
- Date range (FromDate, ToDate)
- Tenant ID (Tenantid)

The "All" filter in the chart UI refers to departments filter.

## Data Availability

The data is available from 2022-09-06 to the present day, with the most comprehensive data for tenant ID 1388. 