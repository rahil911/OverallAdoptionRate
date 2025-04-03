# Database Layer Documentation

## Overview
The database layer is responsible for managing all interactions with the Opus database, providing a robust and efficient interface for accessing adoption rate data. This layer implements connection pooling, secure credential management, and standardized access patterns for stored procedures.

## Directory Structure
```
src/database/
├── __init__.py
├── connection.py
└── data_access.py
```

## Components

### 1. Connection Management (`connection.py`)

#### Purpose
Manages database connections efficiently using connection pooling to minimize resource usage and maximize performance.

#### Key Features
- Connection pooling with configurable pool size
- Automatic connection retry mechanism
- Context managers for safe connection handling
- Error recovery and connection health checks

#### Core Classes and Methods
- `DatabaseConnection`: Main connection management class
  - `get_connection()`: Retrieves a connection from the pool
  - `release_connection(connection)`: Returns a connection to the pool
  - `db_connection()`: Context manager for safe connection usage

#### Usage Example
```python
with DatabaseConnection.db_connection() as conn:
    cursor = conn.cursor()
    # Execute queries
```

### 2. Data Access Layer (`data_access.py`)

#### Purpose
Provides a clean, type-safe interface for interacting with database stored procedures and managing data retrieval operations.

#### Key Features
- Type-safe parameter handling
- Standardized error handling
- Result set transformation
- Query execution monitoring

#### Core Functions
1. `execute_stored_procedure(proc_name, params)`
   - Executes any stored procedure with proper parameter handling
   - Parameters:
     - proc_name: Name of the stored procedure
     - params: Dictionary of parameter names and values
   - Returns: DataFrame containing the result set

2. `get_overall_adoption_rate(from_date, to_date, tenant_id)`
   - Fetches Overall Adoption Rate data
   - Parameters:
     - from_date: Start date (datetime)
     - to_date: End date (datetime)
     - tenant_id: Tenant identifier (int)
   - Returns: DataFrame with adoption rate metrics

3. `get_mau(from_date, to_date, tenant_id)`
   - Fetches Monthly Active Users data
   - Parameters and return type similar to above

4. `get_dau(from_date, to_date, tenant_id)`
   - Fetches Daily Active Users data
   - Parameters and return type similar to above

#### Error Handling
- Connection errors: Automatic retry with exponential backoff
- Query errors: Detailed error messages with query context
- Parameter validation: Type checking and value validation
- Result set validation: Schema verification

## Configuration

### Database Settings
- Server: opusdatdev.database.windows.net
- Database: opusdatdev
- Connection timeout: 30 seconds
- Command timeout: 120 seconds
- Pool size: 10 connections
- Retry attempts: 3

### Required Environment Variables
- `DB_USERNAME`: Database username
- `DB_PASSWORD`: Database password
- `DB_SERVER`: Database server address
- `DB_NAME`: Database name

## Stored Procedures

### SP_OverallAdoptionRate_DWMY
- **Purpose**: Retrieves adoption rate metrics at different time granularities
- **Parameters**:
  - @FromDate (datetime): Start date for data retrieval
  - @ToDate (datetime): End date for data retrieval
  - @Tenantid (int): Tenant identifier
- **Returns**: Multiple metrics including:
  - Daily Active Users (DAU)
  - Weekly Active Users (WAU)
  - Monthly Active Users (MAU)
  - Yearly Active Users (YAU)
  - Corresponding adoption rates for each time period

### SP_DAU
- **Purpose**: Retrieves daily active user counts
- **Parameters**: Same as above
- **Returns**: Daily active user counts for specified period

### SP_MAU
- **Purpose**: Retrieves monthly active user counts
- **Parameters**: Same as above
- **Returns**: Monthly active user counts for specified period

## Best Practices

### Connection Management
1. Always use the context manager for database connections
2. Release connections back to the pool promptly
3. Set appropriate timeouts for long-running queries
4. Monitor connection pool usage

### Query Execution
1. Use parameterized queries to prevent SQL injection
2. Implement retry logic for transient failures
3. Log query performance metrics
4. Validate input parameters before execution

### Error Handling
1. Implement proper error logging
2. Use specific exception types for different error cases
3. Provide meaningful error messages
4. Handle connection timeouts gracefully

## Performance Considerations

### Connection Pooling
- Minimum pool size: 1
- Maximum pool size: 10
- Connection timeout: 30 seconds
- Pool recycle time: 3600 seconds (1 hour)

### Query Optimization
- Use appropriate indexes
- Monitor query execution plans
- Implement query timeouts
- Cache frequently accessed data

## Security

### Credential Management
- Use environment variables for sensitive data
- Implement encryption for connection strings
- Rotate credentials regularly
- Monitor access patterns

### Access Control
- Implement role-based access
- Use least privilege principle
- Audit database access
- Monitor failed login attempts

## Monitoring and Logging

### Metrics Tracked
- Connection pool utilization
- Query execution times
- Error rates
- Result set sizes

### Logging
- Query parameters
- Execution duration
- Error details
- Connection events

## Testing

### Unit Tests
- Connection management tests
- Parameter validation tests
- Error handling tests
- Result set validation tests

### Integration Tests
- Stored procedure execution tests
- Connection pooling tests
- Error recovery tests
- End-to-end data retrieval tests

## Troubleshooting

### Common Issues
1. Connection timeouts
   - Check network connectivity
   - Verify connection string
   - Monitor pool utilization

2. Query timeouts
   - Review query complexity
   - Check parameter values
   - Monitor server load

3. Data type mismatches
   - Verify parameter types
   - Check stored procedure definitions
   - Validate result set schemas

### Debug Tools
- Connection pool monitoring
- Query execution logging
- Error tracking
- Performance profiling

## Future Improvements

### Planned Enhancements
1. Enhanced connection pooling metrics
2. Query result caching
3. Automated failover handling
4. Extended monitoring capabilities

### Technical Debt
1. Standardize error handling
2. Improve parameter validation
3. Enhance logging granularity
4. Optimize connection pool settings 