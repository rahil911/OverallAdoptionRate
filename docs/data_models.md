# Data Models Documentation

## Overview
The data models layer defines the core data structures used throughout the application for representing adoption rate metrics, user activity data, and related analytics. These models ensure type safety, data validation, and consistent data handling across the application.

## Directory Structure
```
src/data_models/
├── __init__.py
└── metrics.py
```

## Core Models

### 1. Adoption Rate Models

#### DailyActiveUsers
```python
class DailyActiveUsers:
    date: datetime
    total_active_users: int
    tenant_id: int
```

**Purpose**: Represents daily active user counts for a specific tenant.

**Attributes**:
- `date`: The specific date for the metrics
- `total_active_users`: Number of active users on that date
- `tenant_id`: Identifier for the tenant

**Methods**:
- `from_dict(data: dict) -> DailyActiveUsers`: Creates instance from dictionary
- `to_dict() -> dict`: Converts instance to dictionary
- `validate() -> bool`: Validates instance data
- `get_active_users() -> int`: Returns the active user count

#### MonthlyActiveUsers
```python
class MonthlyActiveUsers:
    year_month: datetime
    total_active_users: int
    tenant_id: int
```

**Purpose**: Represents monthly active user counts for a specific tenant.

**Attributes**:
- `year_month`: The year and month for the metrics
- `total_active_users`: Number of active users in that month
- `tenant_id`: Identifier for the tenant

**Methods**:
- `from_dict(data: dict) -> MonthlyActiveUsers`: Creates instance from dictionary
- `to_dict() -> dict`: Converts instance to dictionary
- `validate() -> bool`: Validates instance data
- `get_active_users() -> int`: Returns the active user count

#### OverallAdoptionRate
```python
class OverallAdoptionRate:
    date: datetime
    daily_adoption_rate: float
    weekly_adoption_rate: float
    monthly_adoption_rate: float
    yearly_adoption_rate: float
    dau: int
    wau: int
    mau: int
    yau: int
    tenant_id: int
```

**Purpose**: Represents comprehensive adoption rate metrics across different time periods.

**Attributes**:
- `date`: The specific date for the metrics
- `daily_adoption_rate`: Adoption rate for the day (0-100%)
- `weekly_adoption_rate`: Adoption rate for the week (0-100%)
- `monthly_adoption_rate`: Adoption rate for the month (0-100%)
- `yearly_adoption_rate`: Adoption rate for the year (0-100%)
- `dau`: Daily Active Users count
- `wau`: Weekly Active Users count
- `mau`: Monthly Active Users count
- `yau`: Yearly Active Users count
- `tenant_id`: Identifier for the tenant

**Methods**:
- `from_dict(data: dict) -> OverallAdoptionRate`: Creates instance from dictionary
- `to_dict() -> dict`: Converts instance to dictionary
- `validate() -> bool`: Validates instance data
- `get_adoption_rate(period: str) -> float`: Gets adoption rate for specified period
- `get_active_users(period: str) -> int`: Gets active users for specified period

### 2. Collection Models

#### MetricCollection
```python
class MetricCollection:
    metrics: List[Union[DailyActiveUsers, MonthlyActiveUsers, OverallAdoptionRate]]
    metric_type: str
    start_date: datetime
    end_date: datetime
    tenant_id: int
```

**Purpose**: Container for collections of metric instances, providing aggregation and analysis capabilities.

**Attributes**:
- `metrics`: List of metric instances
- `metric_type`: Type of metrics contained
- `start_date`: Start date of the collection
- `end_date`: End date of the collection
- `tenant_id`: Identifier for the tenant

**Methods**:
- `add_metric(metric: Union[DailyActiveUsers, MonthlyActiveUsers, OverallAdoptionRate])`: Adds metric to collection
- `get_metrics() -> List`: Returns all metrics
- `filter_by_date_range(start: datetime, end: datetime) -> MetricCollection`: Filters metrics by date range
- `aggregate_by_period(period: str) -> Dict`: Aggregates metrics by specified period
- `get_statistics() -> Dict`: Calculates basic statistics for the collection

## Data Validation

### Validation Rules

#### Date Validation
- Dates must be valid datetime objects
- Dates cannot be in the future
- Start dates must be before end dates

#### Metric Validation
- Active user counts must be non-negative integers
- Adoption rates must be between 0 and 100
- Tenant ID must be a positive integer

#### Collection Validation
- Collections must not be empty
- All metrics in a collection must be of the same type
- Date ranges must be continuous without gaps

### Validation Methods

```python
def validate_date(date: datetime) -> bool:
    """Validates a date value"""
    pass

def validate_adoption_rate(rate: float) -> bool:
    """Validates an adoption rate value"""
    pass

def validate_active_users(count: int) -> bool:
    """Validates an active users count"""
    pass

def validate_tenant_id(tenant_id: int) -> bool:
    """Validates a tenant ID"""
    pass
```

## Type Conversion

### Database to Model Conversion
```python
def from_db_row(row: Dict) -> Union[DailyActiveUsers, MonthlyActiveUsers, OverallAdoptionRate]:
    """Converts a database row to appropriate model instance"""
    pass
```

### Model to Dictionary Conversion
```python
def to_dict(model: Union[DailyActiveUsers, MonthlyActiveUsers, OverallAdoptionRate]) -> Dict:
    """Converts a model instance to dictionary"""
    pass
```

### JSON Serialization
```python
def to_json(model: Union[DailyActiveUsers, MonthlyActiveUsers, OverallAdoptionRate]) -> str:
    """Converts a model instance to JSON string"""
    pass
```

## Usage Examples

### Creating Model Instances
```python
# Create DailyActiveUsers instance
dau = DailyActiveUsers(
    date=datetime.now(),
    total_active_users=100,
    tenant_id=1388
)

# Create OverallAdoptionRate instance
adoption_rate = OverallAdoptionRate(
    date=datetime.now(),
    daily_adoption_rate=15.5,
    weekly_adoption_rate=20.0,
    monthly_adoption_rate=25.5,
    yearly_adoption_rate=30.0,
    dau=100,
    wau=500,
    mau=1000,
    yau=2000,
    tenant_id=1388
)
```

### Working with Collections
```python
# Create a collection
collection = MetricCollection(metric_type="daily")

# Add metrics
collection.add_metric(dau1)
collection.add_metric(dau2)

# Get statistics
stats = collection.get_statistics()

# Filter by date range
filtered = collection.filter_by_date_range(
    start=datetime(2024, 1, 1),
    end=datetime(2024, 3, 1)
)
```

## Best Practices

### Model Creation
1. Always use factory methods when creating from external data
2. Validate all input data before creating instances
3. Use type hints for better code maintainability
4. Implement proper error handling for invalid data

### Data Handling
1. Use immutable attributes where possible
2. Implement proper serialization methods
3. Handle missing or null values gracefully
4. Use appropriate data types for each field

### Collections
1. Use appropriate collection types for different use cases
2. Implement efficient filtering and aggregation methods
3. Maintain type consistency within collections
4. Provide clear access patterns for collection data

## Error Handling

### Common Errors
1. `ValidationError`: Raised when data validation fails
2. `TypeError`: Raised when incorrect types are provided
3. `ValueError`: Raised when values are out of valid ranges
4. `DateRangeError`: Raised when date ranges are invalid

### Error Handling Example
```python
try:
    metric = DailyActiveUsers.from_dict(data)
    metric.validate()
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
except TypeError as e:
    logger.error(f"Type error: {e}")
except ValueError as e:
    logger.error(f"Value error: {e}")
```

## Testing

### Unit Tests
- Model creation tests
- Validation tests
- Conversion tests
- Collection operation tests

### Integration Tests
- Database conversion tests
- Serialization tests
- Collection aggregation tests
- End-to-end data flow tests

## Future Improvements

### Planned Enhancements
1. Add support for custom metrics
2. Implement metric versioning
3. Add more statistical capabilities
4. Enhance validation rules

### Technical Debt
1. Improve error messages
2. Add more comprehensive validation
3. Optimize collection operations
4. Add support for bulk operations 