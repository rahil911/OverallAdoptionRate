# Database Update Request

## Issue Summary
Our application is currently failing due to database structure changes that have occurred since our code was developed. We were previously using tables and stored procedures that no longer exist or reference tables that no longer exist.

## Investigation Findings

We have identified the following changes that are causing our application to fail:

1. **Missing Tables**: Our code references `StagingEventLogs_20250221` which no longer exists.

2. **Modified Stored Procedures**: The stored procedures (`SP_OverallAdoptionRate_DWMY`, `SP_DAU`, `SP_MAU`) reference the missing table.

3. **Schema Changes**: The new tables have slightly different schema, including:
   - Column name `dateTime` instead of `EventDate`
   - Different data types and nullability settings
   - Additional columns added

4. **New Data Location**: Our test tenant's data (ID 1388) is now in these tables:
   - `[StagingEventLogs_demo_20250325]`
   - `[StagingEventLogs_demo_20250324-VM]`
   - `[StagingEventLogs_demo_20250324-4]`
   
   Each table has 426,401 rows for tenant 1388.

## Requested Solution

To maintain compatibility with our existing codebase and ensure continued functionality of our application, we request the following solution:

### Create Views for Code Compatibility
1. Create a view named `StagingEventLogs_20250221` that maps to the current table containing tenant 1388's data:
   ```sql
   CREATE VIEW StagingEventLogs_20250221 AS
   SELECT * FROM [StagingEventLogs_demo_20250325]
   ```

2. Ensure the stored procedures (`SP_OverallAdoptionRate_DWMY`, `SP_DAU`, `SP_MAU`) continue to reference `StagingEventLogs_20250221` and `OpusTenantsUsers_20250227`.

## Code Requirements and Dependencies

Our codebase has been built and extensively tested with specific database dependencies. To ensure stability and avoid extensive code changes, we need to maintain:

1. **Table/View Names**:
   - `StagingEventLogs_20250221`: Primary source for event data
   - `OpusTenantsUsers_20250227`: User-tenant mapping table
   - `CalenderTable`: System table for date operations

2. **Stored Procedures**:
   - `SP_OverallAdoptionRate_DWMY`
   - `SP_DAU`
   - `SP_MAU`
   All procedures require exactly three parameters:
   - @FromDate (datetime)
   - @ToDate (datetime)
   - @Tenantid (int)

3. **Test Data Requirements**:
   - Tenant ID 1388 must be maintained as our test tenant
   - Data range coverage: 2022-09-06 to present
   - Minimum data volume requirements:
     - Overall Adoption Rate: 731 rows
     - MAU: 25 rows
     - DAU: 579 rows

4. **Column Requirements**:
   The code handles both `dateTime` and `datetime` column names, but requires these essential fields:
   - Event data: dateTime/datetime, userId, tenantId
   - User data: UserId, TenantId, Status

Please maintain these exact names and structures as they serve as the source of truth for our application. Any changes to these would require significant code updates across multiple components.

## Technical Contact
Rahil Harihar
Whatsapp: +91 9902588982

Thank you for your assistance in resolving this issue. 