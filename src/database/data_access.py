import pandas as pd
import logging
from datetime import datetime, timedelta
from src.database.connection import db_connection

# Configure logging
logger = logging.getLogger(__name__)

class DataAccessLayer:
    """
    Data Access Layer for interacting with stored procedures in the Opus database.
    
    Note: Tenant ID 1388 has been verified to have data available for all metrics.
    """
    
    # Default tenant ID with data (found through exploration)
    DEFAULT_TENANT_ID = 1388
    
    @staticmethod
    def execute_stored_procedure(proc_name, params=None):
        """
        Execute a stored procedure and return the result as a pandas DataFrame.
        
        Args:
            proc_name (str): Name of the stored procedure
            params (dict, optional): Parameters for the stored procedure
        
        Returns:
            pandas.DataFrame: Result of the stored procedure
            
        Raises:
            Exception: If execution fails
        """
        try:
            with db_connection() as conn:
                cursor = conn.cursor()
                
                if params:
                    # Build the parameter string for the stored procedure
                    param_str = ', '.join([f"@{k}=?" for k in params.keys()])
                    call_str = f"{{CALL {proc_name}({param_str})}}"
                    logger.info(f"Executing stored procedure: {proc_name} with parameters")
                    
                    # Execute with parameters
                    cursor.execute(call_str, list(params.values()))
                else:
                    logger.info(f"Executing stored procedure: {proc_name} without parameters")
                    cursor.execute(f"{{CALL {proc_name}}}")
                
                # Fetch all results and convert to DataFrame
                columns = [column[0] for column in cursor.description]
                results = cursor.fetchall()
                
                # Log result count
                row_count = len(results)
                logger.info(f"Retrieved {row_count} rows from {proc_name}")
                
                return pd.DataFrame.from_records(results, columns=columns)
                
        except Exception as e:
            logger.error(f"Error executing stored procedure {proc_name}: {str(e)}")
            raise

    @staticmethod
    def get_overall_adoption_rate(from_date=None, to_date=None, tenant_id=None):
        """
        Get Overall Adoption Rate data from SP_OverallAdoptionRate_DWMY stored procedure.
        
        IMPORTANT: Through testing, we've confirmed that this stored procedure only accepts
        three parameters (FromDate, ToDate, Tenantid) despite earlier assumptions that it
        might accept additional filter parameters.
        
        Args:
            from_date (datetime, optional): Start date for data retrieval (defaults to 1 year ago)
            to_date (datetime, optional): End date for data retrieval (defaults to today)
            tenant_id (int, optional): Tenant ID filter (defaults to DEFAULT_TENANT_ID)
            
        Returns:
            pandas.DataFrame: Overall Adoption Rate data with columns:
            - Date: The date of the metrics
            - DAU: Daily Active Users count
            - DOverallAdoptionRate: Daily Overall Adoption Rate
            - WAU: Weekly Active Users count
            - WOverallAdoptionRate: Weekly Overall Adoption Rate
            - MAU: Monthly Active Users count
            - MOverallAdoptionRate: Monthly Overall Adoption Rate
            - YAU: Yearly Active Users count
            - YOverallAdoptionRate: Yearly Overall Adoption Rate
        """
        # Set default dates if not provided
        if from_date is None:
            from_date = datetime.now() - timedelta(days=365)  # Default to 1 year ago
        if to_date is None:
            to_date = datetime.now()  # Default to today
        
        params = {
            'FromDate': from_date,
            'ToDate': to_date
        }
        
        if tenant_id is not None:
            params['Tenantid'] = tenant_id
        else:
            # Use the default tenant ID with data
            params['Tenantid'] = DataAccessLayer.DEFAULT_TENANT_ID
            
        return DataAccessLayer.execute_stored_procedure('SP_OverallAdoptionRate_DWMY', params)
    
    @staticmethod
    def get_mau(from_date=None, to_date=None, tenant_id=None):
        """
        Get Monthly Active Users (MAU) data from SP_MAU stored procedure.
        
        Args:
            from_date (datetime, optional): Start date for data retrieval (defaults to 1 year ago)
            to_date (datetime, optional): End date for data retrieval (defaults to today)
            tenant_id (int, optional): Tenant ID filter (defaults to DEFAULT_TENANT_ID)
            
        Returns:
            pandas.DataFrame: MAU data with columns:
            - Year_MonthNo: The year and month (format: YYYY-MM)
            - TotalActiveUsers: Count of monthly active users
        """
        # Set default dates if not provided
        if from_date is None:
            from_date = datetime.now() - timedelta(days=365)  # Default to 1 year ago
        if to_date is None:
            to_date = datetime.now()  # Default to today
        
        params = {
            'FromDate': from_date,
            'ToDate': to_date
        }
        
        if tenant_id is not None:
            params['Tenantid'] = tenant_id
        else:
            # Use the default tenant ID with data
            params['Tenantid'] = DataAccessLayer.DEFAULT_TENANT_ID
            
        return DataAccessLayer.execute_stored_procedure('SP_MAU', params)
    
    @staticmethod
    def get_dau(from_date=None, to_date=None, tenant_id=None):
        """
        Get Daily Active Users (DAU) data from SP_DAU stored procedure.
        
        Args:
            from_date (datetime, optional): Start date for data retrieval (defaults to 1 year ago)
            to_date (datetime, optional): End date for data retrieval (defaults to today)
            tenant_id (int, optional): Tenant ID filter (defaults to DEFAULT_TENANT_ID)
            
        Returns:
            pandas.DataFrame: DAU data with columns:
            - Date: The date of the metrics
            - TotalActiveUsers: Count of daily active users
        """
        # Set default dates if not provided
        if from_date is None:
            from_date = datetime.now() - timedelta(days=365)  # Default to 1 year ago
        if to_date is None:
            to_date = datetime.now()  # Default to today
        
        params = {
            'FromDate': from_date,
            'ToDate': to_date
        }
        
        if tenant_id is not None:
            params['Tenantid'] = tenant_id
        else:
            # Use the default tenant ID with data
            params['Tenantid'] = DataAccessLayer.DEFAULT_TENANT_ID
            
        return DataAccessLayer.execute_stored_procedure('SP_DAU', params) 