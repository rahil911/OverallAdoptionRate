import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Database configuration
DB_CONFIG = {
    'server': os.getenv('DB_SERVER', 'opusdatdev.database.windows.net'),
    'database': os.getenv('DB_NAME', 'opusdat'),
    'username': os.getenv('DB_USER', 'opusdatdev'),
    'password': os.getenv('DB_PASSWORD', 'Qweasd@123987'),
    'driver': os.getenv('DB_DRIVER', '{ODBC Driver 18 for SQL Server}'),
    'persist_security_info': False,
    'multiple_active_result_sets': True
}

# Connection string template
CONNECTION_STRING_TEMPLATE = (
    'DRIVER={driver};'
    'SERVER={server};'
    'DATABASE={database};'
    'UID={username};'
    'PWD={password};'
    'Persist Security Info={persist_security_info};'
    'MultipleActiveResultSets={multiple_active_result_sets}'
)

def get_connection_string():
    """
    Generate a connection string using the configuration values.
    
    Returns:
        str: Database connection string
    """
    return CONNECTION_STRING_TEMPLATE.format(**DB_CONFIG) 