import pyodbc
import logging
from datetime import datetime
from contextlib import contextmanager
from src.config.db_config import get_connection_string

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Connection pool
# This implementation uses a simple connection pool
connection_pool = []
MAX_POOL_SIZE = 5

def get_connection():
    """
    Get a connection from the pool or create a new one if the pool is empty.
    
    Returns:
        pyodbc.Connection: Database connection
    
    Raises:
        Exception: If connection fails
    """
    try:
        connection_string = get_connection_string()
        
        # Check if there are any connections in the pool
        if connection_pool:
            connection = connection_pool.pop()
            try:
                # Test if the connection is still valid
                connection.cursor().execute("SELECT 1")
                logger.info("Using existing connection from pool")
                return connection
            except Exception:
                # Connection is not valid, close it and create a new one
                logger.info("Connection from pool is no longer valid, creating new connection")
                try:
                    connection.close()
                except:
                    pass
        
        # Create a new connection
        connection = pyodbc.connect(connection_string)
        logger.info("Created new database connection")
        return connection
    
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise

def release_connection(connection):
    """
    Release a connection back to the pool or close it if the pool is full.
    
    Args:
        connection (pyodbc.Connection): Database connection to release
    """
    if len(connection_pool) < MAX_POOL_SIZE:
        connection_pool.append(connection)
        logger.info("Released connection back to pool")
    else:
        connection.close()
        logger.info("Closed connection (pool full)")

@contextmanager
def db_connection():
    """
    Context manager for database connections to ensure proper handling and release.
    
    Yields:
        pyodbc.Connection: Database connection
    
    Usage:
        with db_connection() as conn:
            cursor = conn.cursor()
            # Use cursor for database operations
    """
    connection = None
    try:
        connection = get_connection()
        yield connection
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        if connection:
            release_connection(connection) 