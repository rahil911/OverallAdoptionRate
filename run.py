#!/usr/bin/env python3
"""
Run script for Overall Adoption Rate Chatbot

This script sets up the necessary environment variables and starts the Flask application.
"""

import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for running the application"""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Check for required environment variables
    required_vars = [
        'OPENAI_API_KEY',
        'DB_SERVER',
        'DB_NAME',
        'DB_USER',
        'DB_PASSWORD'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in a .env file or in your environment")
        
        # Set default values for testing if not in production
        if os.getenv('ENVIRONMENT') != 'production':
            logger.warning("Setting default values for testing (NOT RECOMMENDED FOR PRODUCTION)")
            
            if 'OPENAI_API_KEY' in missing_vars and os.getenv('ANTHROPIC_API_KEY'):
                logger.info("Using Anthropic API instead of OpenAI")
            
            for var in missing_vars:
                if var == 'DB_SERVER' and not os.getenv('DB_SERVER'):
                    os.environ['DB_SERVER'] = 'localhost'
                    logger.warning("Set DB_SERVER to default: localhost")
                    
                if var == 'DB_NAME' and not os.getenv('DB_NAME'):
                    os.environ['DB_NAME'] = 'OpusEventLogs'
                    logger.warning("Set DB_NAME to default: OpusEventLogs")
                    
                if var == 'DB_USER' and not os.getenv('DB_USER'):
                    os.environ['DB_USER'] = 'sa'
                    logger.warning("Set DB_USER to default: sa")
                    
                if var == 'DB_PASSWORD' and not os.getenv('DB_PASSWORD'):
                    os.environ['DB_PASSWORD'] = 'password'
                    logger.warning("Set DB_PASSWORD to default: password")
        else:
            return
    
    # Import and run the Flask application
    try:
        from src.ui.app import app, print_startup_message
        
        # Set default port if not specified
        port = int(os.getenv('PORT', '5000'))
        
        # Set host to 0.0.0.0 to make the app accessible externally
        # but only if explicitly allowed
        host = '0.0.0.0' if os.getenv('ALLOW_EXTERNAL_ACCESS', 'false').lower() == 'true' else '127.0.0.1'
        
        # Print startup message
        print_startup_message(host, port)
        
        logger.info(f"Starting app on {host}:{port}")
        app.run(debug=os.getenv('DEBUG', 'false').lower() == 'true', host=host, port=port)
        
    except ImportError as e:
        logger.error(f"Error importing application: {e}")
    except Exception as e:
        logger.error(f"Error starting application: {e}", exc_info=True)

if __name__ == "__main__":
    main() 