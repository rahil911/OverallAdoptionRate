"""
Flask application for Overall Adoption Rate Chatbot UI

This module provides a web interface for interacting with the adoption rate chatbot.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS

from src.chatbot import create_chatbot
from src.ui.api_endpoints import api_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
    static_folder='static', 
    template_folder='templates'
)
CORS(app)

# Register API blueprint
app.register_blueprint(api_bp, url_prefix='/api')

# Set a secret key for session management
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'adoption-rate-chatbot-secret')

# Configure session lifetime
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

# Create chatbot instance
chatbot = create_chatbot()

@app.route('/')
def index():
    """Render the main chatbot interface"""
    # Initialize or reset session data
    session['conversation_id'] = datetime.now().strftime('%Y%m%d%H%M%S')
    session['message_history'] = []
    
    return render_template('index.html')

@app.route('/api/message', methods=['POST'])
def process_message():
    """Process a message from the chat interface"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Process message with chatbot
        response = chatbot.process_query(user_message)
        
        # Update session message history
        if 'message_history' not in session:
            session['message_history'] = []
        
        session['message_history'].append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        session['message_history'].append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Return response
        return jsonify({
            'response': response,
            'conversation_id': session.get('conversation_id', '')
        })
    
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation_history')
def get_conversation_history():
    """Get the current conversation history"""
    history = session.get('message_history', [])
    return jsonify({'history': history})

@app.route('/api/reset_conversation', methods=['POST'])
def reset_conversation():
    """Reset the conversation history"""
    session['conversation_id'] = datetime.now().strftime('%Y%m%d%H%M%S')
    session['message_history'] = []
    
    # Also reset the chatbot's internal message history
    global chatbot
    chatbot = create_chatbot()
    
    return jsonify({'status': 'success'})

def print_startup_message(host, port):
    """Print a helpful startup message for users"""
    message = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                    OVERALL ADOPTION RATE CHATBOT                           ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

The application is running at:
✓ Local:   http://{host}:{port}/
{f"✓ Network: http://0.0.0.0:{port}/" if host == '0.0.0.0' else ""}

You can ask questions about adoption rate trends, such as:
- "What was our monthly adoption rate in March 2024?"
- "Why did our adoption rate drop last quarter?"
- "Predict our adoption rate for the next 6 months"
- "What actions can we take to improve our adoption rate?"

Press CTRL+C to quit
"""
    print(message)

if __name__ == '__main__':
    host = '0.0.0.0' if os.getenv('ALLOW_EXTERNAL_ACCESS', 'false').lower() == 'true' else '127.0.0.1'
    port = int(os.getenv('PORT', '5000'))
    
    # Print startup message
    print_startup_message(host, port)
    
    app.run(debug=True, host=host, port=port) 