# Web Interface Documentation

## Overview
The web interface layer provides a user-friendly interface for interacting with the adoption rate chatbot and visualizing adoption rate data. This layer implements a Flask-based web application with interactive charts, chat functionality, and data visualization components.

## Directory Structure
```
src/ui/
├── __init__.py
├── app.py
├── api_endpoints.py
├── templates/
│   ├── base.html
│   ├── index.html
│   └── components/
│       ├── chat.html
│       └── chart.html
├── static/
│   ├── css/
│   │   ├── styles.css
│   │   └── components/
│   │       ├── chat.css
│   │       └── chart.css
│   ├── js/
│   │   ├── chat.js
│   │   └── chart.js
│   └── img/
└── utils/
    ├── session.py
    └── response_formatter.py
```

## Components

### 1. Flask Application (`app.py`)

#### Purpose
Implements the main web application and routing logic.

#### Key Features
- Route handling
- Session management
- Error handling
- API integration
- Response formatting

#### Core Classes and Methods

##### FlaskApp
```python
class FlaskApp:
    def __init__(
        config: Dict = None
    ):
        """
        Initializes Flask application
        
        Parameters:
            config: Application config
        """

    def register_routes(
        blueprint: Blueprint
    ) -> None:
        """
        Registers application routes
        
        Parameters:
            blueprint: Route blueprint
        """

    def handle_error(
        error: Exception
    ) -> Response:
        """
        Handles application errors
        
        Parameters:
            error: Error instance
            
        Returns:
            Error response
        """

    def format_response(
        data: Any,
        status: int = 200
    ) -> Response:
        """
        Formats API response
        
        Parameters:
            data: Response data
            status: HTTP status
            
        Returns:
            Formatted response
        """
```

### 2. API Endpoints (`api_endpoints.py`)

#### Purpose
Implements REST API endpoints for data access and chat functionality.

#### Key Features
- Chat endpoints
- Data endpoints
- Authentication
- Request validation
- Response formatting

#### Core Classes and Methods

##### APIEndpoints
```python
class APIEndpoints:
    def chat_message(
        message: str,
        context: Dict = None
    ) -> Dict:
        """
        Handles chat messages
        
        Parameters:
            message: User message
            context: Chat context
            
        Returns:
            Chat response
        """

    def get_chart_data(
        metric: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Gets chart data
        
        Parameters:
            metric: Metric name
            start_date: Start date
            end_date: End date
            
        Returns:
            Chart data
        """

    def get_analytics(
        metric: str,
        analysis_type: str
    ) -> Dict:
        """
        Gets analytics data
        
        Parameters:
            metric: Metric name
            analysis_type: Analysis type
            
        Returns:
            Analytics data
        """

    def update_preferences(
        preferences: Dict
    ) -> Dict:
        """
        Updates user preferences
        
        Parameters:
            preferences: User preferences
            
        Returns:
            Update status
        """
```

### 3. Templates (`templates/`)

#### Purpose
Contains HTML templates for rendering web pages and components.

#### Key Features
- Base template
- Component templates
- Responsive design
- Dynamic content
- Error pages

#### Core Templates

##### Base Template (`base.html`)
```html
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}{% endblock %}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
    {% block styles %}{% endblock %}
</head>
<body>
    <header>
        {% include 'components/header.html' %}
    </header>
    
    <main>
        {% block content %}{% endblock %}
    </main>
    
    <footer>
        {% include 'components/footer.html' %}
    </footer>
    
    {% block scripts %}{% endblock %}
</body>
</html>
```

##### Chat Component (`components/chat.html`)
```html
<div class="chat-container">
    <div class="chat-messages" id="chat-messages">
        {% for message in messages %}
            {% include 'components/message.html' %}
        {% endfor %}
    </div>
    
    <div class="chat-input">
        <input type="text" id="message-input" placeholder="Type your message...">
        <button id="send-button">Send</button>
    </div>
</div>
```

### 4. Static Files (`static/`)

#### Purpose
Contains static assets including CSS styles, JavaScript files, and images.

#### Key Features
- Responsive styles
- Interactive scripts
- Asset optimization
- Component styles
- Image assets

#### Core Scripts

##### Chat Script (`js/chat.js`)
```javascript
class ChatManager {
    constructor(options = {}) {
        this.messageContainer = document.getElementById('chat-messages');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        this.options = options;
        this.initializeEventListeners();
    }
    
    async sendMessage(message) {
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            });
            
            const data = await response.json();
            this.displayMessage(data);
        } catch (error) {
            console.error('Error sending message:', error);
        }
    }
    
    displayMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${message.type}`;
        messageElement.textContent = message.content;
        this.messageContainer.appendChild(messageElement);
        this.scrollToBottom();
    }
    
    scrollToBottom() {
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }
    
    initializeEventListeners() {
        this.sendButton.addEventListener('click', () => {
            const message = this.messageInput.value.trim();
            if (message) {
                this.sendMessage(message);
                this.messageInput.value = '';
            }
        });
        
        this.messageInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                const message = this.messageInput.value.trim();
                if (message) {
                    this.sendMessage(message);
                    this.messageInput.value = '';
                }
            }
        });
    }
}
```

##### Chart Script (`js/chart.js`)
```javascript
class ChartManager {
    constructor(options = {}) {
        this.chartContainer = document.getElementById('chart-container');
        this.options = options;
        this.chart = null;
        this.initialize();
    }
    
    async initialize() {
        try {
            const data = await this.fetchChartData();
            this.createChart(data);
        } catch (error) {
            console.error('Error initializing chart:', error);
        }
    }
    
    async fetchChartData() {
        const response = await fetch('/api/chart-data');
        return await response.json();
    }
    
    createChart(data) {
        this.chart = new Plotly.newPlot(this.chartContainer, data, {
            responsive: true,
            displayModeBar: false
        });
    }
    
    updateChart(data) {
        Plotly.update(this.chartContainer, data);
    }
}
```

## Configuration

### Application Settings
```python
APP_CONFIG = {
    'debug': True,
    'host': '0.0.0.0',
    'port': 5000,
    'secret_key': 'your-secret-key',
    'session_type': 'filesystem'
}
```

### UI Settings
```python
UI_CONFIG = {
    'theme': 'light',
    'chart_colors': ['#1f77b4', '#ff7f0e', '#2ca02c'],
    'date_format': 'YYYY-MM-DD',
    'max_messages': 50
}
```

## Error Handling

### UI Errors
```python
class UIError(Exception):
    """Base class for UI errors"""
    pass

class APIError(UIError):
    """Raised when API request fails"""
    pass

class RenderError(UIError):
    """Raised when template rendering fails"""
    pass
```

## Testing

### Unit Tests
1. Route Tests
   - Endpoint responses
   - Error handling
   - Session management

2. API Tests
   - Request validation
   - Response formatting
   - Error cases

3. UI Component Tests
   - Chat functionality
   - Chart updates
   - User interactions

### Integration Tests
1. End-to-end UI Tests
   - Complete user flows
   - Data visualization
   - Error recovery

## Best Practices

### UI Development
1. Follow responsive design
2. Implement progressive enhancement
3. Optimize performance
4. Handle errors gracefully

### API Design
1. Use consistent endpoints
2. Validate all inputs
3. Format responses properly
4. Document thoroughly

### Security
1. Validate user input
2. Protect against XSS
3. Implement CSRF protection
4. Use secure sessions

## Future Improvements

### Planned Enhancements
1. Advanced visualizations
2. Real-time updates
3. Improved mobile support
4. Better error handling

### Technical Debt
1. Optimize asset loading
2. Improve test coverage
3. Enhance documentation
4. Refactor components 